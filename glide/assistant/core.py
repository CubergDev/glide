"""The assistant: text or speech in, an answer or a computer task out, and the result spoken.

`Assistant(config, io=IO(...))` is what the command line and any future interface call. It holds no
vendor: the models, speech-to-text and text-to-speech are the facades of a `GlideConfig`
(providers/config.py), built the first time they are needed, so `glide ask` with no `--speak` never
builds a TTS and `glide chat` never builds an STT.

One request, start to finish:

1. A stop phrase is matched before anything else (router.fast_path): no model, no network, instant.
2. One fast-LLM call picks the route (router.py). Short answers and the acknowledgement of a task come
   back in that same call, so they are spoken without a second one.
3. A longer answer is streamed from the fast LLM, cut into sentences as it arrives and handed to the
   speaker one sentence at a time, so speech starts after the first sentence (speech.py).
4. A computer task runs on a worker thread (tasks.py), as a dry run unless the caller passes
   `act=True`, and its result is spoken when it ends.

Interruption is real, not only a dropped result. Each request owns a `RunControl` (glide/computer/control.py),
the one cancel token of the whole path: the providers called for the request (the router, the streamed answer,
speech recognition) run under it, so cancelling it closes their connections and returns the caller at once
(providers/chain.py); a computer task has its own control, which `stop()` cancels the same way; and the voice
has one, which is cut with the speech. A cancel is silent: it is never shown as an error and never spoken.

`stop()` is safe from any thread at any moment: it cancels every request in flight, the speech, and the task.
`interrupt_speech()` is the barge-in hook, called by whatever hears the person start to talk.

A stop that the person SAID or that the router detected is for what was in progress before it. Every request takes a
ticket, in order, when it is made; a stop inside request N cancels the requests with a ticket up to N, the task they
started, and the speech they queued, and leaves a later request alone (a new utterance that began right after the stop
word is answered). The public `stop()` has no request of its own and takes everything made so far.

A front end that answers by some other means than the router (point-to-ask, glide/assistant/point_voice.py) passes
`responder(text, language)`: a request that is not a stop is handed to it instead of being routed, with the same
tickets, cancellation, stop phrases (the configured extra ones too) and speech as every other request. `on_stop()` is
called when a stop phrase was heard, and when a request was cancelled while the responder had it, so whatever the
responder started ends too. `say_aloud` and `cut_voice` are the public way to use the voice without a request.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..computer.control import RunControl, controlled
from ..providers.base import Audio
from ..providers.config import ConfigError
from ..providers.errors import CANCELLED, ProviderError
from .audio_io import SAMPLE_RATE, Player, rms
from .phrases import say
from .router import DATA_CHARS, Route, Router, answer_messages, fast_path, is_stop, screen_data, stop_phrases
from .speech import SentenceSplitter, Speaker, detect_language, split_sentences
from .tasks import DEFAULT_RUNS_DIR, ComputerTask, TaskBusy, TaskResult, TaskRunner

HISTORY_CHARS = 400  # how much of one earlier message the router and the answer are shown
ANSWER_TOKENS = 1024  # room for a reasoning model's thinking as well as a short spoken answer (see router.ROUTER_TOKENS)
ANSWER_TEMPERATURE = 0.3
UNWIND_S = 2.0  # how long a request waits for the task whose question it dropped to end, so that its own task can start
CLOSE_WAIT_S = 5.0  # how long `close()` waits for a stopped task to unwind before it gives up and says so
CLOSE_POLL_S = 0.05  # how often the wait looks at the clock again
MIN_SPEECH_RMS = 150.0  # audio quieter than this overall is silence: an empty transcript is believed, not retried


def _ignore(*_: object) -> None:
    return None


@dataclass
class IO:
    """Where the assistant's words go. Every callback has a quiet default.

    `player` speaks them (None means text only: no TTS is ever built); `show` gets the text a sentence at a
    time as it is decided; `partial` gets the interim transcript while the person is still speaking and
    `heard` the final one, before anything is answered; `warn` gets anything the user should see that is
    not an answer (a provider that failed, speech that is off).
    `approve(goal, act)` is asked before a computer task starts, a dry run too (it looks at the screen), and the task
    starts only if it returns exactly True; anything else, an exception included, is a refusal and nothing is started.
    It runs on the request's thread, so it may wait for the person. None leaves the decision to the caller's own
    `act` choice, as the terminal commands do (`--act` is their approval).
    The assistant owns the player once it is given: `Assistant.close()` closes it.
    """

    player: Player | None = None
    show: Callable[[str], None] = _ignore
    partial: Callable[[str], None] = _ignore
    heard: Callable[[str], None] = _ignore
    warn: Callable[[str], None] = _ignore
    approve: Callable[[str, bool], bool] | None = None


@dataclass
class Reply:
    """What a request came to. `route` is "answer", "computer", "stop", or "none" (nothing usable was said).

    `text` is what was said or shown for an answer, or the acknowledgement for a task (the task's own result
    is on `task.result` once `task.wait()` returns, and is shown and spoken by the assistant meanwhile).
    `error` is set when something failed; the failure has already been shown through `io.warn`.
    """

    route: str
    text: str = ""
    language: str | None = None
    heard: str | None = None  # the transcript, when the request was speech
    task: ComputerTask | None = None
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


STOPPED = "stopped by the user"
SUPERSEDED = "a newer request replaced it"
SPEECH_CUT = "speech cut"


class _Turn:
    """One request, from the moment it is taken up until its answer is done.

    Its `control` is the cancel token for everything done on its behalf. A stop, or a newer request, cancels it:
    the provider calls in flight are closed and return, and nothing the request has not yet said is said.
    `hearing` is true while speech is still being transcribed, so a barge-in that must not drop the person's
    earlier, still unfinished request can tell it from an answer being written.
    """

    def __init__(self, seq: int, *, hearing: bool = False) -> None:
        self.seq = seq  # the ticket it was made with: a stop inside request N is for the turns with seq up to N
        self.control = RunControl()
        self.hearing = hearing
        self.parts: list[str] = []

    @property
    def cancelled(self) -> bool:
        return self.control.cancelled.is_set()


class _Voice:
    """The TTS as the speaker uses it, with the connection of the sentence being made under a control of its own.

    `cut()` cancels that control, which closes the voice provider's connection and releases the thread waiting on
    it, and gives the sentences said afterwards a fresh one. Sentences are made on the speaker's thread, which
    has no control of its own, so the control is made current here, where each sentence's stream is created.
    """

    def __init__(self, tts) -> None:
        self._tts = tts
        self._lock = threading.Lock()
        self._control = RunControl()

    def stream(self, text, **options):
        with self._lock:
            control = self._control
        with controlled(control):
            return self._tts.stream(text, **options)

    def cut(self) -> None:
        with self._lock:
            old, self._control = self._control, RunControl()
        old.cancel(SPEECH_CUT)


class Assistant:
    def __init__(
        self,
        config,
        *,
        io: IO | None = None,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        history_turns: int = 4,
        clock: Callable[[], float] = time.monotonic,
        clarify: bool = False,
        extra_stop_phrases: Iterable[str] = (),
        close_wait_s: float = CLOSE_WAIT_S,
        responder: Callable[[str, str | None], Reply] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._responder = responder
        self._on_stop = on_stop
        self._close_wait_s = close_wait_s
        self._stops = stop_phrases(extra_stop_phrases)
        self.io = io or IO()
        self._clock = clock
        self._tasks = TaskRunner(config, runs_dir)
        self._history: deque[dict] = deque(maxlen=max(0, history_turns) * 2)
        self._lock = threading.RLock()
        self._live: set[_Turn] = set()  # the requests being heard or answered
        self._seq = 0  # the last ticket given to a request
        self._floor = 0  # a request with a ticket up to this was made before a stop or barge-in and never answers
        self._epoch = 0  # how many stops and barge-ins there have been (observed by the voice loop's tests)
        self._task_seq = 0  # the ticket of the request that started the current task
        self._speech_turn: _Turn | None = None  # the newest request that queued speech since the last cut
        self._speaker: Speaker | None = None
        self._voice: _Voice | None = None
        self._speech_off = False
        self._clarify = clarify  # whether a computer task may put a question to the user (see `answer_pending`)

    def __repr__(self) -> str:
        return f"<Assistant speaking={self.io.player is not None} task_running={self._tasks.running}>"

    # -- requests -------------------------------------------------------------------------------

    def handle_text(self, text: str, *, act: bool = False, wait: bool = True, hint_language: str | None = None) -> Reply:
        """Answer, or do, what `text` asks. A computer task is a dry run unless `act=True`.

        With `wait=True` a computer task has ended when this returns; with `wait=False` it runs on, and its
        result is shown and spoken when it ends. Speech of an answer is always asynchronous: this returns
        when the text is complete, and `wait_idle` waits for the voice.

        A request that is cancelled (a stop, or a newer request) while it is being answered returns
        `Reply("none")` and says nothing, not even that something failed.
        """
        return self._handle(text, act, wait, hint_language, self._ticket())

    def _handle(
        self, text: str, act: bool, wait: bool, hint_language: str | None, ticket: int, turn: _Turn | None = None
    ) -> Reply:
        started = self._clock()
        text = " ".join(text.split())
        if not text:
            return Reply("none")
        if fast_path(text, self._stops) is not None:  # before any model is built or asked: stopping must never wait
            self._stop(turn.seq if turn is not None else ticket)
            if self._on_stop is not None:
                self._on_stop()
            return Reply("stop", timings={"total_s": self._clock() - started})

        own = turn is None
        if own:
            turn = self._enter(ticket, hearing=False)
        try:
            if turn is None or not self._begin(turn):  # a stop or a barge-in came for this request after it was made
                return Reply("none")
            with controlled(turn.control):
                if self._responder is not None:
                    return self._delegate(turn, text, hint_language, started)
                return self._respond(turn, text, act, wait, hint_language, started)
        finally:
            if own and turn is not None:
                self._leave(turn)

    def _delegate(self, turn: _Turn, text: str, hint_language: str | None, started: float) -> Reply:
        """Hand a request that is not a stop to the `responder`. Nothing is routed, asked of a model, remembered or started.

        A stop or a barge-in that came for the request while it was being handed over has the responder's work ended
        (`on_stop`) and the request dropped, so what it started is never answered after a stop that was for it.
        """
        reply = self._responder(text, hint_language)
        if turn.cancelled:
            if self._on_stop is not None:
                self._on_stop()
            return Reply("none")
        reply.timings["total_s"] = self._clock() - started
        return reply

    def _respond(self, turn: _Turn, text: str, act: bool, wait: bool, hint_language: str | None, started: float) -> Reply:
        speaker = self._speaker_or_none()
        if speaker is not None:
            speaker.mark()
        reply = Reply("answer")
        try:
            llm = self._config.llm("fast")
        except ConfigError as exc:
            return self._failed(reply, say("no_llm", hint_language), self._scrub(str(exc)), hint_language, speak=True)

        route = Router(llm, clock=self._clock).route(text, self._messages())
        reply.timings["route_s"] = route.latency_s
        if route.source == "fallback" and not turn.cancelled:  # a request to act becomes an answer: never silently
            why = route.error.kind if route.error is not None else "unreadable reply"
            self.io.warn(f"could not route the request ({why}): answering instead of acting")
        if route.route == "stop":  # the model heard a stop that the fast path did not
            self._stop(turn.seq)
            reply.route = "stop"
            return reply
        language = route.language or hint_language
        if route.route == "computer":
            reply.route = "computer"
            self._computer(turn, text, route, act, wait, reply, started, language)
        else:
            self._answer(turn, llm, text, route, reply, started, language)
        reply.timings["total_s"] = self._clock() - started
        if speaker is not None and speaker.first_audio_at is not None:
            reply.timings["first_audio_s"] = speaker.first_audio_at - started
        return reply

    def handle_audio(
        self,
        chunks: Iterable[bytes],
        *,
        sample_rate: int = SAMPLE_RATE,
        language: str | None = None,
        act: bool = False,
        wait: bool = True,
        stop_only: bool = False,
    ) -> Reply:
        """Transcribe speech as it is captured, then handle the transcript as text.

        `chunks` is the microphone: PCM chunks that end when the person is done. The transcript is streamed,
        so most of it is ready when the last chunk arrives. If the stream fails or comes back empty and the
        audio was not silent, the audio that was kept is sent once more as a single request, because a
        short command is the one case a streaming transcriber is known to get wrong.

        Speech that was begun before a stop or a barge-in is cancelled, even in the middle of being transcribed
        (the connection to the transcriber is closed at once), and is dropped, not answered: whoever interrupted
        wants the newer request. Call `interrupt_speech()` BEFORE starting the speech that interrupts.

        `stop_only=True` is a listen for a stop and nothing else: a partial or final transcript that is a stop
        phrase silences the voice and stops everything, and any other transcript is dropped unheard (not
        shown, not remembered, not answered). The voice loop opens one while Glide is speaking and a sound is
        not yet clearly the person, so that Glide's own transcribed echo can never become a request.
        """
        turn = self._enter(self._ticket(), hearing=True)
        if turn is None:
            return Reply("none")
        try:
            with controlled(turn.control):
                return self._hear(turn, chunks, sample_rate, language, act, wait, stop_only)
        finally:
            self._leave(turn)

    def _hear(
        self, turn: _Turn, chunks: Iterable[bytes], sample_rate: int, language: str | None, act: bool, wait: bool, stop_only: bool
    ) -> Reply:
        try:
            stt = self._config.stt()
        except ConfigError as exc:
            return self._failed(Reply("none"), say("no_speech", language), self._scrub(str(exc)), language, speak=False)

        source = iter(chunks)
        captured = bytearray()

        def tee() -> Iterator[bytes]:
            for chunk in source:
                captured.extend(chunk)
                yield chunk

        final = error = None
        transcripts = stt.stream(tee(), sample_rate=sample_rate, language=language)
        try:
            for transcript in transcripts:
                if not transcript.partial:
                    final = transcript
                    continue
                if not stop_only:
                    self.io.partial(transcript.text)  # a probe's interim text, Glide's own echo among it, is never shown
                if is_stop(transcript.text, self._stops):
                    self._silence(turn)  # a stop that is still being said already silences the voice
        except ProviderError as exc:
            error = exc
        finally:
            close = getattr(transcripts, "close", None)
            if close is not None:
                close()

        heard = final.text.strip() if final is not None else ""
        spoken_language = final.language if final is not None else None
        if turn.cancelled:  # stopped or replaced while it was being heard: no fallback request, no warning, no answer
            return Reply("none", heard=heard)
        if not heard:
            for chunk in source:  # the stream may have died early: the batch request needs the whole utterance
                captured.extend(chunk)
            if captured and rms(bytes(captured)) >= MIN_SPEECH_RMS:
                try:
                    batch = stt.transcribe(Audio(bytes(captured), sample_rate), language=language)
                    heard, spoken_language, error = batch.text.strip(), batch.language, None
                except ProviderError as exc:
                    error = error or exc
        if turn.cancelled:
            return Reply("none", heard=heard)
        if not heard:
            reply = Reply("none", heard="")
            if error is not None:
                reply.error = self._scrub(str(error))
                self.io.warn(f"speech recognition failed: {reply.error}")
            return reply
        if stop_only and not is_stop(heard, self._stops):
            return Reply("none")  # not a stop: nothing of it is kept
        self.io.heard(heard)
        reply = self._handle(heard, act, wait, spoken_language or language, turn.seq, turn)
        reply.heard = heard
        return reply

    # -- control --------------------------------------------------------------------------------

    def stop(self) -> bool:
        """Stop everything: every request in flight, the speech, and the computer task. True if a task was running.

        Safe from any thread. The connection of each provider call in flight is closed and its caller returns at
        once; an answer that still arrives is dropped. A task stops before its next action, and the model call it
        is waiting on is closed the same way. An action that was already sent cannot be taken back.
        """
        with self._lock:
            return self._stop(self._seq)

    def _stop(self, upto: int) -> bool:
        """Stop what was made up to ticket `upto`: those requests, the task if one of them started it, and their speech.

        Everything happens under the lock a request takes to begin or to start a task, so one that is made after
        `upto` is either wholly before this (and is not touched) or wholly after it.
        """
        with self._lock:
            self._epoch += 1
            self._floor = max(self._floor, upto)
            self._cancel_turns(STOPPED, upto=upto)
            had_task = self._task_seq <= upto and self._tasks.stop()  # after the turns, so a result cannot slip out
            self._cut_speech(upto)
        return had_task

    def interrupt_speech(self, *, drop_pending: bool = True) -> None:
        """Barge-in: the person began to talk over Glide. Cut the voice and the answer being written now, closing
        the connections that were making them.

        With `drop_pending` (the default) any request still being heard or routed is cancelled as well, so a late
        answer to it can never be spoken over the person; the transcription connection is closed at once. Pass
        False when an earlier request of the person's is still waiting for its transcript: that one is wanted.
        A running task is left alone (`stop()` stops it).

        Call it BEFORE starting the speech that interrupts: a request begun earlier is cancelled by it, one begun
        later is not.
        """
        with self._lock:
            if drop_pending:
                self._epoch += 1
                self._floor = self._seq
                self._cancel_turns(STOPPED, upto=self._seq)
            else:
                self._cancel_turns(STOPPED, upto=self._seq, hearing=False)
            self._cut_speech(self._seq)

    def _silence(self, turn: _Turn) -> None:
        """A stop is being said inside `turn`: cut the voice and the answers begun before it, without dropping a request
        that is still being heard (this one included) or one made after it."""
        with self._lock:
            self._cancel_turns(STOPPED, upto=turn.seq, hearing=False)
            self._cut_speech(turn.seq)

    def cut_voice(self) -> None:
        """Silence the voice now: the sentence being said and the ones queued. Nothing else is touched: no request is
        cancelled (one still being heard or answered goes on) and the stop and barge-in counts do not change.

        Safe from any thread, and a no-op when nothing is being said or there is no player."""
        with self._lock:
            self._cut_speech(self._seq)

    def say_aloud(self, text: str, *, language: str | None = None) -> bool:
        """Read `text` aloud, a sentence at a time, the way an answer is spoken. False when nothing was queued.

        Without a player this does nothing, and no TTS is built (the same rule as every answer). The sentences are
        queued together under the lock a stop takes, so a stop or `cut_voice` sees all of them or none. They are not
        tied to a request: a later stop, barge-in or `cut_voice` silences them; the caller decides whether an answer
        is still wanted before it calls this (an answer that arrived after a stop must not be passed here).
        """
        speaker = self._speaker_or_none()
        if speaker is None:
            return False
        queued = False
        with self._lock:
            for sentence in split_sentences(text):
                queued = speaker.say(sentence, language=language or detect_language(sentence)) or queued
        return queued

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until everything said so far has been played. False if `timeout` ran out first."""
        return self._speaker.wait_idle(timeout) if self._speaker is not None else True

    @property
    def task(self) -> ComputerTask | None:
        """The task most recently started, running or not."""
        return self._tasks.current

    @property
    def busy(self) -> bool:
        return self._tasks.running

    @property
    def pending_question(self) -> str | None:
        """The question a running task has put to the user and is waiting on, or None. Only with `clarify=True`."""
        task = self._tasks.current
        return task.pending_question if task is not None else None

    def answer_pending(self, text: str) -> bool:
        """Give `text` to the task waiting on a question: the only way a question is ever answered. False when none is.

        A request made through `handle_text` or `handle_audio` is never taken for the answer, whatever it says: it
        is a new request, and it drops the question (the task is stopped, and the question is no longer spoken). The
        front end decides which of the two a line of text or an utterance is.
        """
        task = self._tasks.current
        return task.answer(text) if task is not None else False

    def close(self) -> bool:
        """Stop everything, wait for a running task to unwind, and close the voice. True if nothing was left running.

        A task that is stopped finishes its current action before it reports how it ended, and its thread is a daemon:
        returning at once would let the process end in the middle of a write. So `close` waits up to `close_wait_s`
        (on the assistant's clock) for the task to be done and to have reported (`task.result`). If it has not, that
        is said through `io.warn` and False is returned: the last action may still be in flight.
        """
        deadline = self._clock() + self._close_wait_s
        task = self._tasks.current
        waiting = task is not None and task.running
        self.stop()
        unwound = not waiting or self._wait_for(task, deadline)
        if not unwound:
            self.io.warn(
                "a task was still stopping when Glide closed, so its last action may or may not have happened: check the screen"
            )
        if self._speaker is not None:
            self._speaker.close()
        elif self.io.player is not None:
            self.io.player.close()
        return unwound

    def _wait_for(self, task: ComputerTask, deadline: float) -> bool:
        """Whether `task` has finished (and reported) by `deadline` on the assistant's clock. No lock is held while waiting."""
        while not task.wait(CLOSE_POLL_S):
            if self._clock() >= deadline:
                return False
        return True

    # -- answering ------------------------------------------------------------------------------

    def _answer(self, turn: _Turn, llm, text: str, route: Route, reply: Reply, started: float, language: str | None) -> None:
        try:
            if route.reply:
                for sentence in split_sentences(route.reply):
                    language = self._emit(turn, sentence, language, reply, started)
            else:
                language = self._stream(turn, llm, text, language, reply, started)
        except ProviderError as exc:
            if not turn.cancelled:  # an interrupted answer is not a failure: nothing is shown or said about it
                detail = self._scrub(str(exc))
                self._failed(reply, say("no_llm", language), detail, language, speak=not turn.parts)
        except ConfigError as exc:
            self._failed(reply, say("no_llm", language), self._scrub(str(exc)), language, speak=True)
        reply.text = " ".join(turn.parts)
        reply.language = language
        if reply.text and not turn.cancelled:
            self._remember(text, reply.text)

    def _stream(self, turn: _Turn, llm, text: str, language: str | None, reply: Reply, started: float) -> str | None:
        messages = answer_messages(text, self._messages(), language)
        stream = llm.stream(messages, max_tokens=ANSWER_TOKENS, temperature=ANSWER_TEMPERATURE)
        splitter = SentenceSplitter()
        try:
            for delta in stream:
                if turn.cancelled:
                    return language
                reply.timings.setdefault("first_token_s", self._clock() - started)
                for sentence in splitter.feed(delta):
                    language = self._emit(turn, sentence, language, reply, started)
            for sentence in splitter.flush():
                language = self._emit(turn, sentence, language, reply, started)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()  # a cancelled answer gives its connection back now
        return language

    def _emit(self, turn: _Turn, sentence: str, language: str | None, reply: Reply, started: float) -> str:
        """Show a sentence and hand it to the speaker. Returns the language, decided by the first sentence."""
        language = language or detect_language(sentence)
        if turn.cancelled:
            return language
        turn.parts.append(sentence)
        reply.timings.setdefault("first_sentence_s", self._clock() - started)
        self.io.show(sentence)
        speaker = self._speaker_or_none()
        if speaker is not None:
            with self._lock:  # queued and noted together, so a cut sees what it would be cutting
                if speaker.say(sentence, language=language, only_if=lambda: not turn.cancelled) and (
                    self._speech_turn is None or turn.seq >= self._speech_turn.seq
                ):
                    self._speech_turn = turn
        return language

    # -- computer tasks -------------------------------------------------------------------------

    def _computer(
        self, turn: _Turn, text: str, route: Route, act: bool, wait: bool, reply: Reply, started: float, language: str | None
    ) -> None:
        goal = route.goal or text
        language = language or detect_language(text)
        reply.language = language
        if not self._approved(goal, act):
            reply.error = "the task was not approved: nothing was done"
            if not turn.cancelled:  # a stop that ended the question is silent, like every cancel
                self.io.warn(reply.error)
            return
        if route.reply:
            self._emit(turn, route.reply, language, reply, started)
            reply.text = " ".join(turn.parts)
        try:
            with self._lock:  # checked and started under the lock `stop` takes, so a stop cannot fall between the two
                if turn.cancelled:
                    reply.route = "stop"
                    return
                self._task_seq = turn.seq
                task = self._tasks.start(
                    goal,
                    act=act,
                    on_done=lambda finished: self._finish_task(finished, language),
                    on_question=(lambda finished, question: self._ask_user(finished, question, language))
                    if self._clarify
                    else None,
                )
        except TaskBusy:
            self._failed(reply, say("busy", language), "a task is already running", language, speak=True)
            return
        reply.task = task
        self._remember(text, route.reply or f"(started a computer task: {goal})")
        if wait:
            task.wait()

    def _approved(self, goal: str, act: bool) -> bool:
        """Whether the person said yes to this task (see `IO.approve`). Without an approver the caller's choice stands."""
        approve = self.io.approve
        if approve is None:
            return True
        try:
            return approve(goal, act) is True
        except Exception:  # an approver that breaks is a refusal, never a yes
            return False

    def _ask_user(self, task: ComputerTask, question: str, language: str | None) -> None:
        """A task has a question for the user, on the task's thread: show it and say it. The task waits for `answer_pending`."""
        self.io.show(question)
        speaker = self._speaker_or_none()
        if speaker is not None:
            for sentence in split_sentences(question):
                speaker.say(sentence, language=language or detect_language(sentence), only_if=lambda: not task.stop_requested)

    def _finish_task(self, task: ComputerTask, language: str | None) -> None:
        """A task ended, on its worker thread: say so, unless the user stopped it.

        A task that may have left a write half done is always told, stopped or not: shown, and said after any cut the stop
        made (the sentence is queued under the lock a stop takes, so it is not between the stop and its cut).
        """
        result = task.result
        if result is None:
            return
        if result.uncertain:
            self.io.show(result.summary())
            speaker = self._speaker_or_none()
            if speaker is not None:
                with self._lock:
                    speaker.say(result.spoken(language), language=language or detect_language(result.spoken(language)))
            self._remember_result(result)
            return
        if result.stopped:
            self.io.show(say("stopped", language))
            self._remember_result(result)
            return
        self.io.show(result.summary())
        speaker = self._speaker_or_none()
        if speaker is not None:
            for sentence in split_sentences(result.spoken(language)):
                speaker.say(
                    sentence,
                    language=language or detect_language(sentence),
                    only_if=lambda: not task.stop_requested,  # checked under the lock `stop` takes through the speaker
                )
        self._remember_result(result)

    def _remember_result(self, result: TaskResult) -> None:
        """Note the task's end in the history. What was read off the screen is labelled as data, never as an instruction."""
        note = f"(computer task {result.outcome}: {result.goal})"
        entry = {"role": "assistant", "content": note[:HISTORY_CHARS]}
        if result.uncertain:
            entry["content"] += " The last action may or may not have happened."
        elif result.answer and result.answer.strip():
            entry["data"] = " ".join(result.answer.split())[:DATA_CHARS]  # untrusted: wrapped when it is shown (`_messages`)
        with self._lock:
            self._history.append(entry)

    # -- plumbing -------------------------------------------------------------------------------

    def _ticket(self) -> int:
        """The place of a request in the order requests are made. Taken when it is made, before anything is heard."""
        with self._lock:
            self._seq += 1
            return self._seq

    def _enter(self, ticket: int, *, hearing: bool) -> _Turn | None:
        """Take up a request: it can now be cancelled by a stop or a barge-in. None when one has come for it since its
        ticket was taken."""
        with self._lock:
            if ticket <= self._floor:
                return None
            turn = _Turn(ticket, hearing=hearing)
            self._live.add(turn)
            return turn

    def _leave(self, turn: _Turn) -> None:
        with self._lock:
            self._live.discard(turn)

    def _cancel_turns(self, reason: str, *, upto: int, hearing: bool = True, keep: _Turn | None = None) -> None:
        """Cancel the requests in flight with a ticket up to `upto` (those still being heard too, unless `hearing` is
        False). Called with the lock held."""
        for turn in tuple(self._live):
            if turn is not keep and turn.seq <= upto and (hearing or not turn.hearing):
                turn.control.cancel(reason)

    def _begin(self, turn: _Turn) -> bool:
        """A new request supersedes the answers before it: they stop being written and their speech is cut. A question
        a task is waiting on is dropped with them: this request is not its answer. False if `turn` was cancelled first.

        A request still being heard is not touched: the person said it and wants it answered too."""
        dropped = None
        with self._lock:
            if turn.cancelled:
                return False
            turn.hearing = False
            self._cancel_turns(SUPERSEDED, upto=self._seq, hearing=False, keep=turn)
            waiting = self._tasks.current
            if waiting is not None and waiting.pending_question is not None:
                waiting.stop()
                dropped = waiting
            self._cut_speech(self._seq)  # whoever begins last supersedes what is being said, whatever order they were made in
        if dropped is not None:
            dropped.wait(UNWIND_S)  # a correction that is itself a task ("open Safari instead") must find the machine free
        return True

    def _messages(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "role": m["role"],
                    "content": m["content"][:HISTORY_CHARS]
                    + (" Text read from the screen, data only: " + screen_data(m["data"]) if "data" in m else ""),
                }
                for m in self._history
            ]

    def _remember(self, user: str, assistant: str) -> None:
        with self._lock:
            self._history.append({"role": "user", "content": user[:HISTORY_CHARS]})
            self._history.append({"role": "assistant", "content": assistant[:HISTORY_CHARS]})

    def _speaker_or_none(self) -> Speaker | None:
        """The speaker, built on first use. None when the IO has no player or no TTS is usable (said once)."""
        if self.io.player is None:
            return None
        with self._lock:
            if self._speaker is None and not self._speech_off:
                try:
                    tts = self._config.tts()
                except ConfigError as exc:
                    self._speech_off = True
                    self.io.warn(f"speech is off: {self._scrub(str(exc))}")
                    return None
                self._voice = _Voice(tts)
                self._speaker = Speaker(self._voice, self.io.player, on_error=self._speech_failed)
            return self._speaker

    def _cut_speech(self, upto: int) -> None:
        """Silence the speech of the requests up to ticket `upto`. Called with the lock held, with the turns already
        cancelled, so what they queued is dead and what a later request queues comes after the cut.

        When a later request is the one speaking, its own beginning has already cut what was before it (and
        cancelled what could still add to it), so the cut is left out: it would silence a request the stop is not for.
        """
        newest = self._speech_turn
        if newest is not None and newest.seq > upto and not newest.cancelled:
            return
        self._speech_turn = None
        self._cancel_speech()

    def _cancel_speech(self) -> None:
        """Silence now. The lane is marked dead and its queue drained BEFORE the sentence being made is cut: the thread
        that the cut wakes then finds a dead lane and ends, instead of starting the next queued sentence."""
        if self._speaker is not None:
            self._speaker.cancel()
        elif self.io.player is not None:
            self.io.player.cancel()
        if self._voice is not None:
            self._voice.cut()  # the sentence being made: its connection is closed and its thread released

    def _speech_failed(self, exc: BaseException) -> None:
        if isinstance(exc, ProviderError) and exc.kind == CANCELLED:
            return  # the voice was cut on purpose
        self.io.warn(f"speech failed: {self._scrub(str(exc))}")

    def _failed(self, reply: Reply, sentence: str, detail: str, language: str | None, *, speak: bool) -> Reply:
        """Record a failure: shown through `io.warn`, and, when nothing else has been said, told in one sentence."""
        reply.error = detail
        self.io.warn(detail)
        if speak and (speaker := self._speaker_or_none()) is not None:
            speaker.say(sentence, language=language)
        return reply

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return scrub(text) if callable(scrub) else text
