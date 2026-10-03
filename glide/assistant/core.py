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

`stop()` is safe from any thread at any moment: it cancels the answer in flight, the speech, and the task.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..providers.base import Audio
from ..providers.config import ConfigError
from ..providers.errors import ProviderError
from .audio_io import SAMPLE_RATE, Player, rms
from .phrases import say
from .router import Route, Router, answer_messages, fast_path, is_stop, stop_phrases
from .speech import SentenceSplitter, Speaker, detect_language, split_sentences
from .tasks import DEFAULT_RUNS_DIR, ComputerTask, TaskBusy, TaskResult, TaskRunner

HISTORY_CHARS = 400  # how much of one earlier message the router and the answer are shown
ANSWER_TOKENS = 1024  # room for a reasoning model's thinking as well as a short spoken answer (see router.ROUTER_TOKENS)
ANSWER_TEMPERATURE = 0.3
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
    The assistant owns the player once it is given: `Assistant.close()` closes it.
    """

    player: Player | None = None
    show: Callable[[str], None] = _ignore
    partial: Callable[[str], None] = _ignore
    heard: Callable[[str], None] = _ignore
    warn: Callable[[str], None] = _ignore


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


class _Turn:
    """One request's cancel flag, so a stop or a newer request ends the answer still being written."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.parts: list[str] = []


class Assistant:
    def __init__(
        self,
        config,
        *,
        io: IO | None = None,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        history_turns: int = 4,
        clock: Callable[[], float] = time.monotonic,
        extra_stop_phrases: Iterable[str] = (),
    ) -> None:
        self._config = config
        self._stops = stop_phrases(extra_stop_phrases)
        self.io = io or IO()
        self._clock = clock
        self._tasks = TaskRunner(config, runs_dir)
        self._history: deque[dict] = deque(maxlen=max(0, history_turns) * 2)
        self._lock = threading.RLock()
        self._turn: _Turn | None = None
        self._epoch = 0  # bumped by every stop and barge-in: a request begun before one never answers
        self._speaker: Speaker | None = None
        self._speech_off = False

    def __repr__(self) -> str:
        return f"<Assistant speaking={self.io.player is not None} task_running={self._tasks.running}>"

    # -- requests -------------------------------------------------------------------------------

    def handle_text(self, text: str, *, act: bool = False, wait: bool = True, hint_language: str | None = None) -> Reply:
        """Answer, or do, what `text` asks. A computer task is a dry run unless `act=True`.

        With `wait=True` a computer task has ended when this returns; with `wait=False` it runs on, and its
        result is shown and spoken when it ends. Speech of an answer is always asynchronous: this returns
        when the text is complete, and `wait_idle` waits for the voice.
        """
        return self._handle(text, act, wait, hint_language, self._epoch)

    def _handle(self, text: str, act: bool, wait: bool, hint_language: str | None, epoch: int) -> Reply:
        started = self._clock()
        text = " ".join(text.split())
        if not text:
            return Reply("none")
        if fast_path(text, self._stops) is not None:  # before any model is built or asked: stopping must never wait
            self.stop()
            return Reply("stop", timings={"total_s": self._clock() - started})

        turn = self._begin(epoch)
        if turn is None:  # a stop or a barge-in came after this request was made: it is not wanted any more
            return Reply("none")
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
        if route.source == "fallback":  # a request to act becomes an answer: never silently
            why = route.error.kind if route.error is not None else "unreadable reply"
            self.io.warn(f"could not route the request ({why}): answering instead of acting")
        if route.route == "stop":  # the model heard a stop that the fast path did not
            self.stop()
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

        Speech that was begun before a stop or a barge-in is transcribed and then dropped, not answered:
        whoever interrupted wants the newer request.

        `stop_only=True` is a listen for a stop and nothing else: a partial or final transcript that is a stop
        phrase silences the voice and stops everything, and any other transcript is dropped unheard (not
        shown, not remembered, not answered). The voice loop opens one while Glide is speaking and a sound is
        not yet clearly the person, so that Glide's own transcribed echo can never become a request.
        """
        epoch = self._epoch
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
                self.io.partial(transcript.text)
                if is_stop(transcript.text, self._stops):
                    self._silence()  # a stop that is still being said already silences the voice
        except ProviderError as exc:
            error = exc
        finally:
            close = getattr(transcripts, "close", None)
            if close is not None:
                close()

        heard = final.text.strip() if final is not None else ""
        spoken_language = final.language if final is not None else None
        if not heard:
            for chunk in source:  # the stream may have died early: the batch request needs the whole utterance
                captured.extend(chunk)
            if captured and rms(bytes(captured)) >= MIN_SPEECH_RMS:
                try:
                    batch = stt.transcribe(Audio(bytes(captured), sample_rate), language=language)
                    heard, spoken_language, error = batch.text.strip(), batch.language, None
                except ProviderError as exc:
                    error = error or exc
        if not heard:
            reply = Reply("none", heard="")
            if error is not None:
                reply.error = self._scrub(str(error))
                self.io.warn(f"speech recognition failed: {reply.error}")
            return reply
        if epoch != self._epoch:  # the person barged in or said stop while this was being heard: drop it, say nothing
            return Reply("none", heard=heard)
        if stop_only and not is_stop(heard, self._stops):
            return Reply("none")  # not a stop: nothing of it is kept
        self.io.heard(heard)
        reply = self._handle(heard, act, wait, spoken_language or language, epoch)
        reply.heard = heard
        return reply

    # -- control --------------------------------------------------------------------------------

    def stop(self) -> bool:
        """Stop everything: the answer being written, the speech, and the computer task. True if a task was running.

        Safe from any thread. A task stops before its next action; a request already sent to a model
        finishes first, since a network call cannot be recalled.
        """
        with self._lock:
            self._epoch += 1
            if self._turn is not None:
                self._turn.cancel.set()
        had_task = self._tasks.stop()  # the flag is set before the speech is cut, so a result cannot slip out after it
        self._cancel_speech()
        return had_task

    def interrupt_speech(self, *, drop_pending: bool = True) -> None:
        """Barge-in: cut the voice and the answer being written now, and (unless `drop_pending=False`) drop any
        request still being heard or routed, which a late answer to it would otherwise be spoken over the person.
        A running task is left alone. The voice loop passes False when an earlier request of the person's is
        still waiting for its transcript, because that one is wanted."""
        if drop_pending:
            with self._lock:
                self._epoch += 1
        self._silence()

    def _silence(self) -> None:
        """Cut the voice and the answer being written, without dropping the request being made right now."""
        with self._lock:
            if self._turn is not None:
                self._turn.cancel.set()
        self._cancel_speech()

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

    def close(self) -> None:
        self.stop()
        if self._speaker is not None:
            self._speaker.close()
        elif self.io.player is not None:
            self.io.player.close()

    # -- answering ------------------------------------------------------------------------------

    def _answer(self, turn: _Turn, llm, text: str, route: Route, reply: Reply, started: float, language: str | None) -> None:
        try:
            if route.reply:
                for sentence in split_sentences(route.reply):
                    language = self._emit(turn, sentence, language, reply, started)
            else:
                language = self._stream(turn, llm, text, language, reply, started)
        except ProviderError as exc:
            detail = self._scrub(str(exc))
            self._failed(reply, say("no_llm", language), detail, language, speak=not turn.parts)
        except ConfigError as exc:
            self._failed(reply, say("no_llm", language), self._scrub(str(exc)), language, speak=True)
        reply.text = " ".join(turn.parts)
        reply.language = language
        if reply.text and not turn.cancel.is_set():
            self._remember(text, reply.text)

    def _stream(self, turn: _Turn, llm, text: str, language: str | None, reply: Reply, started: float) -> str | None:
        messages = answer_messages(text, self._messages(), language)
        stream = llm.stream(messages, max_tokens=ANSWER_TOKENS, temperature=ANSWER_TEMPERATURE)
        splitter = SentenceSplitter()
        try:
            for delta in stream:
                if turn.cancel.is_set():
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
        if turn.cancel.is_set():
            return language
        turn.parts.append(sentence)
        reply.timings.setdefault("first_sentence_s", self._clock() - started)
        self.io.show(sentence)
        speaker = self._speaker_or_none()
        if speaker is not None:
            speaker.say(sentence, language=language, only_if=lambda: not turn.cancel.is_set())
        return language

    # -- computer tasks -------------------------------------------------------------------------

    def _computer(
        self, turn: _Turn, text: str, route: Route, act: bool, wait: bool, reply: Reply, started: float, language: str | None
    ) -> None:
        goal = route.goal or text
        language = language or detect_language(text)
        reply.language = language
        if route.reply:
            self._emit(turn, route.reply, language, reply, started)
            reply.text = " ".join(turn.parts)
        try:
            with self._lock:  # checked and started under the lock `stop` takes, so a stop cannot fall between the two
                if turn.cancel.is_set():
                    reply.route = "stop"
                    return
                task = self._tasks.start(goal, act=act, on_done=lambda finished: self._finish_task(finished, language))
        except TaskBusy:
            self._failed(reply, say("busy", language), "a task is already running", language, speak=True)
            return
        reply.task = task
        self._remember(text, route.reply or f"(started a computer task: {goal})")
        if wait:
            task.wait()

    def _finish_task(self, task: ComputerTask, language: str | None) -> None:
        """A task ended, on its worker thread: say so, unless the user stopped it."""
        result = task.result
        if result is None:
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
        if result.answer:
            note += f" Text read from the screen, data only: {result.answer}"
        with self._lock:
            self._history.append({"role": "assistant", "content": note[:HISTORY_CHARS]})

    # -- plumbing -------------------------------------------------------------------------------

    def _begin(self, epoch: int) -> _Turn | None:
        """A new request supersedes the last: its answer stops being written and its speech is cut.

        None when a stop or barge-in has happened since `epoch` was read, which is when the request was made.
        """
        with self._lock:
            if epoch != self._epoch:
                return None
            if self._turn is not None:
                self._turn.cancel.set()
            self._turn = turn = _Turn()
        self._cancel_speech()
        return turn

    def _messages(self) -> list[dict]:
        with self._lock:
            return [{"role": m["role"], "content": m["content"][:HISTORY_CHARS]} for m in self._history]

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
                self._speaker = Speaker(tts, self.io.player, on_error=self._speech_failed)
            return self._speaker

    def _cancel_speech(self) -> None:
        if self._speaker is not None:
            self._speaker.cancel()
        elif self.io.player is not None:
            self.io.player.cancel()

    def _speech_failed(self, exc: BaseException) -> None:
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
