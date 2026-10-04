"""The `glide` command, over fake configuration and a fake terminal, microphone and speaker.

Nothing here reads a real key, opens a device, or touches the screen. Where a test needs the real chains
and switch events it builds a real `GlideConfig` from TOML with fake adapters in place of the vendors.
"""

from __future__ import annotations

import io
import os
import queue
import re
import sys
import threading
from types import SimpleNamespace

import pytest
from test_assistant_audio import FakeInput, tone
from test_assistant_fakes import (
    WAIT,
    FakeConfig,
    FakeLLM,
    FakePlayer,
    FakeSTT,
    FakeTTS,
    route_json,
    wait_until,
)

from glide import cli
from glide.assistant.audio_io import AudioUnavailable, Microphone, chunked
from glide.assistant.core import Reply
from glide.computer import runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState
from glide.providers.base import ChatResult, Usage
from glide.providers.chain import SwitchEvent
from glide.providers.config import ConfigError, GlideConfig
from glide.providers.errors import ProviderError

SECRET = "sk-test-SECRET-9876543210"
EOF_MARK = object()


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No .env is read into the environment, no run folder lands in the repository, no stray secret is around."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("glide.computer.config.load_dotenv", lambda path: None)
    for name in list(os.environ):
        if name.upper().endswith(("KEY", "TOKEN", "SECRET")):
            monkeypatch.delenv(name)


class Keys:
    """A terminal: lines come out when the test puts them in, and `EOFError` when it closes.

    With `settle`, a line is released only after the request for the previous one has been taken up and has ended,
    as a person cannot type the next line before the last request has begun. "Taken up" has to be observed, not
    guessed: `cli.LineReader` reads one line ahead on its own thread, so by the time the chat loop has received a
    line, the next one is already being asked for. Counting live request threads at that moment sees none (the loop
    has not started the thread for the line it was just given) and releases the next line at once, so its
    `interrupt_speech` cancelled a request that had not begun. The loop is seen through `reader()`: a line is
    released only once the loop has received the previous one and has come back asking for the next, which it does
    after it has started that request's thread.
    """

    def __init__(self, *lines, settle: bool = False) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.settle = settle
        self.released = 0  # lines given to the reader thread
        self._seen = threading.Condition()
        self._received = 0  # lines the loop has received
        self._asking = False  # the loop is asking for a line, having handled the last one it received
        for line in lines:
            self.queue.put(line)

    def press(self, line: str = "") -> None:
        self.queue.put(line)

    def close(self) -> None:
        self.queue.put(EOF_MARK)

    def reader(self):
        """A `cli.LineReader` that tells these keys when the loop receives a line and when it asks for the next."""
        keys = self

        class Watched(cli.LineReader):
            def get(self, timeout=None):
                with keys._seen:
                    keys._asking = True
                    keys._seen.notify_all()
                line = super().get(timeout)
                with keys._seen:
                    keys._asking = False
                    keys._received += line is not None
                    keys._seen.notify_all()
                return line

        return Watched

    def _loop_has_taken_up(self, lines: int) -> bool:
        with self._seen:
            return self._seen.wait_for(lambda: self._received >= lines and self._asking, WAIT)

    def __call__(self, prompt: str = "") -> str:
        if self.settle:
            assert self._loop_has_taken_up(self.released), "the chat loop never asked for another line"
            wait_until(lambda: not any(t.name in ("glide-turn", "glide-task") and t.is_alive() for t in threading.enumerate()))
        try:
            item = self.queue.get(timeout=WAIT)
        except queue.Empty:
            raise EOFError from None
        if item is EOF_MARK:
            raise EOFError
        if isinstance(item, BaseException):
            raise item
        self.released += 1
        return item


class Terminal:
    """stdout and stderr of a run, readable while it is still going."""

    def __init__(self, monkeypatch) -> None:
        self.out, self.err = io.StringIO(), io.StringIO()
        monkeypatch.setattr(sys, "stdout", self.out)
        monkeypatch.setattr(sys, "stderr", self.err)

    def __contains__(self, text: str) -> bool:
        return text in self.out.getvalue() or text in self.err.getvalue()

    @property
    def all(self) -> str:
        return self.out.getvalue() + self.err.getvalue()


def start(argv):
    """Run `glide <argv>` on a thread, so a test can type at it while it runs."""
    result: dict = {}

    def target():
        try:
            result["code"] = cli.main(argv)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, result


def finish(thread, result) -> int:
    thread.join(WAIT)
    assert not thread.is_alive(), "the command did not exit"
    assert "error" not in result, result.get("error")
    return result["code"]


def run(argv, monkeypatch, config, keys=None, player=None) -> tuple[int, Terminal]:
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: player or FakePlayer())
    if keys is not None:
        monkeypatch.setattr(cli, "_read_line", keys)
        if keys.settle:
            monkeypatch.setattr(cli, "LineReader", keys.reader())
    terminal = Terminal(monkeypatch)
    return cli.main(argv), terminal


def answering(text="Paris.", **kw) -> FakeConfig:
    return FakeConfig(llm=FakeLLM(route=route_json("answer", reply=text)), tts=FakeTTS(), secret=SECRET, **kw)


# -- parsing ----------------------------------------------------------------------------------------


def test_the_commands_and_their_flags_parse():
    parse = cli.build_parser().parse_args
    ask = parse(["ask", "what", "time", "is", "it", "--speak"])
    assert (ask.command, ask.text, ask.speak, ask.act) == ("ask", ["what", "time", "is", "it"], True, False)
    assert parse(["listen", "--auto", "--lang", "yue"]).auto is True
    assert parse(["chat"]).act is False and parse(["chat", "--act"]).act is True
    assert parse(["doctor", "--live"]).live is True and parse(["doctor"]).live is False
    assert parse(["--config", "x.toml", "status"]).config == "x.toml"


def test_act_is_offered_only_to_the_commands_that_can_do_something():
    parser = cli.build_parser()
    for command in ("ask hi", "listen", "chat"):
        assert parser.parse_args([*command.split(), "--act"]).act is True
    for command in ("doctor", "status"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--act"])


@pytest.mark.parametrize("argv", [[], ["ask"], ["frobnicate"]])
def test_a_missing_command_or_request_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args(argv)
    assert caught.value.code == 2


# -- ask --------------------------------------------------------------------------------------------


def test_ask_prints_the_answer_and_never_builds_a_tts_without_speak(monkeypatch):
    config = answering("Paris is the capital.")
    code, terminal = run(["ask", "capital", "of", "France?"], monkeypatch, config)
    assert code == 0 and "Paris is the capital." in terminal.out.getvalue()
    assert config.calls.tts == 0 and config.closed


def test_ask_prints_no_terminal_control_sequence_a_model_or_a_screen_wrote(monkeypatch):
    # PR15-4175491839: the same sweep for the answer, the notices on stderr and a streamed partial transcript
    hostile = "Paris\x1b]52;c;ZXZpbA==\x07\x1b[2J\x9b31m is\x07 it"
    code, terminal = run(["ask", "capital"], monkeypatch, answering(hostile))
    shown = terminal.out.getvalue() + terminal.err.getvalue()
    assert code == 0 and "Paris" in shown and "is it" in shown
    assert not any(c in shown for c in ("\x1b", "\x07", "\x9b"))
    assert cli.clean("a\x1b[31mb\tc\nd\x07") == "ab\tc\nd"  # tabs and newlines of a multi-line summary stay


def test_ask_speak_speaks_waits_for_the_voice_and_closes_the_player(monkeypatch):
    config, player = answering("One. Two."), FakePlayer()
    code, terminal = run(["ask", "--speak", "hi"], monkeypatch, config, player=player)
    assert code == 0
    assert [t for t, _ in config._tts.calls] == ["One.", "Two."]
    assert player.idle_waits >= 1 and player.closed
    assert "One." in terminal.out.getvalue()


def test_ask_speak_without_audio_support_says_so_and_still_answers(monkeypatch):
    config = answering("Still printed.")
    monkeypatch.setattr(cli, "_load", lambda path: config)

    def no_audio(on_error):
        raise AudioUnavailable("audio needs the 'sounddevice' package")

    monkeypatch.setattr(cli, "_player", no_audio)
    terminal = Terminal(monkeypatch)
    assert cli.main(["ask", "--speak", "hi"]) == 0
    assert "speech is off: audio needs the 'sounddevice' package" in terminal.err.getvalue()
    assert "Still printed." in terminal.out.getvalue() and config.calls.tts == 0


def test_ask_timings_go_to_stderr(monkeypatch):
    code, terminal = run(["ask", "--timings", "hi"], monkeypatch, answering())
    assert code == 0 and "timings: route_s=" in terminal.err.getvalue() and "total_s=" in terminal.err.getvalue()


def test_ask_exits_nonzero_when_the_model_cannot_be_reached_and_says_why_without_the_key(monkeypatch):
    error = ProviderError(f"every llm.fast provider failed (alpha: auth {SECRET})", kind="exhausted")
    llm = FakeLLM(route=error)
    llm.stream_error, llm.deltas, llm.stream_error_after = error, ["x"], 0
    config = FakeConfig(llm=llm, secret=SECRET)
    code, terminal = run(["ask", "hello"], monkeypatch, config)
    assert code == 1
    assert "every llm.fast provider failed" in terminal.err.getvalue()
    assert SECRET not in terminal.all


def test_ask_stopped_by_ctrl_c_stops_everything_and_exits_130(monkeypatch):
    llm = FakeLLM(route=KeyboardInterrupt())
    code, terminal = run(["ask", "hello"], monkeypatch, FakeConfig(llm=llm))
    assert code == 130 and "stopped" in terminal.err.getvalue()


def test_a_stop_phrase_given_to_ask_calls_no_model(monkeypatch):
    config = answering()
    code, _ = run(["ask", "stop"], monkeypatch, config)
    assert code == 0 and config.calls.llm == 0


# -- no command acts without --act ------------------------------------------------------------------


def acting_config(**kw) -> FakeConfig:
    llm = FakeLLM(route=route_json("computer", reply="On it.", goal="Open Safari", language="en"))
    return FakeConfig(llm=llm, tts=FakeTTS(), stt=kw.pop("stt", None), classifier=object(), writer=object(), secret=SECRET)


@pytest.fixture
def loop_calls(monkeypatch):
    """Replace the screen-driving loop with a recorder of what it was asked to do. Nothing real runs."""
    calls = []

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        calls.append(cfg)
        return RunState(outcome="dry run")

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    return calls


def drive(command: str, act: bool, monkeypatch):
    """Run one command with a request that means 'do something', and return the code it exited with."""
    flag = ["--act"] if act else []
    config = acting_config(stt=None)
    if command == "ask":
        return run(["ask", *flag, "open", "safari"], monkeypatch, config)[0]
    if command == "chat":
        return run(["chat", *flag], monkeypatch, config, keys=Keys("open safari", EOF_MARK, settle=True))[0]
    config._stt = FakeSTT(final="open safari")
    monkeypatch.setattr(cli, "_microphone", lambda: Microphone(FakeInput(list(chunked(tone(2000, 0.3))))))
    keys = Keys()
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["listen", *flag, "--text-only"])
    assert wait_until(lambda: prompts(terminal) == 1)
    keys.press("")
    assert wait_until(lambda: prompts(terminal) == 2)  # the recording was handled and the prompt is back
    keys.press("q")
    return finish(thread, result)


@pytest.mark.parametrize("command", ["ask", "chat", "listen"])
@pytest.mark.parametrize("act", [False, True])
def test_no_command_acts_on_the_machine_unless_it_was_given_act(command, act, monkeypatch, loop_calls):
    assert drive(command, act, monkeypatch) == 0
    (cfg,) = loop_calls
    assert cfg.act is act
    assert cfg.goal == "Open Safari"


def test_act_mode_prints_a_banner_so_it_is_never_a_surprise(monkeypatch, loop_calls):
    code, terminal = run(["ask", "--act", "open", "safari"], monkeypatch, acting_config())
    assert code == 0 and "ACT MODE" in terminal.err.getvalue()
    code, terminal = run(["ask", "open", "safari"], monkeypatch, acting_config())
    assert "ACT MODE" not in terminal.all


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        ("done", 0),
        ("dry run", 0),
        ("stalled", 0),
        ("provider failure", 1),
        ("generation unavailable", 1),
        ("desktop unavailable", 1),
        ("crashed", 1),
        ("not permitted", 1),
        ("not configured", 1),
    ],
)
def test_ask_exits_nonzero_when_the_task_could_not_run_at_all(monkeypatch, outcome, code):
    monkeypatch.setattr(runner, "run", lambda cfg, ctx_factory, classifier_factory=None, control=None: RunState(outcome=outcome))
    assert run(["ask", "open", "safari"], monkeypatch, acting_config())[0] == code


def test_a_dry_run_reports_what_it_would_have_done_and_that_it_did_nothing(monkeypatch, loop_calls):
    code, terminal = run(["ask", "open", "safari"], monkeypatch, acting_config())
    assert code == 0
    assert "task dry run: Open Safari" in terminal.out.getvalue()


# -- chat -------------------------------------------------------------------------------------------


def test_chat_answers_each_line_and_leaves_at_the_end_of_input(monkeypatch):
    config = answering("Hello.")
    code, terminal = run(["chat"], monkeypatch, config, keys=Keys("hi", "", "again", EOF_MARK, settle=True))
    assert code == 0 and terminal.out.getvalue().count("Hello.") == 2


def test_act_in_chat_is_off_until_toggled_and_toggles_back(monkeypatch, loop_calls):
    lines = ("open safari", "/act", "open safari", "/act", "open safari", EOF_MARK)
    code, terminal = run(["chat"], monkeypatch, acting_config(), keys=Keys(*lines, settle=True))
    assert code == 0
    assert [c.act for c in loop_calls] == [False, True, False]
    assert "ACT MODE" in terminal.out.getvalue() and "act mode off" in terminal.out.getvalue()


def test_chat_pin_and_unpin_go_to_the_configuration(monkeypatch):
    config = answering()
    lines = ("/pin llm.fast openrouter", "/pin stt eleven strict", "/pin", "/pin a b c d", "/unpin llm.fast", "/unpin", EOF_MARK)
    code, terminal = run(["chat"], monkeypatch, config, keys=Keys(*lines))
    assert code == 0
    assert config.pins == [("llm.fast", "openrouter", False), ("stt", "eleven", True)]
    assert config.unpins == ["llm.fast"]
    text = terminal.out.getvalue()
    assert "llm.fast pinned to openrouter-full" in text and "(strict: nothing else is tried)" in text
    assert text.count("usage: /pin") == 2 and "usage: /unpin" in text


def test_a_pin_the_configuration_refuses_is_a_message_not_a_crash(monkeypatch):
    config = answering()
    config.pin = lambda role, name, strict=False: (_ for _ in ()).throw(ConfigError(f"cannot pin {name!r}: usable: a, b"))
    code, terminal = run(["chat"], monkeypatch, config, keys=Keys("/pin llm.fast zzz", EOF_MARK))
    assert code == 0 and "cannot pin 'zzz': usable: a, b" in terminal.out.getvalue()


def test_chat_stop_status_help_and_unknown_commands(monkeypatch):
    config = answering()
    code, terminal = run(["chat"], monkeypatch, config, keys=Keys("/stop", "/status", "/help", "/wat", "/quit", "never reached"))
    text = terminal.out.getvalue()
    assert code == 0 and "stopped (nothing was running)" in text
    assert "llm.fast-slot" in text and "unavailable: no usable llm.smart provider" in text
    assert text.count("/stop  cut speech") == 2  # /help and the unknown command both print the help


class ScriptedLines:
    """A `LineReader` that plays a script on the calling thread: a line, None for end of input, or an exception to raise."""

    def __init__(self, *items) -> None:
        self.items = list(items)

    def __call__(self, read):  # stands in for the LineReader class: `LineReader(read)` returns this object
        return self

    def get(self, timeout=None):
        if not self.items:
            return None
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.mark.parametrize("command", ["chat", "listen"])
def test_ctrl_c_at_the_prompt_stops_and_a_second_one_leaves(monkeypatch, command):
    player = FakePlayer()
    monkeypatch.setattr(cli, "LineReader", ScriptedLines(KeyboardInterrupt(), KeyboardInterrupt()))
    monkeypatch.setattr(cli, "_microphone", lambda: Microphone(FakeInput([])))
    flag = ["--speak"] if command == "chat" else []
    code, terminal = run([command, *flag], monkeypatch, answering(), player=player)
    assert code == 0 and "stopped (Ctrl-C again to leave)" in terminal.out.getvalue()
    assert player.cancels >= 1


def test_ctrl_c_once_and_then_input_carries_on(monkeypatch):
    monkeypatch.setattr(cli, "LineReader", ScriptedLines(KeyboardInterrupt(), "hi", KeyboardInterrupt(), "hi", None))
    code, terminal = run(["chat"], monkeypatch, answering("Hello."))
    assert code == 0 and terminal.out.getvalue().count("stopped (Ctrl-C again to leave)") == 2  # input in between resets it


def test_a_stop_typed_while_the_answer_is_still_being_written_cuts_it_at_once(monkeypatch):
    gate = threading.Event()
    llm = FakeLLM(route=route_json("answer"), deltas=["One. ", "Two. ", "Three. ", "Four."], gates={2: gate})
    config = FakeConfig(llm=llm, tts=FakeTTS(), secret=SECRET)
    player, keys = FakePlayer(), Keys("count to four")
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: player)
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["chat", "--speak"])
    assert wait_until(lambda: config._tts.calls)  # "One." is being spoken; the stream is held back at "Three. "
    before = player.cancels
    keys.press("stop")
    assert wait_until(lambda: player.cancels > before)  # the voice was cut while the model was still mid-answer
    gate.set()
    keys.close()
    assert finish(thread, result) == 0
    assert llm.stream_closed and [t for t, _ in config._tts.calls] == ["One."]
    assert "stopped" in terminal


def test_a_typed_stop_during_chat_aborts_a_running_task(monkeypatch):
    release, started = threading.Event(), threading.Event()
    seen = {}

    def slow(cfg, ctx_factory, classifier_factory=None, control=None):
        started.set()
        seen["stopped"] = release.wait(WAIT)
        return RunState(outcome="aborted (stopped by the user)")

    monkeypatch.setattr(runner, "run", slow)
    keys = Keys("open safari")
    monkeypatch.setattr(cli, "_load", lambda path: acting_config())
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["chat"])
    assert started.wait(WAIT)
    keys.press("stop")  # typed, with no slash: heard on the terminal thread
    assert wait_until(lambda: "stopped" in terminal.out.getvalue())
    release.set()  # the fake loop cannot see the abort hook, so let it return; the task knows it was stopped
    keys.close()
    assert finish(thread, result) == 0
    assert "Stopped." in terminal and "task aborted" not in terminal.out.getvalue()  # a stopped task reports nothing else


# -- clarify: chat has the channel, ask has not ----------------------------------------------------------


def clarifying_config() -> FakeConfig:
    """Asks which file until the history holds the user's answer, then executes."""

    def route(messages):
        if any("report.pdf" in m["content"] for m in messages):
            return route_json("execute", reply="On it.", goal="Delete the file")
        return route_json("clarify", question="Which file do you mean?")

    return FakeConfig(llm=FakeLLM(route=route), tts=FakeTTS(), classifier=object(), writer=object(), secret=SECRET)


def test_chat_puts_the_routers_question_and_takes_the_next_line_as_its_answer(monkeypatch, loop_calls):
    config = clarifying_config()
    keys = Keys("delete it")
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["chat"])
    assert wait_until(lambda: "Which file do you mean?" in terminal.out.getvalue())
    assert loop_calls == []  # nothing is done while the question is open
    keys.press("report.pdf")  # the next line is the ANSWER, not a new request
    assert wait_until(lambda: len(loop_calls) == 1)
    keys.close()
    assert finish(thread, result) == 0
    assert "Clarifications:" in loop_calls[0].goal and "report.pdf" in loop_calls[0].goal and loop_calls[0].route == "execute"
    assert len(config.fast.chat_calls) == 2  # the request, then the same request with the answer; the answer was not routed alone


def test_chat_treats_a_new_request_after_the_question_as_a_new_request(monkeypatch, loop_calls):
    config = clarifying_config()
    keys = Keys("delete it")
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["chat"])
    assert wait_until(lambda: "Which file do you mean?" in terminal.out.getvalue())
    keys.press("stop")  # a stop is a stop: the question is dropped and nothing is done
    assert wait_until(lambda: "stopped" in terminal.out.getvalue())
    keys.close()
    assert finish(thread, result) == 0
    assert loop_calls == []


def test_leaving_chat_while_a_question_is_open_drops_it_instead_of_waiting_for_the_answer(monkeypatch, loop_calls):
    config = clarifying_config()
    keys = Keys("delete it")
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["chat"])
    assert wait_until(lambda: "Which file do you mean?" in terminal.out.getvalue())
    keys.close()  # end of input with the question still open: nothing will ever answer it
    assert finish(thread, result) == 0  # and the command ends now, not when the question's wait runs out
    assert loop_calls == []


def test_ask_has_no_next_line_so_it_says_what_is_needed_and_does_nothing(monkeypatch, loop_calls):
    code, terminal = run(["ask", "delete", "it"], monkeypatch, clarifying_config())
    assert code == 0 and loop_calls == []
    assert "Which file do you mean?" in terminal.out.getvalue()


# -- listen -----------------------------------------------------------------------------------------


def listen_rig(monkeypatch, stt, config=None, speak=False, chunks=None):
    config = config or answering("Four.")
    config._stt = stt
    player = FakePlayer()
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: player)
    monkeypatch.setattr(
        cli, "_microphone", lambda: Microphone(FakeInput(chunks if chunks is not None else list(chunked(tone(2000, 0.3)))))
    )
    keys = Keys()
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["listen", *([] if speak else ["--text-only"])])
    return SimpleNamespace(config=config, player=player, keys=keys, terminal=terminal, thread=thread, result=result)


def prompts(terminal, prompt: str = "listen> ") -> int:
    return terminal.out.getvalue().count(prompt)


def talk(rig):
    """Enter to talk, and wait until that whole request has been handled and the prompt is back."""
    seen = prompts(rig.terminal)
    rig.keys.press("")
    assert wait_until(lambda: prompts(rig.terminal) > seen)


def test_listen_push_to_talk_transcribes_answers_and_goes_back_to_waiting(monkeypatch):
    rig = listen_rig(monkeypatch, FakeSTT(final="what is two plus two", partials=["what is"]))
    talk(rig)
    rig.keys.press("q")
    assert finish(rig.thread, rig.result) == 0
    out = rig.terminal.out.getvalue()
    assert "you said: what is two plus two" in out and "Four." in out
    assert rig.config._stt.heard  # the microphone's chunks went to the transcriber


def test_pressing_enter_to_talk_cuts_the_voice_at_once(monkeypatch):
    rig = listen_rig(monkeypatch, FakeSTT(final="hello"), speak=True)
    talk(rig)
    first = rig.player.cancels
    talk(rig)  # talking again while Glide may still be speaking
    assert rig.player.cancels > first
    rig.keys.press("q")
    assert finish(rig.thread, rig.result) == 0


def test_typed_text_in_listen_is_a_request_and_slash_stop_stops(monkeypatch):
    rig = listen_rig(monkeypatch, FakeSTT(final=""))
    rig.keys.press("capital of France?")
    assert wait_until(lambda: "Four." in rig.terminal)
    rig.keys.press("/stop")
    assert wait_until(lambda: "stopped (nothing was running)" in rig.terminal)
    rig.keys.close()
    assert finish(rig.thread, rig.result) == 0


def test_a_spoken_stop_never_reaches_a_model(monkeypatch):
    rig = listen_rig(monkeypatch, FakeSTT(final="stop"))
    talk(rig)
    rig.keys.press("q")
    assert finish(rig.thread, rig.result) == 0
    assert rig.config.fast.chat_calls == []


def test_nothing_said_is_reported_and_costs_no_model_call(monkeypatch):
    rig = listen_rig(monkeypatch, FakeSTT(final=""), chunks=[bytes(3200)] * 3)
    talk(rig)
    rig.keys.press("q")
    assert finish(rig.thread, rig.result) == 0
    assert "(nothing heard)" in rig.terminal.out.getvalue() and rig.config.fast.chat_calls == []


def test_a_request_that_was_heard_and_then_cut_is_not_reported_as_nothing_heard(monkeypatch):
    # PR9-4175574734: a cancelled request is now Reply("none"), as its docstring says, and was heard all the same
    monkeypatch.setattr(cli.Assistant, "handle_audio", lambda self, chunks, **kw: Reply("none", heard="what is two and two"))
    rig = listen_rig(monkeypatch, FakeSTT(final="what is two and two"))
    talk(rig)
    rig.keys.press("q")
    assert finish(rig.thread, rig.result) == 0
    assert "(nothing heard)" not in rig.terminal.out.getvalue()


def test_listen_auto_ends_the_recording_when_the_speaker_goes_quiet(monkeypatch):
    chunks = list(chunked(tone(2500, 0.4) + tone(0, 3.0)))  # speech, then a long silence the mic would keep delivering
    config = answering("Hi.")
    config._stt = FakeSTT(final="hello")
    backend = FakeInput(chunks)
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_microphone", lambda: Microphone(backend))
    keys = Keys()
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["listen", "--auto", "--text-only"])
    keys.press("")
    assert wait_until(lambda: "Hi." in terminal)
    assert wait_until(lambda: not any(t.name == "glide-enter" and t.is_alive() for t in threading.enumerate()))
    assert len(config._stt.heard) == 12  # four chunks of speech and eight of the 0.8 s hang time, then the endpointer ended it
    assert len(backend.chunks) == len(chunks) - 12  # and the microphone still had audio to give
    keys.press("q")
    assert finish(thread, result) == 0


def test_listen_without_audio_support_exits_with_a_clear_message(monkeypatch):
    def no_mic():
        raise AudioUnavailable("audio needs the 'sounddevice' package")

    monkeypatch.setattr(cli, "_load", lambda path: answering())
    monkeypatch.setattr(cli, "_microphone", no_mic)
    terminal = Terminal(monkeypatch)
    assert cli.main(["listen"]) == 2
    assert "glide listen: audio needs the 'sounddevice' package" in terminal.err.getvalue()


def test_a_microphone_that_fails_mid_recording_is_a_message_and_the_loop_goes_on(monkeypatch):
    class Broken(FakeInput):
        def read(self):
            raise OSError("device unplugged")

    config = answering()
    config._stt = FakeSTT(final="x")
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_microphone", lambda: Microphone(Broken([])))
    keys = Keys()
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["listen", "--text-only"])
    assert wait_until(lambda: prompts(terminal) == 1)
    keys.press("")
    assert wait_until(lambda: "OSError: device unplugged" in terminal)
    assert wait_until(lambda: prompts(terminal) == 2)
    keys.press("q")
    assert finish(thread, result) == 0


def test_enter_while_an_answer_is_still_being_written_cuts_it_and_starts_listening(monkeypatch):
    gate = threading.Event()
    llm = FakeLLM(route=route_json("answer"), deltas=["One. ", "Two. ", "Three. ", "Four."], gates={2: gate})
    config = FakeConfig(llm=llm, tts=FakeTTS(), stt=FakeSTT(final="count to four"), secret=SECRET)
    player, keys = FakePlayer(), Keys()
    monkeypatch.setattr(cli, "_load", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: player)
    monkeypatch.setattr(cli, "_microphone", lambda: Microphone(FakeInput(list(chunked(tone(2000, 0.3))))))
    monkeypatch.setattr(cli, "_read_line", keys)
    terminal = Terminal(monkeypatch)
    thread, result = start(["listen"])
    assert wait_until(lambda: prompts(terminal) == 1)
    keys.press("")
    assert wait_until(lambda: config._tts.calls)  # "One." is being spoken; the stream is held back at "Three. "
    before = player.cancels
    keys.press("")  # Enter again, over the top of the answer
    assert wait_until(lambda: player.cancels > before)  # the voice is cut at once, with the first answer still unfinished
    assert wait_until(lambda: terminal.out.getvalue().count("listening...") == 2)
    gate.set()
    assert wait_until(lambda: prompts(terminal) >= 2 and llm.stream_closed)
    keys.close()
    assert finish(thread, result) == 0


# -- status, doctor, switches -----------------------------------------------------------------------

TOML = """
[providers.alpha]
kind = "openai_compat"
base_url = "http://localhost:1/v1"
api_key_env = "ALPHA_API_KEY"

[providers.beta]
kind = "openai_compat"
base_url = "http://localhost:2/v1"
api_key_env = "BETA_API_KEY"

[llm.fast]
chain = ["alpha:m1", "beta:m2"]

[llm.smart]
chain = ["beta:m2"]
"""

FULL_TOML = (
    TOML
    + """
[stt]
chain = ["alpha:listen"]

[tts]
chain = ["macos_say"]

[classifier]
chain = ["llm.fast"]
"""
)


class FakeClient:
    """An adapter for one slot. `script` is what each `chat` does: a ChatResult text, or an exception to raise."""

    def __init__(self, name: str, model: str, script) -> None:
        self.name, self.model, self.script, self.sample_rate = name, model, list(script), 22050

    def __repr__(self) -> str:
        return f"<FakeClient {self.name}>"

    def chat(self, messages, **kw) -> ChatResult:
        step = self.script[0] if len(self.script) == 1 else self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return ChatResult(step, Usage(), self.name, self.model, 0.0)

    def stream(self, messages, **kw):
        yield "Streamed."


def real_config(toml: str, scripts: dict[str, list], env: dict[str, str] | None = None) -> GlideConfig:
    """A real GlideConfig over fake adapters: `scripts` maps a slot name ("alpha:m1") to what its `chat` does."""

    def llm(spec, model, key, options):
        return FakeClient(f"{spec.name}:{model}", model, scripts.get(f"{spec.name}:{model}", [route_json("answer", reply="Hi.")]))

    def other(spec, model, key, options):
        return FakeClient(spec.name + (f":{model}" if model else ""), model, [""])

    builders = {
        ("llm", "openai_compat"): llm,
        ("stt", "openai_compat"): other,
        ("tts", "macos_say"): other,
        ("classifier", "typesafe"): other,
    }
    env = {"ALPHA_API_KEY": SECRET, "BETA_API_KEY": "sk-test-BETA-0123456789", **(env or {})}
    return GlideConfig.from_toml(toml, env=env, source="test.toml", builders=builders)


def test_format_switch_names_the_job_both_slots_and_the_reason():
    event = SwitchEvent("llm.fast", "openrouter:x", "openai:y", "rate_limit", "openrouter answered 429: slow down")
    assert (
        cli.format_switch(event) == "fallback: llm.fast openrouter:x -> openai:y (rate_limit: openrouter answered 429: slow down)"
    )


def test_format_switch_when_nothing_is_left_and_when_it_is_a_race():
    last = SwitchEvent("stt", "a", None, "auth", "key refused")
    assert cli.format_switch(last) == "fallback: stt a -> nothing left (auth: key refused)"
    race = SwitchEvent("llm.fast", "a", "b", "slow", "no answer in 2.0s")
    assert cli.format_switch(race) == "racing: llm.fast a is slow (no answer in 2.0s), also trying b"
    assert "also trying nothing else" in cli.format_switch(SwitchEvent("llm.fast", "a", None, "slow", "late"))


def test_a_long_reason_is_cut_to_one_line_and_a_key_in_it_is_scrubbed():
    config = FakeConfig(secret=SECRET)
    event = SwitchEvent("llm.fast", "a", "b", "server", f"line one\nline two {SECRET} " + "x" * 400)
    line = cli.format_switch(event, config)
    assert "\n" not in line and SECRET not in line and "***" in line and len(line) < 260 and line.endswith("...)")


def test_redact_removes_the_value_of_any_secret_looking_variable(monkeypatch):
    monkeypatch.setenv("SOME_OTHER_TOKEN", "tok-0123456789abcdef")
    monkeypatch.setenv("SHORT_KEY", "abc")  # too short to be worth hiding: it would mangle ordinary text
    monkeypatch.setenv("PLAIN_NAME", "plain-value-0123456789")  # not named like a secret: left alone
    assert cli.redact("it said tok-0123456789abcdef and abc") == "it said *** and abc"
    assert cli.redact("plain-value-0123456789") == "plain-value-0123456789"


def test_every_switch_is_printed_to_stderr_as_it_happens_and_the_key_in_it_is_not(monkeypatch):
    rejected = ProviderError(f"alpha answered 429: slow down, key {SECRET}", kind="rate_limit")
    config = real_config(TOML, {"alpha:m1": [rejected]})
    monkeypatch.setattr("glide.providers.config.load_config", lambda path: config)
    monkeypatch.setattr(cli, "_player", lambda on_error: FakePlayer())
    terminal = Terminal(monkeypatch)
    assert cli.main(["ask", "hello"]) == 0
    assert "Hi." in terminal.out.getvalue()
    switch = [line for line in terminal.err.getvalue().splitlines() if line.startswith("fallback:")]
    # the router asks the classifier chain first (its stand-in here is not a real classifier: a visible hop to the fast
    # tier), and the fast chain's own failover from alpha to beta is shown once, as it happens
    own = [line for line in switch if line.startswith("fallback: llm.fast alpha:m1 -> beta:m2 (rate_limit:")]
    assert len(own) == 1 and any(line.startswith("fallback: router classifier -> fast_llm") for line in switch)
    assert SECRET not in terminal.all and "***" in own[0]


def test_a_bad_routing_table_is_one_line_and_exit_2(monkeypatch, tmp_path):
    path = tmp_path / "glide.toml"
    path.write_text("[routing]\nmin_confidance = 0.5\n")
    terminal = Terminal(monkeypatch)
    assert cli.main(["--config", str(path), "status"]) == 2
    err = terminal.err.getvalue()
    assert err.startswith("glide: [routing] has an unknown key 'min_confidance'") and err.count("\n") == 1


def test_a_config_that_cannot_be_loaded_is_one_line_and_exit_2(monkeypatch):
    def broken(path):
        raise ConfigError(f"[llm.fast] has an unknown key 'chian' near {SECRET}")

    monkeypatch.setattr(cli, "_load", broken)
    monkeypatch.setenv("TEST_API_KEY", SECRET)
    terminal = Terminal(monkeypatch)
    assert cli.main(["status"]) == 2
    assert terminal.err.getvalue().startswith("glide: [llm.fast] has an unknown key") and SECRET not in terminal.all


def test_status_lists_every_role_its_slots_and_why_one_is_unavailable(monkeypatch):
    config = real_config(TOML, {})
    code, terminal = run(["status"], monkeypatch, config)
    out = terminal.out.getvalue()
    assert code == 0 and "config: test.toml" in out
    for role in ("llm.fast", "llm.smart", "stt", "tts", "classifier"):
        assert role in out
    assert "alpha:m1  calls 0  failures 0" in out
    assert "unavailable: no usable stt provider" in out  # no key for any stt slot
    assert "skipped: ELEVENLABS_API_KEY is not set" in out
    assert "remembered per process" in out
    assert SECRET not in terminal.all


def test_status_shows_pins_resting_slots_last_errors_and_recent_switches(monkeypatch):
    rejected = ProviderError("alpha answered 429", kind="rate_limit", retry_after=30.0)
    config = real_config(TOML, {"alpha:m1": [rejected]})
    config.llm("fast").chat([{"role": "user", "content": "x"}])  # alpha fails, beta answers: a switch is on record
    config.pin("llm.fast", "beta")
    _, terminal = run(["status"], monkeypatch, config)
    out = terminal.out.getvalue()
    assert re.search(r"resting (29\.\d|30\.0)s", out) and "last error: rate_limit" in out and "[pinned]" in out
    assert "fallback: llm.fast alpha:m1 -> beta:m2" in out


def test_doctor_runs_the_real_check_offline_and_exits_zero_when_every_role_is_usable(monkeypatch):
    config = real_config(FULL_TOML, {})
    code, terminal = run(["doctor"], monkeypatch, config)
    out = terminal.out.getvalue()
    assert code == 0 and "config: test.toml" in out and "ROLE" in out and "ready" in out
    assert "live:" not in out


def test_doctor_exits_nonzero_when_a_role_has_no_usable_provider(monkeypatch):
    code, terminal = run(["doctor"], monkeypatch, real_config(TOML, {}))
    assert code == 1 and "error" in terminal.out.getvalue()


def test_doctor_live_is_passed_through_and_nothing_is_sent_without_it(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_doctor", lambda config, live: seen.append(live) or 0)
    run(["doctor"], monkeypatch, FakeConfig())
    run(["doctor", "--live"], monkeypatch, FakeConfig())
    assert seen == [False, True]


def test_a_secret_in_a_status_error_is_never_printed(monkeypatch):
    config = FakeConfig(secret=SECRET)
    config.chain = lambda role: (_ for _ in ()).throw(ConfigError(f"no usable {role} provider; the key {SECRET} was refused"))
    code, terminal = run(["status"], monkeypatch, config)
    assert code == 0 and SECRET not in terminal.all and "key *** was refused" in terminal.out.getvalue()
