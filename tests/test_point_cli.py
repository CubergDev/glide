"""The point-to-ask command line over a synthetic desktop and fake writers. No screen, no model, no microphone."""

from __future__ import annotations

import _thread
import signal
import threading
from types import SimpleNamespace

import pytest
from point_fakes import Clock, PointTarget, SyntheticDesktop, reply_writer
from test_assistant_fakes import wait_until
from test_pet_core import FakeLoop, PetConfig

from glide.assistant import point_cli
from glide.assistant.point_ask import capture_point
from glide.assistant.point_voice import PointAssistant
from glide.computer.platform_adapter import using
from glide.computer.writer import WriterError
from glide.speech.settings import SpeechSettings as VoiceSettings

SECRET = "selected secret and private provider body"


def config_with(writer):
    return PetConfig(llm=None, writer=writer)


def run(args, *, writer=None, fake=None, **options):
    fake = fake or SyntheticDesktop()
    config = config_with(writer)
    with using(fake):
        code = point_cli.main(["--delay", "0", *args], load=lambda path: config, **options)
    return code, fake, config


def test_the_default_is_a_local_preview_with_no_model_no_files_and_the_text_on_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, fake, config = run([], writer=reply_writer("never used"))
    out = capsys.readouterr()
    assert code == 0 and "label: 錯誤 0007: connection failed" in out.out and "Nothing is sent to a model" in out.err
    assert config.calls.writer == 0  # no writer was even asked for
    assert fake.calls == [("pointer",), ("target", (200.0, 150.0))] and list(tmp_path.iterdir()) == []


HOSTILE = "pwned\x1b]52;c;ZXZpbA==\x07\x1b[2J\x9b31m\x07 label"


def _no_control(text: str) -> bool:
    return not any(c in text for c in ("\x1b", "\x07", "\x9b", "\x9d"))


def test_the_preview_prints_no_terminal_control_sequence_from_the_pointed_item(capsys):
    # PR15-4175491839: an app's accessibility text must not retitle, clear or write the clipboard of the terminal
    desktop = SyntheticDesktop(target=PointTarget("AXStaticText", HOSTILE, value="v\x1b[31m"))
    code, _, _ = run([], writer=reply_writer("never used"), fake=desktop)
    out = capsys.readouterr().out
    assert code == 0 and "pwned" in out and "label" in out and _no_control(out)


def test_a_model_answer_and_a_heard_sentence_print_no_terminal_control_sequence(capsys):
    answer = '{"answer":"Fine \\u001b]0;pwned\\u0007 ok \\u009b31m","uncertain":false}'
    code, _, _ = run(["--allow-model"], writer=reply_writer(answer))
    out = capsys.readouterr()
    assert code == 0 and "Fine" in out.out and "ok" in out.out and _no_control(out.out + out.err)
    point_cli._say(HOSTILE)
    assert _no_control(capsys.readouterr().err)


def test_a_question_goes_to_the_point_answer_with_the_provider_and_what_is_shared_printed_first(capsys):
    code, fake, _ = run(["What is this?", "--allow-model", "--at", "200", "150"], writer=reply_writer())
    out = capsys.readouterr()
    assert code == 0 and out.out.strip() == "This is error 0007. Check the connection."
    assert out.err.index("Answer provider:") < out.err.index("shares your question") < out.err.index("Position the pointer")
    assert [c[0] for c in fake.calls] == ["target"]  # a fixed point: the pointer is not even read


def test_an_uncertain_answer_says_so_on_stderr_only(capsys):
    code, _, _ = run(["--allow-model"], writer=reply_writer('{"answer":"Maybe.","uncertain":true}'))
    out = capsys.readouterr()
    assert code == 0 and out.out.strip() == "Maybe." and "incomplete" in out.err


def test_the_provider_failing_exits_three_and_never_echoes_the_provider_or_the_screen(capsys):
    def generate(request, cancel=None):
        raise WriterError(SECRET)

    code, _, _ = run(["--allow-model"], writer=SimpleNamespace(generate=generate))
    out = capsys.readouterr()
    assert code == 3 and "incomplete answer" in out.err and SECRET not in out.out + out.err and "0007" not in out.out + out.err


def test_a_reply_that_is_not_one_json_object_is_not_shown(capsys):
    code, _, _ = run(["--allow-model"], writer=reply_writer('{"answer":null,"uncertain":false}'))
    out = capsys.readouterr()
    assert code == 3 and "null" not in out.out + out.err and out.out == ""


def test_no_usable_provider_stops_before_the_screen_is_read(capsys):
    code, fake, _ = run(["--allow-model"], writer=None)
    assert code == 2 and "Configure an answer provider" in capsys.readouterr().err and fake.calls == []


def test_a_slot_that_cannot_be_set_up_is_said_to_be_that_and_nothing_is_read(capsys):
    from glide.providers.config import ConfigError

    config = config_with(None)
    config.secret = SECRET

    def broken(timeout=None):
        raise ConfigError(f"[llm.smart] local cannot be set up: bad option {SECRET}")

    config.writer = broken
    fake = SyntheticDesktop()
    with using(fake):
        code = point_cli.main(["--delay", "0", "--allow-model"], load=lambda path: config)
    err = capsys.readouterr().err
    assert code == 2 and "llm.smart" in err and SECRET not in err and "permission" not in err and fake.calls == []


def test_a_text_only_model_refuses_an_image_before_the_screen_is_read(monkeypatch, capsys):
    monkeypatch.setattr(point_cli, "writer_vision", lambda: False)
    code, fake, _ = run(["--allow-model", "--with-image"], writer=reply_writer())
    assert code == 2 and "text-only" in capsys.readouterr().err and fake.calls == []


def test_ctrl_c_during_the_countdown_cancels_before_anything_is_read(monkeypatch, capsys):
    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(point_cli, "_countdown", interrupted)
    fake = SyntheticDesktop()
    with using(fake):
        code = point_cli.main([], load=lambda path: config_with(None))
    assert code == 130 and fake.calls == [] and "cancelled" in capsys.readouterr().err


def test_a_protected_field_or_a_failing_adapter_says_a_fixed_sentence(capsys):
    code, _, _ = run([], fake=SyntheticDesktop(PointTarget("AXTextField", protected=True)))
    assert code == 2 and "Protected fields cannot be read" in capsys.readouterr().err

    class Broken(SyntheticDesktop):
        def point_target(self, point):
            raise RuntimeError(SECRET)

    code, _, _ = run([], fake=Broken())
    out = capsys.readouterr()
    assert code == 2 and "Could not read this point" in out.err and SECRET not in out.err


@pytest.mark.parametrize(
    "args",
    [
        ["--with-image"],
        ["--voice"],
        ["--delay", "nan"],
        ["--radius", "1000"],
        ["--at", "inf", "2"],
        ["--silence-ms", "199"],
        [" "],
    ],
)
def test_invalid_options_make_no_desktop_call_and_load_no_configuration(args):
    fake = SyntheticDesktop()
    with using(fake), pytest.raises(SystemExit) as error:
        point_cli.main(args, load=lambda path: pytest.fail("no configuration for bad options"))
    assert error.value.code == 2 and fake.calls == []


def test_a_bad_configuration_exits_two_without_reading_the_screen(capsys):
    def load(path):
        raise ValueError("the config file x.toml does not exist")

    fake = SyntheticDesktop()
    with using(fake):
        assert point_cli.main(["--delay", "0"], load=load) == 2
    assert "does not exist" in capsys.readouterr().err and fake.calls == []


def test_a_voice_session_binds_the_pin_to_the_assistant_answers_a_spoken_question_and_ends_with_ctrl_c(capsys):
    made = {}

    def factory(config, settings, *, io, act, assistant_factory):
        loop = FakeLoop(assistant_factory(config, io=io))
        made.update(settings=settings, act=act, loop=loop)

        def start():
            loop.calls.append("start")
            loop.assistant.handle_text("what is this", wait=False)  # what VoiceLoop does with a heard turn
            # Nobody presses Ctrl-C in a test: once the answer is in the history, signal the main thread. It is a real
            # signal to that thread (`interrupt_main` only sets a flag, which a main thread already blocked in
            # `finished.wait()` never sees), so the test cannot hang when the machine is slow.
            threading.Thread(
                target=lambda: (
                    wait_until(lambda: loop.assistant._session.history, 30),
                    signal.pthread_kill(threading.main_thread().ident, signal.SIGINT),
                ),
                daemon=True,
            ).start()

        loop.start = start
        return loop

    with using(SyntheticDesktop()):
        code = point_cli.main(
            ["--delay", "0", "--allow-model", "--voice", "--headset", "--silence-ms", "900"],
            voice_factory=factory,
            load=lambda path: config_with(reply_writer()),
        )
    out = capsys.readouterr()
    assert code == 130 and out.out.strip() == "This is error 0007. Check the connection."
    assert made["settings"].headset is True and made["settings"].silence_ms == 900 and made["act"] is False
    # r2 seams 3: the settings are the voice stack's (config.voice), not the providers' view: build_voice reads stop_phrases
    assert isinstance(made["settings"], VoiceSettings) and made["settings"].stop_phrases == ()
    assert isinstance(made["loop"].assistant, PointAssistant) and made["loop"].calls == ["start", "stop"]


def press_ctrl_c() -> None:
    """A real SIGINT to the main thread, which wakes a lock wait; `_thread.interrupt_main` alone does not."""
    if hasattr(signal, "pthread_kill"):
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
    else:
        _thread.interrupt_main()


def voice_run(monkeypatch, on_start, *, watchdog_s=1.0):
    """A voice session whose loop runs `on_start(loop)` when started. A watchdog presses Ctrl-C if the command has not
    stopped the loop by itself, so a command that waits for ever fails the test instead of hanging it.

    The readout cut is a no-op here: these tests are about when the command ends, not about the voice."""
    monkeypatch.setattr(PointAssistant, "cut_voice", lambda self: None)
    made = {}

    def factory(config, settings, *, io, act, assistant_factory):
        loop = FakeLoop(assistant_factory(config, io=io))
        made["loop"] = loop

        def start():
            loop.calls.append("start")
            on_start(loop)
            threading.Thread(
                target=lambda: wait_until(lambda: "stop" in loop.calls, watchdog_s) or press_ctrl_c(), daemon=True
            ).start()

        loop.start = start
        return loop

    with using(SyntheticDesktop()):
        code = point_cli.main(
            ["--delay", "0", "--allow-model", "--voice"], voice_factory=factory, load=lambda path: config_with(reply_writer())
        )
    return code, made["loop"]


def test_a_spoken_stop_ends_the_answer_and_not_the_voice_session(monkeypatch):
    # PR15-4175586924: the session says 'stopped' for the Stop phrase, which used to end the whole command
    code, loop = voice_run(monkeypatch, lambda loop: loop.assistant._session.stop(), watchdog_s=0.3)
    assert code == 130  # only Ctrl-C (here the watchdog's) ended it, and it ended it by the loop being stopped
    assert loop.calls == ["start", "stop"]


def test_a_microphone_that_is_lost_ends_the_voice_session_with_exit_two(monkeypatch):
    # PR15-4175632435: the loop's thread ends on a device fault; nothing else wakes the command
    def lose(loop):
        loop.ended, loop.failure = True, "the microphone was disconnected"

    code, loop = voice_run(monkeypatch, lose)
    assert code == 2 and loop.calls == ["start", "stop"]


def test_a_voice_stack_that_cannot_start_exits_two_with_what_is_missing(capsys):
    def factory(config, settings, **kw):
        raise RuntimeError(f"sounddevice is not installed {SECRET}")

    code, _, _ = run(["--allow-model", "--voice"], writer=reply_writer(), voice_factory=factory)
    err = capsys.readouterr().err
    assert code == 2 and "sounddevice is not installed" in err and "RuntimeError" in err


def test_a_pin_that_expires_ends_a_voice_session_with_exit_two(capsys):
    clock = Clock()

    def factory(config, settings, *, io, act, assistant_factory):
        loop = FakeLoop(assistant_factory(config, io=io))

        def start():
            clock.advance(121)  # the person waited two minutes before speaking
            loop.assistant.handle_text("what is this", wait=False)

        loop.start = start
        return loop

    code, _, _ = run(
        ["--allow-model", "--voice"],
        writer=reply_writer(),
        voice_factory=factory,
        capture=lambda point, **options: capture_point(point, clock=clock, **options),
    )
    assert code == 2 and "expired" in capsys.readouterr().err
