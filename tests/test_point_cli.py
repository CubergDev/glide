"""The point-to-ask command line over a synthetic desktop and fake writers. No screen, no model, no microphone."""

from __future__ import annotations

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
    assert isinstance(made["loop"].assistant, PointAssistant) and made["loop"].calls == ["start", "stop"]


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
