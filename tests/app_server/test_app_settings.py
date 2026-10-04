"""Settings over a real GlideConfig (with fake adapters): names from the file, no key anywhere, a closed set of changes, all or nothing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from app_harness import connect, wait_until
from test_app_bridge import Rig
from test_config import Builders

from glide.app_server.bridge import AppBridge
from glide.app_server.server import AppServer, Limits
from glide.providers.config import GlideConfig

TOML = """
[providers.openai]
api_key_env = "ZETA_API_KEY"

[llm.fast]
chain = ["openai:zeta-model-1", "gemini:zeta-model-2"]

[stt]
chain = [{ provider = "elevenlabs", model = "stt-zeta" }, "openai:stt-omega"]

[tts]
chain = ["macos_say"]

[speech]
silence_ms = 700
language = "yue"
headset = true
"""
KEYS = {"ZETA_API_KEY": "zk-0123456789abcdef-NOT-TO-BE-SENT", "ELEVENLABS_API_KEY": "el-9876543210fedcba-NOT-TO-BE-SENT"}


class FakeLoop:
    def __init__(self, config, settings, io, act, fail=None):
        self.settings, self.act, self.io = settings, act, io
        self.events: list[str] = []
        self.failure = None
        self.assistant = SimpleNamespace(
            io=io,
            pending_question=None,
            task=None,
            busy=False,
            wait_idle=lambda timeout=None: True,
            stop=lambda: self.events.append("stop"),
            interrupt_speech=lambda **kw: self.events.append("interrupt"),
            close=lambda: self.events.append("close"),
            handle_text=lambda text, **kw: self.events.append("text"),
        )

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("loop stopped")

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")


class VoiceFactory:
    def __init__(self):
        self.loops: list[FakeLoop] = []
        self.error: Exception | None = None

    def __call__(self, config, settings, io, act):
        if self.error is not None:
            raise self.error
        loop = FakeLoop(config, settings, io, act)
        self.loops.append(loop)
        return loop


@pytest.fixture
def make(tmp_path):
    rigs: list[Rig] = []

    def build(*, toml: str = TOML, env=None, voice_factory=None, record_content=False):
        config = GlideConfig.from_toml(toml, env=KEYS if env is None else env, builders=Builders().table())
        bridge = AppBridge(config, record_content=record_content, runs_dir=tmp_path / "runs", voice_factory=voice_factory)
        server = AppServer(bridge, core_version="test", limits=Limits(poll_s=0.02), peer_ok=lambda s: True)
        bridge.bind(server)
        client, session = connect(server)
        rig = Rig(bridge=bridge, server=server, client=client, config=config, session=session, tmp=tmp_path)
        rigs.append(rig)
        return rig

    yield build
    for rig in rigs:
        rig.client.close()
        rig.server.stop()
        rig.bridge.close()


def read(rig: Rig) -> tuple[int, dict]:
    ident = f"read{len(rig.frames('settings'))}"
    rig.client.send({"v": 1, "type": "settings_get", "id": ident})
    data = rig.wait_for("settings", lambda f: f.get("reply_to") == ident)["data"]
    return data["revision"], data["settings"]


def role(settings: dict, name: str) -> dict:
    return next(r for r in settings["roles"] if r["role"] == name)


def test_every_role_slot_provider_and_model_name_comes_from_the_file(make):
    rig = make()
    revision, settings = read(rig)
    assert revision == 1
    fast = role(settings, "llm.fast")
    assert [(s["name"], s["provider"], s.get("model")) for s in fast["chain"]] == [
        ("openai:zeta-model-1", "openai", "zeta-model-1"),
        ("gemini:zeta-model-2", "gemini", "zeta-model-2"),
    ]
    stt = role(settings, "stt")
    assert [s["model"] for s in stt["chain"]] == ["stt-zeta", "stt-omega"]
    assert [s["name"] for s in role(settings, "tts")["chain"]] == ["macos_say"]
    assert {r["role"] for r in settings["roles"]} >= {"llm.fast", "llm.smart", "stt", "tts", "classifier"}
    assert settings["voice"] == {
        "hands_free": False,
        "headset": True,
        "silence_ms": 700,
        "language": "yue",
        "silence_ms_range": [200, 2000],
    }
    assert settings["privacy"] == {"record_content": False} and settings["computer"] == {
        "act_enabled": False,
        "engine": "legacy",
        "engines": ["legacy", "structured"],
    }


def test_a_slot_reports_the_name_of_its_key_variable_and_whether_it_is_set_and_nothing_else(make):
    rig = make()
    _, settings = read(rig)
    fast = role(settings, "llm.fast")["chain"]
    assert fast[0]["key_env"] == "ZETA_API_KEY" and fast[0]["key_present"] is True and fast[0]["status"] == "ready"
    assert fast[1]["key_env"] == "GEMINI_API_KEY" and fast[1]["key_present"] is False and fast[1]["status"] == "skipped"
    tts = role(settings, "tts")["chain"][0]
    assert "key_env" not in tts and tts["key_present"] is False and tts["status"] == "ready"  # no key to name


def test_no_key_appears_anywhere_in_anything_the_app_is_sent(make):
    rig = make(record_content=True)
    read(rig)
    rig.settings(("voice.silence_ms", 900))
    rig.settings(("voice.silence_ms", 9))
    rig.settings(("roles.pin", {"role": "llm.fast", "slot": "no-such-slot"}))
    rig.config.pin("llm.fast", "openai:zeta-model-1")
    read(rig)
    text = json.dumps(rig.client.seen)
    for value in KEYS.values():
        assert value not in text
    assert "NOT-TO-BE-SENT" not in text and "api_key" not in text.replace("key_env", "").replace("ZETA_API_KEY", "")


def test_a_resting_slot_is_reported_resting_and_a_pin_is_reported_by_name(make):
    rig = make()
    chain = rig.config.chain("llm.fast")
    rig.config.pin("llm.fast", "openai:zeta-model-1")
    _, settings = read(rig)
    assert role(settings, "llm.fast")["pinned"] == "openai:zeta-model-1"
    chain._health["openai:zeta-model-1"].rest_until = chain._clock() + 60
    _, settings = read(rig)
    assert role(settings, "llm.fast")["chain"][0]["status"] == "resting"


# -- changing ---------------------------------------------------------------------------------------------


def test_a_change_is_applied_and_the_revision_goes_up_and_is_what_the_next_read_reports(make):
    rig = make()
    result = rig.settings(("voice.silence_ms", 900), ("voice.language", "en"), ("voice.headset", False), revision=1)
    assert result == {"ok": True, "revision": 2, "errors": []}
    revision, settings = read(rig)
    assert revision == 2 and settings["voice"]["silence_ms"] == 900 and settings["voice"]["language"] == "en"
    assert settings["voice"]["headset"] is False


def test_a_change_made_against_an_old_revision_is_refused_whole(make):
    rig = make()
    assert rig.settings(("voice.silence_ms", 900), revision=1)["ok"] is True
    result = rig.settings(("voice.silence_ms", 1000), ("computer.act_enabled", True), revision=1)
    assert result["ok"] is False and result["revision"] == 2 and result["errors"][0]["key"] == "revision"
    _, settings = read(rig)
    assert settings["voice"]["silence_ms"] == 900 and settings["computer"]["act_enabled"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("voice.silence_ms", 199),
        ("voice.silence_ms", 2001),
        ("voice.silence_ms", "700"),
        ("voice.silence_ms", True),
        ("voice.hands_free", "yes"),
        ("privacy.record_content", 1),
        ("computer.act_enabled", None),
        ("computer.engine", "turbo"),
        ("computer.engine", True),
        ("voice.language", 5),
        ("voice.language", "not a language"),
        ("roles.pin", "llm.fast"),
        ("roles.pin", {"role": "no.such.role", "slot": None}),
        ("roles.pin", {"role": "llm.fast", "slot": "no-such-slot"}),
        ("roles.pin", {"role": "llm.fast", "slot": 3}),
        ("api_key", "sk-anything"),
        ("providers.openai.api_key_env", "X"),
        ("voice.model", "x"),
    ],
)
def test_a_change_the_core_does_not_accept_is_refused_with_its_key_and_applies_nothing_even_next_to_good_ones(make, key, value):
    rig = make()
    result = rig.settings(("voice.silence_ms", 1500), (key, value), revision=1)
    assert result["ok"] is False and result["revision"] == 1
    assert [e["key"] for e in result["errors"]] == [key]
    assert read(rig)[1]["voice"]["silence_ms"] == 700  # the good change next to it was not applied either


def test_the_engine_starts_where_the_environment_puts_it_and_a_change_reaches_every_task(make, monkeypatch):
    from glide import features

    monkeypatch.delenv("GLIDE_ENGINE", raising=False)
    rig = make()
    assert read(rig)[1]["computer"]["engine"] == "legacy" and features.engine_for(rig.config) == "legacy"
    result = rig.settings(("computer.engine", "structured"), revision=1)
    assert result["ok"] is True and features.engine_for(rig.config) == "structured"
    assert read(rig)[1]["computer"]["engine"] == "structured"
    assert rig.settings(("computer.engine", "legacy"))["ok"] is True and features.engine_for(rig.config) == "legacy"
    monkeypatch.setenv("GLIDE_ENGINE", "structured")
    assert read(make())[1]["computer"]["engine"] == "structured"
    monkeypatch.setenv(
        "GLIDE_ENGINE", "turbo"
    )  # a bad setting does not stop the app: the default shows, doctor and tasks say why
    assert read(make())[1]["computer"]["engine"] == "legacy"


def test_there_is_no_way_to_send_a_key_through_settings(make):
    rig = make()
    for key in ("api_key", "key", "providers.openai.api_key", "roles.key", "voice.key", "env", "ZETA_API_KEY"):
        assert rig.settings((key, "sk-live-0123456789"))["ok"] is False
    assert "sk-live-0123456789" not in json.dumps([f for f in rig.client.seen if f["type"] == "settings"])
    assert rig.config._env["ZETA_API_KEY"] == KEYS["ZETA_API_KEY"]


def test_pinning_and_unpinning_a_role_reaches_the_configuration_and_is_reported(make):
    rig = make()
    result = rig.settings(("roles.pin", {"role": "llm.fast", "slot": "openai:zeta-model-1"}))
    assert result["ok"] is True and rig.config.pinned("llm.fast") == ("openai:zeta-model-1", False)
    assert role(read(rig)[1], "llm.fast")["pinned"] == "openai:zeta-model-1"
    assert rig.settings(("roles.pin", {"role": "llm.fast", "slot": None}))["ok"] is True
    assert rig.config.pinned("llm.fast") is None


def test_a_skipped_slot_cannot_be_pinned(make):
    rig = make()
    result = rig.settings(("roles.pin", {"role": "llm.fast", "slot": "gemini:zeta-model-2"}))
    assert result["ok"] is False and rig.config.pinned("llm.fast") is None


def test_recording_content_is_opt_in_off_by_default_and_follows_the_switch_everywhere(make):
    rig = make()
    assert rig.frames("hello")[0]["data"]["recording_content"] is False
    assert getattr(rig.config, "record_content", False) is False
    assert rig.settings(("privacy.record_content", True))["ok"] is True
    assert rig.config.record_content is True and rig.bridge.recording_content() is True
    assert read(rig)[1]["privacy"] == {"record_content": True}
    rig.bridge._show("now it carries words")
    frame = rig.wait_for("transcript", lambda f: f["data"]["role"] == "assistant")["data"]
    assert frame["text"] == "now it carries words"
    assert rig.settings(("privacy.record_content", False))["ok"] is True
    assert rig.config.record_content is False


def test_settings_are_never_written_to_the_file_or_the_environment(make, tmp_path, monkeypatch):
    rig = make()
    before = dict(rig.config._env)
    rig.settings(("voice.silence_ms", 1000), ("computer.act_enabled", True))
    assert rig.config._env == before
    assert list(tmp_path.glob("*.toml")) == []


# -- hands-free voice -------------------------------------------------------------------------------------


def test_without_a_voice_stack_hands_free_is_refused_with_a_reason_and_nothing_else_changes(make):
    rig = make(voice_factory=None)
    result = rig.settings(("voice.hands_free", True), ("voice.silence_ms", 1200))
    assert result["ok"] is False and result["errors"][0]["key"] == "voice.hands_free"
    assert "not available" in result["errors"][0]["message"]
    assert read(rig)[1]["voice"]["silence_ms"] == 700 and read(rig)[1]["voice"]["hands_free"] is False


def test_hands_free_opens_the_microphone_only_when_asked_and_closes_it_when_asked(make):
    factory = VoiceFactory()
    rig = make(voice_factory=factory)
    assert "voice" in rig.frames("hello")[0]["data"]["capabilities"]
    assert factory.loops == []  # nothing listens until the app says so
    assert rig.settings(("voice.hands_free", True))["ok"] is True
    (loop,) = factory.loops
    assert loop.events == ["start"] and loop.act is False
    assert (loop.settings.headset, loop.settings.silence_ms, loop.settings.language) == (True, 700, "yue")
    assert rig.wait_for("state", lambda f: f["data"]["assistant"] == "listening" and f["data"]["hands_free"] is True) is not None
    assert rig.settings(("voice.hands_free", False))["ok"] is True
    assert loop.events == ["start", "loop stopped", "close"]
    assert (
        rig.wait_for(
            "state",
            lambda f: f["data"]["assistant"] == "idle" and f["data"]["hands_free"] is False and f is rig.frames("state")[-1],
        )
        is not None
    )


def test_a_voice_that_cannot_start_says_why_without_secrets_and_the_setting_stays_off(make):
    factory = VoiceFactory()
    factory.error = RuntimeError(f"no microphone ({KEYS['ZETA_API_KEY']})")
    rig = make(voice_factory=factory)
    result = rig.settings(("voice.hands_free", True))
    assert result["ok"] is False and "no microphone" in result["errors"][0]["message"]
    assert KEYS["ZETA_API_KEY"] not in json.dumps(rig.client.seen)
    assert read(rig)[1]["voice"]["hands_free"] is False and read(rig)[0] == 1


def test_changing_a_voice_option_while_hands_free_restarts_the_loop_with_it(make):
    factory = VoiceFactory()
    rig = make(voice_factory=factory)
    rig.settings(("voice.hands_free", True))
    rig.settings(("voice.headset", False), ("computer.act_enabled", True))
    assert len(factory.loops) == 2 and factory.loops[0].events[-2:] == ["loop stopped", "close"]
    assert factory.loops[1].settings.headset is False and factory.loops[1].act is True
    assert rig.bridge.runtime.assistant is factory.loops[1].assistant  # typed requests go where the voice goes


def test_mute_and_unmute_pause_and_resume_the_loop_and_are_reported(make):
    factory = VoiceFactory()
    rig = make(voice_factory=factory)
    rig.settings(("voice.hands_free", True))
    rig.client.send({"v": 1, "type": "voice_control", "data": {"action": "mute"}})
    assert rig.wait_for("state", lambda f: f["data"]["muted"] is True and f["data"]["assistant"] == "idle") is not None
    rig.client.send({"v": 1, "type": "voice_control", "data": {"action": "unmute"}})
    assert wait_until(lambda: factory.loops[0].events[-2:] == ["pause", "resume"])
    assert (
        rig.wait_for(
            "state",
            lambda f: f["data"]["muted"] is False and f["data"]["assistant"] == "listening" and f is rig.frames("state")[-1],
        )
        is not None
    )


def test_stop_and_interrupt_reach_the_voice_assistant_too(make):
    factory = VoiceFactory()
    rig = make(voice_factory=factory)
    rig.settings(("voice.hands_free", True))
    rig.client.send({"v": 1, "type": "stop"})
    rig.client.send({"v": 1, "type": "interrupt"})
    assert wait_until(lambda: {"stop", "interrupt"} <= set(factory.loops[0].events))


def test_closing_the_bridge_closes_the_voice_loop(make):
    factory = VoiceFactory()
    rig = make(voice_factory=factory)
    rig.settings(("voice.hands_free", True))
    rig.bridge.close()
    assert factory.loops[0].events[-2:] == ["loop stopped", "close"]


def test_a_voice_stack_whose_assistant_has_no_approver_is_refused_not_started(make):
    """Hands-free tasks are gated because the voice stack is given the bridge's IO. If that ever stops being true, fail closed."""

    class NoApprover(VoiceFactory):
        def __call__(self, config, settings, io, act):
            loop = super().__call__(config, settings, io, act)
            loop.assistant.io = SimpleNamespace(approve=None)
            return loop

    factory = NoApprover()
    rig = make(voice_factory=factory)
    result = rig.settings(("voice.hands_free", True))
    assert result["ok"] is False and "approval" in result["errors"][0]["message"]
    assert factory.loops[0].events == ["close"]  # it was never started, and it was closed
    assert read(rig)[1]["voice"]["hands_free"] is False
