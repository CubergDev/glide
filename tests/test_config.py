"""glide.toml to chains: presets, skipping, pinning, options, policy, the classifier's lifetime. No network, no real keys.

Every test passes its own `env` dict, so neither the keys in the shell running the suite nor the real
~/.config/glide/glide.toml can change a result. Adapters are replaced by fakes through the `builders` hook
unless a test says it uses the real ones (those only construct; nothing is sent).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, constants

from glide.computer.generation import GenerationError, GenerationRequest
from glide.providers import config as config_module
from glide.providers.base import Audio, ChatResult, SpeechAudio, Transcript, Usage
from glide.providers.chain import SwitchEvent
from glide.providers.classifier import ChainedClassifier, ClassifierReply, LLMClassifier, classifier_factory
from glide.providers.config import (
    DEFAULT_TOML,
    PRESETS,
    ROLES,
    ConfigError,
    GlideConfig,
    NoUsableProvider,
    SpeechSettings,
    load_config,
    pin_variable,
)
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.llm import OpenAICompatLLM

EXAMPLE = Path(__file__).resolve().parent.parent / "glide.toml.example"

# Distinctive on purpose: a test that looks for a leak must not be fooled by a short string found elsewhere.
KEYS = {
    "OPENAI_API_KEY": "sk-openai-0123456789abcdef",
    "OPENROUTER_API_KEY": "sk-or-v1-0123456789abcdef",
    "GEMINI_API_KEY": "AIzaGemini0123456789abcdef",
    "DEEPSEEK_API_KEY": "sk-deepseek-0123456789abcdef",
    "ELEVENLABS_API_KEY": "el-0123456789abcdef",
    "TYPESAFE_API_KEY": "ts-0123456789abcdef",
}

FAST = """
[llm.fast]
chain = ["openai:gpt-a", "gemini:gem-b"]
"""


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def reply_for(schema: dict | None) -> str:
    """What a chat model would say to a classifier prompt: the first option of every Choice, most likely."""
    if schema is None:
        return "ok"
    out = {}
    for name, prop in schema["properties"].items():
        options = prop["properties"]["top"]["items"]["properties"]["option"]["enum"]
        top = [{"option": o, "p": 0.9 if i == 0 else 0.1 / max(1, len(options) - 1)} for i, o in enumerate(options[:3])]
        out[name] = {"top": top}
    return json.dumps(out)


class FakeClient:
    """One adapter of any job. It records its calls, in `owner.log` too, and fails when `owner.fail` says so."""

    def __init__(self, owner: Builders, name: str, model: str, key: str):
        self.owner, self.name, self.model, self.key = owner, name, model, key
        self.calls: list[str] = []
        self.closed = False
        self.sample_rate = 16000

    def __repr__(self) -> str:
        return f"FakeClient({self.name!r})"  # a real adapter never puts its key in a repr either

    def _go(self, what: str) -> None:
        self.calls.append(what)
        self.owner.log.append(self.name)
        if self.owner.clock is not None:
            self.owner.clock.advance(0.25)
        error = self.owner.fail.get(self.name)
        if error is not None:
            raise error

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, logprobs=False, timeout=None):
        self._go("chat")
        return ChatResult(
            text=self.owner.text if self.owner.text is not None else reply_for(schema),
            usage=Usage(3, 1),
            provider=self.name,
            model=self.model,
            latency_s=0.0,
        )

    def stream(self, messages, **kw):
        self._go("stream")
        yield "x"

    def transcribe(self, audio, *, language=None, prompt=None, timeout=None):
        self._go("transcribe")
        return Transcript(text="", language=None, provider=self.name, model=self.model, latency_s=0.0)

    def synthesize(self, text, *, voice=None, language=None, timeout=None):
        self._go("synthesize")
        return SpeechAudio(pcm=b"\x01\x00" * 8000, sample_rate=self.sample_rate)

    def system_one(self, state, questions, *, model=None, **options):
        self._go("system_one")
        answers = {}
        for name, question in questions.items():
            first = next(iter(question.criteria))
            answers[name] = ChoiceAnswer(choice=first, confidence=0.9, probabilities={first: 0.9})
        return ClassifierReply(answers=answers, usage=Usage(), model=self.model)

    def close(self) -> None:
        self.closed = True


class Builders:
    """Fake `build_client`s for every job, remembering what each was given."""

    def __init__(self, clock: Clock | None = None):
        self.made: list[SimpleNamespace] = []
        self.fail: dict[str, BaseException] = {}
        self.log: list[str] = []
        self.text: str | None = None
        self.clock = clock

    def build(self, family: str):
        def make(spec, model, key, options):
            name = f"{spec.name}:{model}" if model else spec.name
            client = FakeClient(self, name, model, key)
            self.made.append(SimpleNamespace(family=family, spec=spec, model=model, key=key, options=options, client=client))
            return client

        return make

    def table(self) -> dict:
        return {
            ("llm", "openai_compat"): self.build("llm"),
            ("stt", "elevenlabs"): self.build("stt"),
            ("stt", "openai_compat"): self.build("stt"),
            ("tts", "elevenlabs"): self.build("tts"),
            ("tts", "openai_compat"): self.build("tts"),
            ("tts", "macos_say"): self.build("tts"),
            ("classifier", "typesafe"): self.build("classifier"),
        }

    def client(self, name: str) -> FakeClient:
        (found,) = [m.client for m in self.made if m.client.name == name]
        return found


def make(toml: str = "", env: dict | None = None, builders: Builders | None = None) -> tuple[GlideConfig, Builders]:
    fakes = builders or Builders()
    return GlideConfig.from_toml(toml, env=KEYS if env is None else env, builders=fakes.table()), fakes


def chat(llm, **kw):
    return llm.chat([{"role": "user", "content": "hi"}], **kw)


# -- presets and providers -----------------------------------------------------------------------


def test_presets_are_the_documented_endpoints_and_key_variables():
    expected = {
        "openai": ("openai_compat", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        "openrouter": ("openai_compat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "deepseek": ("openai_compat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
        "gemini": ("openai_compat", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
        "elevenlabs": ("elevenlabs", "", "ELEVENLABS_API_KEY"),
        "typesafe": ("typesafe", "https://api.typesafe.ai", "TYPESAFE_API_KEY"),
        "macos_say": ("macos_say", "", ""),
    }
    assert {n: (s.kind, s.base_url, s.api_key_env) for n, s in PRESETS.items()} == expected
    assert all(s.name == n for n, s in PRESETS.items())


def test_the_typesafe_preset_is_what_the_installed_sdk_defaults_to():
    assert PRESETS["typesafe"].base_url == constants.DEFAULT_BASE_URL
    assert PRESETS["typesafe"].api_key_env == constants.API_KEY_ENV


def test_a_user_only_names_the_key_variable_and_the_preset_does_the_rest():
    cfg, fakes = make(
        '[providers.openai]\napi_key_env = "MY_OPENAI"\n' + FAST,
        env={"MY_OPENAI": "sk-mine-0123456789", "OPENAI_API_KEY": "sk-not-this-one-0123"},
    )
    cfg.llm("fast")
    (made,) = fakes.made
    assert made.spec.base_url == "https://api.openai.com/v1" and made.spec.kind == "openai_compat"
    assert made.key == "sk-mine-0123456789"
    assert [i.state for i in cfg.slots("llm.fast")] == ["ready", "skipped"]


def test_a_new_provider_needs_a_kind_and_changing_a_presets_kind_drops_its_url():
    with pytest.raises(ConfigError, match=r"\[providers.mine\] needs a kind"):
        GlideConfig.from_toml('[providers.mine]\nbase_url = "http://x"', env={})
    cfg = GlideConfig.from_toml('[providers.openai]\nkind = "macos_say"', env={})
    assert cfg.providers["openai"].base_url == "" and cfg.providers["openai"].api_key_env == ""
    cfg = GlideConfig.from_toml('[providers.local]\nkind = "openai_compat"\nbase_url = "http://localhost:11434/v1"', env={})
    assert cfg.providers["local"].api_key_env == ""


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('[providers.x]\nkind = "carrier_pigeon"', "kind must be one of"),
        ('[providers.openai]\napi_key_env = "sk-live-abc123"', "NAME of an environment variable"),
        ('[providers.elevenlabs]\napi_key_env = ""', "needs api_key_env"),
        ("[providers.openai]\nbase_url = 3", "must be strings"),
        ('[providers.openai]\nflavour = "x"', "unknown key 'flavour'"),
        ("[providers.openai]\noptions = 3", "must be a table"),
    ],
)
def test_a_mistake_in_a_provider_table_is_an_error_at_load(toml, message):
    with pytest.raises(ConfigError, match=message):
        GlideConfig.from_toml(toml, env={})


def test_a_key_written_in_the_file_is_refused_and_never_echoed():
    secret = "sk-this-must-not-appear-0123456789"
    with pytest.raises(ConfigError) as caught:
        GlideConfig.from_toml(f'[providers.openai]\napi_key = "{secret}"', env={})
    assert "never written in this file" in str(caught.value) and secret not in str(caught.value)
    with pytest.raises(ConfigError) as caught:
        GlideConfig.from_toml(f'[providers.openai]\napi_key_env = "{secret}"', env={})
    assert secret not in str(caught.value)


# -- chain entries -------------------------------------------------------------------------------


def test_an_entry_splits_at_the_first_colon_only_and_a_model_is_optional_for_say_and_typesafe():
    toml = """
    [llm.fast]
    chain = ["openrouter:meta/llama-3:free"]
    [tts]
    chain = ["macos_say"]
    [classifier]
    chain = ["typesafe"]
    """
    cfg, fakes = make(toml)
    assert cfg.chain("llm.fast").names == ["openrouter:meta/llama-3:free"]
    assert fakes.made[0].model == "meta/llama-3:free"
    assert cfg.chain("tts").names == ["macos_say"]
    assert cfg.chain("classifier").names == ["typesafe"]


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('[llm.fast]\nchain = ["nope:gpt"]', "no provider named 'nope'"),
        ('[llm.fast]\nchain = ["openai"]', "needs a model"),
        ('[llm.fast]\nchain = ["typesafe:jev-latest"]', "cannot serve llm.fast"),
        ('[llm.fast]\nchain = ["macos_say"]', "cannot serve llm.fast"),
        ('[stt]\nchain = ["macos_say"]', "cannot serve stt"),
        ('[tts]\nchain = ["typesafe:x"]', "cannot serve tts"),
        ("[llm.fast]\nchain = []", "needs chain"),
        ("[llm.fast]", "needs chain"),
        ("[llm.fast]\nchain = [3]", "must be a table"),
        ('[llm.fast]\nchain = ["llm.smart"]', "only appear in the classifier chain"),
        ('[stt]\nchain = ["llm.fast"]', "only appear in the classifier chain"),
        ('[classifier]\nchain = ["stt"]', "only appear in the classifier chain"),
        ('[llm.fast]\nchain = [{provider = "openai", model = "m", flavour = 1}]', "unknown key 'flavour'"),
        ('[llm.fast]\nchain = [{model = "m"}]', "provider must be a non-empty string"),
        ('[llm.fast]\nchain = ["openai:a", "openai:a"]', "lists 'openai:a' twice"),
        ('[llm.fast]\nchain = ["openai:a"]\nhedge_after = 2', "unknown key 'hedge_after'"),
        ('[llm.medium]\nchain = ["openai:a"]', "is not a role"),
    ],
)
def test_a_mistake_in_a_chain_is_an_error_at_load_naming_it(toml, message):
    with pytest.raises(ConfigError, match=message):
        GlideConfig.from_toml(toml, env=KEYS)


def test_the_same_model_twice_is_allowed_under_distinct_names_and_each_gets_its_own_options():
    toml = """
    [llm.fast]
    chain = [
      { provider = "openai", model = "gpt-a", options = { reasoning_effort = "low" } },
      { provider = "openai", model = "gpt-a", name = "openai:gpt-a:high", options = { reasoning_effort = "high" } },
    ]
    """
    cfg, fakes = make(toml)
    assert cfg.chain("llm.fast").names == ["openai:gpt-a", "openai:gpt-a:high"]
    assert [m.options["reasoning_effort"] for m in fakes.made] == ["low", "high"]


def test_an_unknown_top_level_table_is_a_warning_not_an_error(caplog):
    with caplog.at_level(logging.WARNING, logger="glide.config"):
        cfg = GlideConfig.from_toml('[assistant]\nname = "x"', env={}, source="test.toml")
    assert cfg.warnings == ["test.toml: ignoring the unknown table [assistant]"]
    assert "[assistant]" in caplog.text


def test_invalid_toml_is_a_config_error_naming_the_source():
    with pytest.raises(ConfigError, match=r"broken\.toml is not valid TOML"):
        GlideConfig.from_toml("[llm", env={}, source="broken.toml")


# -- options and policy reach the adapter and the chain ------------------------------------------


def test_inline_table_options_reach_the_adapter_over_the_providers_own():
    toml = """
    [providers.openrouter]
    options = { extra_body = { provider = { order = ["x"] } }, reasoning_effort = "high" }
    [llm.fast]
    chain = [
      { provider = "openrouter", model = "deepseek/x", options = { reasoning_effort = "low" } },
      "openai:gpt-a",
    ]
    """
    cfg, fakes = make(toml)
    cfg.llm("fast")
    first, second = fakes.made
    assert first.options == {"reasoning_effort": "low"}  # the slot's own; the adapter layers them over the spec's
    assert first.spec.options == {"extra_body": {"provider": {"order": ["x"]}}, "reasoning_effort": "high"}
    assert second.options == {} and second.spec.options == {}
    assert first.key == KEYS["OPENROUTER_API_KEY"]  # the key arrives as an argument, nowhere else


def test_the_real_llm_adapter_gets_the_options_url_and_name_from_the_file():
    toml = """
    [llm.fast]
    chain = [{ provider = "gemini", model = "gem-b", options = { reasoning_effort = "low", token_param = "max_completion_tokens" } }]
    """
    cfg = GlideConfig.from_toml(toml, env=KEYS)
    try:
        client = cfg.slots("llm.fast")[0].client
        assert isinstance(client, OpenAICompatLLM)
        assert client.name == "gemini:gem-b" and client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert client.settings["reasoning_effort"] == "low" and client.settings["token_param"] == "max_completion_tokens"
    finally:
        cfg.close()


def test_policy_keys_reach_the_chain():
    toml = """
    [llm.fast]
    chain = ["openai:gpt-a", "gemini:gem-b"]
    order = "latency"
    hedge_after_s = 1.5
    fail_threshold = 3
    cooldown_s = 12
    auth_cooldown_s = 60
    latency_alpha = 0.5
    [stt]
    chain = ["openai:gpt-t"]
    """
    cfg, _ = make(toml)
    policy = cfg.chain("llm.fast").policy
    assert (policy.order, policy.hedge_after_s, policy.fail_threshold) == ("latency", 1.5, 3)
    assert (policy.cooldown_s, policy.auth_cooldown_s, policy.latency_alpha) == (12.0, 60.0, 0.5)
    assert cfg.chain("stt").policy.hedge_after_s is None  # each role has its own


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('order = "random"', "order must be one of"),
        ("hedge_after_s = 0", "more than 0"),
        ("hedge_after_s = -1", "at least 0"),
        ('hedge_after_s = "fast"', "must be a number"),
        ("hedge_after_s = true", "must be a number"),
        ("fail_threshold = 0", "at least 1"),
        ("fail_threshold = 2.5", "whole number"),
        ("cooldown_s = -5", "at least 0"),
        ("latency_alpha = 1.5", "from 0 to 1"),
        ("latency_alpha = 0", "more than 0"),
    ],
)
def test_a_bad_policy_value_is_refused_at_load(line, message):
    with pytest.raises(ConfigError, match=message):
        GlideConfig.from_toml(f'[llm.fast]\nchain = ["openai:gpt-a"]\n{line}', env=KEYS)


def test_an_adapter_that_refuses_its_options_is_a_config_error_and_never_shows_the_key():
    builders = Builders()
    key = KEYS["OPENAI_API_KEY"]

    def refuse(spec, model, api_key, options):
        raise ValueError(f"token_param is wrong (sent with {api_key})")

    table = {**builders.table(), ("llm", "openai_compat"): refuse}
    cfg = GlideConfig.from_toml(FAST, env=KEYS, builders=table)
    with pytest.raises(ConfigError) as caught:
        cfg.llm("fast")
    assert "[llm.fast] openai:gpt-a cannot be set up: token_param is wrong" in str(caught.value)
    assert key not in str(caught.value) and "***" in str(caught.value)
    # and a real adapter's refusal, with the real client class
    bad = GlideConfig.from_toml(
        '[llm.fast]\nchain = [{provider = "openai", model = "m", options = { token_param = "nope" }}]', env=KEYS
    )
    with pytest.raises(ConfigError, match="token_param must be one of"):
        bad.llm("fast")


# -- keys: skipped slots, missing variables ------------------------------------------------------


def test_a_missing_key_skips_only_that_slot_and_says_so(caplog):
    cfg, fakes = make(FAST, env={"GEMINI_API_KEY": KEYS["GEMINI_API_KEY"]})
    with caplog.at_level(logging.WARNING, logger="glide.config"):
        llm = cfg.llm("fast")
    assert llm.chain.names == ["gemini:gem-b"]
    assert [m.spec.name for m in fakes.made] == ["gemini"]  # the skipped one was never even built
    (skipped,) = cfg.skipped
    assert (skipped.name, skipped.state, skipped.short, skipped.missing) == (
        "openai:gpt-a",
        "skipped",
        "no key",
        ("OPENAI_API_KEY",),
    )
    assert "skipping openai:gpt-a for llm.fast: OPENAI_API_KEY is not set" in caplog.text
    assert KEYS["GEMINI_API_KEY"] not in caplog.text


def test_a_blank_or_whitespace_key_counts_as_missing():
    cfg, _ = make(FAST, env={"OPENAI_API_KEY": "   ", "GEMINI_API_KEY": ""})
    with pytest.raises(NoUsableProvider) as caught:
        cfg.llm("fast")
    assert caught.value.missing == ("OPENAI_API_KEY", "GEMINI_API_KEY")


def test_a_role_with_no_usable_slot_is_an_error_naming_the_variables_and_only_when_asked_for():
    cfg, _ = make(FAST, env={})  # loading raised nothing: a partial setup is not a mistake
    with pytest.raises(ConfigError) as caught:
        cfg.llm("fast")
    message = str(caught.value)
    assert "OPENAI_API_KEY" in message and "GEMINI_API_KEY" in message and "no usable llm.fast provider" in message
    assert caught.value.missing == ("OPENAI_API_KEY", "GEMINI_API_KEY")
    assert "built-in default" not in message  # this chain is the file's own
    # the roles the file left out say they are the built-in ones
    with pytest.raises(ConfigError, match="built-in default; add \\[stt\\]"):
        cfg.stt()
    assert cfg.tts().chain.names == ["macos_say"]  # one role being unusable does not touch another


def test_a_slot_with_no_key_variable_is_never_skipped():
    toml = """
    [providers.local]
    kind = "openai_compat"
    base_url = "http://localhost:11434/v1"
    [llm.fast]
    chain = ["local:llama"]
    """
    cfg, fakes = make(toml, env={})
    assert cfg.llm("fast").chain.names == ["local:llama"] and fakes.made[0].key == ""


def test_elevenlabs_speech_without_a_voice_is_skipped_and_with_one_it_is_built():
    base = '[tts]\nchain = [{ provider = "elevenlabs", model = "m"%s }, "macos_say"]'
    cfg, _ = make(base % "")
    assert cfg.tts().chain.names == ["macos_say"]
    (skipped,) = cfg.skipped
    assert skipped.short == "no voice" and "options.voice" in skipped.reason
    for options in (', options = { voice = "abc" }', ', options = { voices = { en = "abc" } }'):
        cfg, _ = make(base % options)
        assert cfg.tts().chain.names == ["elevenlabs:m", "macos_say"]
    cfg, _ = make('[providers.elevenlabs]\noptions = { voice = "abc" }\n' + base % "")  # a provider-level voice counts too
    assert cfg.tts().chain.names == ["elevenlabs:m", "macos_say"]


def test_the_environment_given_is_the_only_one_read(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-real-shell-0123456789")
    cfg, _ = make(FAST, env={})
    with pytest.raises(NoUsableProvider):
        cfg.llm("fast")


# -- the facades ---------------------------------------------------------------------------------


def test_facades_are_built_once_and_chains_are_their_chains():
    cfg, _ = make(FAST + '[llm.smart]\nchain = ["openai:gpt-big"]')
    fast = cfg.llm("fast")
    assert cfg.llm() is fast and cfg.llm("llm.fast") is fast
    assert cfg.llm("smart") is not fast
    assert cfg.chains["llm.fast"] is fast.chain
    assert cfg.chain("fast") is fast.chain
    assert list(cfg.chains) == ["llm.fast", "llm.smart", "stt", "tts", "classifier"]
    with pytest.raises(ConfigError, match="one of fast, smart, planner, research"):
        cfg.llm("medium")
    with pytest.raises(ConfigError, match="is not a role"):
        cfg.chain("nope")


def test_the_chains_property_holds_only_the_roles_that_can_be_built():
    cfg, _ = make(FAST, env={"GEMINI_API_KEY": KEYS["GEMINI_API_KEY"]})
    # llm.smart and stt have no key (chain(role) says why); the classifier still has the fast LLM to run over
    assert list(cfg.chains) == ["llm.fast", "tts", "classifier"]
    assert cfg.chains["classifier"].names == ["llm.fast"]


def test_every_job_gets_its_facade_over_its_slots_in_listed_order():
    cfg, fakes = make(FAST)
    chat(cfg.llm("fast"))
    assert fakes.log == ["openai:gpt-a"]
    stt = cfg.stt()
    assert stt.chain.names == ["elevenlabs:scribe_v2_realtime", "openai:gpt-transcribe"]
    assert stt.transcribe(Audio(b"\x00\x00" * 160)).provider == "elevenlabs:scribe_v2_realtime"
    assert cfg.tts().chain.names == ["macos_say"]  # the ElevenLabs slot has no voice in the built-in chain
    assert cfg.tts().synthesize("hi").sample_rate == 16000


def test_a_failing_slot_falls_back_visibly_and_every_listener_hears_it():
    cfg, fakes = make(FAST)
    heard, also = [], []
    cfg.on_switch(heard.append)
    cfg.on_switch(also.append)
    fakes.fail["openai:gpt-a"] = ProviderError("down", kind="server", provider="openai:gpt-a")
    assert chat(cfg.llm("fast")).provider == "gemini:gem-b"
    assert [(e.role, e.from_slot, e.to_slot, e.kind) for e in heard] == [("llm.fast", "openai:gpt-a", "gemini:gem-b", "server")]
    assert also == heard and isinstance(heard[0], SwitchEvent)


def test_a_listener_added_after_a_chain_was_built_still_hears_it_and_can_be_removed():
    cfg, fakes = make(FAST)
    llm = cfg.llm("fast")  # built first
    heard = []
    remove = cfg.on_switch(heard.append)
    fakes.fail["openai:gpt-a"] = ProviderError("down", kind="timeout", provider="openai:gpt-a")
    chat(llm)
    assert len(heard) == 1
    remove()
    chat(llm)
    assert len(heard) == 1


def test_one_broken_listener_does_not_silence_the_others(caplog):
    cfg, fakes = make(FAST)

    def broken(event):
        raise RuntimeError("boom")

    heard = []
    cfg.on_switch(broken)
    cfg.on_switch(heard.append)
    fakes.fail["openai:gpt-a"] = ProviderError("down", kind="server", provider="openai:gpt-a")
    with caplog.at_level(logging.WARNING, logger="glide.config"):
        assert chat(cfg.llm("fast")).provider == "gemini:gem-b"
    assert len(heard) == 1 and "a switch listener raised" in caplog.text


def test_every_role_reports_to_the_same_listener():
    cfg, fakes = make(FAST + '[stt]\nchain = ["openai:a", "elevenlabs:b"]')
    heard = []
    cfg.on_switch(heard.append)
    fakes.fail["openai:gpt-a"] = ProviderError("x", kind="server", provider="p")
    fakes.fail["openai:a"] = ProviderError("x", kind="server", provider="p")
    chat(cfg.llm("fast"))
    cfg.stt().transcribe(Audio(b"\x00\x00" * 160))
    assert sorted(e.role for e in heard) == ["llm.fast", "stt"]


# -- pinning -------------------------------------------------------------------------------------


def test_pin_variables_are_named_after_the_role():
    assert [pin_variable(r) for r in ROLES] == [
        "GLIDE_PIN_LLM_FAST",
        "GLIDE_PIN_LLM_SMART",
        "GLIDE_PIN_STT",
        "GLIDE_PIN_TTS",
        "GLIDE_PIN_CLASSIFIER",
    ]


def test_a_pin_from_the_environment_by_full_name_and_by_prefix():
    for value in ("gemini:gem-b", "gemini", "gem", "  gemini  "):
        cfg, fakes = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": value})
        assert chat(cfg.llm("fast")).provider == "gemini:gem-b"
        assert fakes.log == ["gemini:gem-b"]
        assert cfg.pinned("llm.fast") == ("gemini:gem-b", False)


def test_a_pin_is_not_strict_so_a_failing_pinned_slot_falls_back_and_says_so():
    cfg, fakes = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "gemini"})
    fakes.fail["gemini:gem-b"] = ProviderError("down", kind="server", provider="gemini:gem-b")
    assert chat(cfg.llm("fast")).provider == "openai:gpt-a"
    assert fakes.log == ["gemini:gem-b", "openai:gpt-a"]
    assert [(e.from_slot, e.to_slot) for e in cfg.chain("llm.fast").events] == [("gemini:gem-b", "openai:gpt-a")]


def test_a_trailing_bang_makes_the_pin_strict_and_nothing_else_is_ever_tried():
    cfg, fakes = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "gemini!"})
    assert cfg.pinned("llm.fast") == ("gemini:gem-b", True)  # asking builds the chain, which applies the pin
    fakes.fail["gemini:gem-b"] = ProviderError("down", kind="server", provider="gemini:gem-b")
    with pytest.raises(AllProvidersFailed):
        chat(cfg.llm("fast"))
    assert fakes.log == ["gemini:gem-b"]  # openai was never called


@pytest.mark.parametrize(
    ("role", "variable", "toml", "pin", "expected"),
    [
        ("llm.smart", "GLIDE_PIN_LLM_SMART", '[llm.smart]\nchain = ["openai:a", "gemini:b"]', "gemini", "gemini:b"),
        ("stt", "GLIDE_PIN_STT", '[stt]\nchain = ["openai:a", "elevenlabs:b"]', "elevenlabs", "elevenlabs:b"),
        ("tts", "GLIDE_PIN_TTS", '[tts]\nchain = ["openai:a", "macos_say"]', "macos_say", "macos_say"),
        ("classifier", "GLIDE_PIN_CLASSIFIER", '[classifier]\nchain = ["typesafe:a", "llm.fast"]', "llm.fast", "llm.fast"),
    ],
)
def test_every_role_has_its_own_pin_variable(role, variable, toml, pin, expected):
    cfg, _ = make(toml, env={**KEYS, variable: pin + "!"})
    assert cfg.pinned(role) == (expected, True)
    assert cfg.chain(role).pinned == expected
    other, _ = make(toml, env={**KEYS, variable: pin})
    assert other.pinned(role) == (expected, False)
    unpinned, _ = make(toml, env=KEYS)
    assert unpinned.pinned(role) is None


def test_a_pin_that_names_no_usable_slot_is_an_error_that_lists_the_slots_and_the_variable():
    cfg, _ = make(FAST, env={"OPENAI_API_KEY": KEYS["OPENAI_API_KEY"], "GLIDE_PIN_LLM_FAST": "gemini"})
    with pytest.raises(ConfigError) as caught:
        cfg.llm("fast")
    message = str(caught.value)
    assert "GLIDE_PIN_LLM_FAST='gemini'" in message
    assert "usable: openai:gpt-a" in message
    assert "skipped: gemini:gem-b (GEMINI_API_KEY is not set)" in message


def test_an_ambiguous_or_empty_pin_is_an_error_too():
    cfg, _ = make('[llm.fast]\nchain = ["openai:gpt-a", "openai:gpt-b"]', env={**KEYS, "GLIDE_PIN_LLM_FAST": "openai"})
    with pytest.raises(ConfigError, match="does not pick exactly one"):
        cfg.llm("fast")
    cfg, _ = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "!"})
    with pytest.raises(ConfigError, match="does not pick exactly one"):
        cfg.llm("fast")
    cfg, _ = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "   "})  # blank means no pin
    assert cfg.llm("fast").chain.pinned is None


def test_pinning_at_runtime_returns_the_name_and_unpin_undoes_it():
    cfg, _ = make(FAST)
    assert cfg.pin("llm.fast", "gem") == "gemini:gem-b"
    assert cfg.pinned("fast") == ("gemini:gem-b", False)
    assert chat(cfg.llm("fast")).provider == "gemini:gem-b"
    cfg.unpin("llm.fast")
    assert cfg.pinned("llm.fast") is None and cfg.chain("llm.fast").pinned is None
    assert chat(cfg.llm("fast")).provider == "openai:gpt-a"
    assert cfg.pin("fast", "gemini:gem-b", strict=True) == "gemini:gem-b"
    assert cfg.pinned("llm.fast") == ("gemini:gem-b", True)
    with pytest.raises(ConfigError, match="cannot pin 'nope'"):
        cfg.pin("fast", "nope")
    assert cfg.pinned("llm.fast") == ("gemini:gem-b", True)  # a refused pin changes nothing


def test_the_runtime_pin_replaces_the_one_from_the_environment():
    cfg, _ = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "gemini!"})
    cfg.llm("fast")
    cfg.pin("llm.fast", "openai")
    assert cfg.pinned("llm.fast") == ("openai:gpt-a", False)
    cfg.unpin("llm.fast")
    assert chat(cfg.llm("fast")).provider == "openai:gpt-a"
    cfg.unpin("tts")  # a role that was never built is not an error to unpin


# -- the classifier ------------------------------------------------------------------------------


def test_the_classifier_chain_is_typesafe_then_the_fast_llm_chain():
    cfg, fakes = make(FAST)
    classifier = cfg.classifier()
    assert isinstance(classifier, ChainedClassifier)
    assert classifier.chain.names == ["typesafe:jev-latest", "llm.fast"]
    assert classifier.chain.role == "classifier"
    assert cfg.classifier() is classifier
    reply = classifier.system_one(state="s", questions={"q": Choice(criteria={"a": None, "b": None})})
    assert reply.answers["q"].choice == "a" and classifier.last_slot == "typesafe:jev-latest"
    assert fakes.log == ["typesafe:jev-latest"]


def test_closing_the_classifier_leaves_the_real_clients_open_for_the_next_run():
    cfg, fakes = make(FAST)
    question = {"q": Choice(criteria={"a": None, "b": None})}
    first = cfg.classifier()
    first.system_one(state="s", questions=question)
    first.close()  # what runner.run() does on exit
    typesafe = fakes.client("typesafe:jev-latest")
    assert typesafe.closed is False
    again = cfg.classifier()
    assert again.system_one(state="s", questions=question).answers["q"].choice == "a"
    with cfg.classifier() as third:  # and the context-manager form
        third.system_one(state="s", questions=question)
    assert typesafe.closed is False
    cfg.close()
    assert typesafe.closed is True and fakes.client("openai:gpt-a").closed is True


def test_a_failing_typesafe_slot_falls_back_to_the_classifier_over_the_fast_llm():
    cfg, fakes = make(FAST)
    heard = []
    cfg.on_switch(heard.append)
    fakes.fail["typesafe:jev-latest"] = ProviderError("down", kind="server", provider="typesafe:jev-latest")
    reply = cfg.classifier().system_one(state="s", questions={"q": Choice(criteria={"a": None, "b": None})})
    assert reply.answers["q"].choice == "a"
    assert cfg.classifier().last_slot == "llm.fast"
    assert fakes.log == ["typesafe:jev-latest", "openai:gpt-a"]  # the LLM slot is the first slot of the fast chain
    assert [(e.role, e.from_slot, e.to_slot) for e in heard] == [("classifier", "typesafe:jev-latest", "llm.fast")]


def test_the_classifier_factory_hands_the_run_the_provider_chain_with_failover_and_a_visible_switch():
    cfg, fakes = make(FAST)
    heard = []
    cfg.on_switch(heard.append)
    fakes.fail["typesafe:jev-latest"] = ProviderError("down", kind="transport", provider="typesafe:jev-latest")
    factory = classifier_factory(cfg)
    questions = {"q": Choice(criteria={"a": None, "b": None})}

    for _ in range(2):  # a run closes what it was given; the next run must still have a classifier
        with factory() as classifier:
            assert isinstance(classifier, ChainedClassifier) and classifier.chain is cfg.chain("classifier")
            assert classifier.system_one(state="s", questions=questions).answers["q"].choice == "a"
            assert classifier.last_slot == "llm.fast"
    assert [(e.role, e.from_slot, e.to_slot, e.kind) for e in heard][0] == (
        "classifier",
        "typesafe:jev-latest",
        "llm.fast",
        "transport",
    )
    assert fakes.client("openai:gpt-a").closed is False  # the slots are lent: closing the run's classifier left them open


def test_a_classifier_factory_with_no_usable_slot_is_a_provider_failure_naming_the_variables():
    cfg, _ = make("", env={})
    with pytest.raises(ProviderError) as caught:
        classifier_factory(cfg)()
    assert caught.value.kind == "auth" and caught.value.provider == "classifier"
    assert "TYPESAFE_API_KEY" in str(caught.value)


def test_the_classifier_over_the_fast_llm_is_skipped_when_the_fast_chain_is_unusable():
    cfg, _ = make(FAST, env={"TYPESAFE_API_KEY": KEYS["TYPESAFE_API_KEY"]})
    assert cfg.classifier().chain.names == ["typesafe:jev-latest"]
    (skipped,) = [s for s in cfg.skipped if s.role == "classifier"]
    assert (skipped.name, skipped.short) == ("llm.fast", "llm.fast unusable")
    assert skipped.missing == ("OPENAI_API_KEY", "GEMINI_API_KEY")


def test_a_classifier_with_neither_slot_usable_names_every_variable_involved():
    cfg, _ = make(FAST, env={})
    with pytest.raises(ConfigError) as caught:
        cfg.classifier()
    assert caught.value.missing == ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")


def test_a_chat_endpoint_can_serve_as_a_classifier_slot_by_itself():
    toml = """
    [classifier]
    chain = [{ provider = "openai", model = "gpt-c", options = { reasoning_effort = "low", max_tokens = 300, timeout = 7 } }]
    """
    cfg, fakes = make(toml)
    classifier = cfg.classifier()
    assert classifier.chain.names == ["openai:gpt-c"]
    (made,) = fakes.made
    assert made.family == "llm" and made.options["reasoning_effort"] == "low"  # the same endpoint adapter as the LLM role
    lent = classifier.chain._slots[0].client  # what the chain holds, to look inside it
    assert isinstance(lent._client, LLMClassifier) and lent._client._max_tokens == 300 and lent._client._timeout == 7
    cfg.close()
    assert made.client.closed is True  # the classifier owns that adapter, and the configuration closes it


# -- the writer ----------------------------------------------------------------------------------


def test_the_writer_is_built_from_the_chains_and_routes_by_role():
    cfg, fakes = make(
        FAST
        + """
        [llm.smart]
        chain = ["openai:gpt-big"]
        [llm.planner]
        chain = ["deepseek:plan-model"]
        [llm.research]
        chain = ["gemini:research-model"]
        """
    )
    fakes.text = "{}"
    writer = cfg.writer(timeout=9.0)

    def ask(role: str):
        request = GenerationRequest("ignored", "be brief", "hi", {"type": "object"}, max_tokens=50, role=role)
        return writer.generate(request)

    answers = [ask(role) for role in ("recovery", "writer", "planner", "research_supervisor", "task_routing")]
    assert [a.model for a in answers] == ["gpt-big", "gpt-a", "plan-model", "research-model", "gpt-a"]
    assert fakes.log == [
        "openai:gpt-big",
        "openai:gpt-a",
        "deepseek:plan-model",
        "gemini:research-model",
        "openai:gpt-a",
    ]
    with pytest.raises(GenerationError, match="no provider chain serves"):
        ask("nobody")


def test_planner_and_research_stand_on_the_smart_chain_when_the_file_gives_them_none():
    cfg, fakes = make(FAST + '[llm.smart]\nchain = ["openai:gpt-big"]')
    assert cfg.llm("planner") is cfg.llm("smart") and cfg.llm("research") is cfg.llm("smart")
    fakes.text = "{}"
    assert cfg.chain("llm.research") is cfg.chain("smart")  # one chain: one health record, one pin, nothing built twice
    assert list(cfg.chains) == ["llm.fast", "llm.smart", "stt", "tts", "classifier"]
    writer = cfg.writer()
    request = GenerationRequest("", "x", "y", {"type": "object"}, role="planner")
    assert writer.generate(request).model == "gpt-big"
    assert fakes.log == ["openai:gpt-big"]


def test_a_planner_chain_in_the_file_is_its_own_chain_with_its_own_pin_and_listing():
    cfg, _ = make(FAST + '[llm.planner]\nchain = ["openai:gpt-p", "gemini:gem-p"]')
    assert cfg.llm("planner") is not cfg.llm("smart")
    assert cfg.chain("planner").names == ["openai:gpt-p", "gemini:gem-p"]
    assert list(cfg.chains) == ["llm.fast", "llm.smart", "stt", "tts", "classifier", "llm.planner"]
    assert cfg.pin("planner", "gemini") == "gemini:gem-p"
    assert cfg.pinned("planner") == ("gemini:gem-p", False) and cfg.pinned("smart") is None


def test_the_role_names_are_checked_and_a_deadline_is_a_cap_on_one_request():
    with pytest.raises(ConfigError, match=r"\[llm.medium\] is not a role"):
        make('[llm.medium]\nchain = ["openai:x"]')
    with pytest.raises(ConfigError, match="deadline_s must be more than 0"):
        make('[llm.research]\nchain = ["openai:x"]\ndeadline_s = 0')
    with pytest.raises(ConfigError, match="unknown key 'deadline_s'"):
        make('[stt]\nchain = ["openai:x"]\ndeadline_s = 5')
    cfg, fakes = make(FAST + '[llm.research]\nchain = ["openai:gpt-r"]\ndeadline_s = 45')
    fakes.text = "{}"
    seen = []
    research = cfg.llm("research")
    real = research.chat
    research.chat = lambda messages, **kw: seen.append(kw["timeout"]) or real(messages, **kw)
    writer = cfg.writer()
    writer.generate(GenerationRequest("", "x", "y", {"type": "object"}, deadline_s=120, role="research_supervisor"))
    writer.generate(GenerationRequest("", "x", "y", {"type": "object"}, deadline_s=20, role="research_supervisor"))
    assert seen == [45, 20]  # the file's number caps; it never stretches a request that asked for less


def test_the_speech_table_is_validated_and_has_no_built_in_values():
    cfg, _ = make("")
    assert cfg.speech == SpeechSettings()
    with pytest.raises(ConfigError, match="vad_model_path and vad_model_sha256: the voice-activity model is not configured"):
        cfg.speech.vad_model()
    digest = "ab" * 32
    cfg, _ = make(
        f"""
        [speech]
        language = "en"
        silence_ms = 700
        headset = true
        vad_model_path = "models/vad.onnx"
        vad_model_url = "https://models.example.test/vad.onnx"
        vad_model_sha256 = "{digest.upper()}"
        """
    )
    assert (cfg.speech.language, cfg.speech.silence_ms, cfg.speech.headset) == ("en", 700, True)
    assert cfg.speech.vad_model() == ("models/vad.onnx", "https://models.example.test/vad.onnx", digest)
    for bad, match in (
        ("silence_ms = 0", "silence_ms must be at least 1"),
        ('headset = "yes"', "headset must be true or false"),
        ('vad_model_url = "http://models.example.test/x"', "must be an https URL"),
        ('vad_model_sha256 = "abc"', "64 hexadecimal digits"),
        ('voice = "x"', "unknown key 'voice'"),
    ):
        with pytest.raises(ConfigError, match=match):
            make(f"[speech]\n{bad}")
    assert "speech" not in "".join(cfg.warnings)  # a known table, not an ignored one


def test_the_writer_names_the_variables_when_the_smart_chain_is_unusable():
    cfg, _ = make(FAST + '[llm.smart]\nchain = ["deepseek:big"]', env={"OPENAI_API_KEY": KEYS["OPENAI_API_KEY"]})
    assert cfg.llm("fast").chain.names == ["openai:gpt-a"]  # the fast chain is fine
    with pytest.raises(ConfigError) as caught:
        cfg.writer()
    assert caught.value.missing == ("DEEPSEEK_API_KEY",)


# -- finding the file ----------------------------------------------------------------------------


def write(directory: Path, name: str, chain: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f'[llm.fast]\nchain = ["{chain}"]\n', encoding="utf-8")
    return path


def names(cfg: GlideConfig) -> list[str]:
    return [s.name for s in cfg.roles["llm.fast"].slots]


def test_load_config_searches_path_then_env_then_here_then_home_then_the_built_in_chains(tmp_path):
    here, home = tmp_path / "here", tmp_path / "home"
    explicit = write(tmp_path, "explicit.toml", "openai:from-path")
    named = write(tmp_path, "named.toml", "openai:from-env")
    write(here, "glide.toml", "openai:from-cwd")
    write(home / ".config" / "glide", "glide.toml", "openai:from-home")
    env = {"GLIDE_CONFIG": str(named), "HOME": str(home)}

    cfg = load_config(explicit, env, cwd=here)
    assert names(cfg) == ["openai:from-path"] and cfg.source == str(explicit)
    assert names(load_config(None, env, cwd=here)) == ["openai:from-env"]
    assert names(load_config(None, {"HOME": str(home)}, cwd=here)) == ["openai:from-cwd"]
    assert names(load_config(None, {"HOME": str(home)}, cwd=tmp_path / "empty")) == ["openai:from-home"]
    assert names(load_config(None, {"HOME": str(home)}, cwd=tmp_path / "empty", home=tmp_path / "nowhere")) == [
        "openrouter:deepseek/deepseek-v4.1-flash",
        "openai:gpt-6-luna",
        "gemini:gemini-3.5-flash-lite",
    ]
    fallback = load_config(None, {}, cwd=tmp_path / "empty")  # no HOME in a given env: the real home is never looked at
    assert fallback.source == "built-in defaults" and fallback.defaulted == ROLES


def test_a_path_or_env_that_names_a_missing_file_is_an_error_not_a_fallback(tmp_path):
    write(tmp_path, "glide.toml", "openai:a")
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.toml", {}, cwd=tmp_path)
    with pytest.raises(ConfigError, match=r"GLIDE_CONFIG names .*nope\.toml"):
        load_config(None, {"GLIDE_CONFIG": str(tmp_path / "nope.toml")}, cwd=tmp_path)
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path, {}, cwd=tmp_path)  # a directory is not a file


def test_an_unreadable_file_and_a_broken_one_are_config_errors(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[llm.fast\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(bad, {})
    bad.write_bytes(b"\xff\xfe[llm]")  # not UTF-8
    with pytest.raises(ConfigError):
        load_config(bad, {})


def test_a_file_that_leaves_roles_out_gets_the_built_in_chain_for_each_of_them(tmp_path):
    path = write(tmp_path, "glide.toml", "openai:mine")
    cfg = load_config(path, KEYS)
    assert cfg.defaulted == ("llm.smart", "stt", "tts", "classifier")
    assert [s.name for s in cfg.roles["stt"].slots] == ["elevenlabs:scribe_v2_realtime", "openai:gpt-transcribe"]


def test_load_config_uses_the_environment_given_for_keys_and_pins(tmp_path):
    path = write(tmp_path, "glide.toml", "openai:mine")
    cfg = load_config(path, {"OPENAI_API_KEY": KEYS["OPENAI_API_KEY"], "GLIDE_PIN_LLM_FAST": "openai!"})
    try:
        assert cfg.llm("fast").chain.pinned == "openai:mine" and cfg.pinned("llm.fast") == ("openai:mine", True)
    finally:
        cfg.close()


# -- the built-in chains and the example file ----------------------------------------------------


def slots_of(cfg: GlideConfig) -> dict[str, list[tuple]]:
    return {role: [(s.name, s.provider, s.model, s.options, s.uses) for s in spec.slots] for role, spec in cfg.roles.items()}


def test_the_built_in_chains_are_the_same_as_the_examples():
    built_in = GlideConfig.from_toml(DEFAULT_TOML, env={})
    example = GlideConfig.from_toml(EXAMPLE.read_text(encoding="utf-8"), env={})
    assert example.defaulted == () and built_in.defaulted == ()
    assert slots_of(example) == slots_of(built_in)
    assert slots_of(GlideConfig.from_toml("", env={})) == slots_of(built_in)  # and that is what no file means


def test_the_example_file_says_every_model_id_is_unverified():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert "UNVERIFIED" in text and "glide doctor --live" in text
    assert "hedge_after_s" in text


def test_the_example_loads_cleanly_with_every_key_faked_and_builds_every_role_with_the_real_adapters(caplog):
    with caplog.at_level(logging.WARNING, logger="glide.config"):
        cfg = load_config(EXAMPLE, KEYS, cwd=EXAMPLE.parent / "nowhere")
        try:
            chains = {role: cfg.chain(role).names for role in ROLES}
            assert chains == {
                "llm.fast": ["openrouter:deepseek/deepseek-v4.1-flash", "openai:gpt-6-luna", "gemini:gemini-3.5-flash-lite"],
                "llm.smart": ["openai:gpt-6.1-sol", "openai:gpt-6-luna"],
                "stt": ["elevenlabs:scribe_v2_realtime", "openai:gpt-transcribe"],
                "tts": ["macos_say"],
                "classifier": ["typesafe:jev-latest", "llm.fast"],
            }
            assert cfg.defaulted == () and cfg.warnings == []
            # the one thing left out is the ElevenLabs voice, which only the user can supply
            assert [(s.role, s.name, s.short) for s in cfg.skipped] == [("tts", "elevenlabs:eleven_v4_turbo", "no voice")]
            assert cfg.chain("llm.fast").policy.hedge_after_s == 2.0
            assert cfg.chain("llm.smart").policy.hedge_after_s is None
            efforts = {i.name: i.options.get("reasoning_effort") for i in cfg.slots("llm.fast")}
            assert set(efforts.values()) == {"low"}
            assert cfg.writer() is not None
        finally:
            cfg.close()
    assert "no voice configured" in caplog.text
    assert all(value not in caplog.text for value in KEYS.values())


def test_the_example_with_a_voice_added_builds_the_elevenlabs_speech_slot():
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        "options = { timeout = 8 }", 'options = { voice = "voice-id-1", timeout = 8 }'
    )
    cfg = GlideConfig.from_toml(text, env=KEYS)
    try:
        assert cfg.tts().chain.names == ["elevenlabs:eleven_v4_turbo", "macos_say"]
    finally:
        cfg.close()


def test_with_no_keys_at_all_the_built_in_chains_still_give_macos_say_and_nothing_else():
    cfg = GlideConfig.from_toml("", env={})
    assert list(cfg.chains) == ["tts"] and cfg.chains["tts"].names == ["macos_say"]


# -- keys never leak -----------------------------------------------------------------------------


def test_keys_stay_out_of_every_repr_and_error_and_log(caplog):
    cfg = GlideConfig.from_toml(FAST, env=KEYS)
    with caplog.at_level(logging.DEBUG):
        cfg.llm("fast")
        cfg.stt()
        try:
            cfg.pin("llm.fast", "nope")
        except ConfigError as e:
            pin_error = str(e)
        text = " ".join(
            [
                repr(cfg),
                repr(cfg.slots("llm.fast")),
                repr(cfg.skipped),
                repr(cfg.chains),
                pin_error,
                caplog.text,
                repr(vars(cfg).keys()),
            ]
        )
    for key in KEYS.values():
        assert key not in text
    cfg.close()


def test_scrub_removes_every_key_it_can_read_and_leaves_other_text_alone():
    cfg = GlideConfig.from_toml(
        FAST, env={**KEYS, "OPENAI_API_KEY": " " + KEYS["OPENAI_API_KEY"] + "\n", "GEMINI_API_KEY": "abc"}
    )
    out = cfg.scrub(f"bad {KEYS['OPENAI_API_KEY']} and {KEYS['ELEVENLABS_API_KEY']} but abc stays")
    assert KEYS["OPENAI_API_KEY"] not in out and KEYS["ELEVENLABS_API_KEY"] not in out
    assert out == "bad *** and *** but abc stays"  # a three-character 'key' would mangle ordinary words, so it is left


def test_the_key_reaches_the_adapter_only_as_an_argument_and_a_slot_never_prints_it():
    cfg, fakes = make(FAST)
    cfg.llm("fast")
    assert fakes.made[0].key == KEYS["OPENAI_API_KEY"]
    assert KEYS["OPENAI_API_KEY"] not in repr(cfg.chain("llm.fast").status())


# -- close ---------------------------------------------------------------------------------------


def test_close_closes_every_adapter_even_when_one_refuses_and_then_raises_the_first_failure():
    cfg, fakes = make(FAST)
    cfg.llm("fast")
    first, second = fakes.client("openai:gpt-a"), fakes.client("gemini:gem-b")

    def refuse():
        raise OSError("stuck")

    first.close = refuse
    with pytest.raises(OSError, match="stuck"):
        cfg.close()
    assert second.closed is True


def test_the_configuration_is_a_context_manager_that_closes_on_exit():
    with make(FAST)[0] as cfg:
        cfg.llm("fast")
        client = cfg.slots("llm.fast")[0].client
    assert client.closed is True


def test_building_a_role_from_two_threads_builds_it_once():
    cfg, fakes = make(FAST)
    out = []
    barrier = threading.Barrier(4)

    def go():
        barrier.wait()
        out.append(cfg.llm("fast"))

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(x) for x in out}) == 1 and len(fakes.made) == 2


def test_config_module_exposes_the_documented_names():
    for name in ("GlideConfig", "ConfigError", "load_config", "PRESETS", "ROLES", "DEFAULT_TOML", "pin_variable"):
        assert hasattr(config_module, name)
