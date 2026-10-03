"""glide doctor: offline rows, live probes through fakes, the table, the command. No network, no real keys.

The live path is only ever run here against fake adapters, or against the real LLM adapter over an
httpx.MockTransport. Nothing reaches a vendor.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from test_config import KEYS, Builders, Clock, FakeClient, make
from typesafe_sdk import ChoiceAnswer

from glide.providers import doctor as doctor_module
from glide.providers.base import SpeechAudio
from glide.providers.classifier import ClassifierReply
from glide.providers.config import ROLES, GlideConfig
from glide.providers.doctor import Row, doctor, failed, format_rows, main
from glide.providers.errors import ProviderError
from glide.providers.llm import OpenAICompatLLM

EVERY_ROLE = """
[llm.fast]
chain = ["openai:gpt-a", "gemini:gem-b"]
[llm.smart]
chain = ["openai:gpt-big"]
[stt]
chain = ["openai:gpt-t", "elevenlabs:scribe"]
[tts]
chain = [{ provider = "elevenlabs", model = "v", options = { voice = "abc" } }, "macos_say"]
[classifier]
chain = ["typesafe:jev-latest"]
"""

FAST = '[llm.fast]\nchain = ["openai:gpt-a", "gemini:gem-b"]'


def by_slot(rows: list[Row]) -> dict[tuple[str, str], Row]:
    return {(r.role, r.slot): r for r in rows}


def live(toml: str = EVERY_ROLE, env: dict | None = None, **kw) -> tuple[list[Row], GlideConfig, Builders]:
    clock = Clock()
    cfg, fakes = make(toml, env=env, builders=Builders(clock))
    return doctor(cfg, live=True, clock=clock, **kw), cfg, fakes


# -- offline -------------------------------------------------------------------------------------


def test_offline_reports_key_presence_per_slot_and_sends_nothing():
    cfg, fakes = make("", env={"OPENAI_API_KEY": KEYS["OPENAI_API_KEY"]})
    rows = by_slot(doctor(cfg))
    assert rows[("llm.fast", "openai:gpt-6-luna")].status == "ready"
    assert rows[("llm.fast", "openai:gpt-6-luna")].detail == "OPENAI_API_KEY is set; adapter built"
    skipped = rows[("llm.fast", "openrouter:deepseek/deepseek-v4.1-flash")]
    assert (skipped.status, skipped.detail) == ("skipped(no key)", "OPENROUTER_API_KEY is not set")
    assert rows[("stt", "elevenlabs:scribe_v2_realtime")].status == "skipped(no key)"
    assert rows[("stt", "openai:gpt-transcribe")].status == "ready"
    assert rows[("tts", "macos_say")].detail == "no key needed; adapter built"
    assert rows[("classifier", "llm.fast")].status == "ready"
    assert "llm.fast chain" in rows[("classifier", "llm.fast")].detail
    assert rows[("classifier", "typesafe:jev-latest")].status == "skipped(no key)"
    assert all(r.latency_s is None for r in rows.values())
    assert fakes.log == [] and all(m.client.calls == [] for m in fakes.made)  # nothing was sent anywhere


def test_a_role_with_nothing_usable_gets_an_error_row_naming_the_variables():
    cfg, _ = make(FAST, env={})
    rows = doctor(cfg, roles=["llm.fast", "llm.smart", "tts"])
    errors = [r for r in rows if r.status == "error"]
    assert [(r.role, r.slot) for r in errors] == [("llm.fast", "-"), ("llm.smart", "-")]
    assert "OPENAI_API_KEY" in errors[0].detail and "GEMINI_API_KEY" in errors[0].detail
    assert [(r.status) for r in rows if r.role == "tts"] == ["skipped(no key)", "ready"]  # say needs nothing: no error row
    assert failed(rows)


def test_a_skipped_slot_without_a_voice_says_why():
    cfg, _ = make(EVERY_ROLE.replace(', options = { voice = "abc" }', ""))
    row = by_slot(doctor(cfg))[("tts", "elevenlabs:v")]
    assert row.status == "skipped(no voice)" and "options.voice" in row.detail


def test_a_slot_the_adapter_refuses_is_an_error_row_for_its_role_not_a_crash():
    cfg = GlideConfig.from_toml(
        '[llm.fast]\nchain = [{provider = "openai", model = "m", options = { token_param = "nope" }}]', env=KEYS
    )
    rows = doctor(cfg, roles=["llm.fast"])
    assert [(r.status, r.slot) for r in rows] == [("error", "-")] and "token_param must be one of" in rows[0].detail


def test_a_pin_that_cannot_be_honoured_shows_as_an_error_row():
    cfg, _ = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "nope"})
    rows = doctor(cfg, roles=["llm.fast"])
    assert [r.status for r in rows] == ["ready", "ready", "error"]
    assert "GLIDE_PIN_LLM_FAST='nope'" in rows[-1].detail
    assert doctor(cfg, live=False, roles=["llm.fast"]) == rows


def test_the_slot_that_is_pinned_says_so():
    cfg, _ = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "gemini!"})
    rows = doctor(cfg, roles=["llm.fast"])
    assert rows[0].detail.endswith("adapter built") and rows[1].detail.endswith("pinned (strict)")


def test_roles_can_be_chosen_and_rows_follow_listed_order():
    cfg, _ = make(EVERY_ROLE)
    assert [(r.role, r.slot) for r in doctor(cfg, roles=["stt", "llm.fast"])] == [
        ("stt", "openai:gpt-t"),
        ("stt", "elevenlabs:scribe"),
        ("llm.fast", "openai:gpt-a"),
        ("llm.fast", "gemini:gem-b"),
    ]
    assert {r.role for r in doctor(cfg)} == set(ROLES)


def test_no_key_value_appears_in_any_offline_row_or_in_the_table():
    cfg, _ = make(EVERY_ROLE)
    text = format_rows(doctor(cfg), detail_width=None) + repr(doctor(cfg))
    assert text and all(key not in text for key in KEYS.values())


# -- live ----------------------------------------------------------------------------------------


def test_live_sends_each_slot_exactly_one_request_and_times_it():
    rows, _, fakes = live()
    assert [(r.role, r.slot, r.status) for r in rows] == [
        ("llm.fast", "openai:gpt-a", "ok"),
        ("llm.fast", "gemini:gem-b", "ok"),
        ("llm.smart", "openai:gpt-big", "ok"),
        ("stt", "openai:gpt-t", "ok"),
        ("stt", "elevenlabs:scribe", "ok"),
        ("tts", "elevenlabs:v", "ok"),
        ("tts", "macos_say", "ok"),
        ("classifier", "typesafe:jev-latest", "ok"),
    ]
    assert {r.latency_s for r in rows} == {0.25}  # the fake clock moves 0.25 s per request
    assert [m.client.calls for m in fakes.made] == [
        ["chat"],
        ["chat"],
        ["chat"],
        ["transcribe"],
        ["transcribe"],
        ["synthesize"],
        ["synthesize"],
        ["system_one"],
    ]
    assert sorted(fakes.log) == sorted(m.client.name for m in fakes.made)  # one request per slot, none fell back to another


def test_each_probe_reaches_its_own_slot_only_because_it_is_pinned_strictly():
    rows, _, fakes = live(FAST, roles=["llm.fast"])
    assert fakes.log == ["openai:gpt-a", "gemini:gem-b"]  # without the strict pin the second probe would hit the first slot
    assert [r.status for r in rows if r.role == "llm.fast"] == ["ok", "ok"]


def test_a_failing_slot_is_reported_by_its_own_error_kind_and_the_next_slot_is_still_probed():
    clock = Clock()
    cfg, fakes = make(FAST, builders=Builders(clock))
    cfg.llm("fast")
    fakes.fail["openai:gpt-a"] = ProviderError(
        "openai:gpt-a answered 401: bad key", kind="auth", provider="openai:gpt-a", status=401
    )
    rows = doctor(cfg, live=True, roles=["llm.fast"], clock=clock)
    assert [(r.slot, r.status) for r in rows] == [("openai:gpt-a", "failed(auth)"), ("gemini:gem-b", "ok")]
    assert "answered 401" in rows[0].detail and rows[0].latency_s == 0.25  # a failure is timed too
    assert fakes.log == ["openai:gpt-a", "gemini:gem-b"]  # no fallback happened inside the probe


@pytest.mark.parametrize(
    "kind", ["auth", "rate_limit", "timeout", "transport", "server", "unsupported", "content", "bad_request"]
)
def test_every_error_kind_comes_out_as_failed_kind_never_as_exhausted(kind):
    cfg, fakes = make(FAST)
    fakes.fail["openai:gpt-a"] = ProviderError("nope", kind=kind, provider="openai:gpt-a")
    rows = doctor(cfg, live=True, roles=["llm.fast"])
    assert rows[0].status == f"failed({kind})"  # a strict pin wraps a failover kind in AllProvidersFailed; the row unwraps it


def test_a_bug_in_an_adapter_is_a_failed_row_naming_its_type_and_the_run_goes_on():
    cfg, fakes = make(FAST)
    fakes.fail["openai:gpt-a"] = RuntimeError("boom")
    rows = doctor(cfg, live=True, roles=["llm.fast"])
    assert [(r.status, r.detail) for r in rows] == [("failed(RuntimeError)", "boom"), ("ok", rows[1].detail)]


def test_the_users_pin_comes_back_after_every_probe():
    cfg, _ = make(FAST)
    cfg.llm("fast")
    cfg.pin("llm.fast", "gemini")
    doctor(cfg, live=True, roles=["llm.fast"])
    assert cfg.pinned("llm.fast") == ("gemini:gem-b", False) and cfg.chain("llm.fast").pinned == "gemini:gem-b"
    cfg.pin("llm.fast", "openai", strict=True)
    doctor(cfg, live=True, roles=["llm.fast"])
    assert cfg.pinned("llm.fast") == ("openai:gpt-a", True)
    cfg.unpin("llm.fast")
    doctor(cfg, live=True, roles=["llm.fast"])
    assert cfg.pinned("llm.fast") is None and cfg.chain("llm.fast").pinned is None


def test_the_pin_comes_back_even_when_the_probe_fails():
    cfg, fakes = make(FAST, env={**KEYS, "GLIDE_PIN_LLM_FAST": "gemini!"})
    fakes.fail["gemini:gem-b"] = RuntimeError("boom")
    doctor(cfg, live=True, roles=["llm.fast"])
    assert cfg.pinned("llm.fast") == ("gemini:gem-b", True)


def test_a_key_echoed_by_a_server_never_reaches_a_row_or_the_table():
    key = KEYS["OPENAI_API_KEY"]
    cfg, fakes = make(FAST)
    fakes.fail["openai:gpt-a"] = ProviderError(f"rejected {key}", kind="auth", provider="openai:gpt-a")
    fakes.fail["gemini:gem-b"] = RuntimeError(f"crashed holding {KEYS['GEMINI_API_KEY']}")
    rows = doctor(cfg, live=True, roles=["llm.fast"])
    text = repr(rows) + format_rows(rows, detail_width=None)
    assert key not in text and KEYS["GEMINI_API_KEY"] not in text and "***" in text


def test_speech_to_text_on_silence_is_ok_whatever_it_hears():
    rows, _, fakes = live('[stt]\nchain = ["openai:gpt-t"]', roles=["stt"])
    assert rows[0].status == "ok" and "no text" in rows[0].detail
    audio_args = fakes.client("openai:gpt-t").calls
    assert audio_args == ["transcribe"]


def test_text_to_speech_that_returns_no_audio_is_a_failure():
    cfg, fakes = make('[tts]\nchain = ["macos_say"]')
    cfg.tts()
    fakes.client("macos_say").synthesize = lambda *a, **k: SpeechAudio(pcm=b"", sample_rate=22050)
    (row,) = doctor(cfg, live=True, roles=["tts"])
    assert row.status == "failed(content)" and "no audio" in row.detail


def test_text_to_speech_reports_the_length_and_rate_of_what_it_made():
    rows, _, _ = live('[tts]\nchain = ["macos_say"]', roles=["tts"])
    assert rows[0].status == "ok" and rows[0].detail == "0.5 s of audio at 16000 Hz"


def test_the_classifier_probe_asks_a_question_with_two_options_and_checks_the_answer():
    seen = []
    cfg, fakes = make('[classifier]\nchain = ["typesafe:jev-latest"]')
    cfg.classifier()
    client = fakes.client("typesafe:jev-latest")
    original = client.system_one

    def spy(state, questions, **kw):
        seen.append(questions)
        return original(state, questions, **kw)

    client.system_one = spy
    (row,) = doctor(cfg, live=True, roles=["classifier"])
    assert row.status == "ok" and row.detail == "chose 'yes' (confidence 0.90)"
    (questions,) = seen
    assert len(questions["q"].criteria) == 2  # one option would be answered without asking anyone


def test_a_classifier_answer_that_is_not_an_option_is_a_content_failure():
    cfg, fakes = make('[classifier]\nchain = ["typesafe:jev-latest"]')
    cfg.classifier()
    fakes.client("typesafe:jev-latest").system_one = lambda state, questions, **kw: ClassifierReply(
        answers={"q": ChoiceAnswer(choice="maybe", confidence=0.5, probabilities={"maybe": 0.5})}
    )
    (row,) = doctor(cfg, live=True, roles=["classifier"])
    assert row.status == "failed(content)"


def test_the_classifier_over_the_fast_llm_is_probed_through_the_fast_chain():
    clock = Clock()
    cfg, fakes = make('[llm.fast]\nchain = ["openai:gpt-a"]\n[classifier]\nchain = ["llm.fast"]', builders=Builders(clock))
    (row,) = doctor(cfg, live=True, roles=["classifier"], clock=clock)
    assert (row.slot, row.status, row.detail) == ("llm.fast", "ok", "chose 'yes' (confidence 0.90)")
    assert fakes.log == ["openai:gpt-a"]


def test_a_timeout_is_passed_to_every_kind_of_probe(monkeypatch):
    seen = []

    def spy(name):
        original = getattr(FakeClient, name)

        def wrapped(self, *args, **kw):
            seen.append((name, kw.get("timeout")))
            return original(self, *args, **kw)

        monkeypatch.setattr(FakeClient, name, wrapped)

    for name in ("chat", "transcribe", "synthesize", "system_one"):
        spy(name)
    cfg, _ = make(EVERY_ROLE)
    doctor(cfg, live=True, timeout=3.5)
    assert {name for name, _ in seen} == {"chat", "transcribe", "synthesize", "system_one"}
    assert all(timeout == 3.5 for _, timeout in seen)
    seen.clear()
    doctor(cfg, live=True)  # left out, the adapters' own limit applies
    assert seen and all(timeout is None for _, timeout in seen)


# -- live, with the real LLM adapter over a mock transport ---------------------------------------


def real_llm(handler):
    transport = httpx.MockTransport(handler)

    def build(spec, model, key, options):
        return OpenAICompatLLM(
            f"{spec.name}:{model}", model, spec.base_url, key, {**spec.options, **options}, transport=transport
        )

    return {("llm", "openai_compat"): build}


def completion(text: str, finish: str = "stop") -> httpx.Response:
    body = {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1},
    }
    return httpx.Response(200, json=body)


REAL = '[llm.fast]\nchain = [{ provider = "openai", model = "gpt-a", options = { reasoning_effort = "low" } }]'


def probe_real(handler, toml=REAL):
    cfg = GlideConfig.from_toml(toml, env=KEYS, builders=real_llm(handler))
    try:
        return doctor(cfg, live=True, roles=["llm.fast"])
    finally:
        cfg.close()


def test_a_five_token_chat_through_the_real_adapter_is_one_small_request():
    seen = []

    def handler(request):
        seen.append(request)
        return completion("ok")

    (row,) = probe_real(handler)
    assert row.status == "ok" and row.detail.startswith("replied 'ok' (9 in, 1 out)")
    assert row.detail.endswith("reasoning_effort=low accepted")
    (request,) = seen
    body = json.loads(request.content)
    assert body["max_tokens"] == 5 and body["messages"] == [{"role": "user", "content": "Reply with the single word: ok"}]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert all(key not in str(request.url) for key in KEYS.values())  # keys go in a header, never in the URL


def test_a_model_that_spends_its_five_tokens_thinking_still_counts_as_reached():
    (row,) = probe_real(lambda request: completion("", finish="length"))
    assert row.status == "ok"
    assert row.detail.startswith("reached, but the reply was cut at 5 tokens")


def test_a_refused_reasoning_effort_is_reported_as_dropped():
    def handler(request):
        if "reasoning_effort" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'reasoning_effort'"}})
        return completion("ok")

    (row,) = probe_real(handler)
    assert row.status == "ok" and row.detail.endswith("reasoning_effort=low refused, so it is dropped")


def test_a_refused_key_is_failed_auth_after_one_request_and_never_shows_the_key():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {KEYS['OPENAI_API_KEY']}"}})

    (row,) = probe_real(handler)
    assert row.status == "failed(auth)" and len(seen) == 1
    assert KEYS["OPENAI_API_KEY"] not in row.detail and "401" in row.detail


def test_an_endpoint_that_cannot_be_reached_is_failed_transport():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    (row,) = probe_real(handler)
    assert row.status == "failed(transport)"


# -- the table -----------------------------------------------------------------------------------


def test_the_table_has_a_header_aligned_columns_and_the_role_once_per_group():
    text = format_rows(
        [
            Row("llm.fast", "openai:gpt-a", "ok", 'replied "ok"', 0.4231),
            Row("llm.fast", "gemini:gem-b", "failed(auth)", "gemini answered 401", 1.0),
            Row("stt", "elevenlabs:scribe", "skipped(no key)", "ELEVENLABS_API_KEY is not set"),
        ]
    )
    lines = text.splitlines()
    assert lines[0].split() == ["ROLE", "SLOT", "STATUS", "TIME", "DETAIL"]
    assert lines[1].startswith("llm.fast") and lines[2].startswith(" ") and lines[3].startswith("stt")
    assert "0.42s" in lines[1] and "1.00s" in lines[2] and " - " in lines[3]
    columns = {line.index(word) for line, word in ((lines[0], "STATUS"), (lines[1], "ok"), (lines[2], "failed(auth)"))}
    assert len(columns) == 1  # the status column lines up
    assert not any(line != line.rstrip() for line in lines)


def test_a_long_detail_is_shortened_unless_asked_for_whole_and_stays_on_one_line():
    row = Row("stt", "s", "failed(server)", "line one\nline two " + "x" * 300)
    short = format_rows([row]).splitlines()
    assert len(short) == 2 and short[1].endswith("…") and "\n" not in short[1]
    whole = format_rows([row], detail_width=None).splitlines()
    assert len(whole) == 2 and "x" * 300 in whole[1]
    assert format_rows([]) == "nothing is configured"


def test_failed_is_true_for_a_live_failure_or_an_unusable_role_and_not_for_a_skip():
    assert not failed([Row("r", "s", "ok", ""), Row("r", "s", "skipped(no key)", ""), Row("r", "s", "ready", "")])
    assert failed([Row("r", "s", "failed(auth)", "")]) and failed([Row("r", "-", "error", "")])


# -- the command ---------------------------------------------------------------------------------


@pytest.fixture
def clean_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no glide.toml and no .env here, whatever the repository holds
    return tmp_path


def test_the_command_prints_the_source_the_built_in_roles_and_the_table(clean_cwd, capsys):
    path = clean_cwd / "mine.toml"
    path.write_text(FAST + "\n", encoding="utf-8")
    code = main(["--config", str(path)], env=KEYS)
    out = capsys.readouterr().out
    assert code == 0
    assert f"config: {path}" in out
    assert "built-in chains in use for: llm.smart, stt, tts, classifier" in out
    assert "openai:gpt-a" in out and "ROLE" in out
    assert all(key not in out for key in KEYS.values())


def test_the_command_exits_one_when_a_role_is_unusable_and_two_when_the_file_cannot_load(clean_cwd, capsys):
    assert main([], env={}) == 1  # no keys at all: the roles that need one are errors
    out = capsys.readouterr().out
    assert "config: built-in defaults" in out and "OPENAI_API_KEY" in out
    assert main(["--config", str(clean_cwd / "missing.toml")], env={}) == 2
    assert "glide doctor:" in capsys.readouterr().err
    assert main(["--role", "tts"], env={}) == 0  # macOS say needs no key, so that role is fine
    assert "macos_say" in capsys.readouterr().out


def test_live_is_only_ever_requested_with_the_flag(clean_cwd, capsys, monkeypatch):
    calls = []

    def fake_doctor(config, **kw):
        calls.append(kw)
        return [Row("llm.fast", "openai:gpt-a", "failed(auth)", "nope", 0.5)]

    monkeypatch.setattr(doctor_module, "doctor", fake_doctor)
    assert main(["--role", "llm.fast"], env=KEYS) == 1
    assert main(["--live", "--role", "llm.fast", "--timeout", "4"], env=KEYS) == 1
    assert [(c["live"], c["roles"], c["timeout"]) for c in calls] == [(False, ["llm.fast"], None), (True, ["llm.fast"], 4.0)]
    out = capsys.readouterr().out
    assert out.count("live: one tiny request") == 1 and "failed(auth)" in out


def test_the_command_reads_dotenv_into_the_real_environment_only(clean_cwd, capsys, monkeypatch):
    (clean_cwd / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv-0123456789\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {"HOME": str(clean_cwd)})  # a private environment, so nothing leaks out of the test
    assert main(["--role", "llm.smart"]) == 0
    assert "OPENAI_API_KEY is set" in capsys.readouterr().out
    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv-0123456789"
    # with an environment given, .env is not read at all
    assert main(["--role", "llm.smart"], env={}) == 1
    capsys.readouterr()


def test_the_command_never_sends_anything_without_live(clean_cwd, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a request was sent")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    assert main([], env=KEYS) in (0, 1)  # real adapters, built and closed, nothing sent
