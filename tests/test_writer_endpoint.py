"""The writer over a real (loopback) chat endpoint, through the provider chains of a glide.toml.

What used to be the OpenAI-compatible writer's own negotiation now lives in the one adapter every role shares
(glide/providers/llm.py); these tests pin that the writer still gets it.
"""

from __future__ import annotations

import json

import pytest
from writer_endpoint import config_for, writer_for

from glide.computer.calls import Calls, MeteredWriter
from glide.computer.writer import WriterError, compose_answer, compose_url, make_writer, provider

ANSWER = '{"achieved": true, "answer": "Sep 19 in Miami.", "focus": "", "question": ""}'


def metered(url, **kw):
    """Metered, as the runner hands it out."""
    return MeteredWriter(writer_for(url, **kw), Calls())


def test_a_request_goes_to_chat_completions_with_the_schema_and_only_the_slots_key(clean_env, endpoint):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    clean_env.setenv("OPENAI_API_KEY", "sk-openai-real")
    writer = writer_for(endpoint.url, key_env="WRITER_TEST_KEY", env={"WRITER_TEST_KEY": "proxy-key"})

    assert compose_url(writer, "open example", []) == "https://example.com"

    (request,) = endpoint.seen
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer proxy-key"
    assert not any("real" in v for v in request["headers"].values())
    body = request["body"]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert '"required": ["ok", "url", "reason"]' in body["messages"][0]["content"]  # json_object needs "JSON" in the prompt
    assert body["response_format"]["json_schema"]["schema"]["required"] == ["ok", "url", "reason"]
    assert body["max_tokens"] == 200 and body["model"] == "fast-model"


def test_an_endpoint_of_its_own_needs_no_key(clean_env, endpoint):
    assert compose_url(writer_for(endpoint.url), "open example", []) == "https://example.com"
    assert "authorization" not in endpoint.seen[0]["headers"]


def test_without_a_usable_slot_there_is_no_writer(clean_env):
    config = config_for("http://127.0.0.1:9", key_env="WRITER_TEST_KEY", env={})  # its key variable is not set
    assert make_writer(config) is None


def test_the_answer_goes_to_the_smart_model_with_the_capture_as_a_data_url(clean_env, endpoint, screen, make_item):
    endpoint.state["reply"] = ANSWER
    answer = compose_answer(
        writer_for(endpoint.url), "find the concert", screen, [make_item(0, "SEP 19")], [], "the goal is achieved"
    )
    assert answer.text == "Sep 19 in Miami." and answer.achieved
    body = endpoint.seen[0]["body"]
    assert body["model"] == "smart-model"
    image, text = body["messages"][1]["content"]
    assert image["type"] == "image_url" and image["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(text["text"])["screen_text_in_reading_order"] == ["SEP 19"]


def test_a_text_only_model_gets_the_screen_text_and_no_capture(clean_env, endpoint, screen, make_item):
    clean_env.setenv("CLICKER_WRITER_VISION", "false")
    endpoint.state["reply"] = ANSWER
    compose_answer(writer_for(endpoint.url), "find the concert", screen, [make_item(0, "SEP 19")], [], "the goal is achieved")
    assert [part["type"] for part in endpoint.seen[0]["body"]["messages"][1]["content"]] == ["text"]


def test_usage_is_counted_with_cached_tokens_apart(clean_env, endpoint):
    endpoint.state["usage"] = {
        "prompt_tokens": 1200,
        "completion_tokens": 30,
        "total_tokens": 1230,
        "prompt_tokens_details": {"cached_tokens": 1000},
    }
    calls = Calls()
    compose_url(MeteredWriter(writer_for(endpoint.url), calls), "open example", [])
    (usage,) = calls.usage.values()
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 200, 1000, 30)


def test_an_endpoint_that_reports_no_usage_still_counts_the_request(clean_env, endpoint):
    calls = Calls()
    compose_url(MeteredWriter(writer_for(endpoint.url), calls), "open example", [])
    (usage,) = calls.usage.values()
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 0, 0, 0)


def test_a_refused_format_steps_down_once_and_stays_down(clean_env, endpoint):
    endpoint.state["reject"] = lambda body: (
        "response_format json_schema is not supported" if body.get("response_format", {}).get("type") == "json_schema" else None
    )
    w = writer_for(endpoint.url)
    assert compose_url(w, "open example", []) == "https://example.com"
    assert compose_url(w, "open example again", []) == "https://example.com"
    formats = [r["body"].get("response_format", {}).get("type") for r in endpoint.seen]
    assert formats == ["json_schema", "json_object", "json_object"]


def test_an_endpoint_with_no_json_mode_at_all_still_gets_the_schema_in_the_prompt(clean_env, endpoint):
    endpoint.state["reject"] = lambda body: "unknown parameter: response_format" if "response_format" in body else None
    endpoint.state["reply"] = 'Sure:\n```json\n{"ok": true, "url": "https://example.com", "reason": ""}\n```'
    assert compose_url(writer_for(endpoint.url), "open example", []) == "https://example.com"
    assert "response_format" not in endpoint.seen[-1]["body"]
    assert '"required": ["ok", "url", "reason"]' in endpoint.seen[-1]["body"]["messages"][0]["content"]


def test_a_model_that_wants_max_completion_tokens_gets_them(clean_env, endpoint):
    endpoint.state["reject"] = lambda body: (
        "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead." if "max_tokens" in body else None
    )
    assert compose_url(writer_for(endpoint.url), "open example", []) == "https://example.com"
    assert endpoint.seen[-1]["body"]["max_completion_tokens"] == 200


def test_reasoning_effort_is_the_slots_own_option_and_dropped_for_good_when_refused(clean_env, endpoint):
    endpoint.state["reject"] = lambda body: (
        "Unrecognized request argument: reasoning_effort" if "reasoning_effort" in body else None
    )
    config = config_for(endpoint.url).__class__.from_toml(
        f"""
        [providers.local]
        kind = "openai_compat"
        base_url = "{endpoint.url}/v1"
        [llm.fast]
        chain = [{{ provider = "local", model = "fast-model", options = {{ reasoning_effort = "low" }} }}]
        [llm.smart]
        chain = ["local:smart-model"]
        """,
        env={},
    )
    w = make_writer(config)
    assert compose_url(w, "open example", []) == "https://example.com"
    assert compose_url(w, "open example again", []) == "https://example.com"
    assert ["reasoning_effort" in r["body"] for r in endpoint.seen] == [True, False, False]


def test_a_refusal_is_a_writer_error_that_says_what_the_endpoint_said_and_is_not_a_retry_loop(clean_env, endpoint):
    endpoint.state["reject"] = lambda body: "model 'nope' not found"
    with pytest.raises(WriterError, match="not found"):
        compose_url(writer_for(endpoint.url), "open example", [])
    assert len(endpoint.seen) == 1


def test_the_startup_line_names_the_slots_and_leaves_urls_and_credentials_out(clean_env):
    line = provider(
        writer_for("http://user:secret@127.0.0.1:9", key_env="WRITER_TEST_KEY", env={"WRITER_TEST_KEY": "sk-key-1234"})
    )
    assert "fast: local:fast-model" in line and "smart: local:smart-model" in line
    assert "secret" not in line and "sk-key-1234" not in line and "127.0.0.1" not in line
