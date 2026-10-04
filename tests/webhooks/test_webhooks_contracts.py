"""The messages and the strict JSON reader. Needs only pydantic."""

import uuid

import pytest
from pydantic import ValidationError

from glide.webhooks.contracts import AgentCall, AgentEvent, Completion, TranslationError, call_id, strict_json
from webhooks.helpers import make_call

LEASE = "t" * 32


def fields(**changes):
    call = make_call("agent.task.requested")
    return {**call.model_dump(), **changes}


def test_only_a_task_request_can_carry_the_action_flag():
    assert AgentCall(**fields(allow_actions=True)).allow_actions is True
    for operation in (
        "github.issue.triage",
        "github.pull_request.review",
        "github.comment.summarize",
        "gmail.mail.changed",
        "outlook.mail.changed",
    ):
        with pytest.raises(ValidationError):
            AgentCall(**fields(operation=operation, allow_actions=True))
        assert AgentCall(**fields(operation=operation)).allow_actions is False


def test_the_action_flag_is_off_unless_it_is_given():
    data = fields()
    data.pop("allow_actions")
    assert AgentCall(**data).allow_actions is False


@pytest.mark.parametrize("identity", [str(uuid.uuid4()).upper(), "not-a-uuid", "{" + str(uuid.uuid4()) + "}", ""])
def test_message_and_task_ids_must_be_canonical_uuids(identity):
    for name in ("id", "task_id"):
        with pytest.raises(ValidationError):
            AgentCall(**fields(**{name: identity}))


@pytest.mark.parametrize("context", [{"text": "x" * 16384}, {"n": float("nan")}, {"n": float("inf")}, {"handler": object()}])
def test_the_context_is_bounded_json(context):
    with pytest.raises(ValidationError):
        AgentCall(**fields(context=context))


def test_a_context_just_inside_the_bound_is_kept():
    context = {"text": "x" * (16384 - len('{"text":""}') - 1)}
    assert AgentCall(**fields(context=context)).context == context


def test_a_call_is_strict_closed_and_immutable():
    for bad in ({"extra": 1}, {"goal": ""}, {"goal": "g" * 4097}, {"event_id": ""}, {"agent_id": "Bad Name"}, {"source": "../x"}):
        with pytest.raises(ValidationError):
            AgentCall(**fields(**bad))
    with pytest.raises(ValidationError):
        AgentCall(**fields(allow_actions="true"))
    call = AgentCall(**fields())
    with pytest.raises(ValidationError):
        call.goal = "changed"
    with pytest.raises(ValidationError):
        AgentCall(**fields(operation="shell.run"))


def test_call_ids_are_stable_and_bound_to_source_and_event():
    assert call_id("a", "1") == call_id("a", "1")
    assert len({call_id("a", "1"), call_id("a", "2"), call_id("b", "1")}) == 3
    assert str(uuid.UUID(call_id("a", "1"))) == call_id("a", "1")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a": 1, "a": 2}',
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b'{"a": -Infinity}',
        b"[1]",
        b'"x"',
        b"null",
        b"\xff\xfe",
        b"{",
        b"[" * 100000,
    ],
)
def test_strict_json_accepts_only_one_plain_object(raw):
    with pytest.raises(TranslationError) as error:
        strict_json(raw)
    assert str(error.value) in {"Invalid JSON object.", "Duplicate JSON keys."}


def test_strict_json_reads_a_plain_object():
    assert strict_json('{"goal": "查看", "n": [1, 2.5, null]}'.encode()) == {"goal": "查看", "n": [1, 2.5, None]}


def test_a_worker_result_and_an_event_need_a_real_lease_token_and_known_words():
    with pytest.raises(ValidationError):
        Completion(lease_token="short", outcome="completed")
    with pytest.raises(ValidationError):
        Completion(lease_token=LEASE, outcome="execute_shell")
    with pytest.raises(ValidationError):
        Completion(lease_token=LEASE, outcome="completed", note="n" * 201)
    assert Completion(lease_token=LEASE, outcome="uncertain").summary == ""
    for bad in (
        {"kind": "shell"},
        {"kind": "progress", "elapsed_s": -1},
        {"kind": "progress", "elapsed_s": float("nan")},
        {"kind": "progress", "elapsed_s": 86401},
        {"kind": "progress", "task_id": "x"},
        {"kind": "progress", "text": "t" * 4097},
    ):
        with pytest.raises(ValidationError):
            AgentEvent(lease_token=LEASE, **bad)
    event = AgentEvent(lease_token=LEASE, kind="progress", elapsed_s=1.5)
    assert event.model_dump(exclude={"lease_token"})["elapsed_s"] == 1.5
