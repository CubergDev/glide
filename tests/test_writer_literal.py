"""Literal navigation needs no generated text, and the planner and research stages ask by role, not by model."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from glide.computer import writer
from glide.computer.generation import GenerationError, GenerationRequest, GenerationResult, GenerationUnavailable
from glide.computer.models import Guidance
from glide.computer.writer import WriterError, WriterUnavailable, compose_url, literal_browser_url


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("open google.com", "https://google.com"),
        ("Open google.com.", "https://google.com"),
        ("Uh, hey, can you open Google.com for me on my Brave browser?", "https://Google.com"),
        ("Please visit https://example.com/docs?q=hello", "https://example.com/docs?q=hello"),
        ("go to example.com in Safari", "https://example.com"),
    ],
)
def test_literal_navigation_never_calls_the_writer(goal, expected):
    fake = SimpleNamespace(generate=lambda *a, **kw: pytest.fail("A literal URL needs no generated text"))
    assert literal_browser_url(goal) == expected
    assert compose_url(fake, goal, []) == expected


@pytest.mark.parametrize(
    "goal",
    [
        "Open google.com and search for cats",
        "Don't open google.com",
        "read the link google.com aloud",
        "open https://name:password@example.com",
        "open javascript:alert(1)",
        "open file:///tmp/private",
        "open http://example.com",
        "open example.com\nUser correction: use another site",
        "open example.com:99999",
        "open example..com",
        "open -example.com",
        "open example.com-",
    ],
)
def test_a_complex_or_unsafe_goal_cannot_take_the_literal_shortcut(goal):
    assert literal_browser_url(goal) == ""


def test_a_correction_in_the_guidance_turns_the_shortcut_off():
    asked = []
    fake = SimpleNamespace(
        generate=lambda request, cancel=None: (
            asked.append(request) or GenerationResult('{"ok": true, "url": "https://other.example", "reason": ""}', "m")
        )
    )
    guided = Guidance().focused("use the other site")
    assert compose_url(fake, "open example.com", [], guided) == "https://other.example"
    assert len(asked) == 1


class Scripted:
    def __init__(self, text: str = "{}", error: Exception | None = None, completed: bool = True):
        self.text, self.error, self.completed = text, error, completed
        self.requests: list[GenerationRequest] = []

    def generate(self, request, cancel=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return GenerationResult(self.text, "m", completed=self.completed)


def test_task_routing_asks_the_task_routing_role_and_no_model():
    fake = Scripted('{"route": "plan", "question": ""}')
    assert writer.route_task(fake, {"goal": "x"}) == ("plan", "")
    (request,) = fake.requests
    assert (request.role, request.model, request.deadline_s) == ("task_routing", "", writer.WRITER_DEADLINE_S)


def test_a_clarify_route_needs_a_question_and_no_other_route_may_have_one():
    assert writer.route_task(Scripted('{"route": "clarify", "question": "which one?"}'), {}) == ("clarify", "which one?")
    for bad in (
        '{"route": "clarify", "question": ""}',
        '{"route": "plan", "question": "why?"}',
        '{"route": "fly", "question": ""}',
    ):
        with pytest.raises(WriterError):
            writer.route_task(Scripted(bad), {})


def test_research_stages_ask_research_roles_and_keep_provider_text_out_of_their_errors():
    fake = Scripted(json.dumps({"supported": True, "complete": False, "feedback": "x"}))
    assert writer.review_research(fake, {})["complete"] is False
    assert fake.requests[0].role == "research_verification" and fake.requests[0].deadline_s == writer.RESEARCH_DEADLINE_S

    for reply in (Scripted("not json PROVIDER-TEXT"), Scripted('{"supported": true}'), Scripted("{}", completed=False)):
        with pytest.raises(WriterError) as caught:
            writer.review_research(reply, {})
        assert "PROVIDER-TEXT" not in str(caught.value)


def test_a_lost_provider_stops_the_research_and_a_missing_writer_says_so():
    with pytest.raises(WriterUnavailable):
        writer.compose_research(Scripted(error=GenerationUnavailable("no credit")), {})
    with pytest.raises(WriterError) as refused:
        writer.compose_research(Scripted(error=GenerationError("bad")), {})
    assert not isinstance(refused.value, WriterUnavailable)
    with pytest.raises(WriterError, match="configured language model"):
        writer.compose_research(None, {})


def test_the_planner_asks_the_planner_role_with_a_schema_built_from_the_execution_contracts(monkeypatch):
    package = types.ModuleType("glide.computer.execution")
    package.__path__ = []
    contracts = types.ModuleType("glide.computer.execution.contracts")
    contracts.EFFECTS = {"url", "tab_created", "scroll", "element_present"}
    contracts.MAX_PLAN_STEPS, contracts.MAX_REPETITIONS = 8, 5
    contracts.safe_url = lambda url: url.startswith("https://")
    monkeypatch.setitem(sys.modules, "glide.computer.execution", package)
    monkeypatch.setitem(sys.modules, "glide.computer.execution.contracts", contracts)
    fake = Scripted('{"question": "", "steps": [], "unsupported": []}')

    assert writer.compose_plan(fake, {"task_context": {}, "observation": {}}) == {"question": "", "steps": [], "unsupported": []}

    (request,) = fake.requests
    assert (request.role, request.model, request.deadline_s) == ("planner", "", writer.PLANNER_DEADLINE_S)
    assert request.schema["required"] == ["question", "steps", "unsupported"]
