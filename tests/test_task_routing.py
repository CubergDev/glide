"""Who owns a task: `routing.decide` as a pure function, and the engine's use of its answer.

The frontier routes here run against `FakeSupervisor` (execution_world.py), put where `research.Supervisor` is; the real
one runs in test_engine_research.py. What is asserted is what the engine owns: which route was taken, who was asked,
what was never touched.
"""

import json

import pytest
from execution_world import Computer, FakeSupervisor, Jev, Reasoner, drive, response

from glide.computer.execution import routing
from glide.computer.execution.contracts import InvalidAction, Milestone
from glide.computer.writer import Answer, WriterError
from glide.providers.errors import ProviderError

B = "https://source-b.test/page"


def classifier(choice, confidence=0.99):
    """A classifier that always picks `choice` for the scope question, with the given confidence."""
    jev = Jev(choice)
    ask = jev.system_one

    def system_one(state, questions):
        result = ask(state, questions)
        result.answers["action"].confidence = confidence
        return result

    jev.system_one = system_one
    return jev


# -- routing.decide: a pure function of the classifier, the writer and the goal ---------------------------------


@pytest.mark.parametrize("workflow", ["execute", "research", "reason", "clarify"])
def test_decide_returns_what_the_classifier_chose_and_asks_no_writer(workflow):
    writer = Reasoner([])
    jev = classifier(workflow)
    scope = routing.decide(jev, writer, "Do the thing", minimum=0.4)
    assert (scope.workflow, scope.source, scope.question) == (workflow, "jev", "")
    assert scope.owner == ("frontier" if workflow in {"research", "reason"} else "jev")
    assert not writer.requests and len(jev.requests) == 1
    assert jev.requests[0] == {"task_scope": True, "goal": "Do the thing"}


def test_decide_passes_clarifications_to_the_classifier():
    jev = classifier("reason")
    routing.decide(jev, Reasoner([]), "Help", minimum=0.4, clarifications=["Which?\nUser: the sums"])
    assert jev.requests[0]["clarifications"] == ["Which?\nUser: the sums"]


def test_decide_asks_the_writer_only_when_the_classifier_is_unsure():
    writer = Reasoner([{"route": "research", "question": ""}])
    scope = routing.decide(classifier("execute", confidence=0.1), writer, "Compare two pages", minimum=0.4)
    assert (scope.workflow, scope.source) == ("research", "writer")
    assert [r.role for r in writer.requests] == ["task_routing"]


def test_decide_keeps_the_writers_clarifying_question():
    writer = Reasoner([{"route": "clarify", "question": "Which site?"}])
    scope = routing.decide(classifier("execute", confidence=0.1), writer, "Open it", minimum=0.4)
    assert (scope.workflow, scope.question) == ("clarify", "Which site?")


def test_decide_does_not_guess_when_the_classifier_chain_is_down():
    writer = Reasoner([])
    jev = Jev()

    def down(state, questions):
        raise ProviderError("every classifier provider failed", kind="exhausted")

    jev.system_one = down
    with pytest.raises(ProviderError):
        routing.decide(jev, writer, "Anything", minimum=0.4)
    assert not writer.requests


def test_decide_refuses_an_unusable_writer_decision():
    writer = Reasoner([{"route": "execute", "question": ""}])
    with pytest.raises(WriterError):
        routing.decide(classifier("execute", confidence=0.1), writer, "Anything", minimum=0.4)


def test_record_names_the_owner_and_never_the_goal(monkeypatch):
    from glide.computer import diagnostics

    recorder = diagnostics.Diagnostics(None)
    with recorder.activate():
        item = routing.record(routing.Scope("research", "jev"))
    assert item == {"workflow": "research", "source": "jev", "owner": "frontier"}
    assert recorder.events[0]["event"] == "task_route"


# -- the engine's use of the scope --------------------------------------------------------------------------------


def test_compound_browser_work_stays_on_the_ordinary_planner_and_jev(monkeypatch, tmp_path):
    computer, writer = (
        Computer(),
        Reasoner(
            [
                response(
                    Milestone("tab", "Open another tab", "tab_created", value=B, quantity=3),
                    Milestone("scroll", "Scroll feed", "scroll", target="Feed", value="down", quantity=2),
                )
            ]
        ),
    )
    state = drive(monkeypatch, tmp_path, computer, writer, Jev(), goal=f"Open three tabs at {B}, then scroll down twice")
    assert state.answer.achieved and len(computer.actions) == 5
    assert [r.role for r in writer.requests] == ["planner"]
    report = json.loads((tmp_path / "run.json").read_text())
    assert all(r["owner"] == "jev" for r in report["routing"]) and "research" not in report


def test_uncertain_action_binding_goes_straight_to_the_planner(monkeypatch, tmp_path):
    jev = Jev("plan")
    classify = jev.system_one

    def uncertain_binding(state, questions):
        result = classify(state, questions)
        if not state.get("task_scope") and "milestone" not in state:
            assert not {"reason", "research"} & questions["action"].criteria.keys()
            result.answers["action"].confidence = 0.1
        return result

    jev.system_one = uncertain_binding
    writer = Reasoner([response(Milestone("tab", "Open the site", "tab_created", value=B))])
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal=f"Open a new tab at {B}")
    assert state.answer.achieved and len(computer.actions) == 1
    assert [r.role for r in writer.requests] == ["planner"]
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["routing"][-1]["source"] == "jev_uncertain" and "research" not in report


def test_a_direct_binding_is_recorded_as_direct(monkeypatch, tmp_path):
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), Jev("navigate"), goal="Open https://different.test")
    assert state.answer.achieved
    assert [r["workflow"] for r in json.loads((tmp_path / "run.json").read_text())["routing"]] == ["execute", "direct"]


def test_missing_browser_capability_does_not_invoke_the_frontier(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([response(unsupported=["drag"])])
    supervisor = FakeSupervisor()
    state = drive(monkeypatch, tmp_path, computer, writer, Jev(), goal="Drag this item to that folder", supervisor=supervisor)
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["drag"]
    assert not computer.actions and supervisor.goal is None
    assert [r.role for r in writer.requests] == ["planner"]
    assert json.loads((tmp_path / "run.json").read_text())["failure_stage"] == "planning"


def test_reasoning_needs_no_browser_and_the_supervisor_gets_the_original_goal(monkeypatch, tmp_path):
    computer = Computer()

    def refuse(_):
        pytest.fail("Pure reasoning attempted to inspect the browser")

    computer.on_inspect = refuse
    supervisor = FakeSupervisor([("answer", Answer("def add(a, b): return a + b", True))])
    goal = "Write a Python function to add two numbers"
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), Jev("reason"), goal=goal, supervisor=supervisor)
    assert state.outcome == "done" and state.answer.text.startswith("def add")
    assert (supervisor.goal, supervisor.route) == (goal, "reason")
    assert supervisor.advances[0]["backend"] is None and supervisor.advances[0]["owner"] == ""
    assert not computer.actions and computer.reads == 0 and not computer.closed
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["transport"] == "none" and report["routing"][0]["owner"] == "frontier"
    assert report["research"] == {"route": "reason", "model_calls": 1}
    assert state.progress == [{"id": "research_answer", "requested": 1, "verified": 1, "remaining": 0}]


def test_research_runs_each_browser_batch_through_jev_and_then_takes_the_answer(monkeypatch, tmp_path):
    computer = Computer()
    batch = [Milestone("open", "Open the source", "url", value=B)]
    supervisor = FakeSupervisor([("browse", batch), ("answer", Answer("Both pages agree.", True, spoken_text="Agree."))])
    jev = Jev("research")
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), jev, goal="Compare the sources", supervisor=supervisor)
    assert state.outcome == "done" and state.answer.text == "Both pages agree."
    assert [a.kind for a in computer.actions] == ["navigate"] and computer.state.url == B
    assert [a["progress"] for a in supervisor.advances] == [
        [],
        [{"id": "research1_open", "requested": 1, "verified": 1, "remaining": 0}],
    ]
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["plan_revisions"] == 1 and report["research"]["model_calls"] == 2
    assert any("milestone" in packet for packet in jev.requests)  # Jev, not the frontier, chose the browser action


def test_the_frontier_failing_is_a_provider_failure_never_a_fake_completion(monkeypatch, tmp_path):
    error = ProviderError("provider answered 500: private server words", kind="server", provider="slot")
    supervisor = FakeSupervisor([("raise", error)])
    computer = Computer()
    state = drive(
        monkeypatch, tmp_path, computer, Reasoner([]), Jev("reason"), goal="Explain this algorithm", supervisor=supervisor
    )
    assert state.outcome == "provider failure" and state.answer is None and not computer.actions
    assert "private server words" not in state.failure and "internal error" in state.failure


def test_a_supervisor_that_blocks_the_task_ends_it_with_its_reason(monkeypatch, tmp_path):
    supervisor = FakeSupervisor([("raise", InvalidAction("This reasoning task needs external evidence; start a research task."))])
    state = drive(monkeypatch, tmp_path, Computer(), Reasoner([]), Jev("reason"), goal="Explain", supervisor=supervisor)
    assert state.outcome == "blocked" and "external evidence" in state.failure


def test_scope_clarification_can_resolve_to_reasoning_without_the_browser(monkeypatch, tmp_path):
    computer, jev = Computer(), Jev("clarify")
    classify = jev.system_one

    def resolve_scope(state, questions):
        jev.route = "reason" if state.get("clarifications") else "clarify"
        return classify(state, questions)

    def refuse(_):
        pytest.fail("An unresolved task or pure reasoning accessed the browser")

    jev.system_one, computer.on_inspect = resolve_scope, refuse
    supervisor = FakeSupervisor([("answer", Answer("The answer is 16.", True))])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([]),
        jev,
        goal="Help me with this",
        ask=lambda _: "Calculate 200 minus 70 minus 90 minus 24.",
        supervisor=supervisor,
    )
    assert state.answer.achieved and not computer.actions and computer.reads == 0
    assert supervisor.goal == "Help me with this"
    assert "Calculate 200 minus 70 minus 90 minus 24" in supervisor.replies[0]


# -- a route the router already decided (RunConfig.route) -------------------------------------------------------


def no_scope_question(choice):
    """A Jev that fails the test if the engine asks the scope question: the router already answered it."""
    jev = Jev(choice)
    ask = jev.system_one

    def system_one(state, questions):
        assert not state.get("task_scope"), "the engine asked who owns the task after the router had decided"
        return ask(state, questions)

    jev.system_one = system_one
    return jev


def test_a_route_from_the_router_is_the_scope_and_the_scope_question_is_never_asked(monkeypatch, tmp_path):
    computer = Computer()
    supervisor = FakeSupervisor([("answer", Answer("Both pages agree.", True))])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([]),
        no_scope_question("execute"),  # what the classifier would say: the router's word wins
        goal="Compare the sources",
        supervisor=supervisor,
        route="research",
    )
    assert state.outcome == "done" and supervisor.route == "research"
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["routing"] == [{"workflow": "research", "source": "router", "owner": "frontier"}]


def test_a_reason_route_from_the_router_needs_no_browser(monkeypatch, tmp_path):
    computer = Computer()
    computer.on_inspect = lambda _: pytest.fail("a routed reason task attempted to inspect the browser")
    supervisor = FakeSupervisor([("answer", Answer("391", True))])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([]),
        no_scope_question("execute"),
        goal="17 times 23",
        supervisor=supervisor,
        route="reason",
    )
    assert state.outcome == "done" and supervisor.route == "reason" and not computer.actions


@pytest.mark.parametrize("route", ["", "clarify", "plan", "anything else"])
def test_without_a_usable_route_the_engine_still_decides_for_itself(monkeypatch, tmp_path, route):
    supervisor = FakeSupervisor([("answer", Answer("def f(): pass", True))])
    state = drive(
        monkeypatch, tmp_path, Computer(), Reasoner([]), Jev("reason"), goal="Write f", supervisor=supervisor, route=route
    )
    assert state.outcome == "done" and supervisor.route == "reason"
    assert json.loads((tmp_path / "run.json").read_text())["routing"][0]["source"] == "jev"


def test_the_task_runner_hands_the_route_to_the_run_config(monkeypatch, tmp_path):
    from glide.assistant import tasks
    from glide.computer import runner
    from glide.computer.platform_adapter import desktop
    from glide.computer.runner import RunState

    seen = []

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        seen.append(cfg.route)
        return RunState(outcome="dry run")

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    config = type("Config", (), {"writer": lambda self: object(), "classifier": lambda self: object()})()
    runner_ = tasks.TaskRunner(config, tmp_path)
    for route in ("research", ""):
        task = runner_.start("do it", route=route)
        assert task.wait(5) and task.route == route
    assert seen == ["research", ""]  # "" is a run nothing routed: the engine decides for itself
