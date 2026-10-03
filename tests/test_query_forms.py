"""Dynamic observed forms, unseen text and destinations through the real engine."""

from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution import progress
from glide.computer.execution.contracts import Element, InvalidAction, Milestone, rebind, validate
from glide.computer.execution.query import FORM_CAPABILITY, QueryForm, intent


class QueryComputer(Computer):
    def __init__(self, destination, parameter):
        super().__init__()
        self.state.app = "browser"
        self.state.url = destination
        self.state.tabs["initial"] = destination
        self.state.capabilities.add("query_form")
        self.state.elements = {"input": Element("input", "Unseen query label", "textbox", "", True)}
        self.state.forms = {"form": QueryForm("form", "input", destination + "/lookup", parameter, (("lang", "auto"),))}


@pytest.mark.parametrize(
    "query", ["best biryani spot near Wong Chuk Hang", "香港 茶餐廳", "São Paulo & café", "2037 budget + 15%", "京都の天気"]
)
@pytest.mark.parametrize(
    "destination,parameter", [("https://google.com", "q"), ("https://unseen.test", "term"), ("https://library.test", "keywords")]
)
def test_quick_queries_compile_observed_endpoints_without_planning(monkeypatch, tmp_path, query, destination, parameter):
    computer = QueryComputer(destination, parameter)
    writer = Reasoner([{"query": query, "destination": destination, "question": "", "compound": False}])
    jev = Jev(route="query")

    # The provenance is code-owned; select the offered navigation through Jev.
    def select(state, options):
        import json

        return next(k for k, v in options.items() if k not in {"key", "replan"} and json.loads(v)["kind"] == "navigate")

    jev.selection = select
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal=f"Open {destination} and search {query}")
    assert state.answer.achieved and state.outcome == "done"
    assert len(computer.actions) == 1 and computer.actions[0].kind == "navigate"
    assert urlsplit(computer.state.url).path == "/lookup"
    assert parse_qs(urlsplit(computer.state.url).query)[parameter] == [query]
    assert writer.requests[0].role == "writer" and not writer.requests[0].model
    assert len(writer.requests) == 1 and all(p["remaining"] == 0 for p in state.progress)


def test_compound_goal_retains_all_requirements_in_planner(monkeypatch, tmp_path):
    computer = QueryComputer("https://unseen.test", "term")
    steps = [
        Milestone("query", "Submit exact query", "query_submitted", target="search form", value="café"),
        Milestone("scroll", "Scroll results", "scroll", target="Feed", value="down", quantity=3),
    ]
    writer = Reasoner([{"query": "", "destination": "https://unseen.test", "question": "", "compound": True}, response(*steps)])
    jev = Jev(route="query")

    def select(state, options):
        import json

        kind = "navigate" if state["milestone"]["effect"] == "query_submitted" else "scroll"
        return next(k for k, v in options.items() if k not in {"key", "replan"} and json.loads(v)["kind"] == kind)

    jev.selection = select
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Search café and scroll three times")
    assert state.answer.achieved and len(computer.actions) == 4
    assert [r.role for r in writer.requests] == ["writer", "planner"]


def test_query_contract_is_semantic_and_bindings_cannot_override_observed_form():
    observed = QueryComputer("https://unseen.test", "text").inspect()
    step = Milestone("query", "Submit query", "query_submitted", target="form description", value="a&b 香港")
    form = observed.forms["form"]
    from glide.computer.execution.policy import candidates

    action = next(a for a in candidates(step, observed) if a.parameter_source == "form:form")
    assert {a.kind for a in candidates(step, observed)} == {"inspect", "navigate"}
    validate(action, observed)
    with pytest.raises(InvalidAction):
        validate(replace(action, value="https://unrelated.test/?q=stolen"), observed)
    fresh = QueryComputer("https://unseen.test", "text").inspect()
    fresh.forms["form"] = replace(form, action="https://unseen.test/changed")
    with pytest.raises(InvalidAction, match="form changed"):
        rebind(action, observed, fresh)


@pytest.mark.parametrize(
    "change",
    [
        {"action": "https://other.test/search"},
        {"action": "javascript:bad()"},
        {"parameter": "api_key"},
        {"values": (("token", "sensitive"),)},
    ],
)
def test_cross_origin_or_credential_forms_are_not_offered(change):
    observed = QueryComputer("https://unseen.test", "q").inspect()
    with pytest.raises(InvalidAction):
        replace(observed.forms["form"], **change).destination(observed, "query")


def test_stop_after_extraction_prevents_navigation(monkeypatch, tmp_path):
    computer = QueryComputer("https://unseen.test", "q")
    control = RunControl("cancel-extraction")
    writer = Reasoner([{"query": "café", "destination": "https://unseen.test", "question": "", "compound": False}])
    original = writer.generate

    def cancel(request, token=None):
        result = original(request, token)
        control.cancel()
        return result

    writer.generate = cancel
    state = drive(monkeypatch, tmp_path, computer, writer, Jev(route="query"), control, goal="Search café")
    assert state.outcome == "aborted" and not computer.actions


def test_extra_goal_cannot_be_discarded_by_a_query_only_request():
    observed = QueryComputer("https://unseen.test", "q").inspect()
    writer = Reasoner([{"query": "food", "destination": "https://unseen.test", "question": "", "compound": True}])
    assert intent(writer, "Search food then compare prices", observed, "https://unseen.test") == (None, "")


@pytest.fixture
def poll_clock(monkeypatch):
    """Exercise readiness deadlines without real sleeps or browser/network access."""
    clock = SimpleNamespace(elapsed=0.0, sleeps=[], on_sleep=None)

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.elapsed += seconds
        if clock.on_sleep:
            clock.on_sleep()

    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: clock.elapsed, sleep=sleep))
    return clock


def select_query_navigation(state, options):
    import json

    return next(k for k, v in options.items() if k != "replan" and json.loads(v)["kind"] == "navigate")


def delayed_query_page(destination, parameter, *, appears_after=5):
    computer = QueryComputer(destination, parameter)
    fields, forms = computer.state.elements, computer.state.forms
    computer.state.url = "about:blank"
    computer.state.tabs["initial"] = "about:blank"
    computer.state.elements, computer.state.forms = {}, {}
    computer.destination_reads = 0

    def hydrate(machine):
        if machine.state.url == destination:
            machine.destination_reads += 1
            if appears_after is not None and machine.destination_reads >= appears_after:
                machine.state.elements, machine.state.forms = fields, forms

    computer.on_inspect = hydrate
    return computer


@pytest.mark.parametrize("route", ["query", "plan"])
@pytest.mark.parametrize("destination,parameter", [("https://google.com", "q"), ("https://unseen.test", "search_text")])
def test_ready_url_waits_for_delayed_query_form(monkeypatch, tmp_path, poll_clock, route, destination, parameter):
    computer = delayed_query_page(destination, parameter)
    text = "nearest good biryani place near Wong Shu Kang"
    writer = Reasoner(
        [{"query": text, "destination": destination, "question": "", "compound": False}]
        if route == "query"
        else [
            response(
                Milestone("nav", "Reach requested site", "url", value=destination),
                Milestone("query", "Submit query", "query_submitted", target="search form", value=text),
            )
        ]
    )
    jev = Jev(route=route, selection=select_query_navigation)
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        jev,
        goal=f"Browser for me and open {destination} and search {text}, please",
        readiness_timeout=0.5,
    )
    assert state.answer.achieved and state.outcome == "done"
    assert [a.kind for a in computer.actions] == ["navigate", "navigate"]
    assert parse_qs(urlsplit(computer.state.url).query)[parameter] == [text]
    assert computer.destination_reads >= 5 and poll_clock.sleeps
    assert len(writer.requests) == 1 and len(jev.requests) == 4  # Includes scope routing; waiting performs no inference.
    assert all(p["remaining"] == 0 for p in state.progress)


def test_missing_query_form_expires_without_submitting_or_fabricating_success(monkeypatch, tmp_path, poll_clock):
    import json

    destination = "https://unseen.test"
    computer = delayed_query_page(destination, "term", appears_after=None)
    writer = Reasoner([{"query": "café", "destination": destination, "question": "", "compound": False}])
    events = []
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_query_navigation),
        RunControl("form-deadline", events.append),
        goal=f"Open {destination} and search café",
        readiness_timeout=0.18,
    )
    assert poll_clock.elapsed == pytest.approx(0.18)
    assert state.outcome == "unsupported" and not state.answer
    assert len(computer.actions) == 1 and computer.state.url == destination
    assert sum(p["remaining"] for p in state.progress) == 1
    assert "didn't submit the search" in events[-1].text
    assert FORM_CAPABILITY not in events[-1].text
    assert json.loads((tmp_path / "run.json").read_text())["unsupported_capability_count"] == 1


def test_stop_during_form_readiness_prevents_query_dispatch(monkeypatch, tmp_path, poll_clock):
    destination = "https://unseen.test"
    computer = delayed_query_page(destination, "term", appears_after=None)
    writer = Reasoner([{"query": "café", "destination": destination, "question": "", "compound": False}])
    control = RunControl("stop-form-wait")
    poll_clock.on_sleep = control.cancel
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_query_navigation),
        control,
        goal=f"Open {destination} and search café",
        readiness_timeout=1,
    )
    assert state.outcome == "aborted" and not state.uncertain
    assert len(computer.actions) == 1 and computer.state.url == destination
    assert poll_clock.elapsed == pytest.approx(0.05)
    assert computer.destination_reads == 2  # No new observation after Stop.


def test_available_query_form_has_no_readiness_delay(monkeypatch, tmp_path, poll_clock):
    computer = QueryComputer("https://unseen.test", "term")
    writer = Reasoner([{"query": "café", "destination": computer.state.url, "question": "", "compound": False}])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_query_navigation),
        goal="Search café",
        readiness_timeout=1,
    )
    assert state.answer.achieved and len(computer.actions) == 1
    assert not poll_clock.sleeps


@pytest.mark.parametrize("unsafe", ["cross_origin", "credential", "disabled"])
def test_readiness_never_treats_an_unsafe_form_as_available(monkeypatch, tmp_path, poll_clock, unsafe):
    computer = QueryComputer("https://unseen.test", "term")
    if unsafe == "cross_origin":
        computer.state.forms["form"] = replace(computer.state.forms["form"], action="https://other.test/search")
    elif unsafe == "credential":
        computer.state.forms["form"] = replace(computer.state.forms["form"], parameter="api_key")
    else:
        computer.state.elements["input"] = replace(computer.state.elements["input"], enabled=False)
    writer = Reasoner([{"query": "café", "destination": computer.state.url, "question": "", "compound": False}])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_query_navigation),
        goal="Search café",
        readiness_timeout=0.1,
    )
    assert state.outcome == "unsupported" and not computer.actions
    assert poll_clock.elapsed == pytest.approx(0.1)


def test_loading_document_retains_the_bounded_readiness_failure(monkeypatch, tmp_path, poll_clock):
    computer = QueryComputer("https://unseen.test", "term")
    computer.state.ready = False
    writer = Reasoner([{"query": "café", "destination": computer.state.url, "question": "", "compound": False}])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_query_navigation),
        goal="Search café",
        readiness_timeout=0.1,
    )
    assert state.outcome == "blocked" and "did not become ready" in state.failure
    assert not computer.actions and poll_clock.elapsed == pytest.approx(0.1)
