"""Research through the whole engine: the real supervisor, the real router, Jev, the effect ledger and the answer checks.

`test_research_execution.py` drives the supervisor alone, with the engine's loop played by a stand-in; these drive the
engine itself over a stateful fake browser, so what is shown is what only the two together decide: that browser
completion is never an answer, that a rejected plan is corrected before a single action, that a read waits for its page,
what the user is told, and what a failed run leaves in its report. Only the browser, the models and Jev are replaced.

Each test is a port of one of the variant-6 engine tests (named in its docstring) onto the merged engine.
"""

import json
from dataclasses import replace

import pytest
from execution_world import Computer, FakeClock, Jev, Reasoner, drive, response
from research_fakes import QUOTE_A, QUOTE_B, SEARCH, A, B, approved, claim, decision
from test_query_forms import QueryComputer

from glide.computer import browser_settings
from glide.computer.config import DEFAULT_READINESS_TIMEOUT
from glide.computer.control import RunControl
from glide.computer.execution import progress
from glide.computer.execution.contracts import Element, Milestone
from glide.computer.execution.research import READ_EFFECTS
from glide.providers.errors import from_status


@pytest.fixture(autouse=True)
def configured_search_page(monkeypatch):
    """The settings the engine reads, set for this test only: the search page is where the fixture's own page is."""
    monkeypatch.setattr(browser_settings, "_table", {})
    monkeypatch.setenv("GLIDE_SEARCH_URL", SEARCH)


class ResearchComputer(QueryComputer):
    """A search page with one query form, linking to two sources, and a `read_page` like the real adapters'."""

    transport = "scripted"

    def __init__(self):
        super().__init__(SEARCH, "term")
        self.evidence_reads = []

    def read_page(self):
        url = self.state.url
        self.evidence_reads.append(url)
        return {
            "url": url,
            "title": "Observed page",
            "text": QUOTE_A if url == A else QUOTE_B if url == B else "Two relevant sources.",
            "links": [{"url": A, "title": "Reviews"}, {"url": B, "title": "Local guide"}],
            "truncated": False,
        }

    def no_page_selected(self):
        """The live session before its first navigation: no tab, so no URL is a destination yet."""
        self.state.url, self.state.active_tab, self.state.tabs = "", "", {}


def research_jev(route="research"):
    """Jev selects the tab or the navigation that a milestone asks for, and nothing else."""

    def choose(state, options):
        desired = "tab_create" if state["milestone"]["effect"] == "tab_created" else "navigate"
        return next(k for k, v in options.items() if k not in {"key", "replan"} and json.loads(v)["kind"] == desired)

    return Jev(route, selection=choose)


def planner_requests(writer):
    return [request for request in writer.requests if request.role == "planner"]


def events_of(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


# -- the happy path, and what is never an answer -----------------------------------------------------------------


@pytest.mark.parametrize("topic", ["Dilwale restaurant in Hong Kong", "accessible museums in Madrid"])
def test_search_read_compare_answer_uses_jev_and_keeps_the_full_goal(monkeypatch, tmp_path, topic):
    """Port of test_research_execution.py::test_search_read_compare_answer_uses_jev_and_keeps_full_goal."""
    computer, events = ResearchComputer(), []
    goal = f"Open a new tab at {SEARCH}, search {topic}, read the sources and give an overall assessment."
    writer = Reasoner(
        [
            decision("browse", goal=f"Open a new tab at {SEARCH} and search {topic}"),
            response(
                Milestone("tab", "Open the requested new tab", "tab_created", value=SEARCH),
                Milestone("query", "Submit the query", "query_submitted", target="search form", value=topic),
            ),
            decision("read"),
            decision("browse", goal=f"Open {A}"),
            response(Milestone("source", "Read the first source", "url", value=A)),
            decision("read"),
            decision("browse", goal=f"Open {B}"),
            response(Milestone("source", "Read the second source", "url", value=B)),
            decision("read"),
            decision(
                "answer",
                claims=[
                    claim("Feedback praises the food but mentions waiting.", "s2"),
                    claim("Another source mentions friendly service and vegetarian options.", "s3", QUOTE_B),
                ],
                limitations="This assessment covers the two pages read.",
            ),
            approved(),
        ]
    )
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        research_jev(),
        goal=goal,
        control=RunControl("research", emit=events.append),
        record_content=True,
    )
    assert state.outcome == "done", state.failure
    assert [a.kind for a in computer.actions] == ["tab_create", "navigate", "navigate", "navigate"]
    assert len(computer.evidence_reads) == 3 and computer.evidence_reads[-2:] == [A, B]
    assert all(p["remaining"] == 0 for p in state.progress)
    assert state.progress[-1]["id"] == "research_answer"
    final = next(e for e in events if e.kind == "completed")
    assert A in final.text and B in final.text and "http" not in final.spoken_text and "[1]" not in final.spoken_text
    packet = json.loads(writer.requests[-1].text)
    assert packet["original_goal"] == goal and len(packet["completed_browser_tasks"]) == 3
    assert {s["id"] for s in packet["evidence"]} == {"request", "s1", "s2", "s3"}
    assert packet["browser_session"]["active_tab"] == computer.state.active_tab
    assert packet["configured_search_url"] == SEARCH  # the engine hands the supervisor the configured address
    assert writer.requests[-1].role == "research_verification"
    assert state.calls.count["writer"] == len(writer.requests) and len(writer.requests) == 11
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["research"]["answer_verified"] and report["research"]["source_count"] == 3


def test_browser_completion_alone_never_completes_research(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_browser_completion_alone_never_completes_research."""
    writer = Reasoner(
        [decision("browse", goal=f"Open {SEARCH}"), response(Milestone("tab", "Open", "tab_created", value=SEARCH))]
    )
    state = drive(monkeypatch, tmp_path, ResearchComputer(), writer, research_jev(), steps=2)
    assert state.outcome == "step limit" and state.answer is None
    assert state.progress[0]["verified"] == 1 and state.progress[-1]["remaining"] == 1
    assert state.progress[-1]["id"] == "research_answer"


def test_the_research_budget_of_the_run_ends_the_run_without_an_answer(monkeypatch, tmp_path):
    """The budget the engine passes the supervisor (`RunConfig.research_calls`) is the one that bounds its model calls."""
    writer = Reasoner([decision("read")] * 5)
    state = drive(monkeypatch, tmp_path, ResearchComputer(), writer, research_jev(), research_calls=2, goal="Review it.")
    assert state.outcome == "blocked" and state.answer is None and "budget" in state.failure
    assert len(writer.requests) == 2


def test_a_misrouted_answer_fails_explicitly_without_routing_again(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_misrouted_answer_fails_explicitly_without_recursive_routing."""
    computer = ResearchComputer()
    writer = Reasoner([response(unsupported=["summarize the evidence"])])
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev("plan"))
    assert [r.role for r in writer.requests] == ["planner"]
    assert state.outcome == "unsupported" and not computer.actions
    assert "summarize the evidence" in state.failure


# -- what a person is told ------------------------------------------------------------------------------------------


def test_the_capability_list_of_an_unsupported_research_step_is_kept_in_the_report_and_not_spoken(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_internal_capability_list_is_logged_but_not_spoken, with the research
    supervisor planning the browser batch that cannot be done."""
    events = []
    computer = ResearchComputer()
    writer = Reasoner([decision("browse", goal=f"Open {SEARCH} and drag the first result"), response(unsupported=["drag"])])
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev(), control=RunControl("unsupported", emit=events.append))
    assert state.outcome == "unsupported" and not computer.actions
    assert "Available capabilities" in state.failure
    blocked = next(e for e in events if e.kind == "blocked")
    assert "Available capabilities" not in blocked.text and "task report" in blocked.text


def test_failure_counts_stay_in_the_report_and_the_supervisors_reason_is_what_is_spoken(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_failure_progress_counts_remain_in_report_but_are_not_spoken."""
    events, reason = [], "I couldn't read the requested sources."
    state = drive(
        monkeypatch,
        tmp_path,
        ResearchComputer(),
        Reasoner([decision("blocked", reason=reason)]),
        research_jev(),
        control=RunControl("failure-readout", emit=events.append),
    )
    assert "Verified 0 effect(s); 1 remain." in state.failure
    assert next(e for e in events if e.kind == "blocked").text == reason


def test_an_account_with_no_credit_ends_a_research_run_in_one_plain_sentence_with_nothing_done(monkeypatch, tmp_path):
    """The research counterpart of test_research_execution.py::test_exhausted_api_credits_are_spoken_clearly_...:
    one request, no retry, no action, no read, no answer, and the sentence names the cause and no server word."""
    events, computer = [], ResearchComputer()
    spent = from_status(402, "SERVER WORDS the user must not hear", provider="research:slot")
    writer = Reasoner([spent])
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev(), control=RunControl("quota", emit=events.append))
    assert state.outcome == "provider failure" and state.answer is None
    assert not computer.actions and not computer.evidence_reads and len(writer.requests) == 1
    assert json.loads((tmp_path / "run.json").read_text())["failure_code"] == "model_unavailable"
    spoken = next(e for e in events if e.kind == "blocked").text
    assert "no credit or quota" in spoken and "SERVER WORDS" not in spoken and "No action was repeated." in spoken


# -- the plan is corrected before any action ----------------------------------------------------------------------

# Exact planner response from live task 63fc668f-9182-4f86-861a-1da093d9ab9b: the first step's "new tab" is not a URL.
RECORDED_PLAN = {
    "question": "",
    "steps": [
        {
            "id": "m1",
            "goal": "Open one new browser tab for the Google search.",
            "effect": "tab_created",
            "target": "browser tab",
            "value": "new tab",
            "quantity": 1,
        },
        {
            "id": "m2",
            "goal": "Search Google for the official Hong Kong Space Museum and Hong Kong Science Museum visitor information pages.",
            "effect": "query_submitted",
            "target": "Google search form",
            "value": "official Hong Kong Space Museum visitor information and Hong Kong Science Museum visitor information",
            "quantity": 1,
        },
    ],
    "unsupported": [],
}


def test_the_recorded_research_plan_is_corrected_before_any_action(monkeypatch, tmp_path):
    """Port of test_planning_repair.py::test_recorded_research_plan_is_corrected_before_any_action."""
    computer = ResearchComputer()
    computer.no_page_selected()
    fixed = json.loads(json.dumps(RECORDED_PLAN))
    fixed["steps"][0]["value"] = SEARCH
    goal = "Find the two museums' official visitor information, compare Wednesday admission for two adults, and cite both."
    writer = Reasoner(
        [
            decision(
                "browse", goal="Open a new tab and search the configured search page for both museums' visitor information."
            ),
            RECORDED_PLAN,
            fixed,
            decision("blocked", reason="Fixture ends after search; the assessment still needs page evidence."),
        ]
    )
    generate = writer.generate

    def checked_generate(request, cancel=None):
        if request.role == "planner":
            assert not computer.actions  # nothing is dispatched while a plan is being made or corrected
        return generate(request, cancel)

    writer.generate = checked_generate
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev(), goal=goal, record_content=True)
    assert [a.kind for a in computer.actions] == ["tab_create", "navigate"], state.failure
    assert state.answer is None and state.progress[-1]["remaining"] == 1
    assert all(p["remaining"] == 0 for p in state.progress[:-1])
    repair = json.loads(writer.requests[2].text)
    assert repair["task_context"]["original_goal"] == goal
    assert repair["configured_search_url"] == SEARCH
    assert "" not in repair["task_context"]["observed_urls"]
    assert "Invalid milestone URL" in repair["plan_correction"]["validation_error"]
    assert repair["plan_correction"]["rejected_plan"] == RECORDED_PLAN
    kinds = [e["event"] for e in events_of(tmp_path)]
    assert "plan_validation_failed" in kinds and "plan_correction_completed" in kinds


def test_a_plan_that_stays_invalid_stops_the_research_run_after_one_correction(monkeypatch, tmp_path):
    computer, writer = ResearchComputer(), Reasoner([decision("browse", goal=f"Open {SEARCH}"), RECORDED_PLAN])
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev())
    assert state.outcome == "blocked" and "Invalid milestone URL" in state.failure
    assert len(planner_requests(writer)) == 2 and not computer.actions


def test_a_search_with_no_selected_page_requires_navigation_before_dispatch(monkeypatch, tmp_path):
    """Port of test_planning_repair.py::test_search_without_selected_page_requires_navigation_before_dispatch."""
    computer = ResearchComputer()
    computer.no_page_selected()
    query = Milestone("search", "Search museum visitor information", "query_submitted", target="search form", value="museums")
    writer = Reasoner(
        [
            decision("browse", goal="Search for the museums' visitor information"),
            response(query),
            response(Milestone("open", "Open search page", "tab_created", value=SEARCH), query),
            decision("blocked", reason="Fixture ends after the verified search."),
        ]
    )
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev())
    assert [a.kind for a in computer.actions] == ["tab_create", "navigate"], state.failure
    assert "selected web page" in json.loads(writer.requests[2].text)["plan_correction"]["validation_error"]
    assert state.answer is None and state.progress[-1]["remaining"] == 1


def test_a_research_plan_cannot_use_an_effect_the_supervisor_did_not_offer(monkeypatch, tmp_path):
    """Port of test_planning_repair.py::test_live_museum_subtask_rejects_unadvertised_effect_before_dispatch_then_reads
    (regression for qualification-20261004/museums: an extra element_present check)."""
    computer = ResearchComputer()
    computer.no_page_selected()
    opening = Milestone("open-search", "Open the search page in a new browser tab.", "tab_created", value=SEARCH)
    query = Milestone(
        "search-hours",
        "Submit the requested search.",
        "query_submitted",
        target="search form",
        value="Hong Kong Space Museum official opening hours",
    )
    extra = Milestone(
        "find-official-page",
        "Locate an official Hong Kong Space Museum visitor-hours page in the search results.",
        "element_present",
        target="Search result for the Hong Kong Space Museum's official opening-hours or visitor-hours page",
    )
    writer = Reasoner(
        [
            decision("browse", goal="Search for Hong Kong Space Museum official opening hours."),
            response(opening, query, extra),
            response(opening, query),
            decision("read"),
            decision("blocked", reason="Fixture stops after reading actual search results."),
        ]
    )
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev())
    assert [a.kind for a in computer.actions] == ["tab_create", "navigate"], state.failure
    assert computer.evidence_reads == [computer.state.url]
    assert state.answer is None and state.progress[-1]["remaining"] == 1
    requests = planner_requests(writer)
    assert len(requests) == 2
    for request in requests:
        assert set(json.loads(request.text)["available_effects"]) == READ_EFFECTS
        offered = {
            effect
            for variant in request.schema["properties"]["steps"]["items"]["anyOf"]
            for effect in variant["properties"]["effect"]["enum"]
        }
        assert offered == READ_EFFECTS - {"disclosure_expanded"}  # no disclosure controller is observed here
    correction = json.loads(requests[-1].text)["plan_correction"]
    assert correction["rejected_plan"]["steps"][-1]["effect"] == "element_present"
    assert "only supports effects" in correction["validation_error"]


def test_a_recorded_disclosure_description_is_corrected_to_the_observed_controller_before_registration(monkeypatch, tmp_path):
    """Port of test_planning_repair.py::test_recorded_disclosure_description_is_corrected_to_observed_controller_before_
    registration (live museums-7 confused controller 21 with its controlled panel 22)."""
    computer = ResearchComputer()
    computer.state.elements = {"21": Element("21", "Opening hours", "a", expanded=False, controls=("22",))}
    malformed = Milestone(
        "expand-opening-hours",
        "Expand the observed Opening hours disclosure to reveal its hours and closure information.",
        "disclosure_expanded",
        target="Opening hours (controls element 22)",
    )
    corrected = replace(malformed, target="21")
    writer = Reasoner(
        [
            decision("browse", goal="Expand the visible Opening hours disclosure (element 21, controlling element 22)."),
            response(malformed),
            response(corrected),
            decision("read"),
            decision("blocked", reason="Fixture ends after reading the revealed source."),
        ]
    )

    def expand(machine, action):
        assert action.kind == "click" and action.target == "21"
        machine.state.elements["21"] = replace(machine.state.elements["21"], expanded=True)
        return ""

    def select(state, options):
        return next(key for key, value in options.items() if key != "replan" and json.loads(value)["target"] == "21")

    computer.on_execute = expand
    generate = writer.generate

    def no_early_dispatch(request, cancel=None):
        if request.role == "planner":
            assert not computer.actions
        return generate(request, cancel)

    writer.generate = no_early_dispatch
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("research", selection=select))
    assert len(computer.actions) == 1 and computer.actions[0].target == "21", state.failure
    assert computer.evidence_reads and state.answer is None
    assert state.progress[0]["verified"] == 1 and state.progress[-1]["remaining"] == 1
    requests = planner_requests(writer)
    assert len(requests) == 2
    for request in requests:
        disclosure = next(
            item
            for item in request.schema["properties"]["steps"]["items"]["anyOf"]
            if "disclosure_expanded" in item["properties"]["effect"]["enum"]
        )
        assert disclosure["properties"]["target"]["enum"] == ["21", "Opening hours"]
        assert disclosure["properties"]["value"]["enum"] == [""]
    correction = json.loads(requests[1].text)["plan_correction"]
    assert "exactly match one observed controller" in correction["validation_error"]
    assert correction["rejected_plan"]["steps"][0]["target"] == malformed.target
    assert json.loads((tmp_path / "run.json").read_text())["recoveries"] == 0


# -- the executor does what the supervisor allows, and Jev still chooses ---------------------------------------------


def test_research_expands_an_observed_disclosure_before_reading_its_evidence(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_research_expands_observed_disclosure_before_reading_its_evidence."""
    computer = ResearchComputer()
    computer.state.url = A
    hours_text = "The gallery opens from 11am to 6pm on weekdays."
    computer.state.elements = {
        "hours": Element("hours", "Opening hours", "button", expanded=False, controls=("hours-panel",)),
        "delete": Element("delete", "Delete account", "button"),
    }

    def expand(machine, action):
        assert action.kind == "click" and action.target == "hours"
        machine.state.elements["hours"] = replace(machine.state.elements["hours"], expanded=True)
        return ""

    computer.on_execute = expand
    read = computer.read_page

    def read_expanded():
        assert computer.state.elements["hours"].expanded  # the page is read only after the section was opened
        return {**read(), "text": hours_text}

    computer.read_page = read_expanded

    def select(state, options):
        offered = [json.loads(value) for key, value in options.items() if key != "replan"]
        assert {action["kind"] for action in offered} <= {"inspect", "click"}
        assert all(action["target"] in {"", "hours"} for action in offered)  # never the unrelated button
        return next(key for key, value in options.items() if key != "replan" and json.loads(value)["target"] == "hours")

    writer = Reasoner(
        [
            decision("browse", goal="Expand the observed Opening hours section so its information can be read."),
            response(Milestone("hours", "Reveal Opening hours", "disclosure_expanded", target="Opening hours")),
            decision("read"),
            decision("answer", claims=[claim("Weekday hours are 11am to 6pm.", quote=hours_text)]),
            approved(),
        ]
    )
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("research", selection=select))
    assert state.answer.achieved and computer.evidence_reads == [A]
    assert len(computer.actions) == 1 and computer.actions[0].parameter_source == "disclosure:hours"
    assert all(item["remaining"] == 0 for item in state.progress)


def test_a_research_search_control_fallback_stays_jev_selected(monkeypatch, tmp_path):
    """Port of test_research_execution.py::test_research_search_control_fallback_stays_jev_selected."""
    computer = ResearchComputer()
    computer.state.forms = {}
    computer.state.elements["input"] = replace(computer.state.elements["input"], search=True)
    computer.submissions["input"] = (SEARCH + "/results", "text")
    writer = Reasoner(
        [
            decision("browse", goal="Search the selected page for official visitor information"),
            response(Milestone("search", "Search the site", "query_submitted", target="search", value="visitor information")),
            decision("read"),
            decision("blocked", reason="The fixture stops before a source-backed assessment."),
        ]
    )

    def choose(state, options):
        return next(k for k, value in options.items() if k != "replan" and json.loads(value)["kind"] in {"type", "key"})

    state = drive(monkeypatch, tmp_path, computer, writer, Jev("research", selection=choose))
    assert [action.kind for action in computer.actions] == ["type", "key"], state.failure
    assert all(action.parameter_source == "query:input" for action in computer.actions)
    assert computer.evidence_reads == [computer.state.url]
    assert state.answer is None and state.progress[0]["verified"] == 1 and state.progress[-1]["remaining"] == 1


# -- readiness ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("becomes_ready", [True, False])
def test_research_waits_for_the_new_document_before_reading_it_and_does_not_reopen(monkeypatch, tmp_path, becomes_ready):
    """Port of test_research_execution.py::test_research_waits_for_new_document_before_reading_without_reopening
    (a replay of live task 50326c73: navigation verified, then reading failed after 1.5 s)."""
    clock = FakeClock(monkeypatch)
    computer = ResearchComputer()

    def observe(machine):
        if machine.actions:
            machine.state.ready = becomes_ready and clock.now >= 3.0

    computer.on_inspect = observe
    writer = Reasoner(
        [
            decision("browse", goal=f"Open {A} in a new tab"),
            response(Milestone("source", "Open source", "tab_created", value=A)),
            decision("read"),
            decision("answer", claims=[claim("Visitors praise the food.")]),
            approved(),
        ]
    )
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        research_jev(),
        goal=f"Open {A}, read it and summarize visitor feedback.",
        readiness_timeout="default",
        record_content=True,  # the events of the run are written only then
    )
    timeout = DEFAULT_READINESS_TIMEOUT
    assert [action.kind for action in computer.actions] == ["tab_create"]
    if becomes_ready:
        assert state.outcome == "done", state.failure
        assert computer.evidence_reads == [A]
        assert 3.0 <= clock.now < 3.1  # it returns when ready, without waiting out the deadline
        assert len(writer.requests) == 5
    else:
        assert state.outcome == "blocked" and state.answer is None
        assert not computer.evidence_reads and clock.now == timeout
        assert "did not become ready" in state.failure
        failure = next(e["details"] for e in events_of(tmp_path) if e["event"] == "readiness_timed_out")
        assert failure["timeout_s"] == timeout and failure["ready"] is False and failure["url"] == A


def test_the_engine_gives_the_supervisor_the_executors_own_tools(monkeypatch, tmp_path):
    """The supervisor is built with the executor's planner, readiness wait, literal-URL reader and search check, the
    run's budget and the configured search address, and with nothing else of the engine."""
    from execution_world import FakeSupervisor

    from glide.computer.execution import planning, query

    supervisor = FakeSupervisor([("raise", ValueError("stop here"))])
    drive(
        monkeypatch,
        tmp_path,
        ResearchComputer(),
        Reasoner([]),
        research_jev(),
        supervisor=supervisor,
        research_calls=5,
        goal=f"Read {A} and report.",
    )
    built = supervisor.built
    assert built["budget"] == 5 and built["search_url"] == SEARCH
    tools = built["tools"]
    assert (
        tools.plan is planning.plan
        and tools.wait_ready is progress.wait_ready
        and tools.is_search_action is query.is_search_action
    )
    assert tools.urls_in(f"see {A}, and ftp://files.test/x and https://user:pw@b.test/") == [
        A
    ]  # code-owned, filtered by what is a safe address


def test_with_no_search_address_configured_the_supervisor_is_told_there_is_none(monkeypatch, tmp_path):
    from execution_world import FakeSupervisor

    monkeypatch.delenv("GLIDE_SEARCH_URL")
    supervisor = FakeSupervisor([("raise", ValueError("stop here"))])
    drive(monkeypatch, tmp_path, ResearchComputer(), Reasoner([]), research_jev(), supervisor=supervisor)
    assert supervisor.built["search_url"] == ""


def test_the_ledger_not_the_supervisor_decides_that_a_planned_step_is_done(monkeypatch, tmp_path):
    """Browser work is counted only from a fresh observation: a navigation the page never shows is not progress, and the
    supervisor is not asked for its next stage until every planned step is verified."""
    computer = Computer()
    computer.state.capabilities.add("query_form")
    writer = Reasoner(
        [decision("browse", goal=f"Open {SEARCH}"), response(Milestone("tab", "Open", "tab_created", value=SEARCH))]
    )

    def stuck(machine, action):
        machine.next_id += 1
        return str(machine.next_id)  # reports a new tab, shows none

    computer.on_execute = stuck
    state = drive(monkeypatch, tmp_path, computer, writer, research_jev(), steps=6, goal="Review it.")
    assert state.outcome != "done" and state.answer is None
    assert len([r for r in writer.requests if r.role == "research_supervisor"]) == 1  # not asked again: the step is open
    assert state.progress[0]["verified"] == 0
