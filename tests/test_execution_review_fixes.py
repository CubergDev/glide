"""Regressions for review findings against the structured engine (ids in each test name's docstring)."""

import json
import sqlite3
from dataclasses import replace

import pytest
from execution_world import Computer, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution.contracts import (
    Action,
    Container,
    Element,
    InvalidAction,
    Media,
    Milestone,
    Navigation,
    Observation,
    effect,
    rebind,
)
from glide.computer.execution.progress import Ledger


def test_pending_write_blocks_a_restart_under_a_new_task_id(tmp_path):
    """PR10-4175413682: the CLI mints a task id per invocation, so the journal must not be keyed on it."""
    step = Milestone("m", "Open", "url", value="https://example.org")
    first = Ledger(tmp_path / "progress.sqlite3", "task-one")
    first.register([step], 0)
    first.begin(step, Action("navigate", "x", value="https://example.org"))
    first.close()
    second = Ledger(tmp_path / "progress.sqlite3", "task-two")
    assert second.unresolved()
    second.close()


def test_restart_with_another_control_dispatches_nothing(monkeypatch, tmp_path):
    """PR10-4175413682, through the engine: a second invocation with a fresh RunControl is blocked before any action."""
    computer = Computer()

    def unknown(c, action):
        raise OSError("connection lost after write")

    computer.on_execute = unknown
    planner = Reasoner([response(Milestone("nav", "Reach", "url", value="https://example.net"))])
    first = drive(monkeypatch, tmp_path, computer, planner, control=RunControl("first-invocation"))
    assert first.uncertain and len(computer.actions) == 1
    second = drive(monkeypatch, tmp_path, computer, planner, control=RunControl("second-invocation"))
    assert second.outcome == "blocked" and len(computer.actions) == 1
    assert "unresolved operation" in second.failure


def test_planner_milestone_ids_stay_out_of_stored_run_data(monkeypatch, tmp_path):
    """PR10-4175413687: a planner id can echo the goal, so neither progress.sqlite3 nor run.json holds it."""
    sentinel = "email_alice_medical_results"
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([response(Milestone(sentinel, "Open the page", "url", value="https://example.net"))]),
    )
    assert state.outcome == "done" and state.progress[0]["id"] == sentinel  # the planner still sees its own ids
    assert sentinel not in (tmp_path / "run.json").read_text()
    assert json.loads((tmp_path / "run.json").read_text())["progress"][0]["id"] == "m1"
    with sqlite3.connect(tmp_path / "progress.sqlite3") as db:
        assert sentinel not in str(list(db.iterdump()))


def test_observation_packet_redacts_credentials_in_every_address(monkeypatch):
    """PR10-4175625125: model packets carry no credential-bearing query value, user info or configured secret."""
    monkeypatch.setenv("GLIDE_TEST_API_KEY", "configured-secret-value")
    page = "https://example.org/cb?access_token=tok123&page=2&note=configured-secret-value"
    obs = Observation(
        "browser",
        "doc:1",
        page,
        "t1",
        {"t1": page, "t2": "https://user:hunter2@example.org/x?api_key=abc123"},
        {"link": Element("link", "Next", "link", href="https://example.org/n?password=pw9&x=1")},
        navigation=Navigation("https://example.org/r?token=nav1", "t1", "f", "l"),
        canonical_url="https://example.org/c?secret=can1",
    )
    packet = json.dumps(obs.packet())
    for leaked in ("tok123", "abc123", "hunter2", "pw9", "nav1", "can1", "configured-secret-value"):
        assert leaked not in packet
    assert "page=2" in packet and "x=1" in packet  # the rest of the address stays useful
    assert obs.url == page  # the engine itself keeps the raw value


def page(owner="doc:1", url="https://example.org/", **kw):
    return Observation("browser", owner, url, kw.pop("active_tab", "t1"), kw.pop("tabs", {"t1": url}), **kw)


def test_media_playing_with_an_empty_target_verifies_the_single_media_element():
    """PR11-4175426267: the planner may leave the target empty for one media element."""
    paused, playing = Media("v", "Video", True, False, 4, 0.0), Media("v", "Video", False, False, 4, 1.5)
    before = page(elements={"play": Element("play", "Play", "button")}, media={"v": paused})
    after = page(elements={"play": Element("play", "Play", "button")}, media={"v": playing})
    click = Action("click", before.identity, "play")
    assert effect(Milestone("m", "Play it", "media_playing"), click, before, after)
    two = page(media={"v": playing, "w": replace(playing, id="w")})
    assert not effect(Milestone("m", "Play it", "media_playing"), click, before, two)  # ambiguous stays unverified


def test_element_absent_needs_the_same_document():
    """PR11-4175426270: leaving the page also makes the old element absent; it is not a removal."""
    delete = {"delete": Element("delete", "Delete", "button")}
    before = page(elements=delete)
    milestone = Milestone("m", "Remove it", "element_absent", target="Delete")
    assert effect(milestone, None, before, page(elements={}))
    assert not effect(milestone, None, before, page(owner="doc:2", elements={}))


def test_targetless_navigation_is_not_rebound_onto_a_tab_that_appeared():
    """PR11-4175634939: a create-a-tab navigation must not overwrite a tab opened before preflight."""
    before = Observation("browser", "", "", capabilities={"inspect", "navigate"})
    action = Action("navigate", before.identity, value="https://example.org/")
    fresh = Observation(
        "browser", "doc:1", "https://user.example/", "t9", {"t9": "https://user.example/"}, capabilities={"inspect", "navigate"}
    )
    with pytest.raises(InvalidAction):
        rebind(action, before, fresh)
    assert rebind(action, before, Observation("browser", "", "", capabilities={"inspect", "navigate"})).kind == "navigate"


def test_tab_closed_needs_a_working_tab_inventory():
    """PR11-4175634943: after a failed enumeration the tabs are empty, which proves nothing about the closed one."""
    before = page(tabs={"t1": "https://example.org/", "t2": "https://example.net/"}, capabilities={"inspect", "tab_close"})
    milestone = Milestone("m", "Close it", "tab_closed", target="t2")
    close = Action("tab_close", before.identity, "t2")
    working = page(tabs={"t1": "https://example.org/"}, capabilities={"inspect", "tab_close"})
    failed = page(tabs={}, capabilities={"inspect"})
    assert effect(milestone, close, before, working)
    assert not effect(milestone, close, before, failed)


def test_scroll_needs_the_same_document():
    """PR12-4175623612: a farther container of the same id on another document is not this scroll."""
    top, farther = Container("feed", "Feed", 0, 1000), Container("feed", "Feed", 300, 1000)
    before = page(containers={"feed": top}, capabilities={"inspect", "scroll"})
    milestone = Milestone("m", "Scroll", "scroll", target="Feed", value="down")
    scroll = Action("scroll", before.identity, "feed", "down")
    assert effect(milestone, scroll, before, page(containers={"feed": farther}, capabilities={"inspect", "scroll"}))
    assert not effect(milestone, scroll, before, page(owner="doc:2", containers={"feed": farther}))


def test_an_existing_page_on_another_host_or_path_does_not_satisfy_a_url_milestone():
    """PR10-4175602716: www and trailing slashes are ignored after a navigation, never for a page that is merely there."""
    here = page(url="https://example.test/account")
    milestone = Milestone("m", "Go", "url", value="https://www.example.test/account/")
    assert not effect(milestone, None, here, here)
    assert not effect(Milestone("m", "Go", "url", value="https://example.test/account/"), None, here, here)
    assert effect(
        Milestone("m", "Go", "url", value="https://example.test/"),
        None,
        page(url="https://example.test"),
        page(url="https://example.test"),
    )
    go = Action("navigate", here.identity, value=milestone.value)
    assert effect(milestone, go, here, here)  # the redirect to www is still accepted once a navigation was dispatched


def test_a_satisfied_milestone_then_a_write_with_one_step(monkeypatch, tmp_path):
    """PR10-4175602709: the satisfied URL is recorded without an action, then the one step is the write."""
    computer = Computer()
    planner = Reasoner(
        [
            response(
                Milestone("here", "Be on the page", "url", value="https://example.org"),
                Milestone("tabs", "Open a tab", "tab_created", value="https://example.net"),
            )
        ]
    )
    state = drive(monkeypatch, tmp_path, computer, planner, steps=1)
    assert state.outcome == "done" and [a.kind for a in computer.actions] == ["tab_create"]


def test_an_invalid_budget_is_still_written_to_run_json(monkeypatch, tmp_path):
    """PR10-4175602711: a run refused before it starts leaves a report saying so."""
    state = drive(monkeypatch, tmp_path / "out", Computer(), Reasoner([]), steps=0)
    assert state.outcome == "blocked"
    report = json.loads((tmp_path / "out" / "run.json").read_text())
    assert (
        report["outcome"] == "blocked"
        and report["failure_code"] == "invalid_budget"
        and "Invalid execution budget" in report["failure"]
    )


def test_a_failing_close_neither_skips_the_backend_nor_crashes_a_finished_run(monkeypatch, tmp_path):
    """PR10-4175625126: ledger.close() raising still closes the backend, and the run's state is returned."""
    computer = Computer()
    original = Ledger.close

    def broken(self):
        original(self)
        raise sqlite3.ProgrammingError("close failed")

    monkeypatch.setattr(Ledger, "close", broken)
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([response(Milestone("nav", "Reach", "url", value="https://example.net"))]),
    )
    assert state.outcome == "done" and computer.closed and (tmp_path / "run.json").exists()


def test_a_page_with_hundreds_of_controls_still_offers_its_search_form():
    """PR10-4175413700: the choice limit applies to the filtered actions of a query milestone."""
    from glide.computer.execution import policy
    from glide.computer.execution.query import QueryForm

    buttons = {f"b{i}": Element(f"b{i}", f"Button {i}", "button") for i in range(300)}
    field = Element("q", "Search", "searchbox", "", True, search=True)
    form = QueryForm("f", "q", "https://example.org/search", "q", ())
    obs = Observation(
        "browser",
        "doc:1",
        "https://example.org/",
        "t",
        {"t": "https://example.org/"},
        {**buttons, "q": field},
        forms={"f": form},
        capabilities={"inspect", "click", "type", "key", "navigate"},
    )
    found = policy.candidates(Milestone("m", "Search", "query_submitted", target="Search", value="cats"), obs)
    assert any(a.parameter_source.startswith("form:") for a in found)
