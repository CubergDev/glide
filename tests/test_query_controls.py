"""Observed search inputs and playback evidence without a browser or provider call."""

import copy
import json
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution.contracts import Action, Element, InvalidAction, Media, Milestone, effect, rebind, validate
from glide.computer.execution.query import available, bind, candidates


class SearchControlComputer(Computer):
    def __init__(self, destination="https://catalog.test"):
        super().__init__()
        self.state.app = "browser"
        self.state.url = destination
        self.state.tabs["initial"] = destination
        self.state.capabilities.add("query_form")
        self.state.elements = {
            "query": Element("query", "Find in catalogue", "searchbox", "", True, search=True),
            "message": Element("message", "Private message", "textbox", "", True),
        }
        self.state.focus = ""
        self.submissions["query"] = (destination + "/results", "keywords")


def select_search(state, options):
    return next(k for k, text in options.items() if k != "replan" and json.loads(text)["kind"] in {"type", "key"})


@pytest.mark.parametrize("destination,text", [("https://catalog.test", "香港 café & tea"), ("https://unseen.test", "video 27")])
def test_search_without_get_form_uses_jev_and_verifies_exact_result_url(monkeypatch, tmp_path, destination, text):
    computer = SearchControlComputer(destination)
    writer = Reasoner([{"query": text, "destination": destination, "question": "", "compound": False}])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev(route="query", selection=select_search),
        goal=f"Search {destination} for {text}",
    )
    assert state.answer.achieved and state.outcome == "done"
    assert [a.kind for a in computer.actions] == ["type", "key"]
    assert all(a.parameter_source == "query:query" for a in computer.actions)
    assert parse_qs(urlsplit(computer.state.url).query)["keywords"] == [text]
    assert computer.state.elements["message"].value == ""
    assert [r.role for r in writer.requests] == ["writer"]


def test_typing_is_not_submission_and_missing_result_evidence_stops(monkeypatch, tmp_path):
    computer = SearchControlComputer()
    computer.submissions = {}  # Enter only changes focus; the page never exposes query results.
    writer = Reasoner([{"query": "tea", "destination": computer.state.url, "question": "", "compound": False}])
    state = drive(
        monkeypatch, tmp_path, computer, writer, Jev(route="query", selection=select_search), goal="Search tea", steps=3
    )
    assert not state.answer or not state.answer.achieved
    assert state.progress[-1]["remaining"] == 1
    assert computer.state.url == "https://catalog.test"


@pytest.mark.parametrize(
    "change", [{"search": False}, {"enabled": False}, {"secret": True}, {"typeable": False}, {"label": "API key"}]
)
def test_query_fallback_never_offers_unrelated_unavailable_or_secret_fields(change):
    observed = SearchControlComputer().inspect()
    observed.elements["query"] = replace(observed.elements["query"], **change)
    step = Milestone("query", "Search", "query_submitted", target="search", value="tea")
    assert not available(observed, step.value)
    assert candidates(step, observed) == []


def submission():
    before = SearchControlComputer().inspect()
    before.elements["query"] = replace(before.elements["query"], value="tea")
    before.focus = "query"
    step = Milestone("query", "Search", "query_submitted", target="search", value="tea")
    action = candidates(step, before)[0]
    return step, action, before


@pytest.mark.parametrize(
    "url",
    [
        "https://other.test/?q=tea",
        "https://catalog.test/results?q=coffee",
        "https://catalog.test/results?q=tea&q=coffee",
        "https://catalog.test",
    ],
)
def test_unrelated_navigation_never_verifies_a_submitted_query(url):
    step, action, before = submission()
    after = copy.deepcopy(before)
    after.url = url
    assert not effect(step, action, before, after)


def test_changed_text_or_focus_is_rejected_before_enter():
    step, action, before = submission()
    assert bind(step, action, before) == step
    for change in ("text", "focus"):
        fresh = copy.deepcopy(before)
        if change == "text":
            fresh.elements["query"] = replace(fresh.elements["query"], value="other")
        else:
            fresh.focus = "message"
        with pytest.raises(InvalidAction):
            rebind(action, before, fresh)
        with pytest.raises(InvalidAction):
            bind(step, action, fresh)


def test_query_provenance_cannot_be_used_for_arbitrary_keys_or_targets():
    _, action, before = submission()
    for forged in (
        replace(action, value="delete"),
        replace(action, modifiers=("command",)),
        replace(action, kind="type", target="message", value="tea"),
    ):
        with pytest.raises(InvalidAction):
            validate(forged, before)


def test_stop_after_typing_never_submits(monkeypatch, tmp_path):
    computer = SearchControlComputer()
    control = RunControl("stop-before-submit")

    def stop(machine):
        if machine.actions and machine.actions[-1].kind == "type":
            control.cancel()

    computer.on_inspect = stop
    writer = Reasoner([{"query": "tea", "destination": computer.state.url, "question": "", "compound": False}])
    state = drive(
        monkeypatch, tmp_path, computer, writer, Jev(route="query", selection=select_search), control, goal="Search tea"
    )
    assert state.outcome == "aborted" and [a.kind for a in computer.actions] == ["type"]


@pytest.mark.parametrize(
    "change", [{"paused": True}, {"ended": True}, {"ready_state": 1}, {"current_time": 0}, {"current_time": float("nan")}]
)
def test_media_playback_requires_actual_playing_state(change):
    before = SearchControlComputer().inspect()
    before.elements["result"] = Element("result", "First result", "link", href="https://catalog.test/watch/first")
    action = Action("click", before.identity, "result")
    after = copy.deepcopy(before)
    after.url = "https://catalog.test/watch/first"
    media = Media("video", "Selected video", False, False, 4, 0.5)
    after.media = {"video": media}
    step = Milestone("play", "Play selected result", "media_playing")
    assert effect(step, action, before, after)
    after.media["video"] = replace(media, **change)
    assert not effect(step, action, before, after)


def test_ambiguous_media_requires_a_selected_target():
    observed = SearchControlComputer().inspect()
    observed.media = {
        "first": Media("first", "First", False, False, 4, 1),
        "second": Media("second", "Second", False, False, 4, 2),
    }
    assert not effect(Milestone("play", "Play", "media_playing"), None, observed, observed)
    before = copy.deepcopy(observed)
    before.media["first"] = replace(before.media["first"], paused=True)
    action = Action("key", before.identity, value="space")
    assert effect(Milestone("play", "Play first", "media_playing", target="first"), action, before, observed)
    assert not effect(Milestone("play", "Play", "media_playing"), action, before, observed)


def test_existing_autoplay_or_wrong_destination_cannot_complete_playback():
    before = SearchControlComputer().inspect()
    before.elements["result"] = Element("result", "First result", "link", href="https://catalog.test/watch/first")
    before.media["preview"] = Media("preview", "Preview", False, False, 4, 2)
    step = Milestone("play", "Play first result", "media_playing")
    assert not effect(step, None, before, before)
    action = Action("click", before.identity, "result")
    after = copy.deepcopy(before)
    after.url = "https://catalog.test/watch/unrelated"
    assert not effect(step, action, before, after)


@pytest.mark.parametrize(
    "destination,valid",
    [
        ("https://catalog.test/watch?item=123", True),
        ("https://catalog.test/watch?item=456", False),
        ("https://catalog.test/watch?item=123&new=bad", False),
        ("https://unrelated.test/watch?item=123", False),
        ("https://catalog.test/other?item=123", False),
        ("https://catalog.test/watch", False),
        ("https://catalog.test/watch?tracking=abc", False),
    ],
)
def test_playback_allows_discarded_query_parameters_but_not_changed_resource(destination, valid):
    before = SearchControlComputer().inspect()
    before.elements["result"] = Element("result", "First result", "link", href="https://catalog.test/watch?item=123&tracking=abc")
    action = Action("click", before.identity, "result")
    after = copy.deepcopy(before)
    after.url = destination
    after.canonical_url = "https://catalog.test/watch?item=123"
    after.media = {"video": Media("video", "Selected video", False, False, 4, 0.3)}
    step = Milestone("play", "Play first result", "media_playing")
    assert bool(effect(step, action, before, after)) == valid


@pytest.mark.parametrize(
    "canonical",
    ["", "https://catalog.test/watch?item=456", "https://catalog.test/watch", "https://unrelated.test/watch?item=123"],
)
def test_removed_parameters_need_matching_page_declared_resource_identity(canonical):
    from glide.computer.execution.contracts import observed_link_destination

    requested = "https://catalog.test/watch?item=123&tracking=abc"
    assert not observed_link_destination(requested, "https://catalog.test/watch?item=123", canonical)
    assert not observed_link_destination(requested, "https://catalog.test/watch?tracking=abc", canonical)


def test_actual_video_normalization_retains_page_declared_canonical_identity():
    from glide.computer.execution.contracts import observed_link_destination

    selected = "https://www.youtube.com/watch?v=lpdRqn6xwiM&list=RDlpdRqn6xwiM&start_radio=1&pp=ygUGWmFsaW1hoAcB"
    committed = "https://www.youtube.com/watch?v=lpdRqn6xwiM&list=RDlpdRqn6xwiM&start_radio=1"
    canonical = "https://www.youtube.com/watch?v=lpdRqn6xwiM"
    assert observed_link_destination(selected, committed, canonical)
    assert not observed_link_destination(selected, committed)
    assert not observed_link_destination(selected, committed.replace("v=lpdRqn6xwiM&", ""), canonical)


def test_search_and_play_first_result_runs_through_jev_with_observed_link_and_media(monkeypatch, tmp_path):
    class VideoComputer(SearchControlComputer):
        def execute(self, action, observed):
            receipt = super().execute(action, observed)
            if action.kind == "click" and action.target == "result":
                self.state.url = self.state.elements["result"].href
                self.state.tabs[self.state.active_tab] = self.state.url
                self.state.owner = "selected-result-document"
                self.state.media = {"video": Media("video", "First result", False, False, 4, 0.3)}
            return receipt

    computer = VideoComputer()
    computer.state.capabilities.add("media_state")
    computer.state.elements["result"] = Element("result", "First result", "link", href="https://catalog.test/watch/first")
    computer.state.media = {"preview": Media("preview", "Unrelated preview", False, False, 4, 5)}
    steps = [
        Milestone("search", "Submit tea query", "query_submitted", target="search", value="tea"),
        Milestone("play", "Open first result and start playback", "media_playing"),
    ]

    def select(state, options):
        if state["milestone"]["effect"] == "query_submitted":
            return select_search(state, options)
        return next(k for k, text in options.items() if k not in {"replan", "key"} and json.loads(text)["target"] == "result")

    writer = Reasoner([response(*steps)])
    state = drive(
        monkeypatch, tmp_path, computer, writer, Jev(route="plan", selection=select), goal="Search tea then play first result"
    )
    assert state.answer.achieved and state.outcome == "done"
    assert [a.kind for a in computer.actions] == ["type", "key", "click"]
    assert computer.state.url == "https://catalog.test/watch/first"
    assert [r.role for r in writer.requests] == ["planner"]
