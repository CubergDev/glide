"""Observed-target binding and generic query submission through the production engine."""

import json
from dataclasses import replace
from urllib.parse import urlencode

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.execution.contracts import Action, Element, InvalidAction, Milestone, effect, validate
from glide.computer.execution.policy import candidates


@pytest.mark.parametrize(
    "destination,parameter,query,label",
    [
        ("https://google.com", "q", "how do you make biryani", "Search"),
        ("https://docs.example.test/find", "term", "async task cancellation", "Look up documentation"),
        ("https://library.example.test/catalog", "keywords", "番茄 炒蛋 recipe", "Catalogue keywords"),
        ("https://travel.example.test/results", "destination", "São Paulo & 香港", "Destination"),
    ],
)
def test_navigation_field_binding_and_submission_are_general(monkeypatch, tmp_path, destination, parameter, query, label):
    computer, jev = Computer(), Jev()
    computer.state.elements = {"input": Element("input", label, "textbox", "", True)}
    computer.state.focus = ""
    computer.submissions["input"] = (destination, parameter)
    steps = [
        Milestone("navigate", "Reach the requested site", "url", value=destination),
        Milestone("input", "Enter the user's exact query", "field_value", target="the query input", value=query),
        Milestone(
            "submit", "Verify submitted query", "url_query", target=destination + "?" + urlencode({parameter: query}), value=query
        ),
    ]
    reasoner = Reasoner([response(*steps)])
    state = drive(monkeypatch, tmp_path, computer, reasoner, jev, goal=f"Open {destination} and search for {query}")
    assert state.outcome == "done" and state.answer.achieved
    assert [a.kind for a in computer.actions] == ["navigate", "type", "key"]
    assert computer.actions[1].target == "input" and computer.actions[1].target_contract == steps[1].contract
    assert computer.actions[2].value == "return" and not computer.actions[2].modifiers
    assert computer.state.url == destination + "?" + urlencode({parameter: query, "source": "offline"})
    assert len(reasoner.requests) == 1 and all(p["remaining"] == 0 for p in state.progress)


def test_descriptive_field_binding_also_handles_unrelated_desktop_form(monkeypatch, tmp_path):
    computer = Computer()
    step = Milestone("edit", "Change the document name", "field_value", target="document name input", value="Q4 résumé")
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]))
    assert state.answer.achieved and computer.state.elements["name"].value == "Q4 résumé"


def test_exact_observed_field_restricts_bindings_and_credentials_are_never_offered():
    observed = Computer().inspect()
    observed.elements["other"] = Element("other", "Other field", "textbox", "", True)
    observed.elements["secret"] = Element("secret", "Password", "textbox", "", True, True)
    exact = Milestone("edit", "Edit name", "field_value", target="Name", value="new")
    descriptive = replace(exact, target="document name input")
    assert [a.target for a in candidates(exact, observed) if a.kind == "type"] == ["name"]
    assert {a.target for a in candidates(descriptive, observed) if a.kind == "type"} == {"name", "other"}


def test_wrong_field_or_changed_description_cannot_satisfy_bound_effect():
    computer = Computer()
    step = Milestone("input", "Enter query", "field_value", target="query input", value="requested")
    before = computer.inspect()
    action = next(a for a in candidates(step, before) if a.kind == "type")
    computer.execute(action, before)
    after = computer.inspect()
    assert effect(step, action, before, after)
    assert not effect(replace(step, target="another input"), action, before, after)
    assert not effect(step, replace(action, target_contract=""), before, after)
    after.elements["name"] = replace(after.elements["name"], value="wrong")
    assert not effect(step, action, before, after)


@pytest.mark.parametrize("binding", [None, 0, "made-up", "f" * 63, "z" * 64])
def test_invalid_target_provenance_is_rejected(binding):
    obs = Computer().inspect()
    with pytest.raises(InvalidAction, match="target binding"):
        validate(Action("type", obs.identity, "name", "value", target_contract=binding), obs)


@pytest.mark.parametrize(
    "url,ready,valid",
    [
        ("https://www.google.com/search?q=how+do+you+make+biryani&source=header", True, True),
        ("https://google.com/search?q=how%20do%20you%20make%20biryani#results", True, True),
        ("https://google.com/search?q=how+do+you+make+biryani+rice", True, False),
        ("https://google.com/search?q=how+do+you+make+biryani&q=wrong", True, False),
        ("https://google.com/search?utm_content=how+do+you+make+biryani", True, False),
        ("https://google.com/search?q=wrong&source=how+do+you+make+biryani", True, False),
        ("https://unrelated.test/search?q=how+do+you+make+biryani", True, False),
        ("https://google.com/search?q=how+do+you+make+biryani", False, False),
    ],
)
def test_query_effect_checks_exact_decoded_text_destination_and_readiness(url, ready, valid):
    obs = Computer().inspect()
    obs.url, obs.ready = url, ready
    step = Milestone(
        "submit",
        "Verify query",
        "url_query",
        target="https://google.com/search?q=how+do+you+make+biryani",
        value="how do you make biryani",
    )
    assert bool(effect(step, None, obs, obs)) == valid


def test_typing_success_is_not_reported_as_submission_success(monkeypatch, tmp_path):
    computer = Computer()
    steps = [
        Milestone("input", "Type query", "field_value", target="query input", value="a new query"),
        Milestone("submit", "Submit query", "url_query", target="https://example.test/search?q=a+new+query", value="a new query"),
    ]
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*steps)]), handoffs=0, steps=5)
    assert not state.answer and state.outcome != "done"
    assert state.progress[0]["verified"] == 1 and state.progress[1]["remaining"] == 1


def test_root_query_url_does_not_match_an_unrelated_path():
    obs = Computer().inspect()
    obs.url = "https://example.test/logout?term=requested"
    step = Milestone("query", "Verify query URL", "url_query", target="https://example.test/?term=requested", value="requested")
    assert not effect(step, None, obs, obs)


@pytest.mark.parametrize("target", ["", "https://example.test/search", "https://example.test/search?q=other"])
def test_query_effect_needs_a_concrete_expected_parameter(target):
    with pytest.raises(InvalidAction, match="named parameter"):
        Milestone("query", "Verify query", "url_query", target=target, value="requested")


def test_tab_creation_does_not_wait_for_previous_document_loading(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.ready = False
    step = Milestone("tabs", "Create three tabs", "tab_created", value="https://new.example.test", quantity=3)
    monkeypatch.setattr("glide.computer.execution.progress.time.sleep", lambda _: pytest.fail("Waited for unrelated load"))
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), readiness_timeout=1)
    assert state.answer.achieved and len(computer.actions) == 3


def test_failed_plan_records_stage_and_type_without_task_content(monkeypatch, tmp_path):
    private = "private user instruction"
    state = drive(monkeypatch, tmp_path, Computer(), Reasoner([response()]), goal=private)
    assert state.outcome == "blocked"
    data = json.loads((tmp_path / "run.json").read_text())
    assert data["failure_stage"] == "planning" and data["error_type"] == "InvalidAction"
    assert private not in json.dumps(data)
    assert data["phases"]["planning"]["calls"] == 1
    assert all(p["seconds"] >= 0 for p in data["phases"].values())
