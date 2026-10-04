"""Unseen parameter combinations run through the real engine, with scripted model bindings.

These cases demonstrate composability/validation, not live-model language accuracy.
"""

import json
from dataclasses import replace

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.execution import policy
from glide.computer.execution.contracts import Action, Container, Element, InvalidAction, Milestone, rebind
from glide.computer.execution.grounding import extract


@pytest.mark.parametrize("quantity", [2, 17, 27, 64, 97, 100])
def test_unseen_tab_quantities_urls_and_query_numbers(monkeypatch, tmp_path, quantity):
    url = f"https://batch-{quantity}.test/archive?q={quantity * 31}"
    computer, jev = Computer(), Jev("tab_create", bindings={"quantity": str(quantity)})
    writer = Reasoner([RuntimeError("A resolved template needs no generative call")])
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal=f"Show {quantity} fresh pages of {url}")
    assert state.answer.achieved and len(computer.actions) == quantity
    assert list(computer.state.tabs.values()).count(url) == quantity
    assert state.progress == [{"id": "direct", "requested": quantity, "verified": quantity, "remaining": 0}]
    assert not writer.requests and len(jev.requests) == 2  # Scope, then observed direct binding; no LLM.


@pytest.mark.parametrize("phrase,quantity", [("twenty-seven", 27), ("六十四", 64), ("ninety seven", 97)])
def test_non_decimal_quantities_have_no_english_word_catalog(monkeypatch, tmp_path, phrase, quantity):
    url = "https://unseen-spoken-count.test/"
    computer, jev = Computer(), Jev("tab_create", bindings={"quantity": str(quantity)})
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), jev, goal=f"Create {phrase} tabs of {url}")
    assert state.answer.achieved and len(computer.state.tabs) == quantity + 1
    assert "quantity" in jev.questions[1] and jev.questions[1]["quantity"].criteria[str(quantity)]["requested_count"] == quantity


def test_uncertain_workflow_routes_to_planner_without_dispatching_a_guess(monkeypatch, tmp_path):
    computer, jev = Computer(), Jev("tab_create", bindings={"quantity": "17"})
    classify = jev.system_one

    def uncertain_route(state, questions):
        reply = classify(state, questions)
        if "milestone" not in state:
            reply.answers["action"].confidence = 0.36
            assert not computer.actions
        return reply

    jev.system_one = uncertain_route
    steps = [Milestone("tabs", "Create requested tabs", "tab_created", value="https://unseen.test", quantity=17)]
    writer = Reasoner([{"route": "plan", "question": ""}, response(*steps)])
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Open seventeen tabs at https://unseen.test")
    assert state.answer.achieved and len(computer.actions) == 17
    assert [r.role for r in writer.requests] == ["task_routing", "planner"]
    assert len(jev.requests) == 2  # Uncertain route, then a checked selection before the cached repeat.
    assert state.progress == [{"id": "tabs", "requested": 17, "verified": 17, "remaining": 0}]


@pytest.mark.parametrize("a,b,up,down", [(3, 8, 11, 17), (19, 7, 23, 29), (31, 13, 5, 2)])
def test_unseen_compound_mix_preserves_each_quantity_and_observed_target(monkeypatch, tmp_path, a, b, up, down):
    computer = Computer()
    field, scroll = f"field-{a}-{b}", f"container-{up}-{down}"
    label = f"Notes for group {a}"
    first, second = f"https://first-{a}.test/report", f"https://second-{b}.test/queue"
    text = f"Fresh text {a}+{b}, multilingual 文本, literal identifiers 00731"
    computer.state.elements = {field: Element(field, label, "field", "old", True)}
    computer.state.focus = field
    computer.state.containers = {scroll: Container(scroll, "Activity panel", 5000, 20000)}
    steps = [
        Milestone("a", "First batch", "tab_created", value=first, quantity=a),
        Milestone("b", "Second batch", "tab_created", value=second, quantity=b),
        Milestone("up", "Review earlier activity", "scroll", target="Activity panel", value="up", quantity=up),
        Milestone("down", "Review later activity", "scroll", target="Activity panel", value="down", quantity=down),
        Milestone("edit", "Set observed notes", "field_value", target=label, value=text),
        Milestone("switch", "Return to original tab", "tab_active", target="initial"),
        Milestone("close", "Close original tab", "tab_closed", target="initial"),
    ]
    writer = Reasoner([response(*steps)])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        goal=f"Open {a} pages of {first}, {b} pages of {second}; review activity both ways, edit notes, switch and close the original tab",
        steps=a + b + up + down + 10,
    )
    assert state.answer.achieved and len(writer.requests) == 1
    assert len(computer.state.tabs) == a + b and "initial" not in computer.state.tabs
    assert list(computer.state.tabs.values()).count(first) == a
    assert list(computer.state.tabs.values()).count(second) == b
    assert computer.state.containers[scroll].position == 5000 + 100 * (down - up)
    assert computer.state.elements[field].value == text
    assert [p["verified"] for p in state.progress] == [a, b, up, down, 1, 1, 1]
    assert all(p["remaining"] == 0 for p in state.progress)
    assert {action.target for action in computer.actions if action.kind == "scroll"} == {scroll}


@pytest.mark.parametrize(
    "key,mods",
    [
        ("g", ("alt", "control", "shift")),
        ("j", ("alt", "command")),
        ("u", ("control", "shift")),
        ("r", ("alt", "command", "control", "shift")),
    ],
)
@pytest.mark.parametrize("source", ["request", "observed_ui"])
def test_unseen_shortcut_combinations_bind_in_one_jev_call(monkeypatch, tmp_path, key, mods, source):
    names = {"alt": "Alt", "command": "Command", "control": "Control", "shift": "Shift"}
    chord = "+".join([*(names[m] for m in mods), key.upper()])
    computer = Computer()
    computer.state.elements["format"] = Element(
        "format", "Format selection", "button", shortcuts=(chord,) if source == "observed_ui" else ()
    )
    output = "Formatting applied"

    def key_effect(c, action):
        assert action.kind == "key" and action.value == key and action.modifiers == mods
        c.state.elements["status"] = Element("status", output, "status", enabled=False)
        return ""

    computer.on_execute = key_effect
    jev = Jev(selection=lambda *_: "key", bindings={"key": key, "modifiers": json.dumps(list(mods))})
    goal = f"Apply formatting using {chord}" if source == "request" else "Apply the formatting shortcut shown in the current UI"
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([response(Milestone("format", "Format selection", "element_present", target=output))]),
        jev,
        goal=goal,
    )
    assert state.answer.achieved and len(computer.actions) == 1
    assert set(jev.questions[-1]) == {"action", "key", "modifiers"}
    assert len(jev.questions[-1]["modifiers"].criteria) == 17  # General modifier powerset + unsupported.
    assert {"key": key, "modifiers": mods} in jev.requests[-1]["grounded_shortcuts"]


def test_guessing_an_unobserved_shortcut_is_not_an_executable_binding(monkeypatch, tmp_path):
    computer = Computer()
    step = Milestone("format", "Format selection", "element_present", target="Formatted")
    writer = Reasoner([response(step), response(unsupported=["observed formatting shortcut"])])
    jev = Jev(selection=lambda *_: "key", bindings={"key": "g", "modifiers": json.dumps(["alt", "control"])})
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Format the selected text")
    assert state.outcome == "unsupported" and not computer.actions
    assert "absent from the request" in json.loads(writer.requests[-1].text)["contradiction"]


@pytest.mark.parametrize("capability", ["drag", "clipboard_read", "filesystem verification", "horizontal scroll"])
def test_unavailable_capabilities_are_explicit_without_partial_execution(monkeypatch, tmp_path, capability):
    computer = Computer()
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([response(unsupported=[capability])]),
        goal=f"Perform work requiring {capability}",
    )
    assert state.outcome == "unsupported" and not state.answer and not computer.actions
    assert state.unsupported_capabilities == [capability] and capability in state.failure
    # The planner's words can echo the request, so the stored report keeps only a count unless content is recorded (D3).
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["unsupported_capability_count"] == 1 and "unsupported_capabilities" not in report
    assert capability not in report["failure"]


def test_the_named_capability_is_stored_when_content_recording_is_on(monkeypatch, tmp_path):
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([response(unsupported=["read unavailable file"])]),
        record_content=True,
    )
    report = json.loads((tmp_path / "run.json").read_text())
    assert state.outcome == "unsupported" and report["unsupported_capabilities"] == ["read unavailable file"]
    assert "read unavailable file" in report["failure"]


def test_missing_tab_capability_is_detected_even_when_planner_omits_the_report(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.capabilities.remove("tab_create")
    steps = [
        Milestone("nav", "Navigate", "url", value="https://new.test"),
        Milestone("tabs", "Create new tabs", "tab_created", value="https://new.test", quantity=27),
    ]
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*steps)]))
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["tab_create"]
    assert not computer.actions and "Available capabilities" in state.failure


def test_unavailable_verification_cannot_be_substituted_with_a_url_visit(monkeypatch, tmp_path):
    computer = Computer()
    plan = response(Milestone("nav", "Visit", "url", value="https://files.test"))
    plan["steps"].append(
        {"id": "file", "goal": "Verify local file", "effect": "file_exists", "target": "report.pdf", "value": "", "quantity": 1}
    )
    state = drive(monkeypatch, tmp_path, computer, Reasoner([plan]))
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["verification:file_exists"]
    assert not computer.actions


def test_model_cannot_overwrite_a_numeric_count_literal(monkeypatch, tmp_path):
    computer = Computer()
    step = Milestone("tabs", "Create exactly 43 tabs", "tab_created", value="https://new.test", quantity=43)
    writer = Reasoner([response(step)])
    jev = Jev("tab_create", bindings={"quantity": "44"})
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Create 43 tabs of https://new.test")
    assert state.answer.achieved and len(writer.requests) == 1
    assert len(computer.state.tabs) == 44 and state.progress[0]["verified"] == 43


@pytest.mark.parametrize("count", ["0", "-3", "2.5", "1001"])
def test_invalid_count_literals_are_not_rounded_or_capped(count):
    literals = extract(f"Create {count} pages of https://new.test/path?q=8")
    assert literals.needs_plan


def test_all_literal_arguments_are_preserved_for_planning(monkeypatch, tmp_path):
    computer = Computer()
    urls = [f"https://unseen-{n}.test" for n in range(5)]
    steps = [Milestone(f"tab-{n}", "Create requested tab", "tab_created", value=url) for n, url in enumerate(urls)]
    writer = Reasoner([response(*steps)])
    state = drive(monkeypatch, tmp_path, computer, writer, goal=", ".join(urls))
    assert state.answer.achieved and len(writer.requests) == 1
    assert set(computer.state.tabs.values()) == {"https://example.org", *urls}
    assert extract(", ".join(urls)).urls == urls


@pytest.mark.parametrize("raw,value", [('"escaped \\"quote\\" 0731"', 'escaped "quote" 0731'), ('""', "")])
def test_literal_text_is_exact_and_not_a_count(monkeypatch, tmp_path, raw, value):
    computer, writer = Computer(), Reasoner([])
    computer.state.elements["name"] = replace(computer.state.elements["name"], value="old")
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("type"), goal=f"Set Name to {raw}")
    assert state.answer.achieved and computer.state.elements["name"].value == value and not writer.requests


def test_explicit_ip_url_and_port_do_not_become_quantities(monkeypatch, tmp_path):
    url = "https://192.0.2.73:9443/archive?ref=891"
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), Jev("navigate"), goal=f"Go to {url}")
    assert state.answer.achieved and computer.state.url == url


def test_choice_limit_is_reported_without_silently_dropping_late_handles():
    computer = Computer()
    computer.state.elements = {str(n): Element(str(n), f"Element {n}", "button") for n in range(260)}
    with pytest.raises(InvalidAction, match="safety limit"):
        policy.candidates(Milestone("focus", "Focus last observed item", "focus", target="259"), computer.inspect())


def test_scope_answer_prevents_a_direct_action_from_completing_half_a_goal(monkeypatch, tmp_path):
    computer = Computer()
    writer = Reasoner(
        [
            response(
                Milestone("nav", "Navigate", "url", value="https://new.test"),
                Milestone("scroll", "Scroll later", "scroll", target="Feed", value="down", quantity=7),
            )
        ]
    )
    jev = Jev("navigate", bindings={"scope": "plan"})
    state = drive(
        monkeypatch, tmp_path, computer, writer, jev, goal="https://new.test; afterwards review seven more sections of the feed"
    )
    assert state.answer.achieved and len(writer.requests) == 1
    assert computer.state.containers["feed"].position == 700


def test_changed_observed_shortcut_source_fails_fresh_preflight():
    before = Computer().inspect()
    before.elements["shortcut"] = Element("shortcut", "Format", "button", shortcuts=("Control+Alt+g",))
    fresh = Computer().inspect()
    fresh.elements["shortcut"] = replace(before.elements["shortcut"], shortcuts=("Control+Alt+j",))
    action = Action("key", before.identity, value="g", modifiers=("alt", "control"), parameter_source="observed:shortcut")
    with pytest.raises(InvalidAction, match="observed shortcut parameters changed"):
        rebind(action, before, fresh)


def test_unsupported_keyboard_symbol_is_reported_before_input(monkeypatch, tmp_path):
    computer = Computer()
    jev = Jev(selection=lambda *_: "key", bindings={"key": "unsupported", "modifiers": json.dumps(["control"])})
    step = Milestone("outcome", "Reach requested view", "element_present", target="Ready")
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), jev, goal="Use Control+F8 to reach Ready")
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["keyboard:f8"]
    assert not computer.actions


def test_partial_supported_plan_is_not_executed_when_other_capability_is_missing(monkeypatch, tmp_path):
    computer = Computer()
    reply = response(Milestone("nav", "Navigate", "url", value="https://new.test"), unsupported=["drag"])
    state = drive(monkeypatch, tmp_path, computer, Reasoner([reply]))
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["drag"] and not computer.actions


def test_model_cannot_claim_an_available_backend_capability_is_missing(monkeypatch, tmp_path):
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(unsupported=["click"])]))
    assert state.outcome == "blocked" and not state.unsupported_capabilities and not computer.actions
    assert "available capability" in state.failure


def test_horizontal_parameter_is_explicitly_unsupported_even_if_report_is_omitted(monkeypatch, tmp_path):
    computer = Computer()
    plan = response()
    plan["steps"] = [
        {"id": "scroll", "goal": "Scroll left", "effect": "scroll", "target": "Feed", "value": "left", "quantity": 1}
    ]
    state = drive(monkeypatch, tmp_path, computer, Reasoner([plan]))
    assert state.outcome == "unsupported" and state.unsupported_capabilities == ["scroll:left"] and not computer.actions


def test_measured_distance_cannot_become_a_repetition_count(monkeypatch, tmp_path):
    computer = Computer()
    jev = Jev()
    writer = Reasoner([response(unsupported=["scroll:exact pixel distance"])])
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Scroll Feed by 50 pixels")
    assert state.outcome == "unsupported" and not computer.actions
    assert extract("Scroll Feed by 50 pixels").needs_plan
    observed_request = next(request for request in jev.requests if "observation" in request)
    assert observed_request["observation"]["parameter_domains"]["scroll"]["exact_distance_parameters"] is False
