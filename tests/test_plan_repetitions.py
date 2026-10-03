"""Repeated effects remain quantities, without rewriting a durable recovery plan."""

from dataclasses import replace

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.execution import planning
from glide.computer.execution.contracts import MAX_REPETITIONS, InvalidAction, Milestone


@pytest.mark.parametrize("quantity", [2, 10, 16])
def test_initial_identical_tab_effects_become_one_quantity(quantity):
    first = Milestone("first", "Create requested tabs", "tab_created", value=f"https://unseen-{quantity}.test/page")
    steps = [replace(first, id=f"tab-{n}", goal=f"Create tab {n}") for n in range(quantity)]
    writer = Reasoner([response(*steps)])
    planned, question = planning.plan(writer, "Create these tabs", Computer().state)
    assert planned == [replace(steps[0], quantity=quantity)]
    assert not question
    assert "ONE milestone" in writer.requests[0].instructions


@pytest.mark.parametrize("quantity", [17, 64, MAX_REPETITIONS])
def test_initial_repeated_quantities_aggregate_beyond_raw_plan_step_limit(quantity):
    first = Milestone("first", "Create tabs", "tab_created", value="https://unseen.test/page", quantity=quantity // 2)
    second = replace(first, id="second", quantity=quantity - first.quantity)
    planned, _ = planning.plan(Reasoner([response(first, second)]), "Create all tabs", Computer().state)
    assert planned == [replace(first, quantity=quantity)]


def test_initial_repeated_scroll_quantities_add_without_merging_different_targets_or_directions():
    first = Milestone("feed-a", "Scroll feed", "scroll", target="Feed", value="down", quantity=7)
    again = replace(first, id="feed-b", quantity=11)
    upward = replace(first, id="feed-up", value="up", quantity=3)
    sidebar = replace(first, id="sidebar", target="Sidebar", quantity=5)
    planned, _ = planning.plan(Reasoner([response(first, again, upward, sidebar)]), "Review both panels", Computer().state)
    assert planned == [replace(first, quantity=18), upward, sidebar]


def test_nonadjacent_or_different_url_effects_are_not_reordered():
    first = Milestone("first", "Create first page", "tab_created", value="https://first.test")
    second = Milestone("second", "Create another page", "tab_created", value="https://second.test")
    last = replace(first, id="last")
    planned, _ = planning.plan(Reasoner([response(first, second, last)]), "Create pages in order", Computer().state)
    assert planned == [first, second, last]


def test_nonrepeatable_effects_keep_separate_milestones():
    first = Milestone("first", "Reach page", "url", value="https://same.test")
    second = replace(first, id="second")
    planned, _ = planning.plan(Reasoner([response(first, second)]), "Reach page twice", Computer().state)
    assert planned == [first, second]


def test_coalescing_refuses_overflow_without_capping_the_requested_quantity():
    first = Milestone("first", "Create all pages", "tab_created", value="https://large.test", quantity=MAX_REPETITIONS)
    second = replace(first, id="second", quantity=1)
    with pytest.raises(InvalidAction, match="repetition safety limit"):
        planning.plan(Reasoner([response(first, second)]), "Create too many pages", Computer().state)


def test_duplicate_ids_are_rejected_before_coalescing():
    first = Milestone("duplicate", "Create requested tabs", "tab_created", value="https://same.test")
    with pytest.raises(InvalidAction, match="Duplicate milestone IDs"):
        planning.plan(Reasoner([response(first, first)]), "Create pages", Computer().state)


def test_recovery_retains_existing_repeat_ids_and_contracts():
    first = Milestone("first", "Create requested tabs", "tab_created", value="https://same.test")
    second = replace(first, id="second")
    original = [first, second]
    planned, _ = planning.plan(
        Reasoner([response(*original)]), "Create pages", Computer().state, original, reason="Retry selection"
    )
    assert planned == original
    assert [(s.id, s.contract) for s in planned] == [(s.id, s.contract) for s in original]


def test_compacted_initial_plan_runs_all_verified_repetitions_with_one_action_selection(monkeypatch, tmp_path):
    first = Milestone("tab-0", "Create tabs", "tab_created", value="https://unrelated.test/manual")
    planned = [replace(first, id=f"tab-{i}") for i in range(10)]
    computer, writer, jev = Computer(), Reasoner([response(*planned)]), Jev()
    state = drive(monkeypatch, tmp_path, computer, writer, jev, goal="Create ten copies of the manual")
    assert state.answer.achieved
    assert len(computer.actions) == 10
    assert state.progress == [{"id": "tab-0", "requested": 10, "verified": 10, "remaining": 0}]
    assert len(writer.requests) == 1 and len(jev.requests) == 3  # Scope, workflow, one cached action selection.
