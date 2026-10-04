"""execute(): approval, per-path re-validation at run time, never overwriting, per-action verification, partial reports."""

from __future__ import annotations

import errno
import json
import os
import stat
from dataclasses import replace

import pytest
from files_world import build_world

from glide.files import ORGANIZE_BY_TYPE, Approval, Move, Plan, Refused, engine, execute, plan


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def run(world, p, **kw):
    return execute(p, Approval(p.plan_hash), manifest_dir=world.runs, home=world.home, **kw)


def planned(world, *names):
    world.make(*names)
    return plan(world.root, ORGANIZE_BY_TYPE, home=world.home)


def statuses(report):
    return [a.status for a in report.actions]


def sealed(p: Plan, moves) -> Plan:
    """A hand-built plan whose hash is correct: the hash proves nothing about whether the paths are safe."""
    bare = Plan(root=p.root, intent=p.intent, moves=tuple(moves), plan_hash="")
    return replace(bare, plan_hash=bare.compute_hash())


def test_executes_an_approved_plan_and_verifies_every_action(world):
    p = planned(world, "a.pdf", "b.png", "c.pdf")
    report = run(world, p)
    assert report.status == "ok" and statuses(report) == ["done"] * 3
    assert world.listing() == ["Documents/", "Documents/a.pdf", "Documents/c.pdf", "Images/", "Images/b.png"]
    assert (world.root / "Documents" / "a.pdf").read_text() == "a.pdf"


def test_no_approval_means_nothing_runs(world):
    p = planned(world, "a.txt")
    before = world.listing()
    for approval in (None, Approval("0" * 32), Approval(""), "yes", p.plan_hash):
        with pytest.raises(Refused):
            execute(p, approval, manifest_dir=world.runs, home=world.home)
    assert world.listing() == before and not world.runs.exists()


def test_a_plan_whose_content_no_longer_matches_its_hash_is_refused(world):
    p = planned(world, "a.txt", "b.txt")
    stale = replace(p, moves=p.moves[:1])  # keeps the old plan_hash
    with pytest.raises(Refused):
        execute(stale, Approval(p.plan_hash), manifest_dir=world.runs, home=world.home)
    assert world.listing() == ["a.txt", "b.txt"]


@pytest.mark.parametrize(
    "where",
    ["../outside/a.txt", "Documents/../../a.txt", "Documents/sub/a.txt", ".Hidden/a.txt", "Documents/.a.txt", "Library/a.txt/"],
)
def test_a_plan_that_points_outside_root_is_refused_even_with_a_correct_hash(world, tmp_path, where):
    p = planned(world, "a.txt")
    source = p.moves[0].source
    bad = sealed(p, [Move(source, f"{p.root}/{where}", p.moves[0].stamp)])
    before = world.listing()
    with pytest.raises(Refused):
        run(world, bad)
    assert world.listing() == before and not (tmp_path / "outside").exists()


def test_a_source_outside_root_is_refused_even_with_a_correct_hash(world, tmp_path):
    (tmp_path / "secret.txt").write_text("s")
    p = planned(world, "a.txt")
    bad = sealed(p, [Move(str(tmp_path / "secret.txt"), f"{p.root}/Documents/secret.txt", (0, 0, 0))])
    with pytest.raises(Refused):
        run(world, bad)
    assert (tmp_path / "secret.txt").exists()


def test_a_root_that_became_a_symlink_is_refused(world, tmp_path):
    p = planned(world, "a.txt")
    elsewhere = world.home / "Elsewhere"
    world.root.rename(elsewhere)
    world.root.symlink_to(elsewhere)
    with pytest.raises(Refused):
        run(world, p)
    assert (elsewhere / "a.txt").exists()
    # and a root that now points outside the home is refused for that reason
    world.root.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    world.root.symlink_to(outside)
    with pytest.raises(Refused):
        run(world, p)


def test_a_source_replaced_by_a_symlink_is_not_followed(world, tmp_path):
    p = planned(world, "a.txt", "b.txt")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (world.root / "a.txt").unlink()
    (world.root / "a.txt").symlink_to(outside)
    report = run(world, p)
    assert statuses(report) == ["failed", "done"] and report.status == "partial"
    assert "symlink" in report.actions[0].reason
    assert outside.read_text() == "secret" and (world.root / "a.txt").is_symlink()
    assert not (world.root / "Documents" / "a.txt").exists()


def test_a_destination_folder_that_is_a_symlink_is_refused(world, tmp_path):
    p = planned(world, "a.txt", "b.png")
    outside = tmp_path / "outside"
    outside.mkdir()
    (world.root / "Documents").symlink_to(outside)
    report = run(world, p)
    assert statuses(report) == ["failed", "done"]
    assert list(outside.iterdir()) == [] and (world.root / "a.txt").exists()


def test_a_destination_folder_that_is_a_file_fails_the_action(world):
    p = planned(world, "a.txt")
    (world.root / "Documents").write_text("i am a file")
    report = run(world, p)
    assert report.status == "failed" and (world.root / "a.txt").exists()


def test_a_file_that_appeared_at_the_destination_is_never_overwritten(world):
    p = planned(world, "a.txt")
    (world.root / "Documents").mkdir()
    (world.root / "Documents" / "a.txt").write_text("precious")
    report = run(world, p)
    assert report.status == "ok" and report.actions[0].destination.endswith("Documents/a (2).txt")
    assert (world.root / "Documents" / "a.txt").read_text() == "precious"
    assert (world.root / "Documents" / "a (2).txt").read_text() == "a.txt"


def test_a_name_taken_between_resolution_and_rename_fails_instead_of_overwriting(world):
    (world.root / "Documents").mkdir()
    p = planned(world, "a.txt", "b.txt")
    taken = world.root / "Documents" / "a.txt"

    def taker():  # runs before each action, after the destinations were chosen
        if not taken.exists():
            taken.write_text("late arrival")
        return False

    report = run(world, p, should_stop=taker)
    assert statuses(report) == ["failed", "done"] and "exists" in report.actions[0].reason
    assert taken.read_text() == "late arrival" and (world.root / "a.txt").exists()


def test_collisions_inside_one_plan_each_get_their_own_name(world):
    world.make("a.txt", "A.TXT")  # one name on a case-insensitive disk
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    report = run(world, p)
    assert report.status == "ok"
    assert len({a.destination.lower() for a in report.actions}) == len(report.actions)


def test_a_file_removed_after_planning_fails_only_its_own_action(world):
    p = planned(world, "a.txt", "b.txt")
    (world.root / "a.txt").unlink()
    report = run(world, p)
    assert statuses(report) == ["failed", "done"] and report.status == "partial"
    assert "gone" in report.actions[0].reason


def test_a_file_replaced_after_planning_is_not_moved(world):
    p = planned(world, "a.txt")
    (world.root / "a.txt").unlink()
    (world.root / "a.txt").write_text("a completely different and longer file")
    report = run(world, p)
    assert report.status == "failed" and "changed" in report.actions[0].reason
    assert (world.root / "a.txt").read_text().startswith("a completely")
    assert not (world.root / "Documents").exists()  # no folder for a move that did not happen


def test_cross_volume_is_refused_not_copied(world, monkeypatch):
    p = planned(world, "a.txt")
    monkeypatch.setattr(engine, "_device", lambda path: 7 if path.endswith("a.txt") else 8)
    report = run(world, p)
    assert report.status == "failed" and "volume" in report.actions[0].reason
    assert (world.root / "a.txt").exists()


def test_a_rename_that_reports_a_different_volume_is_failed_with_the_source_intact(world, monkeypatch):
    p = planned(world, "a.txt")

    def exdev(src, dst):
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(engine.os, "rename", exdev)
    report = run(world, p)
    assert report.status == "failed" and "volume" in report.actions[0].reason
    assert (world.root / "a.txt").exists()


def test_the_manifest_is_written_before_the_first_move(world, monkeypatch):
    p = planned(world, "a.pdf", "b.png")
    real, seen = os.rename, []

    def spy(src, dst):
        files = list(world.runs.glob("*.json"))
        seen.append(json.loads(files[0].read_text())["entries"][len(seen)]["status"] if files else None)
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", spy)
    report = run(world, p)
    assert seen == ["planned", "planned"] and report.manifest is not None
    data = json.loads(report.manifest.read_text())
    assert [e["status"] for e in data["entries"]] == ["done", "done"]
    assert sorted(data["folders_created"]) == ["Documents", "Images"]
    assert data["plan_hash"] == p.plan_hash


def test_the_manifest_holds_paths_and_nothing_else(world):
    p = planned(world, "a.pdf")
    (world.root / "a.pdf").write_text("the secret body of the file")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    report = run(world, p)
    text = report.manifest.read_text()
    assert "secret body" not in text
    assert stat.S_IMODE(report.manifest.stat().st_mode) == 0o600


def test_a_failure_midway_leaves_an_exact_partial_report(world, monkeypatch):
    p = planned(world, "a.txt", "b.txt", "c.txt")
    real, calls = os.rename, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise PermissionError(errno.EACCES, "denied")
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", flaky)
    report = run(world, p)
    assert report.status == "partial" and statuses(report) == ["done", "failed", "done"]
    assert (world.root / "b.txt").exists() and not (world.root / "Documents" / "b.txt").exists()
    data = json.loads(report.manifest.read_text())
    assert [e["status"] for e in data["entries"]] == ["done", "failed", "done"]


def test_everything_failing_is_failed(world, monkeypatch):
    p = planned(world, "a.txt", "b.txt")

    def deny(src, dst):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(engine.os, "rename", deny)
    assert run(world, p).status == "failed"
    assert world.listing()[-2:] == ["a.txt", "b.txt"]


def test_an_outcome_that_cannot_be_observed_is_uncertain_and_stops_the_run(world, monkeypatch):
    p = planned(world, "a.txt", "b.txt", "c.txt")
    real, calls = os.rename, []

    def lying(src, dst):
        calls.append(src)
        if len(calls) == 2:
            return None  # reports success, does nothing: source still there, destination missing
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", lying)
    report = run(world, p)
    assert statuses(report) == ["done", "uncertain", "skipped"] and report.status == "partial"
    assert len(calls) == 2  # never retried, never continued
    assert json.loads(report.manifest.read_text())["entries"][1]["status"] == "uncertain"


def test_a_rename_that_raised_after_it_worked_is_seen_as_done(world, monkeypatch):
    p = planned(world, "a.txt")
    real = os.rename

    def worked_then_raised(src, dst):
        real(src, dst)
        raise OSError(errno.EIO, "late error")

    monkeypatch.setattr(engine.os, "rename", worked_then_raised)
    report = run(world, p)
    assert statuses(report) == ["done"] and (world.root / "Documents" / "a.txt").exists()


def test_a_look_that_fails_is_uncertain(world, monkeypatch):
    p = planned(world, "a.txt")
    real_look, real_rename, moved = engine._look, os.rename, []
    monkeypatch.setattr(engine.os, "rename", lambda a, b: (real_rename(a, b), moved.append(1))[0])
    monkeypatch.setattr(engine, "_look", lambda path: None if moved else real_look(path))  # the look after the move is blind
    assert statuses(run(world, p)) == ["uncertain"]


def test_stopping_midway_reports_exactly_what_was_done(world):
    p = planned(world, "a.txt", "b.txt", "c.txt")
    asked = []

    def stop_after_one():
        asked.append(1)
        return len(asked) > 1

    report = run(world, p, should_stop=stop_after_one)
    assert statuses(report) == ["done", "skipped", "skipped"] and report.status == "partial"
    assert report.actions[1].reason == "stopped"
    assert (world.root / "b.txt").exists() and (world.root / "Documents" / "a.txt").exists()


def test_stopping_before_the_first_action_changes_nothing(world):
    p = planned(world, "a.txt")
    report = run(world, p, should_stop=lambda: True)
    assert report.status == "failed" and world.listing() == ["a.txt"]


def test_a_plan_with_nothing_to_do_writes_no_manifest_and_makes_no_folder(world):
    world.make(".hidden")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    report = run(world, p)
    assert report.status == "ok" and report.actions == () and report.manifest is None
    assert not world.runs.exists() and world.listing() == [".hidden"]


def test_folders_are_made_only_for_moves_that_happen(world):
    p = planned(world, "a.txt", "b.png")
    (world.root / "b.png").unlink()
    report = run(world, p)
    assert report.status == "partial" and world.listing() == ["Documents/", "Documents/a.txt"]


def test_a_home_that_is_not_safe_is_refused(world):
    p = planned(world, "a.txt")
    with pytest.raises(Refused):
        execute(p, Approval(p.plan_hash), manifest_dir=world.runs, home=world.root)  # root is the home: not below it
