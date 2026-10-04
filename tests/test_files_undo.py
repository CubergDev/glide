"""undo(): reverses a run from its manifest, refusing whatever no longer fits, and reports the same statuses."""

from __future__ import annotations

import errno
import json
import os

import pytest
from files_world import build_world

from glide.files import ORGANIZE_BY_TYPE, Approval, Refused, engine, execute, plan, undo


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def applied(world, *names):
    world.make(*names)
    before = world.listing()
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    report = execute(p, Approval(p.plan_hash), manifest_dir=world.runs, home=world.home)
    assert report.status == "ok"
    return report.manifest, before


def statuses(report):
    return [a.status for a in report.actions]


def test_undo_restores_everything_and_removes_the_folders_the_plan_made(world):
    manifest, before = applied(world, "a.pdf", "b.png", "c.pdf")
    report = undo(manifest, home=world.home)
    assert report.status == "ok" and statuses(report) == ["done"] * 3
    assert world.listing() == before
    assert sorted(report.folders_removed) == ["Documents", "Images"]
    assert (world.root / "a.pdf").read_text() == "a.pdf"


def test_undo_runs_in_reverse_order(world, monkeypatch):
    manifest, _ = applied(world, "a.pdf", "b.pdf", "c.pdf")
    real, order = os.rename, []

    def spy(src, dst):
        order.append(os.path.basename(dst))
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", spy)
    undo(manifest, home=world.home)
    assert order == ["c.pdf", "b.pdf", "a.pdf"]


def test_a_folder_that_was_already_there_is_kept(world):
    (world.root / "Documents").mkdir()
    manifest, _ = applied(world, "a.pdf")
    report = undo(manifest, home=world.home)
    assert report.folders_removed == () and (world.root / "Documents").is_dir()


def test_a_folder_with_something_else_in_it_is_kept(world):
    manifest, _ = applied(world, "a.pdf")
    (world.root / "Documents" / "mine.txt").write_text("not from the plan")
    report = undo(manifest, home=world.home)
    assert report.status == "ok" and report.folders_removed == ()
    assert (world.root / "Documents" / "mine.txt").exists()


def test_a_gone_destination_is_refused_and_the_rest_still_undo(world):
    manifest, _ = applied(world, "a.pdf", "b.pdf")
    (world.root / "Documents" / "a.pdf").unlink()
    report = undo(manifest, home=world.home)
    assert statuses(report) == ["done", "failed"] and report.status == "partial"  # last move is undone first
    assert "gone" in report.actions[1].reason
    assert (world.root / "b.pdf").exists() and not (world.root / "a.pdf").exists()
    assert report.folders_removed == ("Documents",)  # the plan made it and it is empty now


def test_an_original_path_that_exists_again_is_never_overwritten(world):
    manifest, _ = applied(world, "a.pdf")
    (world.root / "a.pdf").write_text("new file with the old name")
    report = undo(manifest, home=world.home)
    assert report.status == "failed" and "exists" in report.actions[0].reason
    assert (world.root / "a.pdf").read_text() == "new file with the old name"
    assert (world.root / "Documents" / "a.pdf").exists()


def test_a_destination_replaced_by_a_symlink_is_not_followed(world, tmp_path):
    manifest, _ = applied(world, "a.pdf")
    outside = tmp_path / "outside.txt"
    outside.write_text("out")
    (world.root / "Documents" / "a.pdf").unlink()
    (world.root / "Documents" / "a.pdf").symlink_to(outside)
    report = undo(manifest, home=world.home)
    assert report.status == "failed" and outside.read_text() == "out"
    assert not (world.root / "a.pdf").exists()


def test_a_destination_folder_replaced_by_a_symlink_is_refused(world, tmp_path):
    manifest, _ = applied(world, "a.pdf")
    outside = tmp_path / "outside"
    outside.mkdir()
    (world.root / "Documents" / "a.pdf").rename(outside / "a.pdf")
    (world.root / "Documents").rmdir()
    (world.root / "Documents").symlink_to(outside)
    report = undo(manifest, home=world.home)
    assert report.status == "failed" and (outside / "a.pdf").exists()


def test_undo_twice_does_nothing_the_second_time(world):
    manifest, before = applied(world, "a.pdf", "b.png")
    undo(manifest, home=world.home)
    again = undo(manifest, home=world.home)
    assert statuses(again) == ["skipped", "skipped"] and world.listing() == before


def test_entries_that_did_not_happen_or_are_unknown_are_left_alone(world, monkeypatch):
    world.make("a.txt", "b.txt", "c.txt")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    real, calls = os.rename, []

    def lying(src, dst):
        calls.append(src)
        if len(calls) == 2:
            return None
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", lying)
    manifest = execute(p, Approval(p.plan_hash), manifest_dir=world.runs, home=world.home).manifest
    monkeypatch.setattr(engine.os, "rename", real)
    report = undo(manifest, home=world.home)
    assert statuses(report) == ["skipped", "skipped", "done"]
    assert "reconcile" in report.actions[1].reason
    assert world.listing() == ["a.txt", "b.txt", "c.txt"]


def test_a_failed_undo_move_is_a_partial_report(world, monkeypatch):
    manifest, _ = applied(world, "a.pdf", "b.pdf")
    real, calls = os.rename, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "denied")
        return real(src, dst)

    monkeypatch.setattr(engine.os, "rename", flaky)
    report = undo(manifest, home=world.home)
    assert statuses(report) == ["failed", "done"] and report.status == "partial"
    assert report.folders_removed == ()  # b.pdf's folder still holds a.pdf


def test_undo_can_be_stopped_midway(world):
    manifest, _ = applied(world, "a.pdf", "b.pdf", "c.pdf")
    asked = []

    def stop():
        asked.append(1)
        return len(asked) > 1

    report = undo(manifest, home=world.home, should_stop=stop)
    assert statuses(report) == ["done", "skipped", "skipped"] and report.folders_removed == ()


def test_a_manifest_that_points_outside_root_is_refused_whole(world, tmp_path):
    manifest, _ = applied(world, "a.pdf")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("x")
    data = json.loads(manifest.read_text())
    data["entries"][0]["source"] = str(outside / "gone.txt")
    manifest.write_text(json.dumps(data))
    with pytest.raises(Refused):
        undo(manifest, home=world.home)
    assert (world.root / "Documents" / "a.pdf").exists()


@pytest.mark.parametrize("field", ["destination", "folders_created"])
def test_a_manifest_with_traversal_names_is_refused(world, field):
    manifest, _ = applied(world, "a.pdf")
    data = json.loads(manifest.read_text())
    if field == "destination":
        data["entries"][0]["destination"] = f"{data['root']}/Documents/../../a.pdf"
    else:
        data["folders_created"] = ["../Elsewhere"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(Refused):
        undo(manifest, home=world.home)


def test_a_manifest_for_an_unsafe_root_or_unreadable_file_is_refused(world, tmp_path):
    manifest, _ = applied(world, "a.pdf")
    data = json.loads(manifest.read_text())
    data["root"] = str(world.home)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(Refused):
        undo(bad, home=world.home)
    bad.write_text("{")
    with pytest.raises(Refused):
        undo(bad, home=world.home)
    with pytest.raises(Refused):
        undo(tmp_path / "missing.json", home=world.home)
