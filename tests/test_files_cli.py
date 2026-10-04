"""`python -m glide.files`: plan, preview, apply (only with the hash preview printed) and undo."""

from __future__ import annotations

import json

import pytest
from files_world import build_world

from glide.files.cli import main


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = build_world(tmp_path)
    monkeypatch.setenv("HOME", str(w.home))
    monkeypatch.setenv("USERPROFILE", str(w.home))
    return w


def test_the_whole_flow_needs_the_printed_hash(world, tmp_path, capsys):
    world.make("a.pdf", "b.png")
    plan_file = tmp_path / "plan.json"
    assert main(["plan", str(world.root), "--intent", "type", "--out", str(plan_file)]) == 0
    assert world.listing() == ["a.pdf", "b.png"]

    capsys.readouterr()
    assert main(["preview", str(plan_file)]) == 0
    shown = capsys.readouterr().out
    plan_hash = json.loads(plan_file.read_text())["plan_hash"]
    assert plan_hash in shown and "Documents/" in shown and world.listing() == ["a.pdf", "b.png"]

    with pytest.raises(SystemExit) as no_flag:  # --approve is required: there is no blind apply
        main(["apply", str(plan_file), "--manifest-dir", str(world.runs)])
    assert no_flag.value.code == 2
    assert main(["apply", str(plan_file), "--approve", "0" * 32, "--manifest-dir", str(world.runs)]) == 1
    assert world.listing() == ["a.pdf", "b.png"]

    capsys.readouterr()
    assert main(["apply", str(plan_file), "--approve", plan_hash, "--manifest-dir", str(world.runs)]) == 0
    assert "ok" in capsys.readouterr().out
    assert world.listing() == ["Documents/", "Documents/a.pdf", "Images/", "Images/b.png"]

    manifest = next(world.runs.glob("*.json"))
    assert main(["undo", str(manifest)]) == 0
    assert world.listing() == ["a.pdf", "b.png"]


def test_plan_prints_json_to_stdout_without_out(world, capsys):
    world.make("a.pdf")
    assert main(["plan", str(world.root), "--intent", "type"]) == 0
    assert json.loads(capsys.readouterr().out)["moves"][0]["source"].endswith("a.pdf")


def test_named_folders_come_from_a_json_file(world, tmp_path, capsys):
    world.make("invoice.pdf", "other.txt")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"Money": ["invoice"]}))
    assert main(["plan", str(world.root), "--intent", "named", "--folders", str(rules)]) == 0
    moves = json.loads(capsys.readouterr().out)["moves"]
    assert [m["destination"].endswith("Money/invoice.pdf") for m in moves] == [True]


def test_refusals_exit_nonzero_with_a_message_and_touch_nothing(world, tmp_path, capsys):
    world.make("a.pdf")
    assert main(["plan", str(world.home), "--intent", "type"]) == 1
    assert "refused" in capsys.readouterr().err.lower()
    assert main(["plan", str(world.root), "--intent", "named"]) == 1  # named folders need --folders
    bad = tmp_path / "plan.json"
    bad.write_text("{}")
    assert main(["preview", str(bad)]) == 1
    assert main(["undo", str(tmp_path / "missing.json")]) == 1
    assert world.listing() == ["a.pdf"]
