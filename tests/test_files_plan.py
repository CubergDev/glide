"""The file planner: what plan() picks up, where it sends it, and what preview() says without touching anything."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from files_world import build_world

from glide.files import NAMED_FOLDERS, ORGANIZE_BY_TYPE, Plan, Refused, plan, preview


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def targets(p: Plan) -> dict[str, str]:
    return {m.source.rsplit("/", 1)[1]: m.destination.removeprefix(p.root + "/") for m in p.moves}


def test_organize_by_type_uses_the_extension_table(world):
    world.make("a.pdf", "b.PNG", "c.xlsx", "d.pptx", "e.mp3", "f.mov", "g.zip", "h.py", "i.weird", "j")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    assert targets(p) == {
        "a.pdf": "Documents/a.pdf",
        "b.PNG": "Images/b.PNG",
        "c.xlsx": "Spreadsheets/c.xlsx",
        "d.pptx": "Presentations/d.pptx",
        "e.mp3": "Audio/e.mp3",
        "f.mov": "Video/f.mov",
        "g.zip": "Archives/g.zip",
        "h.py": "Code/h.py",
        "i.weird": "Other/i.weird",
        "j": "Other/j",
    }


def test_the_table_is_configuration(world):
    world.make("a.pdf", "b.txt")
    p = plan(world.root, ORGANIZE_BY_TYPE, categories={"Reading": (".pdf",)}, home=world.home)
    assert targets(p) == {"a.pdf": "Reading/a.pdf", "b.txt": "Other/b.txt"}


def test_only_direct_visible_regular_files(world):
    world.make("a.txt", ".hidden.txt", ".DS_Store")
    (world.root / "folder").mkdir()
    (world.root / "folder" / "inner.txt").write_text("x")
    (world.root / "link.txt").symlink_to(world.root / "a.txt")
    assert targets(plan(world.root, ORGANIZE_BY_TYPE, home=world.home)) == {"a.txt": "Documents/a.txt"}


def test_named_folders_match_keywords_and_extensions(world):
    world.make("Invoice-March.pdf", "photo.JPG", "notes.txt")
    folders = {"Money": ["invoice"], "Pictures": [".jpg", ".png"]}
    p = plan(world.root, NAMED_FOLDERS, folders=folders, home=world.home)
    assert targets(p) == {"Invoice-March.pdf": "Money/Invoice-March.pdf", "photo.JPG": "Pictures/photo.JPG"}  # notes.txt stays


def test_named_folders_first_match_wins_and_empty_keywords_are_refused(world):
    world.make("invoice.pdf")
    p = plan(world.root, NAMED_FOLDERS, folders={"A": ["inv"], "B": ["invoice"]}, home=world.home)
    assert targets(p) == {"invoice.pdf": "A/invoice.pdf"}
    for bad in ({"A": []}, {"A": [""]}):
        with pytest.raises(Refused):
            plan(world.root, NAMED_FOLDERS, folders=bad, home=world.home)


def test_unknown_intent_and_missing_folders_are_refused(world):
    with pytest.raises(Refused):
        plan(world.root, "delete everything", home=world.home)
    with pytest.raises(Refused):
        plan(world.root, NAMED_FOLDERS, home=world.home)


@pytest.mark.parametrize("name", ["../x", "a/b", ".hid", "", ".", "..", "a\x00b", "a\\b", "/abs"])
def test_folder_names_that_could_escape_are_refused(world, name):
    world.make("a.txt")
    with pytest.raises(Refused):
        plan(world.root, NAMED_FOLDERS, folders={name: ["a"]}, home=world.home)
    with pytest.raises(Refused):
        plan(world.root, ORGANIZE_BY_TYPE, categories={name: (".txt",)}, home=world.home)


def test_collisions_get_a_unique_suffix_at_plan_time(world):
    world.make("a.txt")
    (world.root / "Documents").mkdir()
    (world.root / "Documents" / "a.txt").write_text("old")
    (world.root / "Documents" / "a (2).txt").write_text("old")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    assert targets(p) == {"a.txt": "Documents/a (3).txt"}


def test_the_hash_is_stable_and_covers_every_field(world):
    world.make("a.txt", "b.png")
    first = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    assert plan(world.root, ORGANIZE_BY_TYPE, home=world.home).plan_hash == first.plan_hash
    (world.root / "c.txt").write_text("new")
    assert plan(world.root, ORGANIZE_BY_TYPE, home=world.home).plan_hash != first.plan_hash
    assert replace(first, intent="other").compute_hash() != first.plan_hash
    assert first.compute_hash() == first.plan_hash


def test_a_plan_round_trips_through_json_and_a_tampered_one_is_refused(world):
    world.make("a.txt")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    assert Plan.from_json(p.to_json()) == p
    data = json.loads(p.to_json())
    data["moves"][0]["destination"] = data["moves"][0]["destination"].replace("Documents", "Elsewhere")
    with pytest.raises(Refused):
        Plan.from_json(json.dumps(data))
    with pytest.raises(Refused):
        Plan.from_json("not json")


def test_the_root_must_be_a_safe_place(world, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (world.home / ".config").mkdir()
    (world.home / "Library").mkdir()
    (world.home / "Library" / "Mail").mkdir()
    (world.home / "Documents").mkdir()
    (world.home / "Documents" / ".private").mkdir()
    link = world.home / "link"
    link.symlink_to(outside)
    for bad in (
        world.home,
        outside,
        tmp_path,
        world.home / ".config",
        world.home / "Library",
        world.home / "Library" / "Mail",
        world.home / "Documents" / ".private",
        link,
        world.home / "missing",
        world.root / "x.txt",
    ):
        with pytest.raises(Refused):
            plan(bad, ORGANIZE_BY_TYPE, home=world.home)


def test_a_file_root_and_a_filesystem_root_home_are_refused(world):
    world.make("a.txt")
    with pytest.raises(Refused):
        plan(world.root / "a.txt", ORGANIZE_BY_TYPE, home=world.home)
    with pytest.raises(Refused):
        plan(world.root, ORGANIZE_BY_TYPE, home="/")


def test_preview_counts_per_destination_and_touches_nothing(world):
    world.make("a.pdf", "b.pdf", "c.png")
    before = world.listing()
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    text = preview(p)
    assert p.plan_hash in text
    assert "Documents/" in text and "2 files" in text and "Images/" in text and "1 file" in text
    assert "nothing has been moved" in text.lower()
    assert world.listing() == before  # no folder was made either


def test_a_plan_with_no_matches_previews_as_nothing_to_do(world):
    world.make(".hidden")
    p = plan(world.root, ORGANIZE_BY_TYPE, home=world.home)
    assert p.moves == ()
    assert "nothing to move" in preview(p).lower()
    assert world.listing() == [".hidden"]


def test_named_folders_with_no_match_create_nothing(world):
    world.make("a.txt")
    p = plan(world.root, NAMED_FOLDERS, folders={"Money": ["invoice"]}, home=world.home)
    assert p.moves == ()
    assert world.listing() == ["a.txt"]
