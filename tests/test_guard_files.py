"""Files: a test opens a file for writing only under the temp directory (or a tool's cache directory, or /dev/null).

`open(..., 'w'/'a'/'x'/'+')` and `Path.write_text`/`write_bytes` outside it refuse before anything is created, so a test
that writes into the repository, the person's home or the working directory fails and says where.

Not covered, and said so in `conftest._no_writes_outside_temp`: `os.open`, `sqlite3`, `os.rename` and friends, and
`io.FileIO` used directly. Reading is never refused.
"""

# ruff: noqa: SIM115, UP020  (the refused `open(...)` and `io.open(...)` are the subject, not resources to manage)
from __future__ import annotations

import builtins
import io
import os
import shutil
from pathlib import Path

import conftest
import pytest

pytestmark = pytest.mark.expect_refusals

REPO = Path(__file__).resolve().parents[1]
OUTSIDE = REPO / "tests" / "guard_probe_do_not_create.txt"
PROBES = (OUTSIDE, REPO / "guard_probe_relative.txt", REPO / "tests" / "guard_probe_through_link.txt")


@pytest.fixture(autouse=True)
def the_guard_is_live_and_no_probe_file_survives():
    """If the guard were broken these tests would write into the repository: check it is there first, clean up after."""
    assert builtins.open is not conftest.REAL_OPEN, "the write guard is not installed: these tests would write to the repo"
    yield
    for probe in PROBES:
        probe.unlink(missing_ok=True)


@pytest.mark.parametrize("mode", ["w", "a", "x", "wb", "ab", "xb", "r+", "w+", "a+", "rb+", "wt"])
def test_open_for_writing_outside_the_temp_directory_refuses(mode):
    with pytest.raises(RuntimeError, match="for writing"):
        open(OUTSIDE, mode)
    assert not OUTSIDE.exists()


def test_the_mode_may_be_a_keyword_and_the_path_a_string_or_bytes():
    with pytest.raises(RuntimeError, match="for writing"):
        open(str(OUTSIDE), mode="w")
    with pytest.raises(RuntimeError, match="for writing"):
        open(os.fsencode(OUTSIDE), "wb")
    with pytest.raises(RuntimeError, match="for writing"):
        io.open(OUTSIDE, "w")  # `io.open` is `open`: both are refused
    assert not OUTSIDE.exists()


def test_path_write_helpers_and_path_open_refuse():
    with pytest.raises(RuntimeError, match="for writing"):
        OUTSIDE.write_text("x")
    with pytest.raises(RuntimeError, match="for writing"):
        OUTSIDE.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="for writing"):
        OUTSIDE.open("w")
    with pytest.raises(RuntimeError, match="for writing"):
        OUTSIDE.open(mode="a")
    assert not OUTSIDE.exists()


def test_a_relative_path_under_the_working_directory_is_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    with pytest.raises(RuntimeError, match="for writing"):
        open("guard_probe_relative.txt", "w")
    assert not (REPO / "guard_probe_relative.txt").exists()
    monkeypatch.chdir(tmp_path)
    Path("inside.txt").write_text("fine")  # the same relative spelling is fine under a temp working directory
    assert (tmp_path / "inside.txt").read_text() == "fine"


def test_a_copy_into_the_repository_refuses():
    source = Path(__file__)
    with pytest.raises(RuntimeError, match="for writing"):
        shutil.copyfile(source, OUTSIDE)
    assert not OUTSIDE.exists()


def test_a_path_that_climbs_out_of_the_temp_directory_is_outside(tmp_path):
    with pytest.raises(RuntimeError, match="for writing"):
        open(tmp_path.joinpath(*[".."] * 12, "guard_probe_escape.txt"), "w")  # resolves to the filesystem root


def test_a_symlink_out_of_the_temp_directory_is_followed_to_where_it_lands(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(REPO / "tests")
    with pytest.raises(RuntimeError, match="for writing"):
        (link / "guard_probe_through_link.txt").write_text("x")
    assert not (REPO / "tests" / "guard_probe_through_link.txt").exists()


def test_writing_under_the_temp_directory_works(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    with open(tmp_path / "c.txt", "a") as handle:
        handle.write("appended")
    with open(tmp_path / "c.txt", "r+") as handle:
        assert handle.read() == "appended"
    assert (tmp_path / "a.txt").read_text() == "text"
    sub = tmp_path / "deep" / "er"
    sub.mkdir(parents=True)
    (sub / "d.txt").write_text("nested")


def test_reading_is_never_refused_and_neither_is_devnull_or_a_descriptor(tmp_path):
    assert Path(__file__).read_text(encoding="utf-8")
    with open(__file__, "rb") as handle:
        assert handle.read(1)
    with open(os.devnull, "w") as sink:
        sink.write("discarded")
    fd = os.open(tmp_path / "fd.txt", os.O_WRONLY | os.O_CREAT)
    with open(fd, "w") as handle:  # a descriptor the caller already holds
        handle.write("x")


def test_the_tools_cache_directories_inside_the_repository_are_writable_places():
    for name in conftest.WRITE_CACHE_DIRS:
        assert conftest.writable_place(REPO / name / "probe")
    assert conftest.writable_place(REPO / "glide" / "__pycache__" / "x.pyc")
    assert not conftest.writable_place(REPO / "glide" / "x.py")
    assert not conftest.writable_place(Path.home() / ".glide-guard-probe")
    assert not conftest.writable_place(Path("/etc/hosts"))
    assert not conftest.writable_place(None)


def test_one_replacement_serves_every_name_the_function_is_known_by():
    assert builtins.open is io.open
    assert builtins.open is not conftest.REAL_OPEN
    assert Path.write_text.__module__.startswith("pathlib")
