"""Mutation-testing follow-up for `glide/computer/diagnostics.py` (see docs/MUTATION_REPORT.md).

The scrubbing is what keeps a credential, a URL's path or the content of a task out of what a run stores. A mutation
run over it showed which edges were never pinned: the length a credential must have to be recognised and the order they
are removed in, how deep and how wide a value may be before it is cut, what an exception chain keeps, which settings
the runtime snapshot may list, what a failed write leaves behind, and what a report drops when it is too big. Every test
here fails on a mutant of one of those lines. Files go to `tmp_path`; nothing reaches the machine.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from glide.computer import diagnostics as d
from glide.providers.errors import ProviderError

# -- credentials ------------------------------------------------------------------------------------------------------


def test_an_environment_secret_is_recognised_from_six_characters():
    os.environ["MUT_ONE_TOKEN"] = "12345"
    os.environ["MUT_TWO_TOKEN"] = "123456"
    try:
        assert d.scrub_text("a 12345 b 123456 c") == f"a 12345 b {d.REDACTED} c"
    finally:
        del os.environ["MUT_ONE_TOKEN"], os.environ["MUT_TWO_TOKEN"]


def test_a_longer_secret_is_removed_before_a_shorter_one_it_contains():
    os.environ["MUT_SHORT_TOKEN"] = "abcdef"
    os.environ["MUT_LONG_TOKEN"] = "abcdefghij"
    try:
        assert d.scrub_text("key abcdefghij end") == f"key {d.REDACTED} end"
    finally:
        del os.environ["MUT_SHORT_TOKEN"], os.environ["MUT_LONG_TOKEN"]


def test_a_url_that_cannot_be_parsed_is_hidden_and_never_raises():
    assert d.scrub_text("see http://[::1 now") == f"see {d.REDACTED} now"
    assert d.redact("see http://[::1 now") == f"see {d.REDACTED} now"


def test_text_is_cut_after_the_limit_and_not_at_it():
    assert d.scrub_text("a" * 10, limit=10) == "a" * 10
    assert d.scrub_text("a" * 11, limit=10) == "a" * 10 + "..."


def test_scrubbed_text_is_one_line_and_urls_keep_only_their_host_unless_asked_otherwise():
    assert d.scrub_text("open  https://host.example/a/b?x=1\nnow") == "open https://host.example now"
    # a URL with credentials: whatever is shown for it, the user and the password are not part of it
    hidden = d.scrub_text("open https://user:pw@host.example/a/b?token=1 now")
    assert isinstance(hidden, str) and "pw" not in hidden and "user" not in hidden and "token=1" not in hidden
    kept = d.scrub_text("open https://user:pw@host.example/a?token=1&q=2", urls=False)
    assert kept == f"open https://{d.REDACTED}@host.example/a?token={d.REDACTED}"  # user, password and the credential parameter


def nest(levels: int, leaf: object, *, as_dict: bool):
    value = leaf
    for _ in range(levels):
        value = {"k": value} if as_dict else [value]
    return value


def depth_of_marker(value) -> int | None:
    depth = 0
    while True:
        if value == "[nested value omitted]":
            return depth
        if isinstance(value, dict):
            value = value["k"]
        elif isinstance(value, list) and value:
            value = value[0]
        else:
            return None
        depth += 1


@pytest.mark.parametrize("as_dict", [True, False])
def test_redact_follows_structures_thirty_levels_deep_and_not_one_more(as_dict):
    assert depth_of_marker(d.redact(nest(30, "x", as_dict=as_dict))) is None
    assert depth_of_marker(d.redact(nest(31, "x", as_dict=as_dict))) == 31
    assert depth_of_marker(d.redact(nest(16, "x", as_dict=as_dict))) is None  # depth is counted one level at a time


def test_redact_removes_a_credential_wherever_it_hides_in_a_structure():
    os.environ["MUT_NESTED_SECRET"] = "hunter2hunter2"
    try:
        value = {
            "outer": [{"deep": ("see hunter2hunter2", 3)}],
            "api_key": "sk-visible-nowhere",
            "hunter2hunter2": "named by the secret",
            "number": 7,
            None: "none key",
        }
        out = d.redact(value)
    finally:
        del os.environ["MUT_NESTED_SECRET"]
    assert out["outer"] == [{"deep": [f"see {d.REDACTED}", 3]}]
    assert out["api_key"] == d.REDACTED and out["number"] == 7 and out["None"] == "none key"
    assert d.REDACTED in out and "hunter2hunter2" not in json.dumps(out)


def test_bounded_values_are_cut_at_the_documented_depth_width_and_length():
    assert depth_of_marker(d._bounded(nest(d.MAX_DEPTH, "x", as_dict=False), content=False)) is None
    assert depth_of_marker(d._bounded(nest(d.MAX_DEPTH + 1, "x", as_dict=False), content=False)) == d.MAX_DEPTH + 1
    assert depth_of_marker(d._bounded(nest(d.MAX_DEPTH + 1, "x", as_dict=True), content=False)) == d.MAX_DEPTH + 1

    fields = {f"field{i}": i for i in range(d.MAX_ITEMS)}
    assert d._bounded(fields, content=False) == fields
    one_more = d._bounded({**fields, "extra": 1}, content=False)
    assert one_more["_omitted_fields"] == 1 and len(one_more) == d.MAX_ITEMS + 1 and "extra" not in one_more
    many = d._bounded({f"f{i}": i for i in range(d.MAX_ITEMS + 5)}, content=False)
    assert many["_omitted_fields"] == 5 and len(many) == d.MAX_ITEMS + 1

    items = list(range(d.MAX_ITEMS))
    assert d._bounded(items, content=False) == items
    assert d._bounded([*items, 99], content=False) == [*items, {"_omitted_items": 1}]
    assert d._bounded(list(range(d.MAX_ITEMS + 7)), content=False)[-1] == {"_omitted_items": 7}
    long_key = d._bounded({"k" * 100: 1}, content=False)
    assert list(long_key) == ["k" * 64 + "..."]


def test_without_content_recording_content_fields_and_url_paths_are_never_kept():
    out = d._bounded(
        {"goal": "buy shoes", "page_text": "secret page", "ok": "visit https://host.example/private/path?x=1", "n": 3},
        content=False,
    )
    assert out["goal"] == out["page_text"] == d.CONTENT_OMITTED
    assert out["ok"] == "visit https://host.example" and out["n"] == 3
    assert d._bounded({"api_key": "sk-1"}, content=False) == {"api_key": d.REDACTED}


def test_an_object_is_never_shown_by_its_repr_and_a_path_is_scrubbed_like_text():
    class Client:
        def __repr__(self):
            return "Client(api_key='sk-leaked-in-a-repr')"

    assert d._bounded(Client(), content=False) == "<Client>"
    assert d._bounded({"client": Client()}, content=True) == {"client": "<Client>"}
    assert d._bounded(Path("/tmp/with  spaces"), content=False) == "/tmp/with spaces"
    assert (
        d._bounded(None, content=False) is None
        and d._bounded(True, content=False) is True
        and d._bounded(2.5, content=False) == 2.5
    )


def test_an_exception_keeps_its_type_always_and_its_message_only_with_content_recording():
    error = ValueError("secret detail https://host.example/private")
    assert d._bounded(error, content=False) == {"type": "ValueError", "message": "ValueError"}
    assert d._bounded(error, content=True) == {"type": "ValueError", "message": "secret detail https://host.example/private"}

    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no message")

    assert d._message(Unprintable()) == "message could not be formatted"


# -- exception chains --------------------------------------------------------------------------------------------------


def raised(depth: int):
    if depth:
        return raised(depth - 1)
    raise ValueError("deep")


def test_an_exception_chain_keeps_the_last_thirty_frames_and_says_how_many_it_dropped():
    try:
        raised(40)
    except ValueError as error:
        (item,) = d._chain(error)
    assert len(item["frames"]) == d.MAX_FRAMES and item["omitted_frames"] > 0
    assert item["frames"][-1]["function"] == "raised"
    try:
        raised(3)
    except ValueError as error:
        (short,) = d._chain(error)
    assert "omitted_frames" not in short and len(short["frames"]) == 5


def test_an_exception_chain_stops_at_eight_and_at_a_cycle():
    first = last = ValueError("0")
    for index in range(1, 12):
        error = ValueError(str(index))
        last.__cause__ = error
        last = error
    assert len(d._chain(first)) == d.MAX_EXCEPTIONS
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert [item["message"] for item in d._chain(a)] == ["a", "b"]


def test_an_implicit_context_is_followed_unless_the_raise_suppressed_it():
    try:
        try:
            raise KeyError("first")
        except KeyError:
            raise ValueError("second") from None
    except ValueError as error:
        assert [i["type"] for i in d._chain(error)] == ["ValueError"]
    try:
        try:
            raise KeyError("first")
        except KeyError:
            raise ValueError("second")  # noqa: B904  (the implicit context is what is under test)
    except ValueError as error:
        assert [i["type"] for i in d._chain(error)] == ["ValueError", "KeyError"]


def test_the_codes_an_error_carries_are_kept_but_a_bool_is_not_one_and_a_string_is_cut():
    error = ProviderError("x", kind="server", status=502)
    item = d._chain(error)[0]
    assert (item["kind"], item["status"]) == ("server", 502)
    odd = ValueError("x")
    odd.code, odd.errno, odd.request_id = True, 5, "r" * 200  # type: ignore[attr-defined]
    item = d._chain(odd)[0]
    assert "code" not in item and item["errno"] == 5 and item["request_id"] == "r" * 128 + "..."


# -- the runtime snapshot ------------------------------------------------------------------------------------------------


def test_the_runtime_lists_the_names_of_glide_settings_never_a_value_nor_another_variable(monkeypatch):
    # A value cannot be known to be harmless (glide.toml lets any capitals-only name hold a provider key), so only the
    # NAMES of GLIDE_* variables are reported: a key under a harmless-looking name cannot leak through this snapshot.
    monkeypatch.setenv("GLIDE_MODE", "quiet-value")
    monkeypatch.setenv("GLIDE_ACME", "sk-live-0123456789abcdef")
    monkeypatch.setenv("GLIDE_API_KEY", "should-not-be-listed")
    monkeypatch.setenv("UNRELATED_SETTING", "nope")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-0123456789")
    monkeypatch.setattr("sys.argv", ["/some/where/glide", "--flag"])
    runtime = d._runtime()
    assert runtime["entrypoint"] == "glide"
    settings = runtime["settings"]
    assert isinstance(settings, list) and settings == sorted(settings)
    assert {"GLIDE_MODE", "GLIDE_ACME", "GLIDE_API_KEY"} <= set(settings)
    assert all(name.startswith("GLIDE_") for name in settings)
    rendered = json.dumps(runtime)
    for value in ("quiet-value", "sk-live-0123456789abcdef", "should-not-be-listed", "nope", "sk-other-0123456789"):
        assert value not in rendered
    assert "UNRELATED_SETTING" not in rendered and "OPENAI_API_KEY" not in rendered
    assert "httpx" in runtime["packages"] and runtime["python"] and runtime["platform"]
    monkeypatch.setattr("sys.argv", [""])
    assert d._runtime()["entrypoint"] == ""
    monkeypatch.setattr("sys.argv", [])
    assert d._runtime()["entrypoint"] == ""


# -- files -------------------------------------------------------------------------------------------------------------


def test_json_is_written_as_utf8_with_indent_and_refuses_nan():
    data = d._json({"a": "café"})
    assert data == b'{\n  "a": "caf\xc3\xa9"\n}\n'
    with pytest.raises(ValueError):
        d._json({"a": float("nan")})


def test_a_private_file_is_flushed_synced_0600_and_leaves_no_temporary_behind(tmp_path, monkeypatch):
    seen = []
    real_fsync = os.fsync

    def fsync(fd):
        seen.append(os.fstat(fd).st_size)  # what has reached the file when it is synced
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    target = tmp_path / "run.json"
    d.write_private(target, "hello")
    assert target.read_bytes() == b"hello" and stat.S_IMODE(target.stat().st_mode) == 0o600
    assert seen == [5]
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_a_failed_private_write_leaves_neither_the_target_nor_a_temporary(tmp_path, monkeypatch):
    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        d.write_private(tmp_path / "run.json", "hello")
    assert list(tmp_path.iterdir()) == []


def recorder(tmp_path, **kw):
    return d.Diagnostics(tmp_path / "diag", record_content=True, task_id="task-1", **kw)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_an_event_is_written_whole_utf8_and_0600_and_its_name_is_bounded(tmp_path):
    rec = recorder(tmp_path)
    rec.event("n" * 300, note="café", value=3)
    path = rec.root / "events.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and "café" in path.read_text(encoding="utf-8")
    (item,) = lines(path)
    assert item["event"] == "n" * 256 + "..." and item["details"] == {"note": "café", "value": 3}


def test_an_event_over_the_size_limit_is_stored_as_its_field_names_only(tmp_path):
    rec = recorder(tmp_path)
    rec.event("big", **{f"f{i}": "x" * 1000 for i in range(30)}, tail="café")
    (item,) = lines(rec.root / "events.jsonl")
    assert item["details"]["truncated"] is True and len(item["details"]["fields"]) == 20


def test_the_events_file_rotates_once_it_would_pass_its_limit_and_not_before(tmp_path, monkeypatch):
    first = recorder(tmp_path)
    first.event("same", a=1)
    size = (first.root / "events.jsonl").stat().st_size
    monkeypatch.setattr(d, "MAX_EVENTS_BYTES", 2 * size)  # the second line makes the file exactly full: no rotation
    first.event("same", a=1)
    assert not (first.root / "events.previous.jsonl").exists() and len(lines(first.root / "events.jsonl")) == 2
    first.event("same", a=1)  # the third would pass the limit
    assert (first.root / "events.previous.jsonl").exists() and len(lines(first.root / "events.jsonl")) == 1


def test_a_failed_write_is_recorded_by_its_type_and_never_stops_the_run(tmp_path, monkeypatch):
    rec = recorder(tmp_path)

    def broken(path, data):
        raise PermissionError("denied")

    monkeypatch.setattr(d, "_append", broken)
    rec.event("x", a=1)
    assert rec.write_error == "PermissionError" and len(rec.events) == 1


def test_an_exception_is_recorded_as_an_event_with_a_bounded_stage(tmp_path):
    rec = recorder(tmp_path)
    rec.exception(ValueError("boom"), stage="s" * 100)
    assert rec.error_stage == "s" * 64 + "..."
    assert rec.events[-1]["event"] == "exception" and rec.events[-1]["details"]["exception"]["type"] == "ValueError"
    assert rec.error["message"] == "boom"


def finished(tmp_path, summary, *, error=None, events=0, **kw):
    rec = recorder(tmp_path, **kw)
    for index in range(events):
        rec.event("step", index=index, detail="x" * 200)
    rec.finish(summary, error=error)
    return rec, json.loads((rec.root / "diagnostic.json").read_text(encoding="utf-8"))


def test_the_report_keeps_the_summary_content_it_was_given_with_recording_on(tmp_path):
    rec, report = finished(tmp_path, {"outcome": "done", "goal": "open the report"})
    assert report["schema"] == 1 and report["summary"]["goal"] == "open the report"
    assert report["task_id"] == "task-1" and report["artifact_root"] == str(rec.requested_root)
    assert report["diagnostic_write_error"] == ""
    assert report["runtime"]["python"]


def test_a_failed_runs_report_is_markdown_with_the_stage_and_the_task(tmp_path):
    rec = recorder(tmp_path)
    rec.exception(ValueError("boom"), stage="selection")
    rec.finish({"outcome": "blocked"}, error=ValueError("boom"))
    text = (rec.root / "failure.md").read_text(encoding="utf-8")
    assert "- Stage: `selection`" in text and "- Task: `task-1`" in text and "- Outcome: `blocked`" in text
    rec2 = d.Diagnostics(tmp_path / "other", record_content=True)
    rec2.finish({"outcome": "blocked"})
    text = (rec2.root / "failure.md").read_text(encoding="utf-8")
    assert "- Stage: `not recorded`" in text and "- Task: `not assigned`" in text
    rec3 = d.Diagnostics(tmp_path / "third", record_content=True)
    rec3.finish({"outcome": "blocked", "failure_stage": "readiness"})
    assert "- Stage: `readiness`" in (rec3.root / "failure.md").read_text(encoding="utf-8")


def test_a_report_too_big_loses_its_oldest_events_first_and_counts_them(tmp_path, monkeypatch):
    _, full = finished(tmp_path / "full", {"outcome": "done"}, events=30)
    monkeypatch.setattr(d, "MAX_REPORT_BYTES", len(d._json(full)) - 3000)
    _, report = finished(tmp_path / "small", {"outcome": "done"}, events=30)
    kept, omitted = report["recent_events"], report["events_omitted_for_size"]
    assert omitted >= 1 and len(kept) + omitted == len(full["recent_events"])
    assert kept[-1]["details"]["index"] == 29  # the newest survive
    assert len(d._json(report)) <= len(d._json(full)) - 3000


def test_a_summary_too_big_for_the_report_keeps_only_the_core_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "MAX_REPORT_BYTES", 4000)
    summary = {
        "outcome": "blocked",
        "failure_stage": "selection",
        "failure_code": "x" * 300,
        "error_type": "ValueError",
        "goal": "g" * 9000,
    }
    error = ValueError("boom")
    try:
        raised(3)
    except ValueError as caught:
        error = caught
    _, report = finished(tmp_path, summary, error=error)
    assert report["content_omitted_for_size"] is True and report["runtime"] == {}
    assert set(report["summary"]) == {"outcome", "failure_stage", "failure_code", "error_type"}
    assert report["summary"]["failure_code"] == "x" * 256 + "..."
    assert report["exception"] and all(set(e) == {"type", "frames"} and len(e["frames"]) == 1 for e in report["exception"])

    _, nothing = finished(tmp_path / "no-exception", summary)
    assert nothing["content_omitted_for_size"] is True and nothing["exception"] is None


def test_nothing_is_stored_without_content_recording_but_the_events_stay_in_memory(tmp_path):
    rec = d.Diagnostics(tmp_path / "off", record_content=False)
    rec.event("step", goal="buy shoes")
    assert rec.finish({"outcome": "done"}) is None
    assert not (tmp_path / "off").exists()
    assert rec.events[-1]["details"] == {"goal": d.CONTENT_OMITTED}


def test_a_report_that_cannot_be_written_is_recorded_by_its_error_type_and_returns_nothing(tmp_path, monkeypatch):
    rec = recorder(tmp_path)

    def broken(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(d, "write_private", broken)
    assert rec.finish({"outcome": "done"}) is None
    assert rec.write_error == "OSError" and rec.report_path is None


def test_an_event_with_a_value_json_cannot_hold_is_not_written_as_invalid_json(tmp_path):
    rec = recorder(tmp_path)
    rec.event("step", ratio=float("nan"))
    assert rec.write_error == "ValueError"
    path = rec.root / "events.jsonl"
    assert not path.exists() or "NaN" not in path.read_text(encoding="utf-8")
    assert len(rec.events) == 1  # it is still in memory


def test_the_truncated_form_of_an_oversize_event_keeps_field_names_as_text(tmp_path):
    rec = recorder(tmp_path)
    rec.event("big", **{"caf\u00e9": "x" * 4000}, **{f"f{i}": "y" * 1000 for i in range(30)})
    assert "caf\u00e9" in (rec.root / "events.jsonl").read_text(encoding="utf-8")


def test_a_timestamp_is_utc_with_microseconds():
    stamp = d._now()
    assert stamp.endswith("+00:00") and len(stamp.split(".")[1]) == len("123456+00:00")


def test_exactly_thirty_frames_are_all_kept_and_one_more_is_counted_as_omitted():
    def chain_with(frames):
        for depth in range(60):
            try:
                raised(depth)
            except ValueError as error:
                item = d._chain(error)[0]
                if len(item["frames"]) + item.get("omitted_frames", 0) == frames:
                    return item
        raise AssertionError("no depth gives this many frames")

    exact = chain_with(d.MAX_FRAMES)
    assert len(exact["frames"]) == d.MAX_FRAMES and "omitted_frames" not in exact
    over = chain_with(d.MAX_FRAMES + 1)
    assert len(over["frames"]) == d.MAX_FRAMES and over["omitted_frames"] == 1
