"""The wire format: golden lines shared with the Swift codec, parsing, size limits and text hygiene. No sockets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glide.app_server import wire

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "app_protocol"


def fixture(side: str, name: str) -> dict:
    return json.loads((FIXTURES / side / f"{name}.json").read_text())


def line(frame: dict) -> bytes:
    return json.dumps(frame).encode()


def as_frame(message: wire.Outgoing) -> dict:
    return json.loads(wire.encode(message))


SETTINGS = fixture("core", "settings")["data"]["settings"]

CORE_BUILT = {
    "hello": wire.hello(
        core_version="0.1.0", session_id="s1", capabilities=["text_input", "voice"], recording_content=False, reply_to="a1"
    ),
    "state": wire.state("listening", hands_free=True, muted=False),
    "state_detail": wire.state("error", hands_free=True, muted=True, detail="voice_failed"),
    "transcript_redacted": wire.transcript("u1", "user", "open the calendar", partial=False, record_content=False),
    "transcript_partial_redacted": wire.transcript("u2", "user", "open the", partial=True, record_content=False),
    "transcript_text": wire.transcript("a3", "assistant", "Sure.", partial=False, record_content=True),
    "speech_started": wire.speech("started", "a3"),
    "speech_interrupted": wire.speech("interrupted"),
    "task_started": wire.task("t1", "started"),
    "task_attempted": wire.task("t1", "attempted", step=2),
    "task_verified": wire.task("t1", "verified", step=2, verified=True),
    "task_reconcile_required": wire.task("t1", "reconcile_required"),
    "task_failed": wire.task("t1", "failed", summary="stalled"),
    "switch": wire.switch(role="tts", from_slot="alpha:m1", to_slot="beta", kind="timeout", reason="alpha timed out"),
    "switch_nothing_left": wire.switch(role="stt", from_slot="beta", to_slot=None, kind="auth", reason="key refused"),
    "approval_request": wire.approval_request(
        "p1", "screen", "Look at this Mac's screen, without clicking or typing, to: read the page title", expires_in_s=30
    ),
    "approval_closed": wire.approval_closed("p1", "denied"),
    "settings": wire.settings(3, SETTINGS, reply_to="a2"),
    "settings_result": wire.settings_result(
        False, 3, [("voice.silence_ms", "must be a whole number from 200 to 2000")], reply_to="a4"
    ),
    "error": wire.error("busy", "too many requests are being answered", reply_to="a5"),
    "ping": wire.ping(),
}


def test_every_core_fixture_is_what_the_encoder_makes():
    on_disk = sorted(p.stem for p in (FIXTURES / "core").glob("*.json"))
    assert on_disk == sorted(CORE_BUILT)  # a message without a golden line, or a line without a message, fails here
    for name, message in CORE_BUILT.items():
        assert as_frame(message) == fixture("core", name), name


@pytest.mark.parametrize("path", sorted((FIXTURES / "app").glob("*.json")), ids=lambda p: p.stem)
def test_every_app_fixture_parses(path):
    frame = json.loads(path.read_text())
    parsed = wire.parse(line(frame))
    assert isinstance(parsed, wire.Command)
    assert parsed.type == frame["type"]
    assert parsed.id == frame.get("id")


def test_the_app_fixtures_cover_every_command_the_protocol_names():
    names = {json.loads(p.read_text())["type"] for p in (FIXTURES / "app").glob("*.json")}
    assert names == {
        "hello",
        "text_input",
        "interrupt",
        "stop",
        "approval_response",
        "voice_control",
        "settings_get",
        "settings_set",
        "pong",
    }


def test_the_envelope_is_one_line_of_ascii_with_a_newline():
    raw = wire.encode(wire.transcript("u1", "user", "caf\u00e9 \u2028 \u202e", partial=False, record_content=True))
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    raw.decode("ascii")
    assert json.loads(raw)["v"] == 1


# -- content --------------------------------------------------------------------------------------


def test_a_content_free_transcript_has_a_length_and_no_words():
    said = "my password is hunter2"
    frame = as_frame(wire.transcript("u1", "user", said, partial=False, record_content=False))
    assert frame["data"] == {"utterance_id": "u1", "role": "user", "partial": False, "redacted": True, "chars": len(said)}
    assert "hunter2" not in json.dumps(frame)


def test_a_transcript_with_recording_on_carries_the_cleaned_words():
    frame = as_frame(wire.transcript("u1", "user", "hello\x00 \u202eworld", partial=True, record_content=True))
    assert frame["data"]["text"] == "hello world" and frame["data"]["redacted"] is False and "chars" not in frame["data"]


def test_only_a_partial_transcript_may_be_lost():
    assert wire.transcript("u", "user", "x", partial=True, record_content=False).droppable
    assert not wire.transcript("u", "user", "x", partial=False, record_content=False).droppable
    assert not wire.task("t", "attempted").droppable and not wire.state("idle", hands_free=False, muted=False).droppable


def test_a_task_summary_and_a_switch_reason_are_bounded_and_cleaned():
    frame = as_frame(wire.task("t\x00" + "x" * 500, "failed", summary="a\nb\x1b[31m" + "y" * 1000))
    assert len(frame["data"]["task_id"]) <= wire.MAX_ID_CHARS
    assert "\n" not in frame["data"]["summary"] and "\x1b" not in frame["data"]["summary"]
    assert len(frame["data"]["summary"]) <= wire.MAX_SUMMARY_CHARS
    reason = as_frame(wire.switch(role="r", from_slot="a", to_slot=None, kind="k", reason="z" * 5000))["data"]["reason"]
    assert len(reason) <= wire.MAX_REASON_CHARS and reason.endswith("\u2026")


def test_clean_text_removes_what_can_make_one_command_read_as_another():
    assert wire.clean_text("open\u202e Safari\u200b \u0007now\n\tplease", 100) == "open Safari now please"
    assert wire.clean_text("a\u2028b\u2029c\ud800d", 100) == "a b cd"
    assert wire.clean_text(12345, 100) == "12345"
    assert wire.clean_text("abcdef", 4) == "abc\u2026"


def test_an_approval_command_is_bounded_and_the_kind_is_one_the_protocol_names():
    data = as_frame(wire.approval_request("p", "teleport", "go " * 1000))["data"]
    assert data["kind"] == "other" and len(data["command"]) <= wire.MAX_COMMAND_CHARS
    assert "expires_in_s" not in data and "task_id" not in data


def test_a_line_that_would_exceed_the_limit_is_not_encoded():
    with pytest.raises(wire.FrameError) as caught:
        wire.encode(wire.Outgoing("settings", {"x": "a" * (wire.MAX_LINE_BYTES + 1)}))
    assert caught.value.code == "line_too_long"


# -- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[1, 2]",
        b"42",
        b'"hello"',
        b"\xff\xfe",
        b"{}",
        b'{"v":1}',
        b'{"v":1,"type":5}',
        b'{"v":"1","type":"hello"}',
        b'{"v":true,"type":"hello"}',
        b'{"v":1,"type":"hello","data":[]}',
        b'{"v":1,"type":"hello","data":{"protocol":"1"}}',
        b'{"v":1,"type":"text_input","data":{"text":5}}',
        b'{"v":1,"type":"text_input","data":{"text":"   "}}',
        b'{"v":1,"type":"approval_response","data":{"approval_id":"p1","decision":"always"}}',
        b'{"v":1,"type":"approval_response","data":{"approval_id":"","decision":"approve"}}',
        b'{"v":1,"type":"voice_control","data":{"action":"shout"}}',
        b'{"v":1,"type":"settings_set","data":{"base_revision":"1","changes":[]}}',
        b'{"v":1,"type":"settings_set","data":{"base_revision":1,"changes":[{"key":"voice.hands_free"}]}}',
        b'{"v":1,"type":"hello","id":7,"data":{"protocol":1}}',
    ],
)
def test_a_line_that_is_not_a_command_raises_and_never_echoes_itself(raw):
    with pytest.raises(wire.FrameError) as caught:
        wire.parse(raw)
    assert caught.value.code in ("malformed", "text_too_long")
    assert not caught.value.fatal


def test_the_message_of_an_error_never_contains_the_secret_in_the_line():
    secret = "hunter2-the-secret"
    with pytest.raises(wire.FrameError) as caught:
        wire.parse(line({"v": 1, "type": "text_input", "data": {"text": 5, "extra": secret}}))
    assert secret not in str(caught.value)
    with pytest.raises(wire.FrameError) as caught:
        wire.parse(secret.encode())
    assert secret not in str(caught.value)


def test_another_version_is_refused_as_fatal():
    with pytest.raises(wire.UnsupportedVersion) as caught:
        wire.parse(b'{"v":2,"type":"hello","data":{"protocol":2}}')
    assert caught.value.fatal and caught.value.code == "unsupported_version"
    with pytest.raises(wire.UnsupportedVersion):
        wire.parse(b'{"v":0,"type":"ping"}')


def test_an_unknown_type_is_not_an_error():
    parsed = wire.parse(b'{"v":1,"type":"future_thing","id":"a9","data":{"secret":"hunter2"}}')
    assert parsed == wire.Unknown("future_thing", "a9")


def test_an_unknown_type_is_cleaned_before_it_can_be_logged():
    parsed = wire.parse(json.dumps({"v": 1, "type": "x\u202e\x1b[2Jy" * 3}).encode())
    assert isinstance(parsed, wire.Unknown) and "\x1b" not in parsed.type and "\u202e" not in parsed.type


def test_unknown_fields_are_ignored_and_absent_data_is_empty():
    parsed = wire.parse(b'{"v":1,"type":"interrupt","future":true}')
    assert parsed == wire.Command("interrupt", None, {})
    stop = wire.parse(b'{"v":1,"type":"stop","id":"a1"}')
    assert stop.data == {"task_id": None}
    assert wire.parse(b'{"v":1,"type":"stop","data":{"task_id":"t9","x":1}}').data == {"task_id": "t9"}


def test_a_typed_request_is_trimmed_and_a_long_one_is_refused_not_cut():
    assert wire.parse(b'{"v":1,"type":"text_input","data":{"text":"  open   the \\n calendar "}}').data == {
        "text": "open the calendar"
    }
    with pytest.raises(wire.FrameError) as caught:
        wire.parse(line({"v": 1, "type": "text_input", "data": {"text": "x" * (wire.MAX_TEXT_CHARS + 1)}}))
    assert caught.value.code == "text_too_long"


def test_a_line_over_the_limit_is_refused_by_the_parser_too():
    with pytest.raises(wire.FrameError) as caught:
        wire.parse(b" " * (wire.MAX_LINE_BYTES + 1))
    assert caught.value.code == "line_too_long"


def test_a_hello_without_a_numeric_protocol_is_malformed_and_a_wrong_one_parses_for_the_handshake_to_refuse():
    assert wire.parse(b'{"v":1,"type":"hello","data":{"protocol":2,"client":"c"}}').data["protocol"] == 2
    with pytest.raises(wire.FrameError):
        wire.parse(b'{"v":1,"type":"hello","data":{"client":"c"}}')


def test_settings_changes_are_bounded():
    changes = [{"key": "voice.headset", "value": True}] * (wire.MAX_CHANGES + 1)
    with pytest.raises(wire.FrameError):
        wire.parse(line({"v": 1, "type": "settings_set", "data": {"base_revision": 1, "changes": changes}}))
