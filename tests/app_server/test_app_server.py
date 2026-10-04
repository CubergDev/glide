"""The protocol server over in-process connections: handshake, version, framing, limits, approvals, slow and idle clients.

Every connection is a `socketpair`, so no name is bound. The one test that binds a real socket is in test_app_transport.py.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import threading
import time

from app_harness import WAIT, RecordingBackend, connect, make_server, wait_until

from glide.app_server import wire
from glide.app_server.server import AppServer, Limits, Session
from glide.computer.control import RunControl, controlled

# -- the handshake --------------------------------------------------------------------------------------


def test_the_core_says_nothing_before_hello_and_answers_it_with_what_it_speaks():
    backend = RecordingBackend(record_content=True)
    server = make_server(backend)
    client, session = connect(server, handshake=False)
    assert client.read(0.2) is None  # nothing is sent to a client that has not said hello
    assert server.emit(wire.state("idle", hands_free=False, muted=False)) is False  # and no event is queued for it
    client.hello(ident="a1")
    frame = client.read()
    assert frame["type"] == "hello" and frame["v"] == 1 and frame["reply_to"] == "a1"
    assert frame["data"]["protocol"] == 1 and frame["data"]["capabilities"] == ["text_input"]
    assert frame["data"]["recording_content"] is True and frame["data"]["core_version"] == "test"
    assert wait_until(lambda: backend.ready == [session])  # the backend sends its first state only after hello is queued
    server.stop()


def test_a_hello_for_another_protocol_is_a_fatal_error_and_a_close():
    server = make_server()
    client, _ = connect(server, handshake=False)
    client.hello(protocol=2, ident="a7")
    frame = client.read()
    assert frame["type"] == "error" and frame["data"]["fatal"] is True and frame["reply_to"] == "a7"
    assert frame["data"]["code"] == "unsupported_protocol"
    assert client.is_closed()
    server.stop()


def test_an_envelope_with_another_version_is_a_fatal_error_and_a_close_at_any_time():
    server = make_server()
    client, _ = connect(server)
    client.raw(b'{"v":2,"type":"interrupt"}\n')
    frame = client.read_type("error")
    assert frame["data"]["fatal"] is True and frame["data"]["code"] == "unsupported_version"
    assert client.is_closed()
    server.stop()


def test_anything_but_hello_first_is_refused():
    backend = RecordingBackend()
    server = make_server(backend)
    client, _ = connect(server, handshake=False)
    client.send({"v": 1, "type": "text_input", "data": {"text": "hi"}})
    frame = client.read()
    assert frame["data"]["code"] == "expected_hello" and frame["data"]["fatal"] is True
    assert client.is_closed() and backend.commands == []
    server.stop()


def test_no_hello_in_time_closes_the_connection():
    server = make_server(limits=Limits(poll_s=0.02, handshake_s=0.2))
    client, _ = connect(server, handshake=False)
    assert client.is_closed()
    server.stop()


def test_a_second_client_is_refused_while_one_is_connected_and_admitted_after():
    server = make_server()
    first, _ = connect(server)
    second, session = connect(server, handshake=False)
    assert session is None and second.is_closed()  # closed without a word: the app retries
    assert second.seen == []
    first.close()
    assert wait_until(lambda: server.current is None)
    third, _ = connect(server)  # the handshake inside connect() proves it is served
    third.close()
    server.stop()


def test_a_peer_that_is_another_user_or_unknown_is_refused_before_anything_is_read():
    server = make_server(peer_ok=lambda sock: False)
    client, session = connect(server, handshake=False)
    assert session is None and client.is_closed()
    server.stop()


# -- framing --------------------------------------------------------------------------------------------


def test_malformed_lines_are_dropped_one_error_each_and_never_echoed():
    backend = RecordingBackend()
    server = make_server(backend)
    client, _ = connect(server)
    secret = "hunter2-the-secret"
    for raw in (f"{secret}\n", f'["{secret}"]\n', f'{{"v":1,"type":"text_input","data":{{"text":5,"x":"{secret}"}}}}\n'):
        client.raw(raw.encode())
        frame = client.read_type("error")
        assert frame["data"]["code"] == "malformed" and frame["data"]["fatal"] is False
        assert secret not in str(frame)
    client.send({"v": 1, "type": "text_input", "id": "a2", "data": {"text": "still here"}})  # the connection stayed up
    assert wait_until(lambda: [c.data["text"] for c in backend.commands] == ["still here"])
    server.stop()


def test_too_many_unusable_lines_drop_the_connection():
    server = make_server(limits=Limits(poll_s=0.02, bad_lines=3))
    client, _ = connect(server)
    with contextlib.suppress(OSError):  # the core may close the connection before the last line is sent: that is the point
        for _ in range(6):
            client.raw(b"nonsense\n")
    assert client.is_closed()
    server.stop()


def test_an_oversized_line_is_dropped_up_to_its_newline_reported_once_and_the_connection_stays():
    backend = RecordingBackend()
    server = make_server(backend, limits=Limits(poll_s=0.02, max_line_bytes=1000))
    client, _ = connect(server)
    client.raw(b'{"v":1,"type":"text_input","data":{"text":"' + b"x" * 5000 + b'"}}\n')
    frame = client.read_type("error")
    assert frame["data"]["code"] == "line_too_long" and frame["data"]["fatal"] is False
    assert "xxxx" not in str(frame)
    client.send({"v": 1, "type": "interrupt"})
    assert wait_until(lambda: [c.type for c in backend.commands] == ["interrupt"])  # the next line is read normally
    assert backend.commands[0].type == "interrupt" and len(backend.commands) == 1
    server.stop()


def test_a_line_split_across_reads_and_two_lines_in_one_read_both_work():
    backend = RecordingBackend()
    server = make_server(backend)
    client, _ = connect(server)
    client.raw(b'{"v":1,"type":"inter')
    time.sleep(0.05)
    client.raw(b'rupt"}\r\n{"v":1,"type":"stop"}\n\n')
    assert wait_until(lambda: [c.type for c in backend.commands] == ["interrupt", "stop"])
    server.stop()


def test_an_unknown_message_is_ignored_and_logged_without_its_content(caplog):
    backend = RecordingBackend()
    server = make_server(backend)
    client, _ = connect(server)
    with caplog.at_level(logging.INFO, logger="glide.app_server"):
        client.raw(b'{"v":1,"type":"future_thing","id":"a9","data":{"words":"hunter2-the-secret"}}\n')
        client.send({"v": 1, "type": "interrupt"})
        assert wait_until(lambda: [c.type for c in backend.commands] == ["interrupt"])
    assert backend.commands[0].type == "interrupt"
    assert client.read(0.2) is None  # no error came back
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "future_thing" in text and "hunter2" not in text and "words" not in text
    server.stop()


def test_a_backend_that_raises_does_not_end_the_connection_and_the_client_is_told_the_type_only():
    class Broken(RecordingBackend):
        def handle(self, session, command):
            raise ValueError("hunter2-the-secret")

    server = make_server(Broken())
    client, _ = connect(server)
    client.send({"v": 1, "type": "interrupt", "id": "a3"})
    frame = client.read_type("error")
    assert frame["data"]["code"] == "internal" and frame["reply_to"] == "a3" and "hunter2" not in str(frame)
    client.send({"v": 1, "type": "pong"})
    assert not client.is_closed(0.2)
    server.stop()


# -- slow and idle clients ------------------------------------------------------------------------------


def test_a_client_that_never_reads_is_dropped_and_the_core_never_waits_for_it():
    server = make_server(limits=Limits(poll_s=0.02, queue_size=8, send_timeout_s=0.3))
    client, session = connect(server)
    big = wire.Outgoing("settings", {"filler": "a" * 200_000})
    started = time.monotonic()
    for _ in range(100):
        session.send(big)
    assert time.monotonic() - started < 1.0  # every send returned at once
    assert wait_until(lambda: session.closed)
    assert server.current is None
    client.close()
    server.stop()


def queue_only_session(**limits):
    """A session whose threads are not running: what is sent stays queued, so the queue's rules can be seen exactly."""
    server = make_server(limits=Limits(**limits))
    a, b = socket.socketpair()
    return Session(server, b), a


def test_a_message_that_may_be_lost_is_skipped_when_the_queue_is_filling_and_the_client_stays():
    session, peer = queue_only_session(queue_size=16, droppable_queue=2)
    partial = wire.transcript("u1", "user", "x", partial=True, record_content=False)
    assert [session.send(partial) for _ in range(6)] == [True, True, False, False, False, False]
    assert session.dropped == 4 and not session.closed  # losing a partial transcript is not a reason to drop anyone
    peer.close()


def test_a_message_that_must_not_be_lost_is_queued_past_the_droppable_mark_and_a_full_queue_drops_the_client():
    session, peer = queue_only_session(queue_size=16, droppable_queue=2)
    state = wire.state("idle", hands_free=False, muted=False)
    assert all(session.send(state) for _ in range(16))
    assert session.dropped == 0 and not session.closed
    assert session.send(state) is False and session.closed  # one more than fits: this client cannot keep up
    peer.close()


def test_a_quiet_client_is_pinged_and_one_that_answers_stays():
    server = make_server(limits=Limits(poll_s=0.02, ping_s=0.1, idle_s=0.5))
    client, session = connect(server)
    end = time.monotonic() + 1.2
    pings = 0
    while time.monotonic() < end:
        frame = client.read(0.2)
        if frame is not None and frame["type"] == "ping":
            pings += 1
            client.send({"v": 1, "type": "pong"})
    assert pings >= 3 and not session.closed
    client.close()
    server.stop()


def test_a_client_that_says_nothing_is_dropped_as_idle():
    server = make_server(limits=Limits(poll_s=0.02, ping_s=0.1, idle_s=0.4))
    client, session = connect(server)
    assert client.is_closed()
    assert wait_until(lambda: session.closed)
    server.stop()


def test_stopping_the_server_closes_the_connection():
    server = make_server()
    client, session = connect(server)
    server.stop()
    assert client.is_closed() and session.closed


# -- approvals ------------------------------------------------------------------------------------------


def ask_in_thread(server, *args, control=None, **kwargs):
    """`approvals.ask` on its own thread, as the assistant's request thread calls it. Returns (result list, thread)."""
    result: list[bool] = []

    def work():
        if control is None:
            result.append(server.approvals.ask(*args, **kwargs))
        else:
            with controlled(control):
                result.append(server.approvals.ask(*args, **kwargs))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    return result, thread


def test_an_approval_asks_the_app_for_exactly_the_command_and_an_approve_grants_it_once():
    server = make_server()
    client, _ = connect(server)
    result, thread = ask_in_thread(server, "input", "Control this Mac to: open Notes", task_id=None)
    request = client.read_type("approval_request")
    data = request["data"]
    assert data["kind"] == "input" and data["command"] == "Control this Mac to: open Notes"
    assert data["expires_in_s"] == 2.0 and data["approval_id"].startswith("p")
    assert server.approvals.waiting == 1
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": data["approval_id"], "decision": "approve"}})
    thread.join(WAIT)
    assert result == [True]
    closed = client.read_type("approval_closed")
    assert closed["data"] == {"approval_id": data["approval_id"], "outcome": "approved"}
    # the same answer again changes nothing, and does not approve a new request
    result2, thread2 = ask_in_thread(server, "input", "Control this Mac to: open Mail")
    second = client.read_type("approval_request")["data"]
    assert second["approval_id"] != data["approval_id"]
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": data["approval_id"], "decision": "approve"}})
    assert not wait_until(lambda: bool(result2), 0.3)  # still waiting: the old answer is not this request's
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": second["approval_id"], "decision": "deny"}})
    thread2.join(WAIT)
    assert result2 == [False]
    server.stop()


def test_a_deny_is_no():
    server = make_server()
    client, _ = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": "deny"}})
    thread.join(WAIT)
    assert result == [False]
    assert client.read_type("approval_closed")["data"]["outcome"] == "denied"
    server.stop()


def test_silence_is_no_and_the_late_answer_is_ignored():
    server = make_server(approval_timeout_s=0.2)
    client, _ = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    thread.join(WAIT)
    assert result == [False]
    assert client.read_type("approval_closed")["data"]["outcome"] == "expired"
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": "approve"}})
    time.sleep(0.1)
    assert server.approvals.waiting == 0
    server.stop()


def test_a_disconnect_while_an_approval_is_open_is_no_at_once():
    server = make_server(approval_timeout_s=30)
    client, _ = connect(server)
    result, thread = ask_in_thread(server, "input", "Control this Mac to: open Notes")
    client.read_type("approval_request")
    started = time.monotonic()
    client.close()
    thread.join(WAIT)
    assert result == [False] and time.monotonic() - started < 2.0  # not after the 30 s
    server.stop()


def test_an_answer_for_an_unknown_id_is_ignored_and_a_stray_approve_never_grants_the_next_request():
    server = make_server()
    client, _ = connect(server)
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": "p-not-asked-yet", "decision": "approve"}})
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    assert ident != "p-not-asked-yet"
    assert not wait_until(lambda: bool(result), 0.3)
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": "deny"}})
    thread.join(WAIT)
    assert result == [False]
    server.stop()


def test_an_approval_belongs_to_the_connection_it_was_asked_of():
    server = make_server(approval_timeout_s=1.0)
    first, _ = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = first.read_type("approval_request")["data"]["approval_id"]
    first.close()
    thread.join(WAIT)
    assert result == [False]
    assert wait_until(lambda: server.current is None)
    second, _ = connect(server)
    second.send(
        {"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": "approve"}}
    )  # a reconnect does not revive it
    time.sleep(0.1)
    assert server.approvals.waiting == 0
    server.stop()


def test_with_nobody_connected_the_answer_is_no_without_waiting():
    server = make_server(approval_timeout_s=30)
    started = time.monotonic()
    assert server.approvals.ask("screen", "Look") is False
    assert time.monotonic() - started < 0.5
    client, _ = connect(server, handshake=False)  # connected but no hello yet: still nobody to ask
    assert server.approvals.ask("screen", "Look") is False
    client.close()
    server.stop()


def test_a_stop_while_an_approval_is_open_is_no_and_the_app_is_told_it_closed():
    server = make_server(approval_timeout_s=30)
    client, _ = connect(server)
    control = RunControl()
    result, thread = ask_in_thread(server, "input", "Control this Mac to: open Notes", control=control)
    ident = client.read_type("approval_request")["data"]["approval_id"]
    control.cancel("stopped by the user")
    thread.join(WAIT)
    assert result == [False]
    assert client.read_type("approval_closed")["data"] == {"approval_id": ident, "outcome": "cancelled"}
    server.stop()


def test_an_approve_that_arrives_after_a_stop_is_still_no():
    server = make_server(approval_timeout_s=30)
    connect(server)
    control = RunControl()
    control.cancel("stopped by the user")  # cancelled before it is even asked
    result, thread = ask_in_thread(server, "input", "x", control=control)
    thread.join(WAIT)
    assert result == [False]
    server.stop()


def test_an_approval_is_never_granted_by_anything_but_the_word_approve():
    server = make_server()
    client, _ = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    for decision in ("yes", "always", "APPROVE", True, 1, None):
        client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": decision}})
        assert client.read_type("error")["data"]["code"] == "malformed"
    assert not result
    client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": "deny"}})
    thread.join(WAIT)
    assert result == [False]
    server.stop()


def test_an_answer_that_comes_after_the_deadline_is_late_even_if_the_waiter_has_not_noticed_yet():
    now = [1000.0]
    server = AppServer(
        RecordingBackend(),
        core_version="test",
        limits=Limits(poll_s=0.02, idle_s=1e9, ping_s=1e9, handshake_s=1e9),
        approval_timeout_s=30,
        peer_ok=lambda sock: True,
        clock=lambda: now[0],
    )
    client, session = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    now[0] += 31  # the deadline has passed; the waiter is still inside its wait
    assert server.approvals.resolve(ident, "approve", session) is False
    assert server.approvals.resolve(ident, "deny", session) is False
    assert not result
    now[0] -= 31
    assert server.approvals.resolve(ident, "deny", session) is True  # inside the window it is taken
    thread.join(WAIT)
    assert result == [False]
    server.stop()


def test_an_answer_from_a_connection_the_request_was_not_asked_of_is_ignored():
    server = make_server()
    client, session = connect(server)
    result, thread = ask_in_thread(server, "screen", "Look")
    ident = client.read_type("approval_request")["data"]["approval_id"]
    stranger_end, other = socket.socketpair()
    stranger = Session(server, other)
    assert server.approvals.resolve(ident, "approve", stranger) is False
    assert not wait_until(lambda: bool(result), 0.2)
    assert server.approvals.resolve(ident, "deny", session) is True
    thread.join(WAIT)
    assert result == [False]
    stranger_end.close()
    server.stop()


def test_a_disconnect_settles_the_request_as_disconnected_not_as_waiting_out_its_time():
    server = make_server(approval_timeout_s=30)
    client, _ = connect(server)
    _, thread = ask_in_thread(server, "screen", "Look")
    client.read_type("approval_request")
    (pending,) = list(server.approvals._open.values())
    client.close()
    thread.join(WAIT)
    assert pending.outcome == "disconnected"
    server.stop()
