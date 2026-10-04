"""A slow settings or voice command must not stall the reader: stop, interrupt and pongs are still read (audit2 finding 0)."""

from __future__ import annotations

import threading

from app_harness import RecordingBackend, connect, make_server, wait_until

from glide.app_server.server import Limits

SETTINGS_SET = {"v": 1, "type": "settings_set", "id": "s1", "data": {"base_revision": 0, "changes": []}}
VOICE = {"v": 1, "type": "voice_control", "id": "v1", "data": {"action": "mute"}}


class BlockingBackend(RecordingBackend):
    """handle() blocks on `release` for the slow commands, as apply_voice can."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    def handle(self, session, command) -> None:
        super().handle(session, command)
        if command.type in ("settings_set", "voice_control"):
            self.entered.set()
            self.release.wait(5)


def test_a_stop_sent_during_a_blocked_settings_set_reaches_the_backend_at_once():
    backend = BlockingBackend()
    server = make_server(backend)
    client, _ = connect(server)
    client.send(SETTINGS_SET)
    assert backend.entered.wait(2)
    client.send({"v": 1, "type": "stop", "id": "x1", "data": {}})
    try:
        assert wait_until(lambda: any(c.type == "stop" for c in backend.commands), timeout=2)
    finally:
        backend.release.set()
        server.stop()


def test_a_pong_waiting_behind_a_slow_handler_keeps_the_session_open():
    now = [0.0]
    backend = BlockingBackend()
    server = make_server(backend, limits=Limits(poll_s=0.02, idle_s=45, ping_s=1e9, handshake_s=1e9))
    server.clock = lambda: now[0]
    client, session = connect(server)
    client.send(VOICE)
    assert backend.entered.wait(2)
    now[0] += 60  # the handler has taken longer than idle_s
    client.send({"v": 1, "type": "pong"})
    backend.release.set()
    assert not wait_until(lambda: session.closed, timeout=0.5)
    server.stop()
