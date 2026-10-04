"""Shared pieces of the app-server tests: a client over a socketpair, a recording backend, and a wait helper.

Nothing here binds a socket by name: a `socketpair` is two ends of one connection inside this process.
"""

from __future__ import annotations

import json
import select
import socket
import time

from glide.app_server import wire
from glide.app_server.server import AppServer, Limits

WAIT = 3.0


def wait_until(predicate, timeout: float = WAIT) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class Client:
    """The app's end of a connection: sends frames, reads frames, notices when the core closed."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.seen: list[dict] = []

    def send(self, frame: dict) -> None:
        self.sock.sendall(json.dumps(frame).encode() + b"\n")

    def raw(self, data: bytes) -> None:
        self.sock.sendall(data)

    def hello(self, protocol: int = 1, ident: str = "a1") -> None:
        self.send({"v": 1, "type": "hello", "id": ident, "data": {"protocol": protocol, "client": "test", "client_version": "0"}})

    def read(self, timeout: float = WAIT) -> dict | None:
        """The next frame, or None when the core closed the connection or nothing came in time."""
        end = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            left = end - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([self.sock], [], [], left)
            if not ready:
                return None
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                return None
            if not chunk:
                return None
            self.buffer += chunk
        line, _, self.buffer = self.buffer.partition(b"\n")
        frame = json.loads(line)
        self.seen.append(frame)
        return frame

    def read_type(self, kind: str, timeout: float = WAIT) -> dict | None:
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            frame = self.read(left)
            if frame is None:
                return None
            if frame["type"] == kind:
                return frame
        return None

    def is_closed(self, timeout: float = WAIT) -> bool:
        """True once the core has closed its end (reads everything still queued first)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            ready, _, _ = select.select([self.sock], [], [], 0.05)
            if ready:
                try:
                    chunk = self.sock.recv(65536)
                except OSError:
                    return True
                if not chunk:
                    return True
                self.buffer += chunk
        return False

    def close(self) -> None:
        self.sock.close()


class RecordingBackend:
    """A `Backend` that records what the server hands it."""

    def __init__(self, record_content: bool = False) -> None:
        self.commands: list[wire.Command] = []
        self.ready: list = []
        self.closed: list = []
        self.record_content = record_content

    def capabilities(self) -> list[str]:
        return ["text_input"]

    def recording_content(self) -> bool:
        return self.record_content

    def on_ready(self, session) -> None:
        self.ready.append(session)

    def on_closed(self, session) -> None:
        self.closed.append(session)

    def handle(self, session, command) -> None:
        self.commands.append(command)


def connect(server: AppServer, *, handshake: bool = True) -> tuple[Client, object]:
    """A new connection to `server`, optionally already through the handshake."""
    a, b = socket.socketpair()
    session = server.attach(b)
    client = Client(a)
    if handshake:
        client.hello()
        assert client.read_type("hello") is not None
    return client, session


def make_server(
    backend=None, *, limits: Limits | None = None, approval_timeout_s: float = 2.0, peer_ok=lambda sock: True
) -> AppServer:
    return AppServer(
        backend or RecordingBackend(),
        core_version="test",
        limits=limits or Limits(poll_s=0.02),
        approval_timeout_s=approval_timeout_s,
        peer_ok=peer_ok,
    )
