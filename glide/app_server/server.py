"""The server of the app protocol: connections, framing, the handshake, approvals, and nothing about what the assistant does.

One client at a time. A second connection is closed without a word (the app retries with backoff, so it does not give up).
Each connection has a reader thread, which parses lines and hands commands on, and a writer thread, which owns the socket's
send side. Everything the core says goes through a bounded queue, and putting a message there never waits:

- a message that may be lost (a level, a partial transcript) is skipped when the queue is getting full;
- any other message that does not fit means the client cannot keep up, and it is disconnected. An approval open on it is
  then a refusal. A slow or stuck app never slows the assistant down.

The reader hands commands to the `Backend` and must not wait for anything: a backend that has work to do starts a thread.

Approvals (`Approvals`) are the one place a person's yes is taken. A request is asked of the connection that is current and of
no other. It is granted only by an `approve` from that same connection, for that id, before it expires, while the request that
asked is still wanted. Silence, a deny, a disconnect, a stop, a request nobody could be asked, a second answer, an answer for
an id that is unknown or old: all are no. There is no default and no "always".
"""

from __future__ import annotations

import contextlib
import logging
import queue
import secrets
import select
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ..computer.control import current_control
from ..computer.models import Abort
from . import wire
from .transport import Listener, same_user

log = logging.getLogger("glide.app_server")

_CLOSE = object()


@dataclass(frozen=True)
class Limits:
    """Every bound of a connection, in one place."""

    max_line_bytes: int = wire.MAX_LINE_BYTES
    queue_size: int = 256  # messages waiting to be sent; one more than this and the client is dropped
    droppable_queue: int = 64  # messages that may be lost are skipped once this many are waiting
    send_timeout_s: float = 5.0  # one message that cannot be handed to the socket in this long drops the client
    handshake_s: float = 5.0  # the app must say hello within this long
    ping_s: float = 15.0  # quiet this long and the core sends a ping
    idle_s: float = 45.0  # nothing heard from the app in this long (an app answers pings) and it is dropped
    bad_lines: int = 20  # unusable lines tolerated on one connection
    recv_bytes: int = 65536
    poll_s: float = 0.25


class Backend(Protocol):
    """What the server needs from whatever acts on the commands (see bridge.py)."""

    def capabilities(self) -> list[str]: ...

    def recording_content(self) -> bool: ...

    def on_ready(self, session: Session) -> None:
        """The handshake is done and `hello` is queued: send the first state."""

    def on_closed(self, session: Session) -> None: ...

    def handle(self, session: Session, command: wire.Command) -> None:
        """A command other than hello, pong and approval_response. Must return quickly."""


class LineBuffer:
    """Bytes in, lines out, bounded. A line over the limit is dropped up to its newline and reported once."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._buf = bytearray()
        self._discarding = False
        self._dropped = 0

    def feed(self, chunk: bytes) -> list[tuple[str, bytes | int]]:
        out: list[tuple[str, bytes | int]] = []
        for byte in chunk:
            if byte == 0x0A:
                if self._discarding:
                    out.append(("too_long", self._dropped))
                    self._discarding, self._dropped = False, 0
                else:
                    if self._buf.endswith(b"\r"):
                        del self._buf[-1]
                    if self._buf:
                        out.append(("line", bytes(self._buf)))
                    self._buf.clear()
            elif self._discarding:
                self._dropped += 1
            else:
                self._buf.append(byte)
                if len(self._buf) > self._limit:
                    self._discarding, self._dropped = True, len(self._buf)
                    self._buf = bytearray()
        return out


_SLOW_COMMANDS = frozenset({"settings_set", "voice_control"})


class Session:
    """One connection of the app."""

    def __init__(self, server: AppServer, sock: socket.socket) -> None:
        self.server = server
        self.limits = server.limits
        self.sock = sock
        sock.settimeout(
            self.limits.send_timeout_s
        )  # the send side's bound: a message the socket cannot take in this long drops the client
        self.client = ""
        self.ready = False  # true once `hello` has been queued: nothing else is sent before it
        self._queue: queue.Queue = queue.Queue(self.limits.queue_size)
        self._closed = threading.Event()
        self._close_lock = threading.Lock()
        self._clock = server.clock
        self._last_rx = self._clock()
        self._last_tx = self._clock()
        self._bad = 0
        self._closing = False  # a last message is queued: nothing more is read
        self._reason: str | None = None
        self._slow: queue.Queue = queue.Queue()  # settings_set / voice_control, one at a time, off the reader
        self._slow_thread: threading.Thread | None = None
        self.dropped = 0  # messages that were allowed to be lost, and were

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # -- sending (any thread, never waits) --------------------------------------------------------

    def send(self, message: wire.Outgoing) -> bool:
        """Queue a message. False when it was not queued: the connection is closed, the message was allowed to be lost and the
        queue is filling, or the queue is full, in which case this client is too slow and is dropped."""
        if self.closed:
            return False
        if message.droppable and self._queue.qsize() >= self.limits.droppable_queue:
            self.dropped += 1
            return False
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            self.close("the client is too slow")
            return False
        return True

    def send_and_close(self, message: wire.Outgoing, reason: str) -> None:
        """Say a last thing, then close once it has been sent (at once if there is no room to say it)."""
        self._closing = True
        self._reason = reason
        try:
            self._queue.put_nowait(message)
            self._queue.put_nowait(_CLOSE)
        except queue.Full:
            self.close(reason)

    def close(self, reason: str) -> None:
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.sock.close()
        self.server._session_closed(self, reason)

    # -- the two threads -----------------------------------------------------------------------

    def run(self) -> None:
        """The reader: runs on the connection's own thread until the connection ends."""
        writer = threading.Thread(target=self._write_loop, name="glide-app-writer", daemon=True)
        writer.start()
        started = self._clock()
        buffer = LineBuffer(self.limits.max_line_bytes)
        reason = "the client closed the connection"
        try:
            while not self.closed:
                now = self._clock()
                if not self.ready and now - started > self.limits.handshake_s:
                    reason = "no hello in time"
                    break
                if now - self._last_rx > self.limits.idle_s:
                    reason = "idle"
                    break
                try:
                    readable, _, _ = select.select([self.sock], [], [], self.limits.poll_s)
                except (OSError, ValueError):
                    break
                if not readable:
                    continue
                try:
                    chunk = self.sock.recv(self.limits.recv_bytes)
                except OSError:
                    break
                if not chunk:
                    break
                self._last_rx = self._clock()
                for kind, item in buffer.feed(chunk):
                    if self.closed:
                        break
                    if kind == "too_long":
                        log.warning("a line of %s bytes was dropped (limit %s)", item, self.limits.max_line_bytes)
                        self.send(wire.error("line_too_long", "a line was over the limit and was dropped"))
                    else:
                        self._line(item)  # type: ignore[arg-type]
                        self._last_rx = self._clock()  # time spent handling a line is not time the client was quiet
        finally:
            self.close(self._reason or reason)
            writer.join(timeout=1.0)

    def _write_loop(self) -> None:
        while not self.closed:
            try:
                message = self._queue.get(timeout=self.limits.poll_s)
            except queue.Empty:
                if self.ready and self._clock() - self._last_tx >= self.limits.ping_s:
                    message = wire.ping()
                else:
                    continue
            if message is _CLOSE:
                self.close(self._reason or "closed")
                return
            try:
                line = wire.encode(message)
            except wire.FrameError:
                log.warning("a %s message was too large to send and was dropped", message.type)
                continue
            try:
                self.sock.sendall(line)
            except (OSError, ValueError):
                self.close("the client stopped reading")
                return
            self._last_tx = self._clock()

    # -- one line ---------------------------------------------------------------------------------

    def _line(self, line: bytes) -> None:
        if self._closing:
            return
        try:
            command = wire.parse(line)
        except wire.UnsupportedVersion as exc:
            self.send_and_close(wire.error(exc.code, str(exc), fatal=True), "unsupported protocol version")
            return
        except wire.FrameError as exc:
            self._bad += 1
            log.warning("an unusable line was dropped (%s)", exc.code)
            if self._bad > self.limits.bad_lines:
                self.send_and_close(wire.error("too_many_errors", "too many unusable lines", fatal=True), "too many bad lines")
                return
            self.send(wire.error(exc.code, str(exc)))
            return
        if isinstance(command, wire.Unknown):
            log.info("a message of an unknown type (%s) was ignored", command.type)
            return
        if not self.ready:
            self._handshake(command)
            return
        if command.type == "pong":
            return
        if command.type == "hello":
            log.info("a second hello was ignored")
            return
        if command.type == "approval_response":
            granted = self.server.approvals.resolve(command.data["approval_id"], command.data["decision"], self)
            if not granted and command.data["decision"] == "approve":
                log.info("an approval for an unknown, answered or expired request was ignored")
            return
        if command.type in _SLOW_COMMANDS:
            # These can block for many seconds (a device, a permission prompt, a join). The reader must keep reading, so
            # that stop, interrupt, approvals and pongs are never queued behind them; they run in order on one thread.
            if self._slow_thread is None:
                self._slow_thread = threading.Thread(target=self._slow_loop, name="glide-app-slow", daemon=True)
                self._slow_thread.start()
            self._slow.put(command)
            return
        self._dispatch(command)

    def _slow_loop(self) -> None:
        while not self.closed:
            try:
                command = self._slow.get(timeout=self.limits.poll_s)
            except queue.Empty:
                continue
            self._dispatch(command)

    def _dispatch(self, command: wire.Command) -> None:
        try:
            self.server.backend.handle(self, command)
        except Exception as exc:  # a bug in the backend must not take the connection down; the type is all that is said
            log.exception("handling %s failed", command.type)
            self.send(
                wire.error("internal", f"the core could not handle {command.type} ({type(exc).__name__})", reply_to=command.id)
            )

    def _handshake(self, command: wire.Command) -> None:
        if command.type != "hello":
            self.send_and_close(wire.error("expected_hello", "the first message must be hello", fatal=True), "no hello first")
            return
        if command.data["protocol"] != wire.VERSION:
            self.send_and_close(
                wire.error("unsupported_protocol", f"this core speaks protocol {wire.VERSION}", fatal=True, reply_to=command.id),
                "unsupported protocol version",
            )
            return
        self.client = command.data["client"]
        server = self.server
        self.send(
            wire.hello(
                core_version=server.core_version,
                session_id=server.session_id,
                capabilities=server.backend.capabilities(),
                recording_content=server.backend.recording_content(),
                reply_to=command.id,
            )
        )
        self.ready = True
        server.backend.on_ready(self)


class _Pending:
    def __init__(self, session: Session, deadline: float) -> None:
        self.session = session
        self.deadline = deadline  # an answer after this is late, however soon the waiter notices
        self.outcome: str | None = None
        self.wake = threading.Event()


class Approvals:
    """Asks the person, over the current connection, and returns True only for an explicit approve. See the module text."""

    def __init__(self, server: AppServer, *, timeout_s: float = 30.0) -> None:
        self._server = server
        self.timeout_s = timeout_s
        self._open: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    @property
    def waiting(self) -> int:
        with self._lock:
            return sum(1 for p in self._open.values() if p.outcome is None)

    def ask(self, kind: str, command: str, *, task_id: str | None = None) -> bool:
        session = self._server.current
        if session is None:
            return False  # nobody to ask
        pending = _Pending(session, self._server.clock() + self.timeout_s)
        approval_id = "p" + secrets.token_hex(6)
        with self._lock:
            self._open[approval_id] = pending
        control = current_control()
        try:
            asked = session.send(wire.approval_request(approval_id, kind, command, task_id=task_id, expires_in_s=self.timeout_s))
            if not asked:
                self._settle(pending, "disconnected")
                return False
            try:
                with control.closing_on_cancel(pending.wake.set) if control is not None else contextlib.nullcontext():
                    pending.wake.wait(self.timeout_s)
            except Abort:
                pass  # the request was stopped before or while it waited
            cancelled = control is not None and control.cancelled.is_set()
            self._settle(pending, "cancelled" if cancelled else "expired", override_approved=cancelled)
            return pending.outcome == "approved" and not cancelled and not session.closed
        finally:
            with self._lock:
                self._open.pop(approval_id, None)
            self._settle(pending, "cancelled")
            session.send(wire.approval_closed(approval_id, pending.outcome or "cancelled"))

    def resolve(self, approval_id: str, decision: str, session: Session) -> bool:
        """An answer from the app. True if it settled an open request; anything else is ignored."""
        with self._lock:
            pending = self._open.get(approval_id)
            if pending is None or pending.session is not session or pending.outcome is not None:
                return False
            if self._server.clock() > pending.deadline:
                return False
            pending.outcome = "approved" if decision == "approve" else "denied"
        pending.wake.set()
        return True

    def session_closed(self, session: Session) -> None:
        """A connection ended: every request open on it is a no."""
        with self._lock:
            mine = [p for p in self._open.values() if p.session is session]
        for pending in mine:
            self._settle(pending, "disconnected")
            pending.wake.set()

    def _settle(self, pending: _Pending, outcome: str, *, override_approved: bool = False) -> None:
        """Set the outcome if nothing has. `override_approved`: a cancel that beat the waiter to the answer wins over 'approved'."""
        with self._lock:
            if pending.outcome is None or (override_approved and pending.outcome == "approved"):
                pending.outcome = outcome


class AppServer:
    """Accepts the app, one connection at a time, and runs each through the protocol."""

    def __init__(
        self,
        backend: Backend,
        *,
        core_version: str,
        listener: Listener | None = None,
        limits: Limits | None = None,
        approval_timeout_s: float = 30.0,
        peer_ok: Callable[[socket.socket], bool] = same_user,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend = backend
        self.core_version = core_version
        self.session_id = "s" + secrets.token_hex(6)
        self.limits = limits or Limits()
        self.clock = clock
        self.approvals = Approvals(self, timeout_s=approval_timeout_s)
        self._listener = listener
        self._peer_ok = peer_ok
        self._lock = threading.Lock()
        self._session: Session | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    # -- life ---------------------------------------------------------------------------------------

    def start(self) -> None:
        """Bind the socket and accept in the background. Raises `SocketRefused` when the path is not acceptable."""
        if self._listener is None:
            raise RuntimeError("no listener: use attach() to serve a connection you made")
        self._listener.open()
        self._thread = threading.Thread(target=self._accept_loop, name="glide-app-accept", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._listener is not None:
            self._listener.close()
        with self._lock:
            session = self._session
        if session is not None:
            session.close("the core is stopping")
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            conn = self._listener.accept()
            if conn is not None:
                self.attach(conn)

    def attach(self, sock: socket.socket) -> Session | None:
        """Serve a connection that has been accepted: check who is on the other end, make sure it is the only one, and run it.
        Returns None when it was refused (and closed)."""
        if not self._peer_ok(sock):
            log.warning("a connection from another user, or one that could not be identified, was refused")
            sock.close()
            return None
        with self._lock:
            if self._session is not None and not self._session.closed:
                busy = True
            else:
                busy = False
                session = self._session = Session(self, sock)
        if busy:
            log.info("a second client was refused: one is connected")
            sock.close()
            return None
        threading.Thread(target=session.run, name="glide-app-session", daemon=True).start()
        return session

    # -- for the backend ----------------------------------------------------------------------------

    @property
    def current(self) -> Session | None:
        """The connection that has said hello and is still open, or None."""
        with self._lock:
            session = self._session
        return session if session is not None and session.ready and not session.closed else None

    def emit(self, message: wire.Outgoing) -> bool:
        """Send to the current connection, if there is one. Never waits."""
        session = self.current
        return session.send(message) if session is not None else False

    def _session_closed(self, session: Session, reason: str) -> None:
        with self._lock:
            if self._session is session:
                self._session = None
        self.approvals.session_closed(session)
        log.info("the app disconnected (%s)", reason)
        try:
            self.backend.on_closed(session)
        except Exception:
            log.exception("closing the backend's side of a connection failed")
