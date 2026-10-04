"""The local socket: where it lives, who may reach it, and who is on the other end.

A Unix domain stream socket and nothing else: no TCP port, no address that another machine could name. The barrier
is the directory: it is made 0700 and owned by the person running Glide, and a socket in it is 0600. On top of that
every connection's peer user id is read from the kernel and a connection from any other user is refused.

A path is refused (never repaired or followed) when:
- it is longer than a `sockaddr_un` holds (103 bytes of path on macOS);
- the directory or the socket is a symbolic link, or is owned by someone else;
- the directory is open to group or others (it would be repaired by `chmod`, but then it was not ours to begin with);
- something other than a socket is there, or a core is already listening on it.
A leftover socket of our own that nothing answers on is removed: that is what an unclean exit leaves.

`bind` creates the socket with the process umask, and changing a umask is process-wide, so it is not changed: the 0700
directory keeps everyone out while the node is made, and `chmod 0600` follows at once.
"""

from __future__ import annotations

import contextlib
import os
import socket
import stat
import struct
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

ENV_SOCKET = "GLIDE_SOCKET"
MAX_PATH_BYTES = 103  # sockaddr_un.sun_path is 104 bytes on macOS, with the terminating NUL
BACKLOG = 2

DEFAULT_RELATIVE = Path("Library") / "Application Support" / "Glide" / "glide.sock"


class SocketRefused(Exception):
    """The socket path or the connection is not acceptable. The message names the problem, never any content."""


def resolve_path(argument: str | None, environ: Mapping[str, str], home: Path | None = None) -> Path:
    """`--socket`, else `GLIDE_SOCKET`, else the per-user default, with `~/` expanded. The app resolves it the same way."""
    chosen = argument or (environ.get(ENV_SOCKET) or "").strip() or None
    base = home if home is not None else Path(environ.get("HOME") or Path.home())
    if chosen is None:
        return base / DEFAULT_RELATIVE
    if chosen == "~":
        return base
    if chosen.startswith("~/"):
        return base / chosen[2:]
    return Path(chosen)


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def prepare(path: Path, *, uid: int | None = None) -> Path:
    """Check `path`, make its directory if it is missing, and remove a dead socket of ours. Raises `SocketRefused`."""
    uid = os.getuid() if uid is None else uid
    path = Path(path)
    if not path.is_absolute():
        raise SocketRefused("the socket path must be absolute")
    if len(os.fsencode(path)) > MAX_PATH_BYTES:
        raise SocketRefused(f"the socket path is longer than {MAX_PATH_BYTES} bytes, the most a Unix socket can have")
    directory = path.parent
    found = _lstat(directory)
    if found is None:
        if _lstat(directory.parent) is None:
            os.makedirs(directory.parent, exist_ok=True)
        os.mkdir(directory, 0o700)
        os.chmod(directory, 0o700)  # mkdir's mode is cut by the umask; this is the mode that matters
        found = _lstat(directory)
    if found is None or stat.S_ISLNK(found.st_mode):
        raise SocketRefused("the socket's directory is a symbolic link")
    if not stat.S_ISDIR(found.st_mode):
        raise SocketRefused("the socket's directory is not a directory")
    if found.st_uid != uid:
        raise SocketRefused("the socket's directory belongs to another user")
    if found.st_mode & 0o077:
        raise SocketRefused("the socket's directory is open to other users: it must be mode 0700")
    node = _lstat(path)
    if node is None:
        return path
    if stat.S_ISLNK(node.st_mode):
        raise SocketRefused("the socket path is a symbolic link")
    if not stat.S_ISSOCK(node.st_mode):
        raise SocketRefused("something that is not a socket is at the socket path")
    if node.st_uid != uid:
        raise SocketRefused("the socket belongs to another user")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        try:
            probe.connect(str(path))
        except (ConnectionRefusedError, FileNotFoundError):
            pass  # nothing listens: the leftover of an unclean exit
        except OSError as exc:
            raise SocketRefused(f"the socket path could not be probed ({type(exc).__name__})") from None
        else:
            raise SocketRefused("a core is already listening on this socket")
    again = _lstat(path)
    if again is not None and (again.st_ino, again.st_dev) == (node.st_ino, node.st_dev):
        os.unlink(path)
    return path


class Listener:
    """The listening socket. `open()` binds it, `accept()` waits for one connection, `close()` unlinks what `open()` made."""

    def __init__(self, path: Path, *, uid: int | None = None) -> None:
        self.path = Path(path)
        self._uid = os.getuid() if uid is None else uid
        self._sock: socket.socket | None = None
        self._identity: tuple[int, int] | None = None

    def open(self) -> None:
        prepare(self.path, uid=self._uid)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self.path))
            made = os.lstat(self.path)
            self._identity = (made.st_ino, made.st_dev)  # from here on, a failure removes exactly this node
            os.chmod(self.path, 0o600)
            node = os.lstat(self.path)
            if not stat.S_ISSOCK(node.st_mode) or node.st_uid != self._uid or stat.S_IMODE(node.st_mode) != 0o600:
                raise SocketRefused("the socket was not made as asked")
            sock.listen(BACKLOG)
            sock.settimeout(0.5)
        except BaseException:
            sock.close()
            self._unlink()
            raise
        self._sock = sock

    def accept(self) -> socket.socket | None:
        """The next connection, or None when none came within the poll interval or the listener is closed."""
        sock = self._sock
        if sock is None:
            return None
        try:
            conn, _ = sock.accept()
        except (TimeoutError, OSError):
            return None
        return conn

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()
        self._unlink()

    def _unlink(self) -> None:
        """Remove the socket only if it is still the one this listener made: never a file someone put there since."""
        identity, self._identity = self._identity, None
        if identity is None:
            return
        node = _lstat(self.path)
        if node is not None and (node.st_ino, node.st_dev) == identity and stat.S_ISSOCK(node.st_mode):
            with contextlib.suppress(OSError):
                os.unlink(self.path)


def peer_uid(sock: socket.socket) -> int | None:
    """The user id of the process on the other end, from the kernel; None when this system cannot say."""
    try:
        if sys.platform == "darwin":
            xucred = sock.getsockopt(0, getattr(socket, "LOCAL_PEERCRED", 1), struct.calcsize("=IIh2x16I"))  # SOL_LOCAL is 0
            return struct.unpack("=IIh2x16I", xucred)[1]
        if hasattr(socket, "SO_PEERCRED"):
            creds = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            return struct.unpack("3i", creds)[1]
    except (OSError, struct.error, AttributeError):
        return None
    return None


def same_user(sock: socket.socket, *, uid: int | None = None, reader: Callable[[socket.socket], int | None] = peer_uid) -> bool:
    """Whether the peer is this user. A peer that cannot be identified is refused."""
    found = reader(sock)
    return found is not None and found == (os.getuid() if uid is None else uid)
