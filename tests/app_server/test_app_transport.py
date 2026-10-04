"""The socket path and the listener: private directory, private socket, no symlinks, no foreign owners, peer identity.

The tests that bind a real Unix socket use a short directory made by `tempfile.mkdtemp()`: the conftest guard lets a test reach a
Unix socket only inside the temp directory, and macOS limits a socket path to about 100 bytes (pytest's `tmp_path` is longer).
"""

from __future__ import annotations

import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path

import pytest
from app_harness import Client, RecordingBackend, wait_until

from glide.app_server import transport
from glide.app_server.server import AppServer, Limits
from glide.app_server.transport import Listener, SocketRefused, prepare, resolve_path


@pytest.fixture
def short_dir():
    directory = Path(tempfile.mkdtemp(prefix="g"))
    if len(os.fsencode(directory)) > 60:
        shutil.rmtree(directory, ignore_errors=True)
        pytest.skip("the temp directory path is too long for a unix socket")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


def mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# -- where it lives ---------------------------------------------------------------------------------------


def test_the_path_is_the_argument_then_the_environment_then_the_per_user_default():
    home = Path("/Users/someone")
    assert resolve_path("/x/a.sock", {"GLIDE_SOCKET": "/y/b.sock"}, home) == Path("/x/a.sock")
    assert resolve_path(None, {"GLIDE_SOCKET": "/y/b.sock"}, home) == Path("/y/b.sock")
    assert resolve_path(None, {"GLIDE_SOCKET": "  "}, home) == home / "Library/Application Support/Glide/glide.sock"
    assert resolve_path(None, {}, home) == home / "Library/Application Support/Glide/glide.sock"
    assert resolve_path("~/g/a.sock", {}, home) == home / "g/a.sock"


# -- what is refused --------------------------------------------------------------------------------------


def test_a_path_that_is_not_absolute_or_is_too_long_for_a_unix_socket_is_refused(short_dir):
    with pytest.raises(SocketRefused, match="absolute"):
        prepare(Path("relative.sock"))
    with pytest.raises(SocketRefused, match="longer than"):
        prepare(short_dir / ("s" * 120))
    assert prepare(short_dir / "s") == short_dir / "s"  # a short one is fine
    edge = short_dir / ("e" * (transport.MAX_PATH_BYTES - len(os.fsencode(short_dir)) - 1))
    assert len(os.fsencode(edge)) == transport.MAX_PATH_BYTES
    prepare(edge)
    with pytest.raises(SocketRefused):
        prepare(Path(str(edge) + "x"))


def test_a_missing_directory_is_made_private(short_dir):
    path = short_dir / "new" / "deeper" / "g.sock"
    prepare(path)
    assert mode(path.parent) == 0o700 and path.parent.is_dir()


def test_a_directory_open_to_others_is_refused_and_not_repaired(short_dir):
    os.chmod(short_dir, 0o755)
    with pytest.raises(SocketRefused, match="0700"):
        prepare(short_dir / "g.sock")
    assert mode(short_dir) == 0o755  # it was not ours to change
    os.chmod(short_dir, 0o770)
    with pytest.raises(SocketRefused, match="0700"):
        prepare(short_dir / "g.sock")


def test_a_directory_that_is_a_symlink_is_refused(short_dir):
    real = short_dir / "real"
    real.mkdir(mode=0o700)
    link = short_dir / "link"
    link.symlink_to(real)
    with pytest.raises(SocketRefused, match="symbolic link"):
        prepare(link / "g.sock")


def test_a_socket_path_that_is_a_symlink_is_refused_and_its_target_untouched(short_dir):
    target = short_dir / "precious"
    target.write_text("keep")
    (short_dir / "g.sock").symlink_to(target)
    with pytest.raises(SocketRefused, match="symbolic link"):
        prepare(short_dir / "g.sock")
    with pytest.raises(SocketRefused, match="symbolic link"):
        Listener(short_dir / "g.sock").open()
    assert target.read_text() == "keep"
    assert (short_dir / "g.sock").is_symlink()


def test_a_dangling_symlink_at_the_socket_path_is_refused_too(short_dir):
    (short_dir / "g.sock").symlink_to(short_dir / "nowhere")
    with pytest.raises(SocketRefused, match="symbolic link"):
        prepare(short_dir / "g.sock")


def test_something_that_is_not_a_socket_is_refused_and_kept(short_dir):
    (short_dir / "g.sock").write_text("a file of the person's")
    with pytest.raises(SocketRefused, match="not a socket"):
        prepare(short_dir / "g.sock")
    assert (short_dir / "g.sock").read_text() == "a file of the person's"


def test_a_directory_or_socket_owned_by_another_user_is_refused(short_dir):
    other = os.getuid() + 1
    with pytest.raises(SocketRefused, match="another user"):
        prepare(short_dir / "g.sock", uid=other)
    listener = Listener(short_dir / "g.sock")
    listener.open()
    listener.close()
    with socket.socket(socket.AF_UNIX) as leftover:
        leftover.bind(str(short_dir / "g.sock"))  # a dead socket of ours
    with pytest.raises(SocketRefused, match="another user"):
        prepare(short_dir / "g.sock", uid=other)  # the directory check comes first, and it names the owner too


def test_a_socket_in_our_directory_that_another_user_owns_is_refused_and_not_removed(short_dir, monkeypatch):
    with socket.socket(socket.AF_UNIX) as leftover:
        leftover.bind(str(short_dir / "g.sock"))
    real = transport._lstat

    def foreign(path):
        found = real(path)
        if found is not None and Path(path).name == "g.sock":
            fields = list(found)
            fields[stat.ST_UID if hasattr(stat, "ST_UID") else 4] = os.getuid() + 1
            return os.stat_result(fields)
        return found

    monkeypatch.setattr(transport, "_lstat", foreign)
    with pytest.raises(SocketRefused, match="belongs to another user"):
        prepare(short_dir / "g.sock")
    assert (short_dir / "g.sock").exists()


def test_a_leftover_socket_that_nothing_answers_on_is_removed(short_dir):
    with socket.socket(socket.AF_UNIX) as leftover:
        leftover.bind(str(short_dir / "g.sock"))
    assert (short_dir / "g.sock").exists()
    prepare(short_dir / "g.sock")
    assert not (short_dir / "g.sock").exists()


def test_a_socket_a_core_is_listening_on_is_refused_and_left_alone(short_dir):
    first = Listener(short_dir / "g.sock")
    first.open()
    try:
        with pytest.raises(SocketRefused, match="already listening"):
            Listener(short_dir / "g.sock").open()
        assert (short_dir / "g.sock").exists()
    finally:
        first.close()


# -- a real listener --------------------------------------------------------------------------------------


def test_the_listener_makes_a_0600_socket_in_a_0700_directory_and_removes_it_on_close(short_dir):
    path = short_dir / "run" / "g.sock"
    listener = Listener(path)
    listener.open()
    assert mode(path.parent) == 0o700 and mode(path) == 0o600
    assert stat.S_ISSOCK(os.lstat(path).st_mode) and os.lstat(path).st_uid == os.getuid()
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(path))
        assert listener.accept() is not None
    assert listener.accept() is None  # nobody came: the poll ends
    listener.close()
    assert not path.exists()
    listener.close()  # closing twice is fine


def test_closing_removes_only_the_socket_this_listener_made(short_dir):
    path = short_dir / "g.sock"
    listener = Listener(path)
    listener.open()
    os.unlink(path)
    path.write_text("someone else's file, put there since")
    listener.close()
    assert path.read_text() == "someone else's file, put there since"


def test_a_failed_bind_leaves_nothing_behind(short_dir, monkeypatch):
    listener = Listener(short_dir / "g.sock")

    def refuse_chmod(*args):
        raise PermissionError("no")

    monkeypatch.setattr(os, "chmod", refuse_chmod)
    with pytest.raises(PermissionError):
        listener.open()
    assert not (short_dir / "g.sock").exists()


def test_a_whole_session_over_the_real_socket_with_the_real_peer_check(short_dir):
    """Listener, accept loop, peer-uid check, handshake and clean shutdown, on a socket this test binds."""
    backend = RecordingBackend()
    path = short_dir / "g.sock"
    server = AppServer(backend, core_version="test", listener=Listener(path), limits=Limits(poll_s=0.02))
    server.start()
    try:
        client_sock = socket.socket(socket.AF_UNIX)
        client_sock.connect(str(path))
        client = Client(client_sock)
        client.hello()
        frame = client.read_type("hello")
        assert frame is not None and frame["data"]["protocol"] == 1  # this OS reported our own uid for the peer
        assert wait_until(lambda: backend.ready != [])
        client.close()
    finally:
        server.stop()
    assert not path.exists()


# -- who is on the other end ------------------------------------------------------------------------------


def test_the_peer_of_a_connection_in_this_process_is_this_user():
    a, b = socket.socketpair()
    with a, b:
        assert transport.peer_uid(a) == os.getuid()
        assert transport.same_user(a)


def test_a_peer_of_another_user_or_one_that_cannot_be_identified_is_refused():
    a, b = socket.socketpair()
    with a, b:
        assert not transport.same_user(a, reader=lambda sock: os.getuid() + 1)
        assert not transport.same_user(a, reader=lambda sock: None)
        assert not transport.same_user(a, uid=os.getuid() + 1)


def test_an_accept_error_other_than_a_timeout_backs_off_instead_of_spinning():
    """audit2 finding 4: EMFILE made accept return at once, so the accept loop spun at 100% CPU."""
    import errno

    pauses: list[float] = []
    listener = Listener(Path("unused.sock"), pause=pauses.append)

    class Full:
        def accept(self):
            raise OSError(errno.EMFILE, "too many open files")

    listener._sock = Full()  # type: ignore[assignment]
    assert listener.accept() is None
    assert pauses and pauses[0] > 0
