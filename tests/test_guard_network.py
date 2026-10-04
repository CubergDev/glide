"""Sockets: loopback and temp-directory unix sockets only, for connecting, listening, datagrams and name lookups.

Every refused call here is refused before anything is sent (the addresses are RFC 5737 documentation addresses or
names that must not even be looked up).
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest
from conftest import local_socket_path

DOC_ADDRESS = "192.0.2.1"  # RFC 5737 TEST-NET-1: never routable


def test_a_connection_to_anything_but_loopback_refuses():
    for address in ((DOC_ADDRESS, 80), ("93.184.215.14", 443), ("::ffff:192.0.2.1", 80)):
        with socket.socket(socket.AF_INET6 if ":" in address[0] else socket.AF_INET) as sock:
            with pytest.raises(RuntimeError, match="real machine"):
                sock.connect(address)
            with pytest.raises(RuntimeError, match="real machine"):
                sock.connect_ex(address)


@pytest.mark.parametrize("host", ["0.0.0.0", "", DOC_ADDRESS])
def test_a_listener_on_a_non_loopback_address_refuses(host):
    with socket.socket() as sock, pytest.raises(RuntimeError, match="real machine"):
        sock.bind((host, 0))


def test_a_listener_on_ipv6_any_refuses():
    with socket.socket(socket.AF_INET6) as sock, pytest.raises(RuntimeError, match="real machine"):
        sock.bind(("::", 0))


def test_a_loopback_listener_a_test_starts_itself_still_works():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=1) as client:
            conn, _ = server.accept()
            with conn:
                client.sendall(b"ping")
                assert conn.recv(4) == b"ping"


def test_a_datagram_to_a_remote_address_refuses_and_to_loopback_works():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        with pytest.raises(RuntimeError, match="real machine"):
            sock.sendto(b"x", (DOC_ADDRESS, 9))
        with pytest.raises(RuntimeError, match="real machine"):
            sock.sendto(b"x", 0, (DOC_ADDRESS, 9))
        with pytest.raises(RuntimeError, match="real machine"):
            sock.sendmsg([b"x"], [], 0, (DOC_ADDRESS, 9))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1)
        sender.sendto(b"hi", receiver.getsockname())
        assert receiver.recv(2) == b"hi"


@pytest.mark.parametrize(
    "lookup",
    [
        lambda: socket.getaddrinfo("example.com", 443),
        lambda: socket.gethostbyname("example.com"),
        lambda: socket.gethostbyname_ex("example.com"),
        lambda: socket.gethostbyaddr(DOC_ADDRESS),
        lambda: socket.getnameinfo((DOC_ADDRESS, 80), 0),
    ],
)
def test_a_name_lookup_that_leaves_the_machine_refuses(lookup):
    with pytest.raises(RuntimeError, match="real machine"):
        lookup()


def test_loopback_lookups_still_work():
    assert socket.gethostbyname("localhost").startswith("127.")
    assert socket.getaddrinfo("127.0.0.1", 80)
    assert socket.getnameinfo(("127.0.0.1", 80), 0)


def test_a_unix_socket_is_only_reachable_inside_the_temp_directory(tmp_path):
    inside = tmp_path / "s.sock"
    outside = Path.home() / ".glide-guard-probe.sock"
    assert local_socket_path(inside)
    assert not local_socket_path(outside)
    assert not local_socket_path("/var/run/docker.sock")
    assert not local_socket_path("")
    assert not local_socket_path("\0abstract")
    assert not local_socket_path(Path(tempfile.gettempdir()) / ".." / "escape.sock")  # a path that leaves the temp directory
    assert not local_socket_path(os.fsencode("/var/run/syslog"))
    with socket.socket(socket.AF_UNIX) as sock:
        with pytest.raises(RuntimeError, match="real machine"):
            sock.connect(str(outside))
        with pytest.raises(RuntimeError, match="real machine"):
            sock.connect_ex("/var/run/syslog")
        with pytest.raises(RuntimeError, match="real machine"):
            sock.bind(str(outside))
    assert not outside.exists()


def test_a_unix_socket_a_test_creates_in_the_temp_directory_works():
    directory = tempfile.mkdtemp()  # short, under the temp root: unix socket paths are limited to about 100 bytes
    path = os.path.join(directory, "s")
    if len(path) > 90:
        pytest.skip("the temp directory path is too long for a unix socket")
    try:
        with socket.socket(socket.AF_UNIX) as server:
            server.bind(path)
            server.listen(1)
            with socket.socket(socket.AF_UNIX) as client:
                client.connect(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)
        os.rmdir(directory)


def test_a_proxy_in_the_environment_is_dropped(monkeypatch):
    import urllib.request

    assert urllib.request.getproxies() == {}
    assert os.environ["NO_PROXY"] == "*"
