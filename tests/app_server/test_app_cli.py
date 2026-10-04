"""`glide app-server`: the command, its options, its seam, and that `glide --help` stays free of it until it runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from glide import cli as root_cli
from glide.app_server import cli as app_cli
from glide.app_server import runtime
from glide.app_server.transport import SocketRefused


def test_the_command_is_in_the_tree_with_one_line_of_help(capsys):
    with pytest.raises(SystemExit) as caught:
        root_cli.main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert "app-server" in out and "socket" in out


def test_the_help_of_the_command_lists_its_two_options(capsys):
    with pytest.raises(SystemExit):
        root_cli.main(["app-server", "--help"])
    out = capsys.readouterr().out
    assert "--socket" in out and "--record-content" in out and "network port" in out


def test_the_command_runs_serve_with_the_resolved_path_and_the_loaded_configuration(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        app_cli, "serve", lambda config, path, record_content=False: seen.append((config, path, record_content)) or 0
    )
    monkeypatch.setattr(root_cli, "_load", lambda path: "the-config")
    monkeypatch.delenv("GLIDE_SOCKET", raising=False)
    assert root_cli.main(["app-server", "--socket", str(tmp_path / "g.sock")]) == 0
    assert seen == [("the-config", tmp_path / "g.sock", False)]
    assert root_cli.main(["app-server", "--socket", str(tmp_path / "g.sock"), "--record-content"]) == 0
    assert seen[-1][2] is True


def test_the_path_falls_back_to_the_environment_and_then_the_per_user_default(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(app_cli, "serve", lambda config, path, record_content=False: seen.append(path) or 0)
    monkeypatch.setenv("GLIDE_SOCKET", str(tmp_path / "env.sock"))
    app_cli.main([], None)
    monkeypatch.delenv("GLIDE_SOCKET")
    monkeypatch.setenv("HOME", str(tmp_path))
    app_cli.main([], None)
    assert seen[0] == tmp_path / "env.sock"
    assert seen[1] == tmp_path / "Library" / "Application Support" / "Glide" / "glide.sock"


def test_a_socket_path_that_is_refused_is_one_line_on_stderr_and_exit_2(monkeypatch, capsys):
    def refuse(config, path, record_content=False):
        raise SocketRefused("the socket path is a symbolic link")

    monkeypatch.setattr(app_cli, "serve", refuse)
    assert app_cli.main(["--socket", "/x/y.sock"], None) == 2
    assert "symbolic link" in capsys.readouterr().err


def test_the_command_imports_nothing_heavy_until_it_runs():
    assert "glide.app_server.cli" in sys.modules  # this test module imports it, so check the root cli's own import list instead
    source = Path(root_cli.__file__).read_text()
    top = source.split("def _dotenv")[0]
    assert "app_server" not in top  # imported inside the command, like the other subcommands


def test_serve_and_the_real_voice_are_refused_in_tests(tmp_path):
    with pytest.raises(RuntimeError, match="real machine"):
        app_cli.serve(None, tmp_path / "g.sock")
    with pytest.raises(RuntimeError, match="real machine"):
        runtime.real_voice(None, None, None, False)
