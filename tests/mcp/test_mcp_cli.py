"""`glide mcp` in-process: serve runs over StringIO, never over real stdio."""

import io
import json

import pytest

from glide.mcp import cli
from glide.mcp.server import PROTOCOL_VERSIONS


def lines(*messages):
    return "".join(json.dumps(m) + "\n" for m in messages)


INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSIONS[0]}}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


@pytest.fixture
def world(tmp_path):
    def make(mcp_table="", memory_table=""):
        config = tmp_path / "glide.toml"
        config.write_text(mcp_table + memory_table)
        return config

    return make, {"GLIDE_DATA_DIR": str(tmp_path / "data")}, tmp_path / "home"


def run(world, config, *args, stdin=""):
    _, env, home = world
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["--config", str(config), *args], environ=env, home=home, stdin=io.StringIO(stdin), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_status_lists_names_only(world):
    make = world[0]
    config = make(
        '[mcp]\nserver_memory = "read"\n[[mcp.servers]]\nname = "files"\ncommand = "some-server"\nenv = ["SOME_TOKEN"]\n'
    )
    code, out, _ = run(world, config, "status")
    assert code == 0 and json.loads(out) == {"server_memory": "read", "client_servers": ["files"]}
    assert "some-server" not in out and "SOME_TOKEN" not in out


def test_default_serve_offers_no_tools_and_stdout_carries_only_protocol(world):
    config = world[0]("")
    code, out, err = run(world, config, "serve", stdin=lines(INIT, LIST))
    assert code == 0 and err == ""
    replies = [json.loads(line) for line in out.splitlines()]
    assert [r["id"] for r in replies] == [1, 2] and replies[1]["result"] == {"tools": []}
    assert not world[1]["GLIDE_DATA_DIR"] or not (world[2].parent / "data").exists()  # memory untouched


def test_memory_tools_need_both_server_memory_and_memory_on(world):
    config = world[0]('[mcp]\nserver_memory = "read"\n')  # memory itself left off
    code, out, err = run(world, config, "serve", stdin=lines(INIT, LIST))
    assert code == 2 and out == "" and "memory is off" in err

    config = world[0]('[mcp]\nserver_memory = "read"\n', "[memory]\nenabled = true\n")
    code, out, _ = run(world, config, "serve", stdin=lines(INIT, LIST))
    names = [t["name"] for t in json.loads(out.splitlines()[1])["result"]["tools"]]
    assert code == 0 and names == ["glide.memory.recall"]

    config = world[0]('[mcp]\nserver_memory = "write"\n', "[memory]\nenabled = true\n")
    code, out, _ = run(world, config, "serve", stdin=lines(INIT, LIST))
    names = [t["name"] for t in json.loads(out.splitlines()[1])["result"]["tools"]]
    assert names == ["glide.memory.recall", "glide.memory.remember", "glide.memory.forget"]


def call(id, name, args):
    return {"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": args}}


def test_write_through_the_server_persists_and_reads_back(world):
    config = world[0]('[mcp]\nserver_memory = "write"\n', "[memory]\nenabled = true\n")
    stdin = lines(INIT, call(2, "glide.memory.remember", {"key": "style", "text": "concise"}), call(3, "glide.memory.recall", {}))
    code, out, _ = run(world, config, "serve", stdin=stdin)
    replies = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and replies[2]["result"]["structuredContent"]["memories"][0]["text"] == "concise"
    # a second server process sees the same memory, so it was persisted (and the file is private)
    code, out, _ = run(world, config, "serve", stdin=lines(INIT, call(2, "glide.memory.recall", {})))
    assert json.loads(out.splitlines()[1])["result"]["structuredContent"]["memories"][0]["text"] == "concise"


def test_a_bad_config_is_a_controlled_error(world):
    config = world[0]('[mcp]\nserver_memory = "everything"\n')
    code, out, err = run(world, config, "status")
    assert code == 2 and out == "" and "server_memory" in err


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)
