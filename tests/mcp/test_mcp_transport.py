"""open_stdio_client with a fake SDK: nothing is started, and the real SDK transports are refused by the guard."""

import asyncio
import importlib.util
import sys
import types
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from glide.mcp import McpServerSpec, transport
from glide.providers.errors import ProviderError

SPEC = McpServerSpec("files", "some-server", ("--flag",), ("SOME_TOKEN", "UNSET_TOKEN"), 7.0)


def run(coro):
    return asyncio.run(coro)


class FakeSDK:
    def __init__(self, monkeypatch, *, connect_error=None, init_error=None, hang_init=False, exit_error=None):
        self.started, self.parameters, self.initialized, self.closed = [], [], False, False
        sdk = self

        class StdioServerParameters:
            def __init__(self, command, args, env):
                self.command, self.args, self.env = command, args, env
                sdk.parameters.append(self)

        class ClientSession:
            def __init__(self, read, write):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                sdk.closed = True
                if exit_error:
                    raise exit_error

            async def initialize(self):
                if hang_init:
                    await asyncio.Event().wait()  # a server that never answers
                if init_error:
                    raise init_error
                sdk.initialized = True

            async def list_tools(self, cursor=None):
                return {"tools": [{"name": "lookup", "inputSchema": {"type": "object"}}]}

            async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None, *, meta=None):
                sdk.timeout = read_timeout_seconds
                return {"content": [{"type": "text", "text": "ok"}]}

        @asynccontextmanager
        async def stdio_client(parameters):
            if connect_error:
                raise connect_error
            sdk.started.append(parameters.command)
            yield ("read", "write")

        package = types.ModuleType("mcp")
        package.ClientSession, package.StdioServerParameters = ClientSession, StdioServerParameters
        stdio = types.ModuleType("mcp.client.stdio")
        stdio.stdio_client = stdio_client
        monkeypatch.setitem(sys.modules, "mcp", package)
        monkeypatch.setitem(sys.modules, "mcp.client", types.ModuleType("mcp.client"))
        monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)


def test_without_the_users_yes_nothing_is_started(monkeypatch):
    sdk = FakeSDK(monkeypatch)

    async def go(verdict):
        async with transport.open_stdio_client(SPEC, approve=lambda spec: verdict):
            raise AssertionError("must not connect")

    for verdict in (False, None, "yes", 1):
        with pytest.raises(ProviderError) as caught:
            run(go(verdict))
        assert caught.value.kind == "unsupported" and "approve" in str(caught.value)
    assert sdk.started == [] and sdk.parameters == []


def test_approval_receives_the_exact_spec(monkeypatch):
    FakeSDK(monkeypatch)
    asked = []

    async def go():
        async with transport.open_stdio_client(SPEC, approve=lambda spec: asked.append(spec) or True):
            pass

    run(go())
    assert asked == [SPEC]


def test_missing_sdk_is_unsupported_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)

    async def go():
        async with transport.open_stdio_client(SPEC, approve=lambda spec: True):
            pass

    with pytest.raises(ProviderError) as caught:
        run(go())
    assert caught.value.kind == "unsupported" and "SDK" in str(caught.value)


def test_connect_passes_only_named_env_and_returns_a_working_client(monkeypatch):
    sdk = FakeSDK(monkeypatch)

    async def go():
        async with transport.open_stdio_client(
            SPEC, approve=lambda spec: True, environ={"SOME_TOKEN": "value-1", "OTHER": "never-passed"}
        ) as client:
            tools = await client.list_tools()
            result = await client.call_tool("lookup", {})
            return tools, result

    tools, result = run(go())
    assert [t.name for t in tools] == ["lookup"] and result["content"][0]["text"] == "ok"
    parameters = sdk.parameters[0]
    assert (parameters.command, parameters.args, parameters.env) == ("some-server", ["--flag"], {"SOME_TOKEN": "value-1"})
    assert sdk.initialized and sdk.closed and sdk.started == ["some-server"]
    assert sdk.timeout is not None  # the configured timeout reaches the SDK call, in the type its signature names


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        ({"connect_error": FileNotFoundError("no such file: some-server")}, "transport"),
        ({"init_error": TimeoutError()}, "timeout"),
    ],
)
def test_setup_failures_are_provider_errors_without_the_original_text(monkeypatch, kwargs, kind):
    FakeSDK(monkeypatch, **kwargs)

    async def go():
        async with transport.open_stdio_client(SPEC, approve=lambda spec: True):
            raise AssertionError("must not connect")

    with pytest.raises(ProviderError) as caught:
        run(go())
    assert caught.value.kind == kind and "no such file" not in str(caught.value)


def test_a_server_that_never_answers_initialize_times_out_as_a_provider_error(monkeypatch):
    """PR5-4175258151: spec.timeout_s bounds setup too, not only the calls made afterwards."""
    sdk = FakeSDK(monkeypatch, hang_init=True)

    async def go():
        async with transport.open_stdio_client(replace(SPEC, timeout_s=0.01), approve=lambda spec: True):
            raise AssertionError("must not connect")

    with pytest.raises(ProviderError) as caught:
        run(go())
    assert caught.value.kind == "timeout" and "connect" in str(caught.value) and sdk.closed


def test_an_sdk_teardown_failure_after_a_clean_body_is_a_provider_error(monkeypatch):
    """PR5-4175258134: only the caller's own exceptions pass through raw; SDK close errors are mapped."""
    FakeSDK(monkeypatch, exit_error=RuntimeError("sdk detail with /private/path"))

    async def go():
        async with transport.open_stdio_client(SPEC, approve=lambda spec: True):
            pass

    with pytest.raises(ProviderError) as caught:
        run(go())
    assert caught.value.kind == "transport" and "close" in str(caught.value) and "private" not in str(caught.value)


def test_an_error_raised_in_the_callers_block_is_not_relabelled(monkeypatch):
    FakeSDK(monkeypatch)

    async def go():
        async with transport.open_stdio_client(SPEC, approve=lambda spec: True):
            raise KeyError("callers own bug")

    with pytest.raises(KeyError):
        run(go())


def _sdk_installed() -> bool:
    # tests/mcp is itself importable as a namespace package named "mcp" when the SDK is absent, so ask for a submodule
    try:
        return importlib.util.find_spec("mcp.client.stdio") is not None
    except ImportError:
        return False


@pytest.mark.expect_refusals
@pytest.mark.skipif(not _sdk_installed(), reason="the MCP SDK is not installed")
def test_the_guard_refuses_the_real_sdk_transport():
    from mcp.client.stdio import stdio_client

    with pytest.raises(RuntimeError, match="real machine"):
        stdio_client(object())


def test_the_guard_refuses_asyncio_subprocesses():
    with pytest.raises(RuntimeError, match="real machine"):
        asyncio.create_subprocess_exec("true")


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)
