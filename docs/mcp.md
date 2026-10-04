# MCP: Glide as a server and as a client

MCP is an optional extra. Nothing here imports the MCP SDK until a real transport is opened, and the server core
needs no SDK at all.

## Settings

```toml
[mcp]
server_memory = "off"     # "off" (default), "read" or "write". Anything but "off" also needs [memory] enabled = true.

[[mcp.servers]]           # servers Glide may use as a client; none by default
name = "files"            # letters, digits, . _ - ; tool namespace mcp:files/...
command = "some-server"   # a local command, started over stdio only after you approve that exact command
args = ["--flag"]
env = ["SOME_TOKEN"]      # NAMES of environment variables to pass on; values are never written anywhere
timeout_s = 60
```

Unknown keys are errors. Keys and tokens never go in this file.

## Glide as a server (`glide.mcp.server`)

`GlideMCPServer` is a small JSON-RPC 2.0 core: `handle(message)` answers one decoded message, `serve_stream` runs it
over newline-delimited stdin/stdout (`glide mcp serve`, or `python -m glide.mcp serve`). Methods: `initialize`, `ping`, `tools/list`, `tools/call`; batches are refused.

What is offered is decided by the host, never by the client:

- By default, **no tools**.
- `server_memory = "read"` registers `glide.memory.recall`; `"write"` adds `glide.memory.remember` and
  `glide.memory.forget`. The user, project and session are fixed by the host (`--user/--project/--session`); the
  tool schemas have no scope argument. Remembered text is labelled `source = "mcp"`; credentials are refused.
- A tool that touches this machine (screen, input, apps, files) must be registered with `needs_approval=True`; the
  server then needs an `approve` callback that returns exactly `True` for that call, or it refuses. Glide registers
  no such tool, so the screen-driving loop is never exposed implicitly. Per-run approval for any future one is a
  host decision (D5).
- Arguments are bounded (1 MiB per message), checked against the tool's schema subset and never echoed in errors. A
  failing tool reports only its exception type. No call is retried.

## Glide as a client (`glide.mcp.client`)

`MCPClient` is the narrow seam: `list_tools()`, `call_tool(name, arguments)`, `aclose()`. `SessionClient` implements
it over a `request(method, params)` coroutine, such as `session_request(sdk_session)`.

- Only `ProviderError` escapes. The message names `mcp:<server>`, the request and the exception type or JSON-RPC
  code, never a key, header, argument or the server's error text. Kinds: `unsupported` (-32601), `bad_request`
  (-32600/-32602/-32700), `server` (other codes), `timeout`, `transport`, `content` (unusable reply), plus the
  httpx mappings (`auth`, `rate_limit`, ...).
- A tool call is a write that may have run. It is never retried. `MCPCallError.outcome_unknown` is `True` when the
  failure could have come after the server started it (timeout, dropped connection, unusable reply): stop and
  reconcile from a fresh observation; do not replay.
- A tool's own `isError: true` result is feedback in the returned dict, not an exception. Media payloads are
  projected to metadata. Tool descriptions and results are untrusted data, never instructions.
- `open_stdio_client(spec, approve=...)` is the only code that starts a process. It needs the SDK and `approve(spec)`
  returning exactly `True` after you were shown that command. Without that, nothing starts.

`MCPBridge` (`glide.mcp.bridge`) is the harness-level client: it turns a connected server's tools into `Tool`
descriptors for `glide.memory.Harness`, with generation checks so a stale tool list cannot be called after
`tools/list_changed`. Its errors are the host's own (the harness audits them); use `SessionClient` where you want
the ProviderError discipline.

## SDK versions

Written and tested against the MCP Python SDK 1.30. `mcp>=1.30,<2` is the intended pin: 2.x renamed `McpError`,
changed `list_tools` (params only) and `read_timeout_seconds` (float). `session_request` adapts the signatures it
can see, but the SDK type tests (`tests/mcp/test_mcp_sdk_types.py`) are skipped on 2.x and the 2.x path is untested.

## Not verified offline

No real server process, stdio pipe, socket or HTTP transport was ever opened: `open_stdio_client` and
`python -m glide.mcp serve` against a real MCP client (for example a desktop assistant) need a live check by you.
