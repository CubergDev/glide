"""Glide as an MCP server: a small JSON-RPC 2.0 core with no SDK, no sockets and no process of its own.

`GlideMCPServer.handle(message)` takes one decoded JSON-RPC message and returns the response (or None for a
notification). `serve_stream` drives it from a line reader and a writer, which is how `glide mcp serve` runs it
over stdio and how the tests run it in memory. The optional MCP SDK is not needed to serve.

What it offers is decided by the host that builds it, never by the remote client:
- Tools are `ServerTool`s the host registers. By default Glide registers none; `memory_tools` adds Glide's
  memory tools only when the user opted into memory AND asked for them ([mcp] server_memory).
- A tool that touches this machine (screen, input, apps, files) must be registered with `needs_approval=True`
  and the server needs an `approve` callback that returns exactly True for that call; without one it refuses.
  Glide registers no such tool; the machine-driving loop is never exposed implicitly.
- The scope (user, project, session) is fixed by the host. Arguments from the client are data: bounded, checked
  against the tool's schema, and never used to pick another scope or to grant anything.
- Errors a client can see are our own short messages (`ToolInputError`) or the exception type; the text of any
  other exception, internals, paths and stack traces are not sent. Tool results are not retried and no call
  replays a write.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, TextIO

from glide.memory.contracts import Scope
from glide.memory.store import Store

# Wire revisions this server can speak, newest first. These are protocol facts, not provider configuration.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MAX_MESSAGE_BYTES = 1048576
# What a remote client may write into Glide's memory (see `memory_tools`). Its notes are data, not instructions.
MCP_MEMORY_MAX_TEXT = 500
MCP_MEMORY_MAX_COUNT = 50
MCP_MEMORY_KEY_PREFIX = "mcp:"

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603
_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _reject_constant(name: str) -> Any:
    raise ValueError("non-finite JSON number")  # NaN, Infinity and -Infinity are not JSON


def _decode(line: str) -> Any:
    """`json.loads` for the wire: no NaN/Infinity, and a nesting bomb is a ValueError like any other bad JSON."""
    try:
        return json.loads(line, parse_constant=_reject_constant)
    except RecursionError:
        raise ValueError("JSON is nested too deeply") from None


def _encode(message: dict[str, Any]) -> str:
    """One response line. ASCII-only, so a lone surrogate in an echoed id or a result cannot break the UTF-8 writer."""
    return json.dumps(message, allow_nan=False) + "\n"


def _version() -> str:
    try:
        return metadata.version("glide")
    except metadata.PackageNotFoundError:
        return "0"


@dataclass(frozen=True)
class ServerTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]  # sync or async; returns a JSON-compatible value
    read_only: bool = False
    needs_approval: bool = False


class ToolInputError(ValueError):
    """The arguments do not fit the tool or its schema. The message is Glide's own text and is shown to the client;
    the text of every other exception (a plain `ValueError` can quote its input) is not."""


def check_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """A deliberately small JSON Schema subset: object, required, property types, additionalProperties false."""
    if not isinstance(arguments, dict):
        raise ToolInputError("arguments must be an object")
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in arguments:
            raise ToolInputError(f"missing argument {key!r}")
    for key, value in arguments.items():
        if key not in properties:
            if schema.get("additionalProperties") is False:
                raise ToolInputError("unknown argument")  # the name is the client's own text: never echoed
            continue
        wanted = properties[key].get("type")
        expected = _JSON_TYPES.get(wanted) if isinstance(wanted, str) else None
        if expected is not None and (
            not isinstance(value, expected)
            or (wanted in ("integer", "number") and isinstance(value, bool))
            or (wanted == "integer" and isinstance(value, float))
        ):
            raise ToolInputError(f"argument {key!r} must be {wanted}")
        allowed = properties[key].get("enum")
        if isinstance(allowed, list) and value not in allowed:
            raise ToolInputError(f"argument {key!r} must be one of {allowed}")
    return arguments


def _error(identifier: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}


def _ok(identifier: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def _feedback(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _check_property_types(tool: ServerTool) -> None:
    """Refuse a schema the argument check cannot enforce: a property `type` outside `_JSON_TYPES` would pass anything."""
    properties = tool.input_schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"tool {tool.name!r} needs a properties object")
    for key, spec in properties.items():
        wanted = spec.get("type") if isinstance(spec, dict) else None
        if not isinstance(spec, dict) or ("type" in spec and not (isinstance(wanted, str) and wanted in _JSON_TYPES)):
            raise ValueError(f"tool {tool.name!r} property {key!r} needs a plain JSON type ({', '.join(_JSON_TYPES)})")


class GlideMCPServer:
    def __init__(
        self,
        tools: Iterable[ServerTool] = (),
        *,
        approve: Callable[[ServerTool, dict[str, Any]], bool | Awaitable[bool]] | None = None,
        instructions: str | None = None,
    ):
        self._tools: dict[str, ServerTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            if not isinstance(tool.input_schema, dict) or tool.input_schema.get("type") != "object":
                raise ValueError(f"tool {tool.name!r} needs an object input schema")
            _check_property_types(tool)
            self._tools[tool.name] = tool
        self._approve, self._instructions = approve, instructions
        self._initialized = False

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def handle(self, message: Any) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            identifier = message.get("id") if isinstance(message, dict) else None
            return _error(identifier if isinstance(identifier, (str, int)) else None, INVALID_REQUEST, "invalid request")
        method, params = message["method"], message.get("params", {})
        if "id" not in message:
            return None  # a notification (initialized, cancelled, ...) needs no answer and starts nothing
        identifier = message["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            return _error(None, INVALID_REQUEST, "request id must be a string or an integer")
        if not isinstance(params, dict):
            return _error(identifier, INVALID_PARAMS, "params must be an object")
        if method == "ping":
            return _ok(identifier, {})
        if method == "initialize":
            return self._initialize(identifier, params)
        if not self._initialized:
            return _error(identifier, INVALID_REQUEST, "the server is not initialized")
        if method == "tools/list":
            return _ok(identifier, {"tools": [self._describe(tool) for tool in self._tools.values()]})
        if method == "tools/call":
            return await self._call(identifier, params)
        return _error(identifier, METHOD_NOT_FOUND, "method not found")

    def _initialize(self, identifier: Any, params: dict[str, Any]) -> dict[str, Any]:
        wanted = params.get("protocolVersion")
        if not isinstance(wanted, str):
            return _error(identifier, INVALID_PARAMS, "protocolVersion is required")
        self._initialized = True
        result: dict[str, Any] = {
            "protocolVersion": wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "glide", "version": _version()},
        }
        if self._instructions:
            result["instructions"] = self._instructions
        return _ok(identifier, result)

    @staticmethod
    def _describe(tool: ServerTool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
            "annotations": {"readOnlyHint": tool.read_only},
        }

    async def _call(self, identifier: Any, params: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(params.get("name")) if isinstance(params.get("name"), str) else None
        if tool is None:
            return _error(identifier, INVALID_PARAMS, "unknown tool")
        try:
            arguments = check_arguments(tool.input_schema, params.get("arguments", {}))
        except ToolInputError as error:
            return _error(identifier, INVALID_PARAMS, str(error))
        try:
            if tool.needs_approval:
                if self._approve is None:
                    return _ok(identifier, _feedback("this tool needs approval from the user and no approver is configured"))
                verdict = self._approve(tool, dict(arguments))
                if inspect.isawaitable(verdict):
                    verdict = await verdict
                if verdict is not True:
                    return _ok(identifier, _feedback("the user declined this tool call"))
            value = tool.handler(dict(arguments))
            if inspect.isawaitable(value):
                value = await value
            text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except ToolInputError as error:  # a handler's own validation message; any other error shows its type only
            return _ok(identifier, _feedback(str(error)))
        except Exception as error:
            return _ok(identifier, _feedback(f"the tool failed ({type(error).__name__})"))
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
        if isinstance(value, dict):
            result["structuredContent"] = value
        return _ok(identifier, result)


def memory_tools(store: Store, scope: Scope, *, write: bool = False) -> list[ServerTool]:
    """Glide's memory as MCP tools, for a fixed host-chosen scope. Writes only when `write` is True.

    A remote client is not the user, so what it writes is kept apart from what the user saved: its notes get the
    source "mcp" and the kind "note" (never a standing preference), keys under "mcp:" so they cannot replace a
    memory of the user's, at most MCP_MEMORY_MAX_TEXT characters and MCP_MEMORY_MAX_COUNT notes, and it can forget
    only its own notes. The user sees and removes them with `glide memory recall` and `glide memory forget`.
    """

    def remember(arguments: dict) -> dict:
        text, key = arguments["text"], MCP_MEMORY_KEY_PREFIX + arguments["key"]
        if len(text) > MCP_MEMORY_MAX_TEXT:
            raise ToolInputError(f"a remote note is limited to {MCP_MEMORY_MAX_TEXT} characters")
        notes = {row["key"] for row in store.memories(scope) if row["source"] == "mcp"}
        if key not in notes and len(notes) >= MCP_MEMORY_MAX_COUNT:
            raise ToolInputError(f"at most {MCP_MEMORY_MAX_COUNT} remote notes may be kept; ask the user to clear some")
        try:  # the store's refusals (credentials, bounds) are fixed Glide text; the host picks the level, never the client
            return {"id": store.remember(scope, key, text, kind="note", source="mcp", level="project")}
        except ValueError as error:
            raise ToolInputError(str(error)) from None

    def forget(arguments: dict) -> dict:
        own = any(row["id"] == arguments["id"] and row["source"] == "mcp" for row in store.memories(scope))
        return {"forgotten": own and store.forget(scope, arguments["id"])}

    tools = [
        ServerTool(
            "glide.memory.recall",
            "List the memories Glide holds for the current user and project.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda _args: {"memories": store.memories(scope)},
            read_only=True,
        )
    ]
    if write:
        tools += [
            ServerTool(
                "glide.memory.remember",
                f"Save a short keyed note (at most {MCP_MEMORY_MAX_TEXT} characters). It is kept as data from you, never as an instruction to Glide. Credentials are refused.",
                {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["key", "text"],
                    "additionalProperties": False,
                },
                remember,
            ),
            ServerTool(
                "glide.memory.forget",
                "Delete one of your own notes by id.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                forget,
            ),
        ]
    return tools


async def serve_stream(
    server: GlideMCPServer,
    readline: Callable[[], Awaitable[str]],
    write: Callable[[str], None],
) -> None:
    """Newline-delimited JSON-RPC: read a line, answer it, until the reader returns an empty string (EOF)."""
    while True:
        line = await readline()
        if not line:
            return
        if not line.strip():
            continue
        if len(line.encode("utf-8", "replace")) > MAX_MESSAGE_BYTES:
            response: dict[str, Any] | None = _error(None, INVALID_REQUEST, "message too large")
        else:
            try:
                message = _decode(line)
            except ValueError:
                response = _error(None, PARSE_ERROR, "parse error")
            else:
                response = await server.handle(message)
        if response is not None:
            write(_encode(response))


def read_bounded_line(stream: TextIO) -> str:
    """`stream.readline()` that never holds more than MAX_MESSAGE_BYTES + 1 characters.

    A line over the bound comes back cut at the bound (so `serve_stream` answers "message too large"), and the rest
    of it is discarded in bounded chunks up to its newline, so an endless line cannot grow memory.
    """
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if len(line) > MAX_MESSAGE_BYTES and not line.endswith("\n"):
        while (chunk := stream.readline(MAX_MESSAGE_BYTES)) and not chunk.endswith("\n"):
            pass
    return line


def text_stream_writer(stream: TextIO) -> Callable[[str], None]:
    def write(text: str) -> None:
        stream.write(text)
        stream.flush()

    return write
