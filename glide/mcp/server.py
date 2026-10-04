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
- Errors a client can see are our own short messages or the exception type; internals, paths and stack traces
  are not sent. Tool results are not retried and no call replays a write.
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

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603
_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


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
    """The arguments do not fit the tool's schema. The message is safe to show the client."""


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
                raise ToolInputError(f"unknown argument {key!r}")
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
        except ValueError as error:  # our own validation messages (never the client's text echoed from storage)
            return _ok(identifier, _feedback(str(error)))
        except Exception as error:
            return _ok(identifier, _feedback(f"the tool failed ({type(error).__name__})"))
        result: dict[str, Any] = {"content": [{"type": "text", "text": text}], "isError": False}
        if isinstance(value, dict):
            result["structuredContent"] = value
        return _ok(identifier, result)


def memory_tools(store: Store, scope: Scope, *, write: bool = False) -> list[ServerTool]:
    """Glide's memory as MCP tools, for a fixed host-chosen scope. Writes only when `write` is True."""
    levels = ["user", "project", "session"]
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
                "Save a keyed memory. Credentials are refused.",
                {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "text": {"type": "string"},
                        "level": {"type": "string", "enum": levels},
                    },
                    "required": ["key", "text"],
                    "additionalProperties": False,
                },
                lambda args: {
                    "id": store.remember(scope, args["key"], args["text"], source="mcp", level=args.get("level", "project"))
                },
            ),
            ServerTool(
                "glide.memory.forget",
                "Delete one memory by id.",
                {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                lambda args: {"forgotten": store.forget(scope, args["id"])},
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
                message = json.loads(line)
            except ValueError:
                response = _error(None, PARSE_ERROR, "parse error")
            else:
                response = await server.handle(message)
        if response is not None:
            write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")


def text_stream_writer(stream: TextIO) -> Callable[[str], None]:
    def write(text: str) -> None:
        stream.write(text)
        stream.flush()

    return write
