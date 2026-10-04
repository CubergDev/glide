"""The MCP client bridge: tools of an already initialized, host-owned MCP session become harness tools.

This is the CLIENT side of Glide (Glide consuming another server's tools); the server side is server.py.
Errors from the host's `request` callback propagate as they are (the harness audits them); the ProviderError-only
client interface is client.py.

No SDK import, connection, resource fetch, task polling or process start occurs here.
The host supplies negotiated server capabilities, grants, relevance and schema validation.
"""

import asyncio
import copy
import inspect
import math
import re
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from glide.memory._callbacks import adrive, drive, require_sync
from glide.memory._json import bounded_json
from glide.memory.contracts import Tool
from glide.memory.events import EventBus, Invocation, current_invocation


class MCPProtocolError(ValueError):
    """A protocol response is erroneous or cannot satisfy the bounded contract."""


class MCPStaleCatalogError(RuntimeError):
    """Refresh and replace the harness tools before executing this catalog."""


def rpc_code(error: BaseException) -> int | None:
    """The JSON-RPC error code an MCP SDK exception carries (its `.error` is ErrorData or a dict), else None."""
    detail = getattr(error, "error", None)
    code = detail.get("code") if isinstance(detail, dict) else getattr(detail, "code", None)
    return code if type(code) is int else None


def _protocol_failure(error: BaseException) -> bool:
    return isinstance(error, MCPProtocolError) or rpc_code(error) is not None


# A tool call is a write. Only an explicit refusal made before anything ran leaves its outcome known: a JSON-RPC
# request, method, params or parse error, or an HTTP status for a request that is malformed, unauthorized, unknown or
# unacceptable. Every other failure of a call that was sent (any 5xx, 408, 429, -32603, -32000..-32099, a timeout, a
# dropped connection, an unusable reply, anything unrecognized) may have come after the tool ran.
REFUSED_RPC_CODES = frozenset({-32600, -32601, -32602, -32700})
REFUSED_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 415, 422})
OUTCOME_UNKNOWN_NOTE = "the call may or may not have happened; do not repeat it without checking its effect first"


def refused_before_running(error: BaseException) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return rpc_code(error) in REFUSED_RPC_CODES or (type(status) is int and status in REFUSED_HTTP_STATUSES)


def mark_call_outcome(error: BaseException) -> None:
    """Say on an exception from a sent call whether it may have run: `outcome_unknown`, and a note when it is True.

    The exception keeps its type and its message; an exception that already carries the flag is left alone.
    """
    if hasattr(error, "outcome_unknown"):
        return
    unknown = not refused_before_running(error)
    with suppress(AttributeError, TypeError):  # an exception type that refuses new attributes stays unmarked
        error.outcome_unknown = unknown
        if unknown:
            error.add_note(OUTCOME_UNKNOWN_NOTE)


def _response(value, asynchronous):
    if asynchronous and not inspect.isawaitable(value):
        raise TypeError("asynchronous MCP callback must return an awaitable")
    return value


def _mapping(value: Any) -> dict:
    if not isinstance(value, dict) and callable(getattr(value, "model_dump", None)):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if not isinstance(value, dict):
        raise MCPProtocolError("MCP response must be a dictionary or SDK-like model")
    if "error" in value:
        raise MCPProtocolError("MCP transport returned a protocol error")
    return value


def wire_json(value: Any, limit: int = 65536) -> Any:
    return bounded_json(value, limit, MCPProtocolError, "MCP payload")


def normalize_mcp_result(result: Any, *, max_bytes: int = 65536) -> dict:
    """Keep tool feedback; project standard base64 media as bounded metadata.

    Resource links and embedded resource URIs remain references. This projection
    neither fetches nor decodes any resource; media rendering belongs to the host.
    """
    value = _mapping(result)
    content = value.get("content")
    if not isinstance(content, list):
        raise MCPProtocolError("MCP tool result requires a content list")
    if "isError" in value and type(value["isError"]) is not bool:
        raise MCPProtocolError("isError must be a boolean")
    projected = []
    for raw in content:
        block = copy.copy(_mapping(raw))
        block.pop("_meta", None)
        kind = block.get("type")
        if not isinstance(kind, str):
            raise MCPProtocolError("MCP content type must be a string")
        if kind == "text":
            if not isinstance(block.get("text"), str):
                raise MCPProtocolError("text content requires a string")
        elif kind in {"image", "audio"}:
            data = block.pop("data", None)
            if not isinstance(data, str) or not isinstance(block.get("mimeType"), str):
                raise MCPProtocolError("media content requires data and mimeType")
            block.update(data_omitted=True, encoded_bytes=len(data))
        elif kind == "resource":
            resource = copy.copy(_mapping(block.get("resource")))
            resource.pop("_meta", None)
            if not isinstance(resource.get("uri"), str):
                raise MCPProtocolError("embedded resource requires a URI")
            if "blob" in resource:
                blob = resource.pop("blob")
                if not isinstance(blob, str):
                    raise MCPProtocolError("resource blob must be a string")
                resource.update(blob_omitted=True, encoded_bytes=len(blob))
            elif not isinstance(resource.get("text"), str):
                raise MCPProtocolError("embedded resource requires text or blob")
            block["resource"] = resource
        elif kind == "resource_link":
            if not isinstance(block.get("uri"), str) or not isinstance(block.get("name"), str):
                raise MCPProtocolError("resource link requires URI and name")
        else:
            raise MCPProtocolError("unsupported MCP content type")
        projected.append(block)
    normalized = {"content": projected}
    for key in ("structuredContent", "isError"):
        if key in value:
            normalized[key] = value[key]
    if "structuredContent" in normalized and not isinstance(normalized["structuredContent"], dict):
        raise MCPProtocolError("structuredContent must be an object")
    return wire_json(normalized, max_bytes)


def _validate_result(result, schema, validator):
    normalized = normalize_mcp_result(result)
    if schema is not None and not normalized.get("isError", False):
        if "structuredContent" not in normalized:
            raise MCPProtocolError("tool with outputSchema requires structuredContent")
        if validator is not None:
            try:
                verdict = require_sync(validator(copy.deepcopy(schema), copy.deepcopy(normalized["structuredContent"])))
                if verdict is False:
                    raise MCPProtocolError("host rejected the structured tool output")
            except Exception as error:
                if isinstance(error, MCPProtocolError):
                    raise
                raise MCPProtocolError("host could not validate the structured tool output") from error
    return normalized


def tool_descriptor(raw):
    descriptor = _mapping(raw)
    name = descriptor.get("name")
    if not isinstance(name, str) or not 1 <= len(name) <= 128:
        raise MCPProtocolError("MCP tool name must contain 1 to 128 characters")
    for key in ("description", "title"):
        if key in descriptor and not isinstance(descriptor[key], str):
            raise MCPProtocolError("MCP tool text metadata must contain strings")
    if "annotations" in descriptor and not isinstance(descriptor["annotations"], dict):
        raise MCPProtocolError("MCP tool annotations must be an object")
    execution = descriptor.get("execution", {})
    support = execution.get("taskSupport", "forbidden") if isinstance(execution, dict) else None
    if not isinstance(support, str) or support not in {"required", "optional", "forbidden"}:
        raise MCPProtocolError("tool execution declaration is invalid")
    schema, output = descriptor.get("inputSchema"), descriptor.get("outputSchema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise MCPProtocolError("MCP tool requires an object inputSchema")
    if output is not None and not isinstance(output, dict):
        raise MCPProtocolError("MCP outputSchema must be a schema object")
    return wire_json(descriptor, 1048576)


def read_tools_page(response: Any, inventory: list, seen: set) -> str | None:
    """Append one `tools/list` page to `inventory`; return the next cursor, or None on the last page."""
    page = wire_json(_mapping(response), 1048576)
    if not isinstance(page.get("tools"), list):
        raise MCPProtocolError("tools/list requires a tools list")
    inventory.extend(page["tools"])
    if len(inventory) > 4096:
        raise MCPProtocolError("MCP inventory exceeds 4096 tools")
    cursor = page.get("nextCursor")
    if cursor is not None:
        if not isinstance(cursor, str) or not cursor or len(cursor) > 4096 or cursor in seen:
            raise MCPProtocolError("MCP pagination cursor is invalid or repeated")
        seen.add(cursor)
    return cursor


def bind_mcp(
    server: str,
    inventory: Iterable[dict[str, Any]],
    call_tool: Callable,
    *,
    permissions: dict[str, frozenset[str]],
    keywords: dict[str, tuple[str, ...]],
    asynchronous: bool = False,
    normalize_results: bool = True,
    validate_output: Callable | None = None,
) -> tuple[Tool, ...]:
    """Bind an existing callback; opt out of normalization only for legacy callbacks."""
    if not isinstance(server, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", server):
        raise ValueError("invalid MCP server identifier")
    if not callable(call_tool) or (validate_output is not None and not callable(validate_output)):
        raise TypeError("MCP callback and optional output validator must be callable")
    if type(asynchronous) is not bool or type(normalize_results) is not bool:
        raise TypeError("MCP callback and normalization flags must be booleans")
    if not isinstance(permissions, dict) or not isinstance(keywords, dict):
        raise TypeError("MCP permissions and keywords must be host mappings")
    result, seen = [], set()
    for raw in inventory:
        descriptor = tool_descriptor(raw)
        name = descriptor["name"]
        if name in seen or len(seen) >= 4096:
            raise MCPProtocolError("MCP tool names must be unique in a bounded inventory")
        seen.add(name)
        if name not in permissions or name not in keywords:
            continue
        if not isinstance(permissions[name], frozenset) or any(
            not isinstance(value, str) or not value.strip() for value in permissions[name]
        ):
            raise TypeError("MCP tool permissions require a frozenset of nonempty strings")
        if (
            not isinstance(keywords[name], tuple)
            or not keywords[name]
            or any(not isinstance(value, str) or not value.strip() for value in keywords[name])
        ):
            raise TypeError("MCP tool keywords require a nonempty tuple of nonempty strings")
        execution = descriptor.get("execution", {})
        if execution.get("taskSupport") == "required":
            continue
        schema, output = descriptor.get("inputSchema"), descriptor.get("outputSchema")
        private_output = None if output is None else wire_json(output)

        def steps(arguments, tool_name, output_schema):
            response = yield lambda: _response(call_tool(tool_name, arguments), asynchronous)
            return _validate_result(response, output_schema, validate_output) if normalize_results else response

        def invoke(arguments, tool_name=name, output_schema=private_output, workflow=steps):
            return drive(workflow(arguments, tool_name, output_schema))

        async def ainvoke(arguments, tool_name=name, output_schema=private_output, workflow=steps):
            return await adrive(workflow(arguments, tool_name, output_schema))

        result.append(
            Tool(
                id=f"mcp:{server}/{name}",
                description=str(descriptor.get("description", name))[:2000],
                keywords=keywords[name],
                permissions=permissions[name],
                schema=wire_json(schema),
                invoke=ainvoke if asynchronous else invoke,
                origin="mcp",
                asynchronous=asynchronous,
                output_schema=None if output is None else wire_json(output),
            )
        )
    return tuple(result)


@dataclass
class _Active:
    context: Invocation | None
    correlation_id: str
    started: float
    name: str
    progress: float = -1


class MCPBridge:
    """Refresh an inventory explicitly; generations reject stale tool closures."""

    def __init__(
        self,
        server: str,
        request: Callable,
        *,
        permissions,
        keywords,
        capabilities,
        event_bus: EventBus | None = None,
        asynchronous=False,
        validate_output=None,
    ):
        if not isinstance(server, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", server) or not callable(request):
            raise ValueError("bridge requires a server identifier and host request callback")
        if type(asynchronous) is not bool or (validate_output is not None and not callable(validate_output)):
            raise TypeError("bridge asynchronous flag must be boolean and validator callable")
        if not isinstance(permissions, dict) or not isinstance(keywords, dict):
            raise TypeError("bridge grants and relevance must be host mappings")
        self.server, self.request = server, request
        self.permissions, self.keywords = copy.deepcopy(permissions), copy.deepcopy(keywords)
        self._capabilities = wire_json(_mapping(capabilities))
        for section in ("tools", "resources", "prompts", "logging"):
            if section in self._capabilities and not isinstance(self._capabilities[section], dict):
                raise ValueError("negotiated capabilities must contain capability objects")
        self.event_bus = EventBus() if event_bus is None else event_bus
        self.asynchronous, self.validate_output = asynchronous, validate_output
        self._lock = RLock()
        self._connected, self._dirty, self._generation = True, True, 0
        self._tools: tuple[Tool, ...] = ()
        self._active: dict[str, _Active] = {}

    @property
    def capabilities(self) -> dict:
        return copy.deepcopy(self._capabilities)

    def _emit(self, kind, data, active: _Active | None = None):
        context = active.context if active is not None else current_invocation()
        with suppress(BaseException):
            self.event_bus.publish(
                f"mcp.{kind}",
                f"mcp:{self.server}",
                data,
                scope=None if context is None else context.scope,
                correlation_id=active.correlation_id if active else (None if context is None else context.call_id),
            )

    def set_connected(self, connected: bool) -> None:
        if type(connected) is not bool:
            raise TypeError("connected must be a boolean")
        with self._lock:
            if connected == self._connected:
                return
            self._connected, self._dirty = connected, True
            self._generation += 1
            generation = self._generation
            self._tools = ()
        self._emit("connection_changed", {"connected": connected, "generation": generation})

    def tools(self) -> tuple[Tool, ...]:
        with self._lock:
            if "tools" not in self._capabilities:
                return ()
            if not self._connected or self._dirty:
                raise MCPStaleCatalogError("refresh the connected MCP inventory before selecting tools")
            return tuple(tool.snapshot() for tool in self._tools)

    def attach_inventory(self, callback: Callable[[tuple[Tool, ...]], None]):
        """Push inventory changes to a host registry without starting any requests.

        The callback must be synchronous and quick, such as an atomic harness
        namespace replacement. Observer faults are isolated by the event bus.
        """
        if not callable(callback):
            raise TypeError("inventory callback must be callable")
        kinds = frozenset({"mcp.inventory_refreshed", "mcp.tools_changed", "mcp.connection_changed"})

        def deliver():
            with self._lock:
                require_sync(callback(self.tools() if self._connected and not self._dirty else ()))

        def changed(event):
            if event.source != f"mcp:{self.server}":
                return
            # Events are wake-ups, not snapshots. Publish current authoritative
            # state so an older event cannot withdraw a newer clean inventory.
            deliver()

        unsubscribe = self.event_bus.subscribe(changed, kinds=kinds)
        try:
            deliver()
        except BaseException:
            unsubscribe()
            raise
        return unsubscribe

    def _begin_refresh(self):
        with self._lock:
            if not self._connected:
                raise MCPStaleCatalogError("MCP host is disconnected")
            if "tools" not in self._capabilities:
                raise MCPProtocolError("host did not negotiate tools capability")
            return self._generation

    def _publish(self, generation, inventory):
        next_generation = generation + 1
        schemas, driver = {}, adrive if self.asynchronous else drive

        def call_tool(name, arguments):
            return driver(self._call_steps(next_generation, name, arguments, schemas[name]))

        tools = bind_mcp(
            self.server,
            inventory,
            call_tool,
            permissions=self.permissions,
            keywords=self.keywords,
            asynchronous=self.asynchronous,
            normalize_results=False,
        )
        schemas.update((tool.id.split("/", 1)[1], copy.deepcopy(tool.output_schema)) for tool in tools)
        with self._lock:
            if not self._connected or generation != self._generation:
                raise MCPStaleCatalogError("MCP inventory changed during refresh")
            self._tools, self._dirty, self._generation = tuple(tools), False, next_generation
        for descriptor in inventory:
            if descriptor.get("execution", {}).get("taskSupport") == "required":
                self._emit(
                    "tool_skipped",
                    {"name": descriptor.get("name"), "status": "skipped", "reason": "task_required_unsupported"},
                )
        self._emit("inventory_refreshed", {"count": len(tools), "generation": next_generation, "operation": "tools/list"})
        return self.tools()

    def refresh(self) -> tuple[Tool, ...]:
        if self.asynchronous:
            raise TypeError("await arefresh() for an asynchronous MCP bridge")
        return drive(self._refresh_steps())

    async def arefresh(self) -> tuple[Tool, ...]:
        if not self.asynchronous:
            raise TypeError("use refresh() for a synchronous MCP bridge")
        return await adrive(self._refresh_steps())

    def _refresh_steps(self):
        generation, inventory, seen, cursor = self._begin_refresh(), [], set(), None
        try:
            for _ in range(128):
                params = {} if cursor is None else {"cursor": cursor}
                response = yield lambda params=params: _response(self.request("tools/list", params), self.asynchronous)
                cursor = read_tools_page(response, inventory, seen)
                if cursor is None:
                    return self._publish(generation, inventory)
            raise MCPProtocolError("MCP inventory exceeds 128 pages")
        except asyncio.CancelledError:
            self._emit("cancelled", {"operation": "tools/list", "status": "cancelled"})
            raise
        except Exception as error:
            self._emit(
                "protocol_failure" if _protocol_failure(error) else "execution_failure",
                {"operation": "tools/list", "status": "error"},
            )
            raise

    def _admit(self, generation, name, arguments):
        arguments = wire_json(arguments)
        if not isinstance(arguments, dict):
            raise MCPProtocolError("MCP tool arguments must be an object")
        context, token = current_invocation(), uuid4().hex
        active = _Active(context, token if context is None else context.call_id, time.monotonic(), name)
        with self._lock:
            if not self._connected or self._dirty or generation != self._generation:
                raise MCPStaleCatalogError("refresh and replace the stale MCP tool inventory")
            self._active[token] = active
        self._emit("call_started", {"name": name, "progressToken": token, "operation": "tools/call"}, active)
        return token, active, {"name": name, "arguments": arguments, "_meta": {"progressToken": token}}

    def _finish(self, token, active, result=None, error=None):
        with self._lock:
            self._active.pop(token, None)
        status = "success"
        if isinstance(error, asyncio.CancelledError):
            status = "cancelled"
        elif error is not None and _protocol_failure(error):
            status = "protocol_failure"
        elif error is not None or (result is not None and result.get("isError", False)):
            status = "execution_failure"
        data = {
            "status": status,
            "success": status == "success",
            "progressToken": token,
            "name": active.name,
            "operation": "tools/call",
            "duration_ms": (time.monotonic() - active.started) * 1000,
        }
        if status != "success":
            self._emit(status, data, active)
        self._emit("call_finished", data, active)

    def _call_steps(self, generation, name, arguments, schema):
        token, active, params = self._admit(generation, name, arguments)
        try:
            response = yield lambda: _response(self.request("tools/call", params), self.asynchronous)
            result = _validate_result(response, schema, self.validate_output)
        except BaseException as error:
            mark_call_outcome(error)
            self._finish(token, active, error=error)
            raise
        self._finish(token, active, result=result)
        return result

    def on_notification(self, method: str, params: Any = None) -> None:
        """Accept host-routed notifications without issuing any requests."""
        data = wire_json({} if params is None else _mapping(params))
        if method == "notifications/progress":
            token, progress, total = data.get("progressToken"), data.get("progress"), data.get("total")
            if not isinstance(token, (str, int)) or isinstance(token, bool):
                raise MCPProtocolError("progress notification requires a valid token")
            with self._lock:
                active = self._active.get(token)
                if active is None:
                    return
                if (
                    type(progress) not in (int, float)
                    or not 0 <= progress <= 1e15
                    or not math.isfinite(progress)
                    or progress <= active.progress
                    or (
                        total is not None
                        and (type(total) not in (int, float) or not progress <= total <= 1e15 or not math.isfinite(total))
                    )
                ):
                    raise MCPProtocolError("progress must increase and have a valid optional total")
                active.progress = progress
            self._emit("progress", {**data, "name": active.name, "operation": "tools/call"}, active)
            return
        required = {
            "notifications/tools/list_changed": ("tools", "listChanged"),
            "notifications/resources/list_changed": ("resources", "listChanged"),
            "notifications/resources/updated": ("resources", "subscribe"),
            "notifications/prompts/list_changed": ("prompts", "listChanged"),
            "notifications/message": ("logging", None),
        }.get(method)
        if required is None or required[0] not in self._capabilities:
            return
        section, flag = required
        if flag is not None and self._capabilities[section].get(flag) is not True:
            return
        if method == "notifications/tools/list_changed":
            with self._lock:
                self._dirty = True
                self._generation += 1
                generation = self._generation
            self._emit("tools_changed", {"generation": generation})
        else:
            self._emit(method.removeprefix("notifications/").replace("/", "."), data)
