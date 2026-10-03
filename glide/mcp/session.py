"""Adapt an initialized MCP SDK session to the bridge's `request(method, params)` callback.

Nothing here imports the SDK at module level, opens a transport or initializes the session; the caller owns
the session. Written against MCP Python SDK 1.30 and tolerant of the 2.x signature changes (params-only
`list_tools`, float `read_timeout_seconds`). Only the `tools/list` and `tools/call` methods are mapped.
"""

import copy
import inspect
from collections import deque
from collections.abc import Callable
from datetime import timedelta
from typing import Any


def _default_page_params(cursor: str):
    try:
        from mcp.types import PaginatedRequestParams  # only reached by a params-only SDK, which has it
    except ImportError:
        raise TypeError("this SDK needs a native page_params factory for pagination") from None
    return PaginatedRequestParams(cursor=cursor)


def _timeout_for(parameter: inspect.Parameter, value: Any) -> Any:
    """Hand the SDK the timeout type its signature names (timedelta in 1.x, seconds in 2.x)."""
    annotation = str(parameter.annotation)
    if "timedelta" in annotation and isinstance(value, (int, float)) and not isinstance(value, bool):
        return timedelta(seconds=value)
    if "timedelta" not in annotation and "float" in annotation and isinstance(value, timedelta):
        return value.total_seconds()
    return value


def session_request(
    session: Any,
    *,
    on_notification: Callable[[str, dict], Any] | None = None,
    page_params: Callable[[str], Any] | None = None,
    read_timeout_seconds: Any = None,
) -> Callable:
    """Map MCPBridge requests to async list_tools/call_tool session methods.

    SDK 1.x cursor pagination is supported directly. For a params-only SDK,
    supply a factory returning its native PaginatedRequestParams for each cursor.
    Notification handlers for logging/resources/list changes remain host-owned.
    """
    if not callable(getattr(session, "list_tools", None)) or not callable(getattr(session, "call_tool", None)):
        raise TypeError("an initialized asynchronous MCP session is required")
    listing = inspect.signature(session.list_tools).parameters
    calling = inspect.signature(session.call_tool).parameters
    notification_errors: deque[dict[str, str]] = deque(maxlen=128)

    async def request(method: str, params: dict):
        if not isinstance(params, dict):
            raise TypeError("request params must be a dictionary")
        if method == "tools/list":
            cursor = params.get("cursor")
            if cursor is None:
                return await session.list_tools()
            if "cursor" in listing:
                return await session.list_tools(cursor=cursor)
            if "params" in listing:
                return await session.list_tools(params=(page_params or _default_page_params)(cursor))
            raise TypeError("this SDK's list_tools cannot take a pagination cursor")
        if method != "tools/call":
            raise ValueError("session adapter supports only tools/list and tools/call")
        kwargs = {}
        metadata = copy.deepcopy(params.get("_meta", {}))
        if metadata:
            if "meta" not in calling:
                raise TypeError("this SDK cannot accept per-call metadata")
            kwargs["meta"] = metadata
        if on_notification is not None and "progress_callback" in calling:
            token = metadata.get("progressToken")

            async def progress(value, total=None, message=None):
                data = {"progressToken": token, "progress": value}
                if total is not None:
                    data["total"] = total
                if message is not None:
                    data["message"] = message
                try:
                    result = on_notification("notifications/progress", data)
                    if inspect.isawaitable(result):
                        await result
                except Exception as error:
                    # Progress is observability; a handler failure must not hide a completed tool effect.
                    # Cancellation still propagates because it inherits BaseException.
                    notification_errors.append({"method": "notifications/progress", "error": type(error).__name__})

            kwargs["progress_callback"] = progress
        if read_timeout_seconds is not None:
            if "read_timeout_seconds" not in calling:
                raise TypeError("this SDK does not expose read_timeout_seconds")
            kwargs["read_timeout_seconds"] = _timeout_for(calling["read_timeout_seconds"], read_timeout_seconds)
        return await session.call_tool(params["name"], copy.deepcopy(params.get("arguments", {})), **kwargs)

    request.notification_errors = notification_errors
    return request
