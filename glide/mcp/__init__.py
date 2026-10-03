"""Glide as an MCP server and as an MCP client. Optional: the MCP SDK is needed only for real transports.

Importing this package imports no SDK, opens nothing and starts nothing.

- `server`: `GlideMCPServer`, a small JSON-RPC core that serves tools the host registers (no SDK, no sockets).
- `client`: `MCPClient`, the minimal discover-and-call interface, and `SessionClient`, which raises only
  `ProviderError`.
- `bridge`: `MCPBridge`, which turns a connected server's tools into harness tools for a model turn.
- `session`: `session_request`, which adapts an initialized SDK session for the bridge and the client.
- `transport`: `open_stdio_client`, the one place a server process is started (needs the SDK and a user yes).
"""

from .bridge import MCPBridge, MCPProtocolError, MCPStaleCatalogError, bind_mcp, normalize_mcp_result
from .client import MCPCallError, MCPClient, RemoteTool, SessionClient
from .config import McpServerSpec, McpSettings
from .server import GlideMCPServer, ServerTool, memory_tools, serve_stream
from .session import session_request

__all__ = [
    "GlideMCPServer",
    "MCPBridge",
    "MCPCallError",
    "MCPClient",
    "MCPProtocolError",
    "MCPStaleCatalogError",
    "McpServerSpec",
    "McpSettings",
    "RemoteTool",
    "ServerTool",
    "SessionClient",
    "bind_mcp",
    "memory_tools",
    "normalize_mcp_result",
    "serve_stream",
    "session_request",
]
