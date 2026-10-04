---
{
  "id": "mcp-demo",
  "description": "An opt-in namespace for the tools of one MCP server, here named demo.",
  "skills": [],
  "tools": ["mcp:demo/lookup"]
}
---
This manifest only enables tool ids; it cannot connect a server or run a tool. The ids are
`mcp:SERVER/TOOL`, the names `glide.mcp.MCPBridge` gives the tools of a connected server. Replace
`demo/lookup` with your server's name and tools when creating your own plugin.
