"""
mcp_/schema.py
==============

Translators between MCP SDK types and our vendor-neutral domain types.

Right now there's only one direction (MCP -> our ToolSpec). Tool call
results travel in the opposite direction but they're plain strings/dicts,
not specialized types — they don't need a translator yet.
"""

from __future__ import annotations

from mcp.types import Tool as MCPTool

from mcp_agent.providers.base import ToolSpec


def mcp_tool_to_spec(tool: MCPTool) -> ToolSpec:
    """Convert one MCP Tool into our ToolSpec.

    MCP's `inputSchema` and our `input_schema` are both JSON Schema dicts
    with the same shape — the translation is a rename plus a safe default
    if the server returned None for the description.

    Args:
        tool: The mcp.types.Tool object returned by session.list_tools().

    Returns:
        A ToolSpec with the same data, ready to be handed to a Provider.
    """
    return ToolSpec(
        name=tool.name,
        # Description is technically optional in MCP; default to empty
        # string so the Provider never has to handle None.
        description=tool.description or "",
        # `inputSchema` is camelCase because MCP uses JSON-RPC conventions;
        # we use snake_case in our domain because we're Python.
        input_schema=tool.inputSchema,
    )


def mcp_tools_to_specs(tools: list[MCPTool]) -> list[ToolSpec]:
    """Convenience: convert a list of MCP tools at once."""
    return [mcp_tool_to_spec(t) for t in tools]
