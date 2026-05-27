"""MCP integration: client management and schema translation.

`schema` translates MCP's `Tool` type into our vendor-neutral `ToolSpec`.
Future modules in this sub-package will own the client session lifecycle
(connecting to one or many MCP servers, executing tool calls).

The trailing underscore in the module name avoids shadowing the official
`mcp` SDK package — `import mcp` should always reach the SDK, not us.
"""
