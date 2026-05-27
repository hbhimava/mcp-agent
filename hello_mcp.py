"""
hello_mcp.py
============

A minimal MCP client. Launches the official filesystem MCP server as a
subprocess, performs the protocol handshake, lists the tools the server
exposes, and calls one of them.

No LLM involved — this is purely "can we speak MCP to a real server?"

Run with:
    uv run python hello_mcp.py

What you should see (roughly):
    1. A list of ~10 tools the filesystem server exposes (read_text_file,
       write_file, list_directory, etc.) with their descriptions and schemas.
    2. The contents of README.md, fetched via the `read_text_file` tool.
"""

import asyncio
import json
from pathlib import Path

# --- MCP SDK imports --------------------------------------------------------
#
# `ClientSession` is the high-level object representing a connection to one
# MCP server. It owns the protocol state machine and exposes the methods we
# care about: initialize(), list_tools(), call_tool(), etc.
#
# `StdioServerParameters` is a small dataclass describing HOW to launch the
# server subprocess: the executable, its arguments, and any environment
# variables to pass. (For stdio transport — there's a separate setup for
# HTTP transport, which we'll meet later.)
#
# `stdio_client` is an async context manager that takes those parameters,
# spawns the subprocess, and yields a (read_stream, write_stream) pair
# wired up to the subprocess's stdout/stdin. The SDK handles JSON-RPC
# framing on those streams so we never see raw protocol messages.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Where the filesystem server is allowed to read/write. The filesystem
# server takes one or more directory arguments and refuses any path that
# escapes them — this sandbox is the whole reason it's safe to expose
# `read_file` to an LLM at all.
#
# We point it at the repo root. `.resolve()` turns it into an absolute path
# (the server requires absolute paths).
ALLOWED_DIR = Path(__file__).parent.resolve()


async def main() -> None:
    """Top-level coroutine. Run with asyncio.run(main())."""

    # ---- Step 1: describe how to launch the server -----------------------
    #
    # We're going to run:
    #     npx -y @modelcontextprotocol/server-filesystem <ALLOWED_DIR>
    #
    # `npx -y` will silently download the npm package on first use and cache
    # it in ~/.npm/_npx/. Subsequent runs are fast.
    #
    # Note: `args` is a list of strings, not a single shell command. The SDK
    # spawns the subprocess directly (not through a shell), which is safer
    # — no string interpolation means no shell-injection risk if any of the
    # args came from user input.
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(ALLOWED_DIR),
        ],
        # `env=None` means "inherit the parent's environment." If we wanted
        # to pass secrets only to the server (e.g. an API key for some other
        # server) we'd populate this dict explicitly.
        env=None,
    )

    print(f"→ Launching filesystem server, sandboxed to: {ALLOWED_DIR}\n")

    # ---- Step 2: open the transport + the session ------------------------
    #
    # One `async with`, two context managers. Equivalent to nesting two
    # `async with` blocks; the parenthesized form (PEP 617, Python 3.10+)
    # is the modern idiom and saves a level of indentation.
    #
    # Lifetime: the outer manager (stdio_client) owns the subprocess; the
    # inner one (ClientSession) owns the protocol state. Listed in this
    # order, they're entered top-to-bottom and exited bottom-to-top — so
    # the session shuts down cleanly before the transport closes.
    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        # ---- Step 3: initialize -------------------------------------
        #
        # The MCP handshake. Client and server exchange:
        #   - Protocol version (so we can fail fast on incompatible peers)
        #   - Capabilities (what each side supports — sampling, roots,
        #     resource subscription, etc. We care about almost none of
        #     this for our use case.)
        #   - Server info (name, version — useful for logging)
        #
        # Until `initialize()` completes, you can't make any other calls.
        init_result = await session.initialize()
        print(
            f"✓ Connected to: {init_result.serverInfo.name} "
            f"v{init_result.serverInfo.version}"
        )
        print(f"  Protocol version: {init_result.protocolVersion}\n")

        # ---- Step 4: list tools -------------------------------------
        #
        # Ask the server what tools it exposes. Returns a list of `Tool`
        # objects, each with:
        #   - .name: str               (e.g. "read_text_file")
        #   - .description: str        (human-readable, used by the LLM)
        #   - .inputSchema: dict       (JSON Schema for the arguments)
        #
        # The `.description` is what the LLM actually reads to decide
        # when to use a tool. Server authors who write good descriptions
        # get good tool use. Bad descriptions → confused agents.
        tools_result = await session.list_tools()
        print(f"✓ Server exposes {len(tools_result.tools)} tools:\n")
        for tool in tools_result.tools:
            # Truncate description to one line for the overview.
            desc_oneline = (tool.description or "").split("\n")[0]
            print(f"  • {tool.name}")
            print(f"      {desc_oneline}")
        print()

        # Print the full schema of the first tool, so you can see what
        # the LLM will eventually receive when we wire one up.
        if tools_result.tools:
            first = tools_result.tools[0]
            print(f"--- Full schema of `{first.name}` ---")
            print(json.dumps(first.inputSchema, indent=2))
            print()

        # ---- Step 5: actually call a tool ---------------------------
        #
        # We'll call `read_text_file` on README.md. The exact tool name
        # may vary by server version — older versions exposed `read_file`,
        # newer ones split it into `read_text_file` / `read_media_file`.
        # We pick whichever is available.
        target_tool = None
        for candidate in ("read_text_file", "read_file"):
            if any(t.name == candidate for t in tools_result.tools):
                target_tool = candidate
                break

        if target_tool is None:
            print(
                "⚠ Neither `read_text_file` nor `read_file` is available "
                "on this server. Skipping the tool-call demo."
            )
            return

        # `call_tool` takes the tool name and a dict of arguments matching
        # the tool's inputSchema. The server validates the args against
        # the schema and returns a `CallToolResult`.
        #
        # The result's `.content` is a list of content blocks (text,
        # image, embedded resource). For a text file we expect one
        # TextContent block.
        readme_path = str(ALLOWED_DIR / "README.md")
        print(f"→ Calling tool `{target_tool}` with path={readme_path}\n")

        result = await session.call_tool(
            target_tool,
            arguments={"path": readme_path},
        )

        # `isError` is set by the server when the tool errors out (e.g.
        # file not found). Distinguish protocol errors (which would
        # have raised) from tool-logic errors (which come back here).
        if result.isError:
            print("✗ Tool returned an error:")
            for block in result.content:
                print(f"  {getattr(block, 'text', block)}")
            return

        # Print the first content block. In real client code we'd loop
        # and handle each block type (text/image/resource).
        print("✓ Tool result (first 500 chars):\n")
        for block in result.content:
            # Each block is one of several types; TextContent has .text.
            text = getattr(block, "text", None)
            if text is not None:
                print(text[:500])
                if len(text) > 500:
                    print(f"\n... [truncated, {len(text) - 500} more chars]")
                break


if __name__ == "__main__":
    # asyncio.run() creates an event loop, runs main() on it, and closes
    # the loop when main() returns. The standard "I have one async entry
    # point" pattern.
    asyncio.run(main())
