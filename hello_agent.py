"""
hello_agent.py
==============

End-to-end demo: an LLM (gpt-4o-mini) answers a user question by deciding
which MCP tool(s) to call, executing them, and producing a final answer.

This is the first script that actually exercises the full picture:

    user question
        |
        v
    [our agent code]  <--->  OpenAI (decides which tool)
        |
        v
    [our agent code]  <--->  MCP filesystem server (executes tool)
        |
        v
    final answer

Differences from a "real" agent (those come in Phase 2):
  - No CLI: the user question is hardcoded.
  - No streaming: we wait for each LLM response in full.
  - No parallel tool execution: if the LLM requests 3 tools, we run them
    sequentially. (The protocol supports parallel; we'll add that later.)
  - Tool-call rounds capped at 5 (so a buggy model can't burn $$$ in a loop).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_agent.mcp_.schema import mcp_tools_to_specs
from mcp_agent.providers.base import ChatMessage, ToolCall
from mcp_agent.providers.openai_provider import OpenAIProvider

# Load OPENAI_API_KEY (and any other env vars) from .env into os.environ.
# Must run before instantiating OpenAIProvider — the OpenAI SDK reads
# OPENAI_API_KEY at construction time.
load_dotenv()


# The directory the MCP filesystem server is allowed to access. Same as
# hello_mcp.py — we point it at the repo root so the model can read README.md
# (and any other files in the project).
ALLOWED_DIR = Path(__file__).parent.resolve()

# The question we're putting to the agent. Hardcoded for now; Phase 2 will
# make this a CLI argument.
USER_QUESTION = (
    "Read the README.md in the current project directory and summarize, in "
    "3 sentences, what the project is for and what it currently supports."
)

# Safety cap on tool-calling rounds. One "round" = the LLM emits N tool calls,
# we execute them all, and feed results back. 5 rounds is generous for this
# demo; in real agents you'd tune this per use case.
MAX_TOOL_ROUNDS = 5


async def execute_tool_call(session: ClientSession, call: ToolCall) -> str:
    """Run one ToolCall against the MCP session and return a string result.

    The agent loop needs string results so it can put them inside a
    ChatMessage(role="tool", content=...). MCP returns structured content
    blocks (text/image/resource); we flatten to text here.

    Errors are converted to text starting with "ERROR:" rather than raised —
    that way the model sees the failure and can recover (try a different
    path, give up gracefully, etc.) instead of the program crashing.
    """
    print(f"  → tool: {call.name}({json.dumps(call.arguments)})")
    try:
        result = await session.call_tool(call.name, arguments=call.arguments)
    except Exception as e:
        # Protocol-level failure (network, invalid args, etc.). Tell the model.
        return f"ERROR: tool call raised {type(e).__name__}: {e}"

    # Concatenate all text content blocks; ignore non-text blocks for now.
    pieces: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            pieces.append(text)
    text_output = "\n".join(pieces) if pieces else "(no text content)"

    if result.isError:
        # Tool-logic failure (file not found, bad path, etc.). Prefix so the
        # model can pattern-match on it and react.
        return f"ERROR: {text_output}"

    # Truncate over-long results so we don't blow the context window with
    # one file read. In real agents this gets more sophisticated (smart
    # truncation, summarization, etc.). 4000 chars is a reasonable cap.
    if len(text_output) > 4000:
        text_output = text_output[:4000] + f"\n... [truncated {len(text_output) - 4000} chars]"
    return text_output


async def main() -> None:
    """Top-level coroutine."""

    # ---- 1. Boot the MCP server (same pattern as hello_mcp.py) -----------
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(ALLOWED_DIR),
        ],
        env=None,
    )

    # ---- 2. Boot the LLM provider ----------------------------------------
    # gpt-4o-mini is the default (set in OpenAIProvider's __init__). Cheap
    # and good enough for tool use; you can override with the
    # MCP_AGENT_OPENAI_MODEL env var if you want to try gpt-4o or others.
    provider = OpenAIProvider()
    print(f"→ Provider: {provider.name} (model: {provider.model})")
    print(f"→ MCP server: filesystem, sandboxed to {ALLOWED_DIR}\n")

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        # ---- 3. Initialize MCP + fetch tool catalog --------------------
        await session.initialize()
        tools_result = await session.list_tools()
        tool_specs = mcp_tools_to_specs(tools_result.tools)
        print(f"✓ {len(tool_specs)} tools available to the model\n")

        # ---- 4. Seed the conversation ----------------------------------
        # System prompt: short, points at the tools, sets expectations.
        # Real agents use much longer prompts; this is the minimum viable.
        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You are an assistant with access to filesystem tools "
                    "via Model Context Protocol. When the user asks about "
                    f"files, use the tools to read them. The current working "
                    f"directory is {ALLOWED_DIR}. Always use absolute paths."
                ),
            ),
            ChatMessage(role="user", content=USER_QUESTION),
        ]

        print(f"→ User: {USER_QUESTION}\n")

        # ---- 5. The agent loop -----------------------------------------
        # Repeatedly: call the LLM; if it wants tools, run them; otherwise
        # we have a final answer. Capped at MAX_TOOL_ROUNDS to prevent
        # runaway tool calling on a misbehaving model.
        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            print(f"--- LLM round {round_num} ---")
            response = await provider.chat(messages, tools=tool_specs)

            # Log what happened in this round.
            if response.usage:
                print(
                    f"  usage: prompt={response.usage['prompt_tokens']}, "
                    f"completion={response.usage['completion_tokens']}, "
                    f"total={response.usage['total_tokens']}"
                )
            print(f"  finish_reason: {response.finish_reason}")
            if response.text:
                # Show a preview if the assistant produced any text in this turn.
                # (Models sometimes interleave a "let me check..." string with
                # tool calls.)
                preview = response.text[:200]
                print(f"  text: {preview}{'...' if len(response.text) > 200 else ''}")

            # Always append the assistant's message to the history — even if
            # it's tool-calls-only. This is how the LLM knows in the NEXT
            # round which tool calls it just made.
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            # No tool calls = we have our final answer. Print and exit.
            if not response.tool_calls:
                print("\n=== FINAL ANSWER ===")
                print(response.text or "(model returned no text)")
                return

            # ---- Execute each tool call, append results --------------
            for call in response.tool_calls:
                result_text = await execute_tool_call(session, call)
                # Preview the result so the user can follow along.
                first_line = result_text.split("\n")[0][:140]
                print(f"    ← {first_line}{'...' if len(result_text) > len(first_line) else ''}")
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result_text,
                        tool_call_id=call.id,
                    )
                )

            print()  # blank line between rounds

        # If we fall out of the loop without returning, we hit the round cap.
        print(
            f"\n⚠ Hit MAX_TOOL_ROUNDS ({MAX_TOOL_ROUNDS}) without a final "
            "answer. Either the task is too complex for this cap, or the "
            "model is looping."
        )


if __name__ == "__main__":
    asyncio.run(main())
