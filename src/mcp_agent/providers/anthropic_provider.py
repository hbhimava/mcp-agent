"""
providers/anthropic_provider.py
===============================

Anthropic Claude implementation of the Provider interface.

STATUS: stubbed. Interface defined, methods raise NotImplementedError until
Anthropic API credits are acquired and the implementation is filled in.

The shape of the eventual implementation:
    - import anthropic and use anthropic.AsyncAnthropic
    - _spec_to_anthropic:  ToolSpec -> {"name", "description", "input_schema"}
      (no "function" wrapping, "input_schema" not "parameters")
    - _message_to_anthropic: ChatMessage -> Anthropic's content-block format,
      which is structurally different from OpenAI's: assistant tool calls
      are "tool_use" content blocks, tool results are "tool_result" content
      blocks. Multi-block content is the norm, not the exception.
    - _response_to_chat_response: parse content blocks, separate text from
      tool_use blocks, map stop_reason ("end_turn" -> "stop", "tool_use" ->
      "tool_calls", "max_tokens" -> "length").

The OpenAI provider is the working reference. When implementing this one,
the goal is identical inputs/outputs in our domain types — the test for
correctness is "the agent loop works with either provider unchanged."
"""

from __future__ import annotations

import os

from mcp_agent.providers.base import (
    ChatMessage,
    ChatResponse,
    Provider,
    ToolSpec,
)

_STUB_MSG = (
    "AnthropicProvider is stubbed. Implementation deferred until Anthropic "
    "API credits are acquired. The interface is defined; only `chat()` and "
    "the private translation helpers need to be filled in. See "
    "providers/openai_provider.py for a working reference."
)


class AnthropicProvider(Provider):
    """Provider that will call Anthropic's Messages API. Not yet implemented."""

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # We deliberately accept (and remember) the same arguments the real
        # implementation will need, so callers wiring up providers don't have
        # to change their code when the stub becomes real.
        self.model = model or os.environ.get("MCP_AGENT_ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self._api_key = api_key  # not used yet; stored for future _client init

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResponse:
        raise NotImplementedError(_STUB_MSG)
