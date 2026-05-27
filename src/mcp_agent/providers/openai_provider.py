"""
providers/openai_provider.py
============================

The OpenAI implementation of the Provider interface.

This is the only file in the package that imports `openai`. If you wanted
to replace the OpenAI SDK with raw HTTP calls, all the damage is contained
here.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from mcp_agent.providers.base import (
    ChatMessage,
    ChatResponse,
    Provider,
    ToolCall,
    ToolSpec,
)


class OpenAIProvider(Provider):
    """Provider that calls OpenAI's chat completions endpoint."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create an OpenAI provider.

        Args:
            model: Model identifier (e.g. "gpt-4o-mini"). If None, read from
                MCP_AGENT_OPENAI_MODEL env var, defaulting to "gpt-4o-mini".
            api_key: API key. If None, the OpenAI SDK reads it from the
                OPENAI_API_KEY env var (which is what we want — keeps secrets
                out of the constructor's call sites).
        """
        self.model = model or os.environ.get("MCP_AGENT_OPENAI_MODEL", "gpt-4o-mini")
        # AsyncOpenAI() with no api_key arg reads OPENAI_API_KEY automatically.
        # We let it do that rather than forwarding the env var ourselves —
        # that way a user passing api_key=... explicitly still works as expected.
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    # ---------------------------------------------------------------------
    # The Provider contract
    # ---------------------------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResponse:
        """Send messages + (optional) tools to OpenAI, return a ChatResponse."""

        # ---- outbound translation: Language B -> Language C ----
        openai_messages = [self._message_to_openai(m) for m in messages]
        openai_tools = [self._spec_to_openai(t) for t in tools] if tools else None

        # Build the request kwargs. We omit `tools` entirely when empty —
        # OpenAI's API rejects `tools=[]` with a validation error.
        request: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }
        if openai_tools:
            request["tools"] = openai_tools
            # `tool_choice="auto"` is the default but we set it explicitly
            # to document intent. Other values: "none" (force no tools),
            # "required" (force at least one tool), or a specific tool name.
            request["tool_choice"] = "auto"

        # ---- the actual API call ----
        completion = await self._client.chat.completions.create(**request)

        # ---- inbound translation: Language C -> Language B ----
        return self._completion_to_response(completion)

    # ---------------------------------------------------------------------
    # Translation helpers (private)
    # ---------------------------------------------------------------------

    @staticmethod
    def _spec_to_openai(spec: ToolSpec) -> dict[str, Any]:
        """ToolSpec -> OpenAI's tool format.

        OpenAI wraps the tool in {"type": "function", "function": {...}} and
        renames `input_schema` to `parameters`. Everything else is verbatim.
        """
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

    @staticmethod
    def _message_to_openai(msg: ChatMessage) -> dict[str, Any]:
        """ChatMessage -> OpenAI message dict.

        OpenAI's schema varies by role:
          - system/user: {role, content}
          - assistant with text only: {role, content}
          - assistant with tool calls: {role, content (may be null), tool_calls}
          - tool: {role, tool_call_id, content}
        """
        if msg.role == "tool":
            # Tool result messages MUST have tool_call_id and content (string).
            if msg.tool_call_id is None:
                raise ValueError("ChatMessage(role='tool') requires tool_call_id")
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content or "",
            }

        out: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            out["content"] = msg.content
        if msg.tool_calls:
            # Assistant message with tool calls: content may be null,
            # tool_calls is a list of OpenAI's tool_call shape.
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # OpenAI expects arguments as a JSON STRING, not a dict.
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    @staticmethod
    def _completion_to_response(completion: Any) -> ChatResponse:
        """OpenAI ChatCompletion -> ChatResponse."""
        choice = completion.choices[0]
        msg = choice.message

        # Parse tool calls if present. OpenAI gives us a list of objects with
        # .id, .function.name, .function.arguments (a JSON string we parse).
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # Models occasionally produce malformed JSON. Surface
                    # the raw string so the agent loop can decide whether
                    # to retry, fail, or feed the error back to the model.
                    args = {"__raw_arguments__": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        # Normalize finish_reason.
        raw_reason = choice.finish_reason
        if raw_reason in ("stop", "length", "tool_calls"):
            finish: Any = raw_reason
        else:
            finish = "other"

        # Token accounting.
        usage = None
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        return ChatResponse(
            text=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
        )
