"""
providers/base.py
=================

The vendor-neutral domain model and abstract Provider interface.

Nothing in this file imports `openai` or `anthropic` or `mcp` — it's pure
contract. The agent loop will depend on these types; concrete providers
will implement the interface. Swapping providers should never require
changes to anything but this file's subclasses.

Domain types:
    ToolSpec     - one tool, as described to the LLM
    ToolCall     - the LLM's request to invoke a tool
    ChatMessage  - one turn in the conversation
    ChatResponse - what Provider.chat() returns
"""

from __future__ import annotations  # forward refs without quotes (PEP 563-ish)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Tool specs and calls
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolSpec:
    """A tool's definition, in vendor-neutral form.

    Built from an MCP `Tool` by `mcp_agent.mcp_.schema.mcp_tool_to_spec()`,
    then handed to a Provider which translates it into the vendor's tool
    format (OpenAI's nested `function`, Anthropic's flat `input_schema`, ...).

    Attributes:
        name: Tool identifier. Must match across MCP and the LLM round-trip.
        description: Human-readable; the LLM reads this to decide when to use
            the tool. Quality of the description directly affects tool-use
            quality.
        input_schema: JSON Schema describing the tool's arguments. This is
            the same shape both MCP and LLM vendors use, so the translator
            usually copies it verbatim under a different key.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """The LLM has decided to call a tool.

    Returned inside a ChatResponse when the model issues tool calls instead
    of (or alongside) a text reply. The agent executes the call via MCP and
    then sends a follow-up ChatMessage with role="tool" to feed the result
    back to the LLM.

    Attributes:
        id: Provider-assigned identifier. Opaque to us, but critical: the
            agent must echo this id back in the corresponding tool-result
            ChatMessage so the LLM can pair request with response.
        name: Which tool the LLM wants to call (matches a ToolSpec.name).
        arguments: Parsed arguments dict. OpenAI returns these as a JSON
            string and Anthropic returns them as a dict; the provider
            normalizes to dict here so the agent loop never has to parse.
    """

    id: str
    name: str
    arguments: dict[str, Any]


# ---------------------------------------------------------------------------
# Chat messages and responses
# ---------------------------------------------------------------------------


# The four roles in a chat conversation. Constraining this with Literal means
# typos like "asistant" become type errors instead of silent runtime bugs.
Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    """One turn in the conversation history.

    Five message shapes the agent will actually use:

    1. System prompt:
           ChatMessage(role="system", content="You are a helpful agent...")

    2. User question:
           ChatMessage(role="user", content="What does my README say?")

    3. Assistant reply (pure text — final answer):
           ChatMessage(role="assistant", content="The README says...")

    4. Assistant reply (tool calls — model wants tools executed):
           ChatMessage(role="assistant", content=None,
                       tool_calls=[ToolCall(id="call_123", name="read_file", ...)])

    5. Tool result (we fed an executed tool's output back to the model):
           ChatMessage(role="tool", content="<file contents>",
                       tool_call_id="call_123")

    Some assistant messages have both content AND tool_calls (model says
    "Let me check..." while also issuing a tool call). Our shape supports it.

    Attributes:
        role: Who's speaking.
        content: Text content of the message, if any.
        tool_calls: Set when an assistant message includes tool requests.
            Use a default_factory list (not `= []`) to avoid the shared
            mutable default pitfall.
        tool_call_id: Set on role="tool" messages to identify which
            ToolCall this is the result of.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass(slots=True)
class ChatResponse:
    """What Provider.chat() returns.

    Attributes:
        text: The model's text output, if any. None when the model only
            wanted to issue tool calls (this happens — strict models
            sometimes return zero text and one tool call).
        tool_calls: Empty list if the model produced no tool calls.
        finish_reason: Why the model stopped generating. Useful for
            distinguishing "answered completely" from "ran out of tokens"
            from "wants tools." Vendor-specific values normalized to:
                "stop"       - finished a text reply
                "tool_calls" - wants tools executed
                "length"     - hit max_tokens (response truncated)
                "other"      - anything else / unknown
        usage: Optional token-accounting dict. Populated when the vendor
            response includes it. Keys: "prompt_tokens", "completion_tokens",
            "total_tokens".
    """

    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Literal["stop", "tool_calls", "length", "other"] = "stop"
    usage: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# The Provider contract
# ---------------------------------------------------------------------------


class Provider(ABC):
    """Abstract LLM provider.

    Concrete providers (OpenAIProvider, AnthropicProvider) inherit from this
    and implement `chat()`. The agent loop depends only on this interface.

    Construction is provider-specific (different SDKs, different config),
    so __init__ is not part of the contract — each subclass takes whatever
    it needs (API key, model name, base URL, etc.).
    """

    # Subclasses set this so logs/evals can record which model was used.
    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the response.

        Args:
            messages: Conversation history. The last message is usually
                role="user" (first turn) or role="tool" (after a tool call).
            tools: Available tools. If None or empty, the model can only
                produce text.

        Returns:
            ChatResponse with either text, tool_calls, or both.

        Raises:
            Exceptions from the underlying SDK are allowed to propagate.
            The agent loop handles retries/timeouts at a higher level
            (Phase 2.2).
        """
        ...
