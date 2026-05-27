"""LLM provider abstraction.

`base` defines the vendor-neutral Provider interface and the domain types
(ToolSpec, ToolCall, ChatMessage, ChatResponse) that the agent loop uses.

Concrete providers translate between those domain types and a vendor's SDK:
    openai_provider.OpenAIProvider     - implemented
    anthropic_provider.AnthropicProvider - stubbed (NotImplementedError)
"""
