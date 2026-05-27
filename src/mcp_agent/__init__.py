"""
mcp-agent: a model-agnostic agent built on Model Context Protocol.

This `__init__.py` only hosts the CLI entry point declared in pyproject.toml's
`[project.scripts]`. The real agent code lives in submodules:

    mcp_agent.providers   - LLM provider abstraction (OpenAI, Anthropic)
    mcp_agent.mcp_        - MCP client + schema bridge
    mcp_agent.agent       - the agent loop (Phase 2)
    mcp_agent.cli         - the CLI (Phase 2)
"""

__version__ = "0.1.0"


def main() -> None:
    """Placeholder CLI entry point.

    Phase 1: prints a status message. Phase 2 will replace this with a real
    CLI that parses args, loads config, and runs the agent loop.
    """
    print("mcp-agent v0.1.0 (Phase 1 scaffolding)")
    print("Try: `uv run python hello_agent.py` for the current demo.")
