# mcp-agent

A model-agnostic agent that uses **Model Context Protocol (MCP)** for all tool calling.

**Status:** 🚧 Active development. Stage 1 of a multi-stage GenAI portfolio project.

## What it is

`mcp-agent` is a small, readable agent loop that:

- Connects to any MCP-compliant tool server (community servers like `filesystem` and `fetch`, plus a custom server wrapping [docs-rag](https://github.com/hbhimava/docs-rag)).
- Talks to LLMs through a thin provider abstraction so the same agent works against OpenAI (implemented) or Anthropic (interface defined, implementation deferred).
- Runs a tool-use loop with iteration, parallel tool calls, and streaming output.

The goal is to demonstrate the **interface layer** that's becoming standard for tool use, rather than wrapping any one vendor's SDK.

## Why MCP?

Before MCP, every agent framework reinvented "how does an LLM call a tool." MCP (introduced by Anthropic in late 2024, now an open standard with SDKs in Python, TypeScript, C#, Java, and Ruby) lets tool providers speak one protocol and any MCP-aware client use them. Build the client once, get the whole ecosystem.

## Status

| Phase | Description | State |
|-------|-------------|-------|
| 1 | Scaffold, MCP concepts, hello-world client, provider abstraction | ✅ Done |
| 2 | Naive agent → real loop with iteration + parallel tool calls → streaming + CLI | ⏳ Planned |
| 3 | Custom MCP server wrapping `docs-rag` | ⏳ Planned |
| 4 | Eval harness, comparison metrics, README polish | ⏳ Planned |
| 5 | Dockerize, GitHub Actions CI | ⏳ Planned |

See [docs-rag](https://github.com/hbhimava/docs-rag) for the companion project this builds on.

## License

MIT — see [LICENSE](./LICENSE).
