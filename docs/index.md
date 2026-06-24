# Documentation Index

Complete documentation for the **Autonomous ML Research Engineer** v2.0 — a multi-agent platform that automates the full ML research lifecycle.

> **Current state:** 15/15 phases complete · 23 agents · 61 tools · 186 Pydantic models · 56 CLI commands · 878 tests passing.

---

## Start here

| Document | Audience | Description |
|----------|----------|-------------|
| [**Quick Start**](QUICKSTART.md) | Everyone | Install, first run, 5-minute tour of every phase. |
| [**README**](../README.md) | Everyone | Project overview, capabilities, badges, demo workflows. |

## Reference

| Document | Audience | Description |
|----------|----------|-------------|
| [**Architecture**](architecture.md) | Engineers, architects | High-level architecture, 15-phase pipeline, component layers, data flow. |
| [**System Design**](system_design.md) | Engineers | Detailed design: domain models, tool contracts, storage schema, enums, error handling, testing strategy. |
| [**Agents**](agents.md) | Engineers | Deep-dive on all 23 agents: responsibilities, constructors, workflows, LLM wiring. |
| [**Tools**](tools.md) | Engineers, contributors | Reference for all 61 typed tools: input/output models, key logic. |
| [**Models**](models.md) | Engineers | Reference for all 186 Pydantic models grouped by phase. |
| [**Memory System**](memory_system.md) | Engineers | Memory types, retrieval strategies, knowledge graph, vector store, repository memory (Phase 12). |
| [**Storage Schema**](storage_schema.md) | Engineers, DBAs | All SQLite tables, columns, relationships, output directory layout. |
| [**CLI Reference**](cli_reference.md) | Users, engineers | All 56 CLI commands with flags and examples. |
| [**LLM Integration**](llm_integration.md) | GenAI engineers | Provider-agnostic LLM layer, Ollama Cloud, per-agent routing, 23-agent config. |

## Guides

| Document | Audience | Description |
|----------|----------|-------------|
| [**Roadmap**](roadmap.md) | Maintainers, contributors | Versioned roadmap — v1.0 through v2.0 and beyond. |
| [**Contributing**](contributing.md) | Contributors | Setup, conventions, how to add providers/tools/agents, PR checklist. |

---

## Conventions used across these docs

- **Phase N** refers to the fifteen-phase architecture (1 = paper analysis … 15 = research workflows).
- All code blocks are Python 3.12+ unless noted.
- All models are Pydantic v2; all enums are `StrEnum`.
- "Patch-first" means code changes are produced as reviewable unified diffs, never applied silently.
- "Routed" (in agent tables) means the agent's LLM provider is resolved by `ModelRouter` from `llm_config.yaml`.

## Verification commands

```bash
uv run pytest -q                              # 878 tests
uv run mypy src/research_engineer/llm         # type-check the LLM layer
uv run ruff check .                            # lint
research-engineer llm status                   # inspect provider/model routing
```
