# Documentation Index

Complete documentation for the **Autonomous ML Research Engineer** v1.0 — a multi-agent platform that automates the full ML research lifecycle.

> **Current state:** 10/10 phases complete · 9 agents · 51 tools · 207 Pydantic models · 49 CLI commands · 690 tests passing.

---

## Start here

| Document | Audience | Description |
|----------|----------|-------------|
| [**Quick Start**](QUICKSTART.md) | Everyone | Install, first run, 5-minute tour of every phase. |
| [**README**](../README_v1.md) | Everyone | Project overview, capabilities, badges, demo workflows. |

## Reference

| Document | Audience | Description |
|----------|----------|-------------|
| [**Architecture**](architecture.md) | Engineers, architects | High-level architecture, phase pipeline, component layers, data flow. |
| [**System Design**](system_design.md) | Engineers | Detailed design: domain models, tool contracts, storage schema, enums, error handling, testing strategy. |
| [**Agents**](agents.md) | Engineers | Deep-dive on all 9 agents: responsibilities, constructors, workflows, LLM wiring. |
| [**Tools**](tools.md) | Engineers, contributors | Reference for all 51 typed tools: input/output models, key logic. |
| [**Models**](models.md) | Engineers | Reference for all 207 Pydantic models grouped by phase. |
| [**Memory System**](memory_system.md) | Engineers | Memory types, retrieval strategies, knowledge graph, vector store. |
| [**Storage Schema**](storage_schema.md) | Engineers, DBAs | All SQLite tables, columns, relationships, output directory layout. |
| [**CLI Reference**](cli_reference.md) | Users, engineers | All 49 CLI commands with flags and examples. |
| [**LLM Integration**](llm_integration.md) | GenAI engineers | Provider-agnostic LLM layer, Ollama Cloud, per-agent routing, config. |

## Guides

| Document | Audience | Description |
|----------|----------|-------------|
| [**Roadmap**](roadmap.md) | Maintainers, contributors | Versioned roadmap from v1.0 to v2.0. |
| [**Contributing**](contributing.md) | Contributors | Setup, conventions, how to add providers/tools/agents, PR checklist. |

---

## Conventions used across these docs

- **Phase N** refers to the ten-phase architecture (1 = paper analysis … 10 = LLM layer).
- All code blocks are Python 3.12+ unless noted.
- All models are Pydantic v2; all enums are `StrEnum`.
- "Patch-first" means code changes are produced as reviewable unified diffs, never applied silently.
- "Routed" (in agent tables) means the agent's LLM provider is resolved by `ModelRouter` from `llm_config.yaml`.

## Verification commands

```bash
uv run pytest -q                              # 690 tests
uv run mypy src/research_engineer/llm         # type-check the LLM layer
uv run ruff check .                            # lint
research-engineer llm status                   # inspect provider/model routing
```