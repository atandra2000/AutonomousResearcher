# Documentation Index

Complete documentation for the **Autonomous ML Research Engineer** v0.9.0 — a multi-agent platform that automates the full ML research lifecycle.

> **Current state:** 15/15 phases complete · 20 agents · 61 typed tools ·
> 17 Pydantic v2 schema modules (251 classes) · 73 CLI commands across 10
> sub-apps · **1451 passing, 2 optional-dependency skipped** · plus the E4 agent-eval harness,
> E8 continuous-improvement loop, and P1/P2 research benchmarks.

## Visual Systems Atlas

Explore the [Interactive Visual Systems Guide](diagrams/research_engineer_visual_guide.html): four verified Archify showcase maps ([Multi-Agent System Architecture](diagrams/research-engineer-architecture.html), [Research & Self-Repair Workflow](diagrams/research-engineer-workflow.html), [Agent Delegation & Trace](diagrams/research-engineer-sequence.html), [Telemetry & Artifact Dataflow](diagrams/research-engineer-dataflow.html)), interactive Research Loop simulator, LLM & Tool compute budget calculator, and [verification receipts](diagrams/RECEIPTS.md).

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
| [**Agents**](agents.md) | Engineers | Deep-dive on all 20 agents: responsibilities, constructors, workflows, LLM wiring. |
| [**Tools**](tools.md) | Engineers, contributors | Reference for all 61 typed tools: input/output models, key logic. |
| [**Models**](models.md) | Engineers | Reference for Pydantic v2 models across 17 schema modules grouped by phase. |
| [**Memory System**](memory_system.md) | Engineers | Memory types, retrieval strategies, knowledge graph, vector store, repository memory (Phase 12). |
| [**Storage Schema**](storage_schema.md) | Engineers, DBAs | All SQLite tables, columns, relationships, output directory layout. |
| [**CLI Reference**](cli_reference.md) | Users, engineers | All 73 CLI commands with flags and examples. |
| [**LLM Integration**](llm_integration.md) | GenAI engineers | Provider-agnostic LLM layer, Ollama Cloud, per-agent routing, 23-component config (20 agents + 3 framework classes). |
| [**Framework-Stack Migration**](framework_stack_migration.md) | Engineers, architects | LangGraph as the only research engine: research graph, Postgres checkpointing, LangChain provider, gateway adapter. |

## Guides

| Document | Audience | Description |
|----------|----------|-------------|
| [**Roadmap**](roadmap.md) | Maintainers, contributors | Versioned roadmap — v1.0 through v2.0 and beyond. |
| [**Contributing**](contributing.md) | Contributors | Setup, conventions, how to add providers/tools/agents, PR checklist. |
| [**P2 Benchmark**](benchmark_p2.md) | Engineers, researchers | The LLM-backed research tier: suites, budgets, fail-closed guarantees. |

## Historical records

Point-in-time reports kept as provenance for measured experiments. They describe
the serving tier that was later removed in `c31c536`; read them as history, not
as current architecture.

| Document | Description |
|----------|-------------|
| [**Implementation Plan**](autonomous_agentic_implementation_plan.md) | The 14-workstream plan that built the autonomous platform (complete). |
| [**P2 Session Notes**](P2_SESSION_NOTES.md) | Working notes from the P2 benchmark build (ephemeral). |
| [**P3 Report**](p3_report.md) | Agent capability optimization experiment over the frozen baseline. |
| [**P3-Short Report**](p3_short_report.md) | Model selection & capability validation across 3 models × 2 repeats. |
| [**P4 Closure Report**](p4_closure_report.md) | Production-readiness closure: grader hardening, E8 gate, CI stabilization. |

---

## Conventions used across these docs

- **Phase N** refers to the fifteen-phase architecture (1 = paper analysis … 15 = research workflows).
- All code blocks are Python 3.12+ unless noted.
- All models are Pydantic v2; all enums are `StrEnum`.
- "Patch-first" means code changes are produced as reviewable unified diffs, never applied silently.
- "Routed" (in agent tables) means the agent's LLM provider is resolved by `ModelRouter` from `llm_config.yaml`.

## Verification commands

```bash
uv run python -m pytest -q                     # 1451 passed, 2 skipped (optional deps)
uv run mypy src/research_engineer/llm         # type-check the LLM layer
uv run ruff check .                            # lint
research-engineer llm status                   # inspect provider/model routing
```
