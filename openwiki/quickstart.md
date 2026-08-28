---
type: navigation guide
title: Quickstart
description: Entry router for installing and operating Autonomous ML Research Engineer, with task-oriented links to architecture, agents, workflows, runtime findings, operations, and tests.
tags: [quickstart, navigation, cli, operations]
verified:
  - by: openwiki/0.4.3
    at: 2026-08-28T12:25:37.430Z
sources:
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-bc61113e76cca6dd3c6e605f
    resource: repo://src/research_engineer/agents/research_loop_agent.py
  - id: openwiki-source-90c485ca6d5ee63a6bb09112
    resource: repo://src/research_engineer/agents/research_workflow.py
  - id: openwiki-source-0cd1dc265699c357afc4b69c
    resource: repo://src/research_engineer/agents/task_agent.py
  - id: openwiki-source-614771b0d5f5df521ba66521
    resource: repo://src/research_engineer/cli/__init__.py
  - id: openwiki-source-583853a90da01124d6384e55
    resource: repo://src/research_engineer/observability/otel.py
  - id: openwiki-source-8839922cb8c4da87bca73c8d
    resource: repo://src/research_engineer/runtime/__init__.py
  - id: openwiki-source-926c37d01f94ffd1a72d86f7
    resource: repo://src/research_engineer/runtime/runtime.py
  - id: openwiki-source-e7bb6404452c79f65c2e483a
    resource: repo://src/research_engineer/service/serve.py
generated: { by: "openwiki/0.4.3", at: "2026-08-28T12:25:37.430Z" }
---

# Quickstart

Use this page to choose the right system boundary before changing or operating the repository. The project is a Python 3.12+ package whose `research-engineer` console script exposes the Typer CLI. Its core work spans paper and repository analysis, planning, patch-oriented implementation, experiment execution and evaluation; later layers add repository memory, delegation/repair, and staged research workflows.

## Start locally

Install with the repository-supported `uv` path, then inspect the CLI and resolved LLM setup:

```bash
uv sync
research-engineer --help
research-engineer llm status
```

`OLLAMA_API_KEY` enables the documented Ollama Cloud setup; `OLLAMA_BASE_URL` and `OLLAMA_MODEL` can override its endpoint and model. For provider routing and other operational knobs, use [Configuration and Runtime Controls](operations/configuration.md) rather than treating these commands as a complete configuration reference.

## Choose a task

| If you need to… | Begin here | Then follow |
|---|---|---|
| Orient yourself in the repository-wide subsystem map or deployed boundary | [Architecture Overview](architecture/overview.md) | [Agent System](concepts/agents.md), [Configuration and Runtime Controls](operations/configuration.md) |
| Understand agent responsibilities, LLM-backed coordination, memory, or tools | [Agent System](concepts/agents.md) | Architecture, then the workflow that matches the change |
| Analyze a paper, repository, plan, execute, evaluate, and iterate toward a research objective | [Research Loop Workflow](workflows/research-loop.md) | Configuration, testing, and runtime findings |
| Make a terminal-first coding change, including repository context, patch review, tests, delegation, or repair | [Terminal Task Agent Workflow](workflows/terminal-task-agent.md) | Agent System, testing, and operations |
| Set LLM, timeout, budget, execution, or service controls | [Configuration and Runtime Controls](operations/configuration.md) | Safety and Architecture as required |
| Select or run focused verification suites | [Testing Overview](testing/overview.md) | The owning workflow or subsystem page |
| Diagnose observed behavior or an integration issue—especially the LangSmith connector evidence pull | **[Runtime Behavior](runtime/runtime-behavior.md)** | Architecture and operations; use this findings page instead of raw traces |

## First commands by workflow

Run `research-engineer <command> --help` before adding flags. These documented starting points deliberately use dry-run where execution could have side effects:

```bash
# Paper → repository-aware experiment plan
research-engineer analyze 2503.12345
research-engineer analyze-repo ./my_model_repo --output-format markdown
research-engineer plan 2503.12345 ./my_model_repo

# Iterative research; dry-run is explicit here
research-engineer loop run "Improve training stability" \
  --repo ./my_model_repo \
  --max-iterations 3 \
  --dry-run

# Terminal-first coding; delegation and repair are optional capabilities
research-engineer task "Add EMA checkpoint support" --repo ./my_repo
research-engineer task "Add EMA checkpoint support" --delegate --max-repairs 3

# Build and query repository memory before context-heavy coding work
research-engineer memory build --repo ./my_repo
research-engineer memory query "checkpoint saving logic" --repo ./my_repo

# Goal → staged research report
research-engineer research "Design a more efficient diffusion transformer"
```

The loop owns iterative research state, approval handling, stored iterations, stopping checks, and reporting; the terminal task agent owns the coding-oriented analyze → plan → patch/diff → optional-test route. Keep those behavioral details in their dedicated workflow pages.

## Navigation map

- **Architecture:** [Architecture Overview](architecture/overview.md) — subsystem and integration boundaries.
- **Concepts:** [Agent System](concepts/agents.md) — responsibilities and collaboration boundaries.
- **Workflows:** [Research Loop Workflow](workflows/research-loop.md) and [Terminal Task Agent Workflow](workflows/terminal-task-agent.md) — choose research iteration or coding execution.
- **Operations:** [Configuration and Runtime Controls](operations/configuration.md) — configuration and runtime knobs.
- **Testing:** [Testing Overview](testing/overview.md) — suites and safe validation scope.
- **Runtime findings:** **[Runtime Behavior](runtime/runtime-behavior.md)** — synthesized LangSmith connector evidence, failures, outliers, and costs without raw trace dumps.

## When the CLI is not the boundary

The optional `service` dependency group supports a FastAPI/worker deployment. The service launcher runs separate `api` and `worker` processes, selects an in-memory, SQLite, or PostgreSQL checkpoint store from configuration, and can construct the gateway/safety chain for worker execution. Start at [Architecture Overview](architecture/overview.md) for the service map and [Configuration and Runtime Controls](operations/configuration.md) before deploying.

The generic `AgentRuntime` is the lower-level async execution boundary: it drives `plan → act → observe → evaluate`, supports budgets, cancellation, recoverable-error handling, checkpoints, tool-gateway routing, safety control, and observability. Consult [Runtime Behavior](runtime/runtime-behavior.md) first when runtime evidence is relevant, and use the architecture and testing pages to plan a safe change.

## Verify the change you make

The repository configures pytest to discover `tests/test_*.py` with `src` on `pythonpath`, applies a 300-second per-test timeout, and marks live-network tests as `network`. The documented baseline checks are:

```bash
uv run python -m pytest -q
uv run ruff check .
scripts/ci_mypy.sh
```

Use [Testing Overview](testing/overview.md) to select focused tests; do not infer current pass counts from this router.
