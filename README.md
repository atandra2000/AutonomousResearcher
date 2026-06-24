<div align="center">

# Autonomous ML Research Engineer

**An agentic system that reads ML papers, understands codebases, plans experiments, writes patches, runs training, evaluates results, and iterates autonomously — all powered by a provider-agnostic LLM layer.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-687%20passing-success)](#testing-statistics)
[![Coverage](https://img.shields.io/badge/phases-1%E2%80%9310%20complete-blue)](#roadmap)

</div>

---

> **v1.0 — Production-ready.** Nine specialized agents, 51 typed tools, 198 Pydantic models, a persistent knowledge graph, a vector memory store, and a self-orchestrating research loop — wired through a single config-driven LLM layer.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Capabilities](#2-key-capabilities)
3. [Architecture](#3-architecture)
4. [Agent Ecosystem](#4-agent-ecosystem)
5. [Tool Ecosystem](#5-tool-ecosystem)
6. [Knowledge Graph Architecture](#6-knowledge-graph-architecture)
7. [Memory System Architecture](#7-memory-system-architecture)
8. [Ollama Cloud Integration](#8-ollama-cloud-integration)
9. [Installation](#9-installation)
10. [Quick Start](#10-quick-start)
11. [End-to-End Examples](#11-end-to-end-examples)
12. [CLI Reference](#12-cli-reference)
13. [Demo Workflows](#13-demo-workflows)
14. [Testing Statistics](#14-testing-statistics)
15. [Roadmap](#15-roadmap)
16. [Limitations](#16-limitations)
17. [Contributing](#17-contributing)

---

## 1. Project Overview

The **Autonomous ML Research Engineer** is a multi-agent platform that automates the full ML research lifecycle — from reading an arXiv paper to running and evaluating experiments against a real codebase. Instead of bolting an LLM onto a single script, it decomposes research work into **nine cooperating agents**, each backed by typed tools, sharing a persistent memory and a knowledge graph, and orchestrated by an autonomous loop that can iterate until a target metric is reached.

It is built for **ML engineers** who want to reproduce or extend papers against their own repositories, **research engineers** who want a repeatable experiment pipeline, **GenAI engineers** who want a clean, provider-agnostic LLM integration pattern, and **open-source contributors** who want a well-tested, typed, async-first codebase to extend.

**Why it's different:**

- **Agent-native, not prompt-native.** Each phase (paper analysis, repo analysis, planning, coding, memory, literature, execution, evaluation, loop) is a discrete agent with a typed contract.
- **Patch-first.** Code changes are produced as reviewable unified diffs — the system never silently mutates your repo.
- **Memory-first.** Every run writes to SQLite + ChromaDB + a knowledge graph, so the platform *learns across runs*.
- **Provider-agnostic.** All agents talk to models through one abstraction; switching from Ollama Cloud to another provider is a YAML edit, not a code change.
- **Safe by default.** Experiment execution is dry-run by default, with a command allowlist, timeouts, and working-directory confinement.

---

## 2. Key Capabilities

| Capability | What it does |
|------------|--------------|
| **Paper analysis** | Ingest arXiv ID / URL / PDF → structured `ResearchSummary` + `EngineeringReport` (no LLM required for extraction). |
| **Repository analysis** | AST scan, dependency graph, training-pipeline extraction, config analysis, knowledge-graph build, documentation generation. |
| **Experiment planning** | 7-dimension compatibility analysis, ordered implementation plan, 5-group experiment matrix, 6-suite validation plan, 7-category risk assessment, GPU-hour + cost estimation, best/likely/worst result prediction. |
| **Code implementation** | Code generation → unified-diff patches → self-review → test generation → migration & rollback planning → implementation report. |
| **Research memory** | 9 memory types, 10 relationship types, SQLite + ChromaDB vector store, 6 retrieval strategies, automatic relationship detection. |
| **Literature intelligence** | Multi-source search (local + arXiv + Semantic Scholar), 7-dimension comparison, structured reviews, trend analysis, paper recommendations, paper↔repo relevance scoring. |
| **Experiment execution** | Subprocess runner with allowlist + dry-run, live monitoring, metric parsing (logs/JSON/CSV), artifact collection with checksums, rule-based failure detection. |
| **Evaluation** | Experiment comparison, training-dynamics analysis (over/underfit, convergence, instability), Welch t-test + Cohen's d + 95% CIs (pure Python, no SciPy), next-experiment recommendations. |
| **Autonomous loop** | State-machine orchestrator: recall → discover → plan → implement → run → evaluate → store → learn → stop-check, with approval gates and a final research report. |
| **LLM layer** | `LLMProvider` ABC, Ollama Cloud provider, per-agent model routing, `${VAR}` env expansion, config-only model switching. |

---

## 3. Architecture

The platform is organized into **ten phases**, each a self-contained layer with agents, tools, and typed models. Phases 1–8 are individual capabilities; Phase 9 orchestrates them; Phase 10 is the LLM substrate every agent sits on.

```
                    Inputs
                 ┌─────────────┐
                 │ arXiv / PDF │
                 │ ML repo     │
                 │ Research    │
                 │ goal        │
                 └──────┬──────┘
                        │
        ┌───────────────┼───────────────┐
        v               v               v
  ResearchAgent  RepositoryAgent  ResearchLoopAgent
  (paper→summary) (repo→struct)   (orchestrator)
        │               │               │
        └───────┬───────┘               │
                v                       │
        ExperimentPlannerAgent           │
        (plan→9-file plan)              │
                │                       │
                v                       │
          CodingAgent                   │
        (patches + tests)               │
                │                       │
                v                       │
        ExperimentAgent                 │
        (run + monitor)                 │
                │                       │
                v                       │
        EvaluationAgent                 │
        (compare + stats)               │
                │                       │
                v                       v
            MemoryAgent ◄── ResearchLoopAgent
        (SQLite + ChromaDB               │
         + Knowledge Graph)              │
                │                       │
                └───────────────────────┘
                        │
                        v
              research_report.md / .json

    LLM Layer (Phase 10)
    llm_config.yaml → ProviderFactory
    → ModelRouter → OllamaCloudProvider
    → resolve_llm() on every agent
```

### Phase pipeline

```
Paper Analysis → Experiment Planning ← Repo Analysis
                      ↓
              Code Implementation
                      ↓
              Experiment Execution
                      ↓
                    Evaluation
                      ↓
         ┌────────── Loop ──────────┐
         │ (recall → discover →     │
         │  plan → implement → run  │
         │  → evaluate → store)     │
         └──────────────────────────┘
         LLM Layer (Phase 10) — provider-agnostic
```

---

## 4. Agent Ecosystem

Nine agents, each with a single responsibility, a typed result model, and an optional LLM provider resolved through the router.

| # | Agent | Phase | Responsibility | LLM |
|---|-------|-------|----------------|-----|
| 1 | `ResearchAgent` | 1 | Acquire paper (arXiv/PDF), parse, produce `ResearchSummary` + `EngineeringReport`, store. | routed |
| 2 | `RepositoryAgent` | 2 | Scan repo, AST analysis, dependency graph, training pipeline, config analysis, knowledge graph, docs. | optional |
| 3 | `ExperimentPlannerAgent` | 3 | Compatibility (7 dims), implementation plan, experiment matrix, validation, risk, compute, prediction. | routed |
| 4 | `CodingAgent` | 4 | Code generation → patches → self-review → tests → migration → rollback → report. **Patch-first.** | routed |
| 5 | `MemoryAgent` | 5 | Store/recall 9 memory types, manage relationships, vector search, knowledge graph. | routed |
| 6 | `LiteratureAgent` | 6 | Multi-source search, 7-dim comparison, reviews, trends, recommendations, relevance scoring. | routed |
| 7 | `ExperimentAgent` | 7 | Launch (allowlisted, dry-run default), monitor, collect metrics + artifacts, detect failures. | routed |
| 8 | `EvaluationAgent` | 8 | Compare runs, training dynamics, statistical significance, next-experiment recommendations. | routed |
| 9 | `ResearchLoopAgent` | 9 | Orchestrate Phases 1–8 in iterative cycles with stopping conditions + approval gates + reports. | routed |

Every agent constructor accepts an optional `llm: LLMProvider` and exposes `agent_name` + `llm_provider`. **No agent instantiates a model directly** — they all go through `resolve_llm()`.

---

## 5. Tool Ecosystem

**51 typed tools** following a uniform `Tool[InputType, OutputType]` ABC (`async execute`, `async validate`, `ToolError`). Inputs and outputs are Pydantic v2 models — fully typed, validated, serializable.

| Phase | Tools |
|-------|-------|
| 1 | `ArxivTool`, `PDFTool`, `PaperParserTool`, `StorageTool` |
| 2 | `RepositoryScannerTool`, `ASTAnalysisTool`, `DependencyGraphTool`, `TrainingPipelineTool`, `ConfigAnalysisTool`, `KnowledgeGraphTool`, `DocumentationTool` |
| 3 | `CompatibilityAnalysisTool`, `ImplementationPlannerTool`, `ImpactAnalysisTool`, `ExperimentDesignTool`, `ValidationPlannerTool`, `RiskAssessmentTool`, `ComputeEstimatorTool`, `ResultPredictionTool` |
| 4 | `CodeGenerationTool`, `PatchGenerationTool`, `SelfReviewTool`, `TestGenerationTool`, `MigrationPlannerTool`, `RollbackPlannerTool`, `PatchApplicationTool`, `ImplementationReportTool` |
| 5 | `MemoryStorageTool`, `VectorStore`/`ChromaVectorStore`, `EmbeddingStrategy`, `QueryProcessor`, `MemoryKnowledgeGraph`/`MemoryGraphTool`, `RelationshipDetector`, 6x `RetrievalStrategy`, `MemoryQueryTool`, `MemoryWriteTool`, `MemoryRecallTool` |
| 6 | `PaperSearchTool`, `PaperComparisonTool`, `LiteratureReviewTool`, `PaperRelationshipTool`, `TrendAnalysisTool`, `PaperRecommendationTool`, `RelevanceScoringTool` |
| 7 | `ExperimentRunnerTool`, `MonitoringTool`, `MetricCollectorTool`, `ArtifactCollectorTool`, `FailureDetectorTool`, `ExperimentStorageTool` |
| 8 | `ExperimentComparisonTool`, `TrainingDynamicsTool`, `StatisticalSignificanceTool`, `NextExperimentTool`, `EvaluationStorageTool` |
| 9 | `LoopStorageTool`, `StoppingConditionChecker`, `ReportGeneratorTool` |
| 10 | `LLMProvider` ABC, `OllamaCloudProvider`, `ProviderFactory`, `ModelRouter` |

### Safety controls in the experiment runner

- **Dry-run by default** — commands are echoed, not executed, unless `dry_run=False`.
- **Command allowlist** — only `python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`.
- **Timeouts** + working-directory confinement.

---

## 6. Knowledge Graph Architecture

The `MemoryKnowledgeGraph` is a directed, typed, weighted graph that captures how every artifact in the platform relates to every other. Nodes are memories; edges are typed relationships with confidence scores.

```
PaperMemory ──cites──→ PaperMemory (reference)
PaperMemory ──implements──→ RepositoryMemory (repo)
PaperMemory ──inspires──→ ExperimentPlanMemory (plan)
ExperimentPlanMemory ──produces──→ PatchMemory (patch)
PatchMemory ──validates──→ ExperimentRecord (run)
ExperimentRecord ──succeeded_with──→ SuccessfulApproachMemory
ExperimentRecord ──failed_with──→ FailedApproachMemory
FailedApproachMemory ──conflicts_with──→ SuccessfulApproachMemory
ExperimentPlanMemory ──extends──→ ResearchInsightMemory
```

**10 relationship types:** `cites`, `implements`, `extends`, `similar_to`, `depends_on`, `conflicts_with`, `validates`, `failed_with`, `succeeded_with`, `inspired_by`.

**Graph statistics** (`GraphStats`): node count, edge count, density, average degree, weakly-connected components, most-central nodes, edge counts by relationship type.

Edges are added **automatically** — every agent that stores a memory also calls `MemoryKnowledgeGraph.add_relationship()`, so the graph stays consistent without manual wiring.

---

## 7. Memory System Architecture

Memory is the platform's long-term brain. It persists across runs and powers cross-run learning in the autonomous loop.

**9 memory types:**
- `PaperMemory`, `RepositoryMemory`, `ExperimentPlanMemory`, `PatchMemory`
- `ArchitectureDecisionMemory`, `ResearchInsightMemory`
- `FailedApproachMemory`, `SuccessfulApproachMemory`
- Pattern / AntiPattern / BestPractice memories

**3 storage backends:**
- **SQLite** — structured records
- **ChromaDB** — SPECTER-style embeddings (`all-mpnet-base-v2`)
- **Knowledge Graph** — typed relationships

**6 retrieval strategies:**
1. `DirectLookup` — exact ID/tag match
2. `SemanticSearch` — vector similarity
3. `GraphTraversal` — relationship walk
4. `TagBasedFilter` — tag intersection
5. `TemporalQuery` — recency-weighted
6. `HybridSearch` — combines vector + graph + tag signals

- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` via `EmbeddingStrategy`.
- **Auto-relationship detection:** `RelationshipDetector` infers links between new and existing memories.
- **Access logging + versioning:** every read is logged; memories are versioned.

---

## 8. Ollama Cloud Integration

All agents reach the model through a **provider-agnostic LLM layer** (Phase 10). The default provider is **Ollama Cloud**, spoken via its OpenAI-compatible Chat Completions endpoint over `httpx` — no LlamaIndex, no vendor SDK.

```
llm_config.yaml ──→ ProviderFactory ──→ ModelRouter ──→ OllamaCloudProvider
(default_provider   builds + caches     for_agent(name)   POST /v1/chat/
 + per-agent        providers,          → _BoundProvider  completions
 models)            ${VAR} expansion
                                            ↑
                                    9 agents
                                    resolve_llm()
```

### Resolution rules

1. An explicit `LLMProvider` passed to an agent constructor **wins**.
2. `llm_enabled=False` (e.g. `RepositoryAgent` default) → no provider attached.
3. Otherwise `ModelRouter.for_agent(agent_name)` resolves from `llm_config.yaml`.

### Per-agent model routing

Each agent can use a **different model**, configured in one file:

```yaml
# llm_config.yaml
default_provider: ollama
default_model: llama3
providers:
  ollama:
    type: ollama
    base_url: https://api.olama.cloud
    api_key: ${OLLAMA_API_KEY}      # expanded from the environment
    default_model: llama3
    timeout: 60
agents:
  ResearchAgent:          {provider: ollama, model: llama3}
  CodingAgent:            {provider: ollama, model: qwen2.5-coder}
  EvaluationAgent:        {provider: ollama, model: llama3}
```

**Switching a model is a config-only change** — no source edits. Adding a new provider is `register_provider_type()` + a YAML block.

### Environment variables

| Variable | Default |
|----------|---------|
| `RE_LLM_CONFIG` | `llm_config.yaml` at repo root |
| `OLLAMA_BASE_URL` | `https://api.olama.cloud` |
| `OLLAMA_API_KEY` | (none) |
| `OLLAMA_MODEL` / `OLLAMA_DEFAULT_MODEL` | `llama3` |
| `OLLAMA_TIMEOUT` | `60` |

---

## 9. Installation

**Prerequisites:** Python >= 3.12, [`uv`](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Clone
git clone https://github.com/<your-org>/AutonomousMLResearchEngineer.git
cd AutonomousMLResearchEngineer

# Install with uv (recommended)
uv sync

# ...or with pip + a venv
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Optional — enable Ollama Cloud:**

```bash
export OLLAMA_API_KEY="your-key"
# Optionally override base URL / model
export OLLAMA_BASE_URL="https://api.olama.cloud"
export OLLAMA_MODEL="llama3"
```

Verify the install:

```bash
research-engineer --help
research-engineer llm status
```

---

## 10. Quick Start

### Analyze a paper

```bash
# From an arXiv ID
research-engineer analyze 2503.12345

# From a URL
research-engineer analyze https://arxiv.org/abs/2503.12345

# From a local PDF
research-engineer analyze ./papers/attention.pdf --output-dir output/
```

### Analyze a repository

```bash
research-engineer analyze-repo ./my_model_repo --output-format markdown
```

### Plan an experiment (paper x repo)

```bash
research-engineer plan 2503.12345 ./my_model_repo
# -> output/plans/<paper_id>_<repo>/  (8 markdown files + plan_result.json)
```

### Run the autonomous research loop

```bash
research-engineer loop run "Improve training stability" \
  --repo ./my_model_repo \
  --max-iterations 3 \
  --dry-run
```

---

## 11. End-to-End Examples

### Example A — Reproduce a paper against your repo

```bash
# 1. Understand the paper
research-engineer analyze 2401.04088 --output-dir output/

# 2. Understand your repo
research-engineer analyze-repo ./my_transformer --output-format json > repo.json

# 3. Generate a 9-file experiment plan
research-engineer plan 2401.04088 ./my_transformer

# 4. Implement the changes as reviewable patches
research-engineer implement --task "Add rotary positional embeddings" \
  --repo ./my_transformer \
  --paper 2401.04088

# 5. (Dry-)run the experiment
research-engineer experiment run \
  --command "python train.py --config configs/exp.yaml" \
  --repo ./my_transformer \
  --dry-run

# 6. Evaluate the run
research-engineer evaluate analyze <experiment_id>
```

### Example B — Autonomous loop with a target metric

```bash
research-engineer loop run "Reduce validation loss below 0.10" \
  --repo ./my_transformer \
  --target-metric loss \
  --target-value 0.10 \
  --higher-is-better false \
  --max-iterations 10 \
  --budget-hours 8.0

# Inspect the result
research-engineer loop list --status stopped
research-engineer loop report <loop_id> --output-dir ./reports
```

The loop will: recall relevant memories -> discover literature -> plan -> implement -> (dry-)run -> evaluate -> store insights -> update the knowledge graph -> check the stopping condition -> repeat. It stops on `target_achieved`, `max_iterations_reached`, `budget_exceeded`, or `no_improvement`.

### Example C — Literature discovery for a topic

```bash
research-engineer literature discover "mixture of experts routing" \
  --repo ./my_moe_repo \
  --max-papers 25
# -> output/literature/<topic>_<timestamp>/  (search, comparison, review, trends, recommendations, relevance)
```

---

## 12. CLI Reference

**49 commands** across 6 sub-apps. Run `research-engineer <command> --help` for full flags.

| Sub-app | Command | Purpose |
|---------|---------|---------|
| **core** | `analyze <paper>` | Analyze a paper (arXiv ID / URL / PDF). |
| | `analyze-repo <path>` | Analyze a repository. |
| | `plan <paper> <repo>` | Generate a 9-file experiment plan. |
| | `implement` | Generate patches + tests + reports. |
| | `get <paper_id>` | Retrieve a stored paper. |
| | `search <query>` | Search stored papers. |
| | `history` / `cache-status` | Analysis history / cache stats. |
| **memory** | `memory search|list|stats|related|graph|export|import|archive` | Query and manage research memories. |
| **literature** | `literature search|compare|review|relationships|trends|recommend|relevance|discover` | Literature intelligence. |
| **experiment** | `experiment run|monitor|list|get|search|cancel|history` | Run and track experiments. |
| **evaluate** | `evaluate run|compare|analyze|dynamics|significance|next|list|get|search` | Evaluate experiments. |
| **loop** | `loop run|list|get|iterations|iteration|search|report` | Autonomous research loops. |
| **llm** | `llm status` / `llm config` | Inspect LLM provider/model routing. |

```bash
# Inspect which model each agent uses
research-engineer llm status
research-engineer llm status --format json

# Dump the resolved config
research-engineer llm config
research-engineer llm config --config path/to/llm_config.yaml
```

---

## 13. Demo Workflows

### Demo 1 — Single-paper, single-repo plan (~30 s, no LLM needed)

```bash
research-engineer analyze 1706.03762            # Attention Is All You Need
research-engineer analyze-repo ./my_transformer
research-engineer plan 1706.03762 ./my_transformer
# Open output/plans/1706.03762_my_transformer/compatibility_analysis.md
```

### Demo 2 — Literature review for a new topic

```bash
research-engineer literature review "sparse mixture of experts" --depth comprehensive
research-engineer literature trends "sparse mixture of experts"
research-engineer literature recommend "sparse mixture of experts" --repo ./my_moe_repo
```

### Demo 3 — Closed-loop autonomous research (dry-run)

```bash
export OLLAMA_API_KEY="..."
research-engineer loop run "Stabilize training at long context" \
  --repo ./my_llm --max-iterations 3 --dry-run
research-engineer loop report <loop_id> --output-dir ./reports
```

### Demo 4 — Evaluate two runs for statistical significance

```bash
research-engineer evaluate compare exp_aaa exp_bbb
research-engineer evaluate significance exp_aaa exp_bbb
research-engineer evaluate dynamics exp_aaa
research-engineer evaluate next exp_aaa exp_bbb
```

---

## 14. Testing Statistics

| Metric | Value |
|--------|-------|
| **Total tests** | 687 passing |
| **Phase 10 (LLM) tests** | 29 |
| **Source files** | 96 Python files, ~34k LOC |
| **Test files** | 37 files, ~9.9k LOC |
| **Pydantic models** | 198 |
| **Typed tools** | 51 |
| **Agents** | 9 + LLM layer |
| **CLI commands** | 49 |
| **Phases complete** | 10 / 10 |

```bash
uv run pytest -q          # 687 passed
uv run mypy src/research_engineer/llm   # clean
uv run ruff check .       # lint
```

Test coverage spans every phase: models, tools, agents, CLI, and end-to-end integration (`test_integration.py`, `test_integration_phases.py`).

---

## 15. Roadmap

- **v1.1** — Additional providers (OpenAI, Anthropic, local Ollama) behind the same `LLMProvider` ABC.
- **v1.2** — Streaming-first agent outputs; structured tool-calling for the CodingAgent.
- **v1.3** — Web UI dashboard for loop monitoring + knowledge-graph visualization.
- **v1.4** — Multi-repo experiment matrices; distributed experiment execution.
- **v2.0** — Self-improving meta-loop: the platform proposes its own research goals from memory trends.

---

## 16. Limitations

- **Experiment execution is sandboxed by design.** The runner uses a command allowlist and dry-run default; it will not run arbitrary shells. Real training requires you to opt out of dry-run.
- **Patch-first, not auto-apply.** `CodingAgent` produces reviewable unified diffs. Applying patches is a separate, explicit, approval-gated step.
- **Paper extraction is rule-based in Phases 1–3** (no LLM cost for parsing); LLM is used where it adds value (e.g. optional repo analysis, code generation).
- **Vector store** uses `sentence-transformers/all-mpnet-base-v2`; large memory corpora may need a dedicated embedding service.
- **Statistical significance** is implemented in pure Python (Welch t-test, Cohen's d, bootstrap CIs) — no SciPy dependency, but not a substitute for a full stats package for production research conclusions.
- **Ollama Cloud** is the default provider; other providers require implementing the `LLMProvider` ABC (a ~80-line class).

---

## 17. Contributing

Contributions are welcome — especially new providers, new tools, and new retrieval strategies.

### Development setup

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run mypy src/research_engineer/llm
```

### Conventions

- **Python >= 3.12**, async-first, Pydantic v2, `StrEnum` for all enums.
- **Typed tools** follow `Tool[Input, Output]` with `async execute` / `async validate`.
- **No direct model calls** — agents obtain providers via `resolve_llm()`; model switching is config-only.
- **Patch-first** — never mutate user code directly; produce reviewable diffs.
- **Repository-agnostic & paper-agnostic** — no hardcoded assumptions about specific repos or paper topics.
- **Tests must pass before PR** (target > 90% coverage).

### Adding a new LLM provider

```python
from research_engineer.llm import LLMProvider, LLMRequest, LLMResponse, register_provider_type

class MyProvider(LLMProvider):
    name = "myprov"
    default_model = "x"
    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

register_provider_type("myprov", MyProvider)
```

```yaml
# llm_config.yaml
providers:
  myprov: {type: myprov, api_key: ${MYP_KEY}, default_model: x}
agents:
  ResearchAgent: {provider: myprov, model: x}
```

No agent code changes required.

### Adding a new tool

1. Subclass `Tool[YourInput, YourOutput]` (Pydantic models for I/O).
2. Implement `async execute()` and optionally `async validate()`.
3. Export from `tools/__init__.py` and wire into the relevant agent.
4. Add tests under `tests/`.

### Project layout

```
src/research_engineer/
├── agents/      # 9 agents + _llm_support.py
├── llm/         # Phase 10: base, ollama_provider, factory, router
├── models/      # 198 Pydantic models across 12 modules
├── tools/       # 51 typed tools
└── cli/         # 49 Typer commands
tests/           # 37 test files, 687 tests
llm_config.yaml  # provider + per-agent model config
docs/llm_integration.md
```

---

<div align="center">

**Built for ML practitioners who want research automation that's typed, testable, and provider-agnostic.**

Star the repo if it's useful · Open an issue if it's not · PRs welcome

</div>
