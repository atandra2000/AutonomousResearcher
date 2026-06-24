<div align="center">

# Autonomous ML Research Engineer

**An agentic platform that reads ML papers, understands codebases, plans experiments, writes patches, runs training, evaluates results, iterates autonomously, and conducts end-to-end research workflows — all powered by a provider-agnostic LLM layer with multi-agent delegation and self-repair.**

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-878%20passing-success)](#14-testing-statistics)
[![Coverage](https://img.shields.io/badge/phases-1%E2%80%9315%20complete-blue)](#15-roadmap)

</div>

---

> **v2.0 — Production-ready with autonomous research workflows.** 23 specialized agents, 61 typed tools, 186 Pydantic models, a persistent knowledge graph, a vector memory store, a symbol-graph repository memory, a multi-agent delegation framework, an autonomous self-repair engine, and an end-to-end research workflow orchestrator — wired through a single config-driven LLM layer with per-agent model routing (qwen3-coder-next:cloud for coding, glm-5.2:cloud for reasoning, minimax-m3:cloud for orchestration).

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

The **Autonomous ML Research Engineer** is a multi-agent platform that automates the full ML research lifecycle — from reading an arXiv paper to running and evaluating experiments against a real codebase, and now from a research goal to a complete research report with literature review, hypotheses, experiments, and conclusions.

It decomposes research work into **fifteen cooperating phases**, each a self-contained layer with agents, typed tools, and Pydantic models. Phases 1–8 are individual capabilities; Phase 9 orchestrates them into an autonomous loop; Phase 10 is the provider-agnostic LLM substrate; Phase 11 adds a terminal-first autonomous coding agent; Phase 12 adds persistent repository memory with hybrid retrieval; Phase 13 adds multi-agent delegation with review/test repair loops; Phase 14 adds autonomous self-repair with structured failure analysis; Phase 15 adds end-to-end autonomous research workflows.

It is built for **ML engineers** who want to reproduce or extend papers against their own repositories, **research engineers** who want a repeatable experiment pipeline, **GenAI engineers** who want a clean, provider-agnostic LLM integration pattern, and **open-source contributors** who want a well-tested, typed, async-first codebase to extend.

**Why it's different:**

- **Agent-native, not prompt-native.** Each phase is a discrete agent with a typed contract.
- **Patch-first.** Code changes are produced as reviewable unified diffs — the system never silently mutates your repo.
- **Memory-first.** Every run writes to SQLite + ChromaDB + a knowledge graph, so the platform *learns across runs*.
- **Provider-agnostic.** All agents talk to models through one abstraction; switching models is a YAML edit, not a code change.
- **Multi-agent delegation.** A generic capability-based router dispatches work to specialized agents — no hardcoded task logic.
- **Autonomous self-repair.** Structured failure analysis, strategy generation, and iterative repair with stagnation detection.
- **End-to-end research workflows.** Literature review → synthesis → hypotheses → experiments → analysis → report.
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
| **Terminal-first coding** (Phase 11) | `TaskAgent` orchestrates analyze → plan → implement → diff → test with `TerminalTool` (run_command, read_file, write_file, search_code, apply_patch, git_status, git_diff). |
| **Repository memory** (Phase 12) | AST-based symbol indexing, semantic chunking, symbol graph (deps, callers, callees, related, tests), hybrid retrieval (semantic + graph + metadata), persistent SQLite storage, incremental updates. |
| **Multi-agent delegation** (Phase 13) | Generic `DelegationFramework` with role/capability routing, `SharedTaskContext` for inter-agent communication, review/test repair loops, `ArchitectAgent`, `ReviewerAgent`, `TestAgent`. |
| **Autonomous self-repair** (Phase 14) | `SelfRepairFramework` with structured `FailureReport`, `RepairStrategist`, `FailureAnalyzer`, configurable retry budgets, stagnation detection, four termination conditions. |
| **Research workflows** (Phase 15) | `ResearchOrchestrator` → literature discovery → knowledge synthesis → hypothesis generation → experiment planning → execution → result analysis → report generation. |

---

## 3. Architecture

The platform is organized into **fifteen phases**, each a self-contained layer with agents, tools, and typed models. Phases 1–8 are individual capabilities; Phase 9 orchestrates them; Phase 10 is the LLM substrate; Phase 11 adds terminal-first coding; Phase 12 adds repository memory; Phase 13 adds multi-agent delegation; Phase 14 adds autonomous self-repair; Phase 15 adds end-to-end research workflows.

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

23 agents across 15 phases, each with a single responsibility, a typed result model, and an optional LLM provider resolved through the router.

| # | Agent | Phase | Responsibility | Model |
|---|-------|-------|----------------|-------|
| 1 | `ResearchAgent` | 1 | Acquire paper (arXiv/PDF), parse, produce `ResearchSummary` + `EngineeringReport`, store. | glm-5.2:cloud |
| 2 | `RepositoryAgent` | 2 | Scan repo, AST analysis, dependency graph, training pipeline, config analysis, knowledge graph, docs. | glm-5.2:cloud |
| 3 | `ExperimentPlannerAgent` | 3 | Compatibility (7 dims), implementation plan, experiment matrix, validation, risk, compute, prediction. | glm-5.2:cloud |
| 4 | `CodingAgent` | 4 | Code generation → patches → self-review → tests → migration → rollback → report. **Patch-first.** | qwen3-coder-next:cloud |
| 5 | `MemoryAgent` | 5 | Store/recall 9 memory types, manage relationships, vector search, knowledge graph. | glm-5.2:cloud |
| 6 | `LiteratureAgent` | 6 | Multi-source search, 7-dim comparison, reviews, trends, recommendations, relevance scoring. | glm-5.2:cloud |
| 7 | `ExperimentAgent` | 7 | Launch (allowlisted, dry-run default), monitor, collect metrics + artifacts, detect failures. | minimax-m3:cloud |
| 8 | `EvaluationAgent` | 8 | Compare runs, training dynamics, statistical significance, next-experiment recommendations. | glm-5.2:cloud |
| 9 | `ResearchLoopAgent` | 9 | Orchestrate Phases 1–8 in iterative cycles with stopping conditions + approval gates + reports. | minimax-m3:cloud |
| 10 | `TaskAgent` | 11 | Terminal-first autonomous coding: analyze → plan → implement → diff → test. | minimax-m3:cloud |
| 11 | `ArchitectAgent` | 13 | Produces implementation plans grounded in repository memory + research context. | glm-5.2:cloud |
| 12 | `ReviewerAgent` | 13 | Reviews generated code changes; LLM + heuristic review producing structured feedback. | glm-5.2:cloud |
| 13 | `TestAgent` | 13 | Executes tests via TerminalTool, parses pytest failures, provides structured feedback. | minimax-m3:cloud |
| 14 | `FailureAnalyzer` | 14 | Diagnoses failures from test/review/impl errors; produces structured `FailureReport`. | glm-5.2:cloud |
| 15 | `RepairStrategist` | 14 | Generates ranked repair strategies from failure reports; category-keyed strategy map. | glm-5.2:cloud |
| 16 | `LiteratureDiscoveryAgent` | 15 | Discovers relevant papers and generates a literature review. | glm-5.2:cloud |
| 17 | `KnowledgeSynthesisAgent` | 15 | Synthesizes key findings, gaps, and trends from discovered papers. | glm-5.2:cloud |
| 18 | `HypothesisGeneratorAgent` | 15 | Generates testable hypotheses from knowledge synthesis. | glm-5.2:cloud |
| 19 | `ResearchExperimentPlannerAgent` | 15 | Designs experiments to test hypotheses. | glm-5.2:cloud |
| 20 | `ExperimentExecutorAgent` | 15 | Executes experiments (dry-run default for safety). | minimax-m3:cloud |
| 21 | `ResultAnalyzerAgent` | 15 | Analyzes experiment results and updates hypothesis status. | glm-5.2:cloud |
| 22 | `ReportGeneratorAgent` | 15 | Generates the final research report with evidence and conclusions. | glm-5.2:cloud |
| 23 | `ResearchOrchestrator` | 15 | Top-level coordinator for end-to-end research workflows. | minimax-m3:cloud |

Every agent constructor accepts an optional `llm: LLMProvider` and exposes `agent_name` + `llm_provider`. **No agent instantiates a model directly** — they all go through `resolve_llm()`.

---

## 5. Tool Ecosystem

**61 typed tools** following a uniform `Tool[InputType, OutputType]` ABC (`async execute`, `async validate`, `ToolError`). Inputs and outputs are Pydantic v2 models — fully typed, validated, serializable.

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
| 11 | `TerminalTool` (run_command, read_file, write_file, search_code, apply_patch, git_status, git_diff) |
| 12 | `RepositoryIndexer`, `SymbolGraph`, `HashingEmbedder`, `InMemoryVectorBackend`, `HybridRetriever`, `RepositoryMemoryStore`, `RepositoryMemory` |
| 13 | `DelegationFramework`, `AgentDescriptor`, `SharedTaskContext` |
| 14 | `SelfRepairFramework`, `FailureAnalyzer`, `RepairStrategist` |
| 15 | `ResearchWorkflowFramework`, `ResearchOrchestrator` |

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
                                    23 agents
                                    resolve_llm()
```

### Resolution rules

1. An explicit `LLMProvider` passed to an agent constructor **wins**.
2. `llm_enabled=False` (e.g. `RepositoryAgent` default) → no provider attached.
3. Otherwise `ModelRouter.for_agent(agent_name)` resolves from `llm_config.yaml`.

### Per-agent model routing

Each agent uses a **different model**, configured in one file. The platform uses three specialized models:

- **qwen3-coder-next:cloud** — coding (CodingAgent)
- **glm-5.2:cloud** — reasoning (Research, Planning, Literature, Evaluation, Architecture, Review, Analysis)
- **minimax-m3:cloud** — orchestration (TaskAgent, ResearchLoop, Experiment, Test, ResearchOrchestrator)

```yaml
# llm_config.yaml
default_provider: ollama
default_model: glm-5.2:cloud
providers:
  ollama:
    type: ollama
    base_url: https://api.olama.cloud
    api_key: ${OLLAMA_API_KEY}      # expanded from the environment
    default_model: glm-5.2:cloud
    timeout: 60
agents:
  CodingAgent:            {provider: ollama, model: qwen3-coder-next:cloud}
  ResearchAgent:          {provider: ollama, model: glm-5.2:cloud}
  TaskAgent:              {provider: ollama, model: minimax-m3:cloud}
  ResearchOrchestrator:   {provider: ollama, model: minimax-m3:cloud}
  # ... 23 agents total
```

**Switching a model is a config-only change** — no source edits. Adding a new provider is `register_provider_type()` + a YAML block.

### Environment variables

| Variable | Default |
|----------|---------|
| `RE_LLM_CONFIG` | `llm_config.yaml` at repo root |
| `OLLAMA_BASE_URL` | `https://api.olama.cloud` |
| `OLLAMA_API_KEY` | (none) |
| `OLLAMA_MODEL` / `OLLAMA_DEFAULT_MODEL` | `glm-5.2:cloud` |
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

### Run a terminal-first autonomous coding task (Phase 11)

```bash
# Analyze → plan → implement → diff → (optionally) test
research-engineer task "Add EMA checkpoint support" --repo ./my_repo

# With multi-agent delegation (Phase 13) + self-repair (Phase 14)
research-engineer task "Add EMA checkpoint support" --delegate --max-repairs 3
```

### Build repository memory (Phase 12)

```bash
# Index the repository for semantic code retrieval
research-engineer memory build --repo ./my_repo

# Query for relevant code
research-engineer memory query "checkpoint saving logic" --repo ./my_repo

# Explore the symbol graph
research-engineer memory symbol-graph "Trainer" --repo ./my_repo
```

### Run an autonomous research workflow (Phase 15)

```bash
# Literature review → synthesis → hypotheses → experiments → analysis → report
research-engineer research "Design a more efficient diffusion transformer"
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

### Example D — Terminal-first autonomous coding with delegation (Phases 11–14)

```bash
export OLLAMA_API_KEY="..."
# Build repository memory first for context-aware coding
research-engineer memory build --repo ./my_repo

# Run a delegated task with self-repair
research-engineer task "Add EMA checkpoint support" \
  --repo ./my_repo \
  --delegate \
  --max-repairs 3 \
  --run-tests
# -> Automatically: analyze → research → architect → code → review → test → repair → report
```

### Example E — End-to-end autonomous research workflow (Phase 15)

```bash
export OLLAMA_API_KEY="..."
research-engineer research "Design a more efficient diffusion transformer architecture" \
  --max-papers 30 \
  --max-hypotheses 5 \
  --dry-run
# -> output/research/<workflow_id>/research_report.md
#    Stages: literature → synthesis → hypotheses → experiments → analysis → report
```

---

## 12. CLI Reference

**56 commands** across 7 sub-apps. Run `research-engineer <command> --help` for full flags.

| Sub-app | Command | Purpose |
|---------|---------|---------|
| **core** | `analyze <paper>` | Analyze a paper (arXiv ID / URL / PDF). |
| | `analyze-repo <path>` | Analyze a repository. |
| | `plan <paper> <repo>` | Generate a 9-file experiment plan. |
| | `implement` | Generate patches + tests + reports. |
| | `task <goal>` | Terminal-first autonomous coding (Phase 11). |
| | `research <goal>` | Autonomous research workflow (Phase 15). |
| | `get <paper_id>` | Retrieve a stored paper. |
| | `search <query>` | Search stored papers. |
| | `history` / `cache-status` | Analysis history / cache stats. |
| **memory** | `memory search|list|stats|related|graph|export|import|archive` | Query and manage research memories. |
| | `memory build|refresh|stats|query|symbol-graph` | Repository memory (Phase 12). |
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

### Demo 5 — Terminal-first autonomous coding with delegation (Phases 11–14)

```bash
export OLLAMA_API_KEY="..."
research-engineer memory build --repo ./my_repo
research-engineer task "Add EMA checkpoint support" \
  --repo ./my_repo --delegate --max-repairs 3 --run-tests
```

### Demo 6 — End-to-end autonomous research workflow (Phase 15)

```bash
export OLLAMA_API_KEY="..."
research-engineer research "Design a more efficient diffusion transformer" \
  --max-papers 30 --max-hypotheses 5 --dry-run
# -> output/research/<workflow_id>/research_report.md
```

---

## 14. Testing Statistics

| Metric | Value |
|--------|-------|
| **Total tests** | 878 passing |
| **Phase 10 (LLM) tests** | 29 |
| **Phase 11 (Task/Terminal) tests** | 60 |
| **Phase 12 (Repository Memory) tests** | 51 |
| **Phase 13 (Delegation) tests** | 31 |
| **Phase 14 (Self-Repair) tests** | 31 |
| **Phase 15 (Research Workflow) tests** | 39 |
| **Source files** | 120+ Python files |
| **Pydantic models** | 186 |
| **Typed tools** | 61 |
| **Agents** | 23 + LLM layer |
| **CLI commands** | 56 |
| **Phases complete** | 15 / 15 |

```bash
uv run pytest -q          # 878 passed
uv run mypy src/research_engineer/llm   # clean
uv run ruff check .       # lint
```

Test coverage spans every phase: models, tools, agents, CLI, and end-to-end integration (`test_integration.py`, `test_integration_phases.py`).

---

## 15. Roadmap

### Completed (v2.0)

| Phase | Status | Description |
|-------|--------|-------------|
| 1–10 | ✅ Complete | Paper analysis through LLM layer (v1.0) |
| 11 | ✅ Complete | Terminal-first autonomous coding agent |
| 12 | ✅ Complete | Repository memory with hybrid retrieval |
| 13 | ✅ Complete | Multi-agent delegation framework |
| 14 | ✅ Complete | Autonomous self-repair with structured failure analysis |
| 15 | ✅ Complete | End-to-end autonomous research workflows |

### Planned

- **v2.1** — Additional providers (OpenAI, Anthropic, local Ollama) behind the same `LLMProvider` ABC.
- **v2.2** — Streaming-first agent outputs; structured tool-calling for the CodingAgent.
- **v2.3** — Web UI dashboard for loop monitoring + knowledge-graph visualization.
- **v2.4** — Multi-repo experiment matrices; distributed experiment execution.
- **v3.0** — Self-improving meta-loop: the platform proposes its own research goals from memory trends.

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
├── agents/      # 23 agents + delegation + self-repair + research workflow
├── llm/         # Phase 10: base, ollama_provider, factory, router
├── memory/      # Phase 12: indexer, symbol_graph, retriever, storage
├── models/      # 186 Pydantic models across 18 modules
├── tools/       # 61 typed tools
└── cli/         # 56 Typer commands
tests/           # 45+ test files, 878 tests
llm_config.yaml  # provider + per-agent model config
docs/            # 13 documentation files
```

---

<div align="center">

**Built for ML practitioners who want research automation that's typed, testable, and provider-agnostic.**

Star the repo if it's useful · Open an issue if it's not · PRs welcome

</div>
