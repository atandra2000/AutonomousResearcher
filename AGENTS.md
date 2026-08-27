# AGENTS.md - Autonomous ML Research Engineer

> **Project:** `AutonomousMLResearchEngineer/` · **Type:** 15-phase
> multi-agent ML research platform + production deployment stack ·
> **Version:** 0.9.0 · **Stats:** 23 agents · 61 typed tools · 17 pydantic
> v2 schema modules (251 classes) · **1464 tests — 1462 passing, 2 network-skipped** · plus agent-eval
> harness (`eval/`, E4), continuous-improvement loop (`improve/`, E8),
> P1/P2 research benchmarks (`benchmark` CLI), and a Docker Compose
> production stack (`deploy/`: FastAPI api + worker + Postgres queue +
> OTel collector, E7). **Stack:** Python 3.12, pydantic v2, typer, httpx,
> arxiv, pymupdf, chromadb, sentence-transformers, pytest-asyncio, ruff,
> mypy.

This file is the **developer reference manual** for the platform. It is
intentionally detailed (architecture diagrams, source tree, CLI command
list, per-phase model inventory) — heavier than the per-project AGENTS.md
in other repos because the platform itself is one massive project.

---

## Quick Commands

```bash
# Paper analysis
research-engineer analyze <arxiv_id_or_url_or_pdf>
research-engineer analyze-repo <path>

# Planning & Implementation
research-engineer plan <paper> <repo>
research-engineer implement --paper|--task|--plan <repo>

# Literature intelligence
research-engineer literature {discover|search|review|compare|trends|recommend|relevance|relationships} <topic>
research-engineer literature discover <topic>     # full workflow

# Experiments
research-engineer experiment run --command <cmd> --repo <path>
research-engineer experiment {monitor|list|get|search|cancel|history} <id>

# Evaluation
research-engineer evaluate {run|compare|analyze|dynamics|significance|next|list|get|search} <ids>

# Autonomous loops
research-engineer loop run <goal> --repo <path> [--max-iterations N] [--target-metric loss --target-value 0.01] [--approval]
research-engineer loop {list|get|iterations|iteration|search|report} <loop_id>

# Memory (Phase 12)
research-engineer memory {build|query|symbol-graph} --repo <path>

# LLM layer (Phase 10)
research-engineer llm {status|config [--config path]}

# Top-level workflows
research-engineer task <goal> --repo <path>      # Phase 11: terminal-first autonomous coding
research-engineer research <goal>                # Phase 15: end-to-end paper→report

# Agent evaluation & self-improvement
research-engineer eval-harness --suite <name>    # E4: graded eval suites (deterministic + LLM-judged)
research-engineer improve <report_id>            # E8: mine failures → propose/promote improvements

# Benchmarks
research-engineer benchmark {p1|p2|list|compare} # P1 deterministic / P2 LLM-backed suites

# Review gate
research-engineer review                         # interactive approval of plans/patches/experiments

# Production deployment (E7 stack: api + worker + postgres + otel-collector)
cd deploy && cp .env.example .env                # set POSTGRES_PASSWORD + RE_SERVICE_API_TOKEN
docker compose up -d --build
curl -s localhost:8000/health && curl -s localhost:8000/ready
scripts/smoke_test.sh [--down]                   # full deploy→run→crash→recover probe

# Dev — NOTE: bare `uv run pytest` resolves to Homebrew's Python 3.14
# pytest on this machine and fails collection; always go through python -m.
uv run python -m pytest
uv run ruff check .
scripts/ci_mypy.sh                               # mypy: fail only on errors NEW vs baseline
# (configs/mypy-baseline.txt carries the legacy mypy debt; the gate
#  guarantees it never grows. Re-baseline deliberately after fixes.)
```

## Input Detection

- **arXiv ID:** `^\d{4}\.\d{5}$` (e.g. `2503.12345`)
- **arXiv URL:** `^https?://arxiv.org/(abs|pdf)/\d{4}\.\d{5}(:?v\d+)?$`
- **PDF file:** `.*\.pdf$`

---

## Architecture (15 phases)

Each phase has an Agent class, a set of Tools, and pydantic v2 schemas.
Phase 10 is the provider-agnostic LLM layer that every LLM-using agent
shares. Phase 12 is repository memory (AST index + symbol graph +
vector store). Phase 14 is the self-repair loop. Phase 15 is the
end-to-end paper→report orchestrator.

### Phase 1: Paper Analysis

```
CLI → ResearchAgent → [ArxivTool | PDFTool] → ParserTool → StorageTool
                                                          ↓
                                                  ResearchSummary (14 fields)
                                                  EngineeringReport
                                                          ↓
                                                  SQLite + output files
```

### Phase 2: Repository Analysis

```
CLI → RepositoryAgent → [ScannerTool | ASTAnalyzerTool | DependencyGraphTool
                          | TrainingPipelineTool | ConfigAnalyzerTool
                          | KnowledgeGraphTool | DocumentationTool]
                                 ↓
                          RepositorySummary → output files
```

### Phase 3: Experiment Planning

```
CLI → ExperimentPlannerAgent
              ↓
      ResearchAgent.analyze() + RepositoryAgent.analyze()
              ↓
      CompatibilityAnalysis → ImplementationPlanner → ImpactAnalysis
      → ExperimentDesign → ValidationPlanner → RiskAssessment
      → ComputeEstimator → ResultPrediction
              ↓
      PlanResult → 8 markdown files + plan_result.json
```

### Phase 4: Code Implementation

```
CLI → CodingAgent
              ↓
      CodeGeneration → PatchGeneration → SelfReview → TestGeneration
      → MigrationPlanner → RollbackPlanner → ImplementationReport
              ↓
      ImplementationResult → patches + tests + reports
```

### Phase 5: Research Memory

```
CLI → MemoryAgent
              ↓
      [MemoryStorage | VectorStore | EmbeddingStrategy | QueryProcessor]
              ↓
      PaperMemory + RepositoryMemory + ExperimentPlanMemory + PatchMemory
      + ArchitectureDecisionMemory + ResearchInsightMemory
      + FailedApproachMemory + SuccessfulApproachMemory
              ↓
      SQLite + ChromaDB + Knowledge Graph → semantic retrieval
```

### Phase 6: Literature Intelligence

```
CLI → LiteratureAgent
              ↓
      PaperSearchTool (local + arXiv + Semantic Scholar)
              ↓
      [PaperComparisonTool | PaperRelationshipTool | TrendAnalysisTool]
              ↓
      LiteratureReviewTool → structured review synthesis
              ↓
      [PaperRecommendationTool | RelevanceScoringTool]
              ↓
      MemoryAgent.store_insight() + MemoryKnowledgeGraph.add_relationship()
              ↓
      LiteratureResult → markdown + JSON output files
```

### Phase 7: Experiment Execution

```
CLI → ExperimentAgent
              ↓
      ExperimentRunnerTool (launch training/eval subprocess)
      MonitoringTool (analyze output, scan checkpoints)
      MetricCollectorTool (parse metrics from logs/json/csv)
      ArtifactCollectorTool (copy checkpoints, logs, plots, configs)
      FailureDetectorTool (classify outcome, detect anomalies)
      ExperimentStorageTool (persist to SQLite + output files)
      MemoryAgent.store_success() / store_failure()
      MemoryKnowledgeGraph.add_relationship() (auto graph update)
```

### Phase 8: Evaluation

```
CLI → EvaluationAgent
      → ExperimentComparisonTool + TrainingDynamicsTool
      + StatisticalSignificanceTool (Welch t-test, Cohen's d, 95% CIs)
      + NextExperimentTool (rule-based recs + paper query)
      → EvaluationStorageTool (SQLite)
      → MemoryAgent.store_insight() / store_success() / store_failure()
```

### Phase 9: Autonomous Research Loop

```
CLI → ResearchLoopAgent (orchestrator)
              ↓
      LoopConfig + LoopState (state machine: created→running→iterating→evaluated→stopped)
              ↓
      ┌─── Iteration cycle (repeated) ───────────────────────────┐
      │  1. MemoryAgent.get_context() (recall past insights)      │
      │  2. LiteratureAgent.discover() (first iteration)          │
      │  3. ExperimentPlannerAgent.plan()                         │
      │  4. CodingAgent.implement()                                │
      │  5. ExperimentAgent.run() (dry-run default)               │
      │  6. EvaluationAgent.analyze()                             │
      │  7. LoopStorageTool (persist iteration)                   │
      │  8. MemoryAgent.store_success/failure/insight()           │
      │  9. MemoryKnowledgeGraph.add_node/add_relationship()      │
      │ 10. StoppingConditionChecker (target/max/budget/no-improve)│
      │ 11. [Approval gates if approval_mode=True]                │
      └────────────────────────────────────────────────────────────┘
              ↓
      ReportGeneratorTool → research_report.md + research_report.json
```

### Phase 10: Provider-Agnostic LLM Layer

```
llm_config.yaml
      ↓
ProviderFactory ──builds──► OllamaCloudProvider (cached)
      ↓
ModelRouter.for_agent(name) ──binds model──► _BoundProvider
      ↓
agents/_llm_support.resolve_llm(agent_name, llm)
      ↓
agent.llm_provider (every agent exposes this)
      ↓
await provider.complete(LLMRequest) → LLMResponse
```

Resolution rules:
1. Explicit `LLMProvider` passed to agent constructor wins.
2. `llm_enabled=False` (e.g. `RepositoryAgent` default) → no provider.
3. Otherwise `ModelRouter.for_agent(agent_name)` resolves from
   `llm_config.yaml`.

### Phase 11: Terminal-First Autonomous Coding

```
CLI → TaskAgent → TerminalTool (7 ops: run_command, read_file,
                write_file, search_code, apply_patch, git_status, git_diff)
                → Analyze → Plan → Implement → Diff → (Optional) Test
                → TaskResult → output/tasks/<task_id>/
```

### Phase 12: Repository Memory

```
CLI → RepositoryMemory
      → RepositoryIndexer (AST parsing → symbols + chunks)
      → SymbolGraph (deps, callers, callees, related, tests)
      → HashingEmbedder (offline lightweight embeddings)
      → InMemoryVectorBackend (no heavy deps)
      → HybridRetriever (semantic + graph + metadata)
      → RepositoryMemoryStore (SQLite-backed, incremental refresh)
```

### Phase 13: Multi-Agent Delegation

```
CLI → TaskAgent --delegate
              ↓
      DelegationFramework (AgentRole, AgentCapability, SharedTaskContext)
              ↓
      ArchitectAgent → CodingAgent → ReviewerAgent → TestAgent
              ↓
      Delegated result with repair loop
```

### Phase 14: Autonomous Self-Repair

```
SelfRepairFramework
      ↓
FailureAnalyzer (FailureReport with root cause + severity)
RepairStrategist (ranked repair strategies by category)
      ↓
Termination: SUCCESS | BUDGET_EXHAUSTED | NO_STRATEGIES | STAGNATION
```

### Phase 15: End-to-End Research Workflows

```
CLI → ResearchOrchestrator
              ↓
      ResearchWorkflowFramework (7 skippable stages)
              ↓
      1. LiteratureDiscoveryAgent
      2. KnowledgeSynthesisAgent
      3. HypothesisGeneratorAgent
      4. ResearchExperimentPlannerAgent
      5. ExperimentExecutorAgent
      6. ResultAnalyzerAgent
      7. ReportGeneratorAgent
              ↓
      output/research/<workflow_id>/research_report.md + .json
```

---

## Source Structure (top-level only — full tree is 200+ files)

```
src/research_engineer/
├── cli/                       # Typer CLI (single main.py, 20 command families, 70+ commands)
├── agents/                    # one file per agent + support (_llm_support, _adapters,
│                              #   _streaming, research_stages, research_workflow, delegation)
├── memory/                    # Phase 12: indexer, symbol graph, embeddings, retriever,
│                              #   vector backends, storage
├── llm/                       # Phase 10: base ABC, factory, router + providers
│                              #   (ollama cloud/local, openai, anthropic), react_loop,
│                              #   streaming, resilience, cost
├── eval/                      # E4 agent-eval harness: graders, metrics, runner, scripted suites
├── improve/                   # E8 continuous improvement: mining, proposals, gate, pipeline
├── gateway/ runtime/ service/ safety/ observability/   # E1–E3/E5–E7 platform infra layers
├── models/                    # 17 pydantic v2 schema modules, 251 classes total
└── tools/                     # 61 typed tools + base.py, base_cache.py, rate_limiter, _stats.py

deploy/                        # E7 production stack: Dockerfile, docker-compose.yml
                               #   (api + worker + postgres + otel-collector), .env.example
scripts/                       # smoke_test.sh (deploy probe), ci_mypy.sh (baseline gate)
configs/                       # mypy-baseline.txt + experiment configs
docs/                          # 20 documentation files (architecture, deployment, CLI, …)
```

Model schema modules (`models/`): paper, summary, plan, planner, repo,
coding, memory, literature, experiment, evaluation, loop, task, repair,
delegation, research, storage, ast_models — the three largest are
`experiment.py`, `evaluation.py`, and `literature.py` (30+ classes each).

---

## Platform Infra Layers (E1–E8)

Beyond the 15 research phases, the repo ships the production hardening
layers (all under `src/research_engineer/` unless noted):

- **E1 runtime/** — `AgentRuntime`: long-running agent execution with
  resumable steps.
- **E2 runtime/checkpoints** — versioned, full-payload checkpoint store
  (Postgres in deployment; recovery resumes from the last checkpoint).
- **E3 gateway/** — tool-call policy/sandboxing (command allowlist,
  working-directory confinement, timeouts).
- **E4 eval/** — graded agent-eval suites (deterministic + LLM-judged).
- **E5 observability/** — structured logs, metrics; optional OpenTelemetry
  export via the `telemetry` extra + `deploy/otel-collector-config.yaml`.
- **E6 safety/** — approval gates and guardrails shared by agents.
- **E7 service/ + deploy/** — FastAPI run API (`POST /runs`,
  status/cancel/resume/result, bearer auth, `/health`, `/ready`),
  Postgres-backed queue (`FOR UPDATE SKIP LOCKED`), worker with heartbeat
  + crash recovery, Docker Compose stack, `scripts/smoke_test.sh`.
- **E8 improve/** — mines E4 reports into improvement candidates with an
  approve/promote gate (PostgreSQL-backed store).

---

## Hard rules (project-specific; workspace rules live in root AGENTS.md)

1. **`uv run` everywhere** — never `python` directly (root rule).
2. **Pytest-then-lint-then-mypy** before declaring any change complete.
3. **Never** reduce test coverage below **1462 passing** (2 network-skipped).
4. **Repository-agnostic** — never hardcode assumptions about specific repos.
5. **Paper-agnostic** — must work for any ML paper (attention, MoE,
   diffusion, etc.).
6. **Patch-first philosophy** — Phase 4 generates patches for review, does
   not directly modify code by default.
7. **No LLM in Phase 1-3** — rule-based extraction only (no API costs).
8. **Pydantic v2** — all I/O models are typed Pydantic models.
9. **Python 3.12+** — `requires-python = ">=3.12"`; **StrEnum** for all
   enums (not `str, Enum`).
10. **Type checking:** `disallow_untyped_defs = true`, `strict_optional = true`.
11. **Line length:** 88 chars (ruff).

---

## Tool Interface Pattern

All tools follow `Tool[Input, Output]` ABC:
- `async execute(input: InputType) -> OutputType`
- `async validate(input: InputType) -> bool`
- `ToolError` on failure

## LLM Layer env vars

| Variable | Used by | Default |
|----------|---------|---------|
| `RE_LLM_CONFIG` | factory | `llm_config.yaml` at repo root |
| `OLLAMA_BASE_URL` | OllamaCloudProvider | `https://ollama.com` |
| `OLLAMA_API_KEY` | OllamaCloudProvider | (none) |
| `OLLAMA_MODEL` / `OLLAMA_DEFAULT_MODEL` | OllamaCloudProvider fallback | `llama3` (routing default comes from `llm_config.yaml`: `glm-5.3-flash`) |
| `OLLAMA_TIMEOUT` | OllamaCloudProvider | `60` |

## Storage

- **DB:** `data/research_engineer.db` (SQLite)
- **Vector store:** `data/vector_store/` (ChromaDB)
- **Per-phase output:** `output/<phase>/<id>/` (markdown + JSON)

## Test Status

**1464 tests — 1462 passing, 2 network-skipped** (verified via
`uv run python -m pytest -q`, 2026-08-27) — never reduce. Includes the
original phase suites plus tests for the E4 eval harness, E8 improvement
loop, the E1–E7 platform layers (runtime, checkpoints, gateway policy,
service API, Postgres queue), LLM layer extensions (openai / anthropic /
local ollama providers, streaming, resilience), and P1/P2 benchmarks.
Coverage target >90%.

## CI / Type-Debt Policy

- `configs/mypy-baseline.txt` records all current mypy errors (normalized,
  line-number-free). `scripts/ci_mypy.sh` fails only on **new** errors —
  the legacy debt can shrink but never grow. Re-baseline deliberately
  after fixing a batch.
- GitHub Actions runs pytest + ruff + the mypy baseline gate on every push
  (see `.github/workflows/ci.yml`).
- `network` pytest marker: tests needing live internet (arXiv API) are
  skipped when offline — CI runs fully offline.