# Agents Reference

Deep-dive into the nine agents that power the Autonomous ML Research Engineer. Each agent has a single responsibility, a typed result model, and an LLM provider resolved through `resolve_llm()`.

> See [Architecture](architecture.md) for how agents fit together and [LLM Integration](llm_integration.md) for provider routing.

---

## Common patterns

Every agent shares these traits:

- **Constructor** accepts optional tool/agent dependencies (dependency injection) and an optional `llm: LLMProvider`.
- **`agent_name`** — canonical name used by `ModelRouter` for LLM routing.
- **`llm_provider`** — resolved provider (explicit > router > disabled); may be a model-bound `_BoundProvider`.
- **Async-first** — all entry-point methods are `async def`.
- **No direct model calls** — agents obtain providers via `resolve_llm()`.

```python
from research_engineer.agents._llm_support import resolve_llm

class SomeAgent:
    def __init__(self, ..., llm: LLMProvider | None = None) -> None:
        self.agent_name = "SomeAgent"
        ...
        self.llm_provider = resolve_llm(self.agent_name, llm)
```

---

## 1. ResearchAgent (Phase 1)

**File:** `agents/research_agent.py`
**Responsibility:** Acquire a paper (arXiv/PDF), parse it, produce a structured `ResearchSummary` + `EngineeringReport`, and store the result.

### Constructor

```python
ResearchAgent(
    arxiv_tool: ArxivTool | None = None,
    pdf_tool: PDFTool | None = None,
    parser_tool: PaperParserTool | None = None,
    storage_tool: StorageTool | None = None,
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async analyze(paper_input, output_dir="output") -> dict` | Main entry point. Detects input type (arXiv ID/URL/PDF), acquires, parses, generates summary + plan, stores. Returns `AnalysisResult` dict. |

### Workflow

1. `_detect_input_type()` — regex on arXiv ID (`^\d{4}\.\d{5}$`), arXiv URL, or `.pdf` path.
2. Acquire via `ArxivTool` or `PDFTool`.
3. Parse via `PaperParserTool` → sections, figures, tables, equations.
4. `_generate_summary()` — rule-based extraction of 14 fields (no LLM).
5. `_generate_plan()` — `EngineeringReport` with complexity, files, effort, dependencies.
6. Store via `StorageTool` (SQLite `papers` table).

### Output

```
output/<paper_id>_summary.json   # 14-field ResearchSummary
output/<paper_id>_plan.json      # EngineeringReport
```

---

## 2. RepositoryAgent (Phase 2)

**File:** `agents/repository_agent.py`
**Responsibility:** Scan a repository, analyze its AST, build dependency/training/config graphs, construct a knowledge graph, and generate documentation.

### Constructor

```python
RepositoryAgent(
    scanner: RepositoryScannerTool | None = None,
    ast_analyzer: ASTAnalysisTool | None = None,
    dependency_graph: DependencyGraphTool | None = None,
    training_pipeline: TrainingPipelineTool | None = None,
    config_analyzer: ConfigAnalysisTool | None = None,
    knowledge_graph: KnowledgeGraphTool | None = None,
    documentation: DocumentationTool | None = None,
    enable_caching: bool = True,
    cache_path: str = ".cache/repo_analysis",
    rate_limit_enabled: bool = False,
    llm_enabled: bool = False,        # LLM off by default
    llm_model: str = "llama3",       # legacy LlamaIndex fallback
    llm: LLMProvider | None = None,
)
```

> **Note:** `RepositoryAgent` is the only agent with `llm_enabled=False` by default. Pass `llm_enabled=True` (or an explicit `llm=`) to enable LLM-powered architecture analysis.

### Methods

| Method | Description |
|--------|-------------|
| `async analyze(repo_path, output_dir="output", enable_llm=None) -> dict` | Full repo analysis. Returns repository_name, project_type, architecture_summary, important_files, generated_files, analysis_time_seconds, output_dir, and optional llm_analysis. |

### Workflow

1. Scan structure (`RepositoryScannerTool`).
2. AST analysis (`ASTAnalysisTool`) — classes, functions, complexity.
3. Dependency graph (`DependencyGraphTool`).
4. Training pipeline extraction (`TrainingPipelineTool`).
5. Config analysis (`ConfigAnalysisTool`) — YAML/JSON/TOML.
6. Knowledge graph (`KnowledgeGraphTool`).
7. File-importance ranking + implementation targets.
8. Documentation generation (`DocumentationTool`).
9. Optional LLM architecture analysis (Phase 10 provider, or legacy LlamaIndex fallback).

---

## 3. ExperimentPlannerAgent (Phase 3)

**File:** `agents/experiment_planner_agent.py`
**Responsibility:** Given a paper and a repository, produce a 9-file experiment plan covering compatibility, implementation, impact, experiments, validation, risk, compute, and predictions.

### Constructor

```python
ExperimentPlannerAgent(
    research_agent: ResearchAgent | None = None,
    repository_agent: RepositoryAgent | None = None,
    compatibility_tool, implementation_tool, impact_tool, experiment_tool,
    validation_tool, risk_tool, compute_tool, prediction_tool,  # all optional
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async plan(paper_input, repo_path, output_dir="output", output_format="markdown") -> PlannerResult` | Runs Phases 1 + 2, then 8 planning tools, writes 9 files. |

### Workflow

1. `research_agent.analyze(paper_input)` → `ResearchSummary`.
2. `repository_agent.analyze(repo_path)` → `RepositorySummary`.
3. `CompatibilityAnalysisTool` (7 dimensions).
4. `ImplementationPlannerTool` (ordered steps).
5. `ImpactAnalysisTool` (7 dimensions).
6. `ExperimentDesignTool` (5 groups: baseline, MVE, ablation, stress, scaling).
7. `ValidationPlannerTool` (6 suites).
8. `RiskAssessmentTool` (7 categories).
9. `ComputeEstimatorTool` (GPU hours, memory, storage, cost).
10. `ResultPredictionTool` (best/likely/worst + failure modes).
11. Write 8 markdown files + `plan_result.json` to `output/plans/<paper>_<repo>/`.

---

## 4. CodingAgent (Phase 4)

**File:** `agents/coding_agent.py`
**Responsibility:** Generate code changes as reviewable patches, with self-review, tests, migration, and rollback plans. **Patch-first** — never mutates code directly.

### Constructor

```python
CodingAgent(
    research_agent: ResearchAgent | None = None,
    repository_agent: RepositoryAgent | None = None,
    code_generation_tool, patch_generation_tool, self_review_tool,
    test_generation_tool, migration_planner_tool, rollback_planner_tool,
    implementation_report_tool,  # all optional
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async implement(task_description, repo_path, paper_input=None, implementation_plan=None, output_dir="output", constraints=None, requirements=None) -> CodingAgentResult` | Full implementation workflow. |
| `async apply_patches(implementation_id, approved=False, dry_run=True) -> dict` | Apply previously-generated patches (explicit, approval-gated). |

### Workflow

1. (Optional) `research_agent.analyze(paper_input)` for context.
2. `repository_agent.analyze(repo_path)` for repo context.
3. `CodeGenerationTool` → `CodeChange` list.
4. `PatchGenerationTool` → unified-diff `GeneratedPatch` list.
5. `SelfReviewTool` → `ReviewResult` (architecture/style/performance checks).
6. `TestGenerationTool` → `TestSuite` list.
7. `MigrationPlannerTool` → data/checkpoint migration plan.
8. `RollbackPlannerTool` → rollback steps.
9. `ImplementationReportTool` → markdown reports.

### Output

```
output/<implementation_id>/
├── implementation_report.md
├── code_change_summary.md
├── patch_review.md
├── migration_plan.md
├── test_plan.md
├── rollback_plan.md
└── (patches + tests)
```

---

## 5. MemoryAgent (Phase 5)

**File:** `agents/memory_agent.py`
**Responsibility:** Store and retrieve 9 memory types, manage relationships, power semantic search, and maintain the knowledge graph. The platform's long-term brain.

### Constructor

```python
MemoryAgent(
    config: MemoryConfig | None = None,
    llm: LLMProvider | None = None,
)
# MemoryConfig: db_path, vector_store_path, embedding_model,
#               auto_consolidate, auto_detect_relationships,
#               min_relationship_confidence, batch_size, log_access
```

### Methods

| Method | Description |
|--------|-------------|
| `async store_paper(paper, summary, plan) -> str` | Store a `PaperMemory`. |
| `async store_repository(repo, analysis) -> str` | Store a `RepositoryMemory`. |
| `async store_plan(plan_result) -> str` | Store an `ExperimentPlanMemory`. |
| `async store_patch(...) -> str` | Store a `PatchMemory`. |
| `async store_decision(context, decision, rationale, alternatives=None) -> str` | Store an `ArchitectureDecisionMemory`. |
| `async store_insight(content, insight_type, confidence, tags) -> str` | Store a `ResearchInsightMemory`. |
| `async store_failure(content, failure_mode, context, tags) -> str` | Store a `FailedApproachMemory`. |
| `async store_success(content, outcome, context, tags) -> str` | Store a `SuccessfulApproachMemory`. |
| `async retrieve(memory_id) -> dict \| None` | Fetch by ID. |
| `async search(query, filters=None, limit=10) -> list[MemoryResult]` | Hybrid search. |
| `async get_context(current_task, limit=5) -> list[MemoryResult]` | Recall relevant memories (used by the loop). |
| `async get_related(memory_id, max_depth=2) -> list[MemoryResult]` | Graph traversal. |
| `async get_stats() -> MemoryStats` | Memory statistics. |

### Backends

- **SQLite** — `memories`, `memory_relationships`, `memory_access_log`, `memory_versions` tables.
- **ChromaDB** — vector embeddings (`all-mpnet-base-v2`).
- **Knowledge graph** — `MemoryKnowledgeGraph` with 10 relationship types.

See [Memory System](memory_system.md) for the full deep dive.

---

## 6. LiteratureAgent (Phase 6)

**File:** `agents/literature_agent.py`
**Responsibility:** Search, compare, review, relate, trend-analyze, recommend, and score relevance of papers — across local store, arXiv, and Semantic Scholar.

### Constructor

```python
LiteratureAgent(
    memory_agent: Any | None = None,
    search_tool, comparison_tool, review_tool, relationship_tool,
    trend_tool, recommendation_tool, relevance_tool,  # all optional
    config: LiteratureConfig | None = None,
    llm: LLMProvider | None = None,
)
# LiteratureConfig: max_papers, review_depth, store_findings, ...
```

### Methods

| Method | Description |
|--------|-------------|
| `async discover(topic, repo_path=None, output_dir=None, max_papers=None) -> LiteratureResult` | Full workflow: search → compare → review → relationships → trends → recommend → relevance. |
| `async search_papers(query, sources=None, max_papers=None) -> PaperSearchOutput` | Multi-source search. |
| `async compare_papers(paper_ids) -> PaperComparisonOutput` | 7-dimension comparison. |
| `async detect_relationships(paper_ids) -> PaperRelationshipOutput` | Citation/extension/similarity/contradiction. |
| `async analyze_trends(topic) -> TrendAnalysisOutput` | Rising/stable/declining trends. |
| `async generate_review(topic, papers=None, depth=None) -> LiteratureReviewOutput` | Structured review. |
| `async recommend_papers(topic, criteria=None) -> PaperRecommendationOutput` | Ranked recommendations. |
| `async score_relevance(paper_id, repo_path) -> RelevanceScoringOutput` | 6-dimension paper↔repo relevance. |

All findings are stored in `MemoryAgent` and the knowledge graph.

---

## 7. ExperimentAgent (Phase 7)

**File:** `agents/experiment_agent.py`
**Responsibility:** Launch experiments safely (dry-run default, command allowlist), monitor them, collect metrics + artifacts, detect failures, and persist results.

### Constructor

```python
ExperimentAgent(
    memory_agent: Any | None = None,
    runner_tool, monitoring_tool, metric_tool, artifact_tool,
    failure_tool, storage_tool,  # all optional
    config: ExperimentConfig | None = None,
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async run(command, repo_path, paper_id=None, plan_id=None, patch_id=None, implementation_id=None, experiment_type=TRAINING, timeout_seconds=None, env_vars=None, dry_run=None, output_dir=None) -> ExperimentResult` | Full execution workflow. |
| `async run_training(...)` / `run_evaluation(...)` / `run_validation(...)` | Convenience wrappers for typed runs. |
| `async list_experiments(status=None, limit=50) -> list[ExperimentRecord]` | List runs. |
| `async get_experiment(experiment_id) -> ExperimentRecord \| None` | Fetch one. |
| `async search_experiments(query, limit=20) -> list[ExperimentRecord]` | Text search. |
| `async cancel_experiment(experiment_id) -> bool` | Cancel a running experiment. |
| `async history(paper_id=None, limit=20) -> list[ExperimentRecord]` | History (optionally per paper). |

### Safety controls

- **Dry-run default** — commands are echoed, not executed, unless `dry_run=False`.
- **Command allowlist:** `python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`.
- **Timeouts** + working-directory confinement.

---

## 8. EvaluationAgent (Phase 8)

**File:** `agents/evaluation_agent.py`
**Responsibility:** Compare experiments, analyze training dynamics, run statistical significance tests, recommend next experiments, and store conclusions.

### Constructor

```python
EvaluationAgent(
    memory_agent: Any | None = None,
    literature_agent: Any | None = None,
    comparison_tool, dynamics_tool, significance_tool, next_tool,
    storage_tool,  # all optional
    config: EvaluationConfig | None = None,
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async analyze(experiments, paper_id=None, repo_path=None, output_dir=None) -> EvaluationResult` | Full evaluation: compare + dynamics + significance + next. |
| `async evaluate_single(experiment, ...) -> EvaluationResult` | Evaluate one experiment. |
| `async compare(experiments) -> ExperimentComparisonOutput` | Metric deltas, winner, shared/unique metrics. |
| `async dynamics_analysis(experiment) -> TrainingDynamicsOutput` | Over/underfit, convergence, instability, divergence, plateau. |
| `async significance_test(experiments) -> StatisticalSignificanceOutput` | Welch t-test, Cohen's d, 95% CIs (pure Python). |
| `async next_experiments(experiments, paper_id=None) -> NextExperimentOutput` | Rule-based recommendations + paper suggestions via `LiteratureAgent`. |
| `async list_evaluations(...) / get_evaluation(...) / search_evaluations(...)` | Query stored evaluations. |

### Statistical methods

Implemented in pure Python (`tools/_stats.py`) — **no SciPy dependency**:
- Welch's t-test (unequal variances)
- Cohen's d effect size
- 95% confidence intervals (normal approximation)

---

## 9. ResearchLoopAgent (Phase 9)

**File:** `agents/research_loop_agent.py`
**Responsibility:** Orchestrate Phases 1–8 in iterative cycles with memory recall, stopping conditions, optional approval gates, and a final research report.

### Constructor

```python
ResearchLoopAgent(
    memory_agent, literature_agent, repository_agent, research_agent,
    planner_agent, coding_agent, experiment_agent, evaluation_agent,  # all optional
    storage_tool: LoopStorageTool | None = None,
    stopping_checker: StoppingConditionChecker | None = None,
    report_generator: ReportGeneratorTool | None = None,
    config: LoopConfig | None = None,
    llm: LLMProvider | None = None,
)
```

### Methods

| Method | Description |
|--------|-------------|
| `async run(goal, repo_path, config=None, approval_callback=None) -> LoopResult` | Start a new loop. |
| `async generate_report(loop_id, output_dir=None) -> ReportOutput` | Generate `research_report.md` + `.json`. |
| `async list_loops(...) / get_loop(...) / search_loops(...)` | Query stored loops. |
| `async list_iterations(...) / get_iteration(...)` | Query iterations. |

### Loop state machine

```
created → running → iterating → (awaiting_approval) → evaluated → stopped
```

### Iteration cycle

1. `_recall_context(goal)` — `MemoryAgent.get_context()`.
2. `_run_literature()` — `LiteratureAgent.discover()` (first iteration only, unless `skip_literature_after_first=False`).
3. `_run_planning()` — `ExperimentPlannerAgent.plan()`.
4. `_run_implementation()` — `CodingAgent.implement()`.
5. `_run_experiment()` — `ExperimentAgent.run()` (dry-run default).
6. `_run_evaluation()` — `EvaluationAgent.analyze()`.
7. `LoopStorageTool` — persist the iteration.
8. `MemoryAgent.store_success/failure/insight()`.
9. `MemoryKnowledgeGraph.add_node/add_relationship()`.
10. `StoppingConditionChecker` — target/max/budget/no-improvement.
11. (Optional) `_request_approval()` if `approval_mode=True`.

### Stopping conditions

| Condition | Trigger |
|-----------|---------|
| `target_achieved` | `primary_metric_value` meets/exceeds `target_metric_value`. |
| `max_iterations_reached` | `iteration_count >= max_iterations`. |
| `budget_exceeded` | `cumulative_cost_hours >= budget_hours` or `cumulative_cost_usd >= budget_cost`. |
| `no_improvement` | No improvement over `stagnation_window` iterations (threshold = `improvement_threshold`). |

### Approval gates

`ApprovalGate` values: `plan`, `implementation`, `next_iteration`. When `approval_mode=True`, the loop calls the async `approval_callback` (if provided) and waits for human approval before proceeding.

### Output

```
output/loops/<loop_id>/
├── research_report.md    # executive summary, methodology, iteration history, findings, failures, conclusions, config
└── research_report.json  # full LoopResult serialized
```

---

## Agent LLM routing summary

| Agent | `agent_name` | Default routing |
|-------|--------------|-----------------|
| ResearchAgent | `ResearchAgent` | ollama / llama3 |
| RepositoryAgent | `RepositoryAgent` | *(none unless `llm_enabled=True`)* |
| ExperimentPlannerAgent | `ExperimentPlannerAgent` | ollama / llama3 |
| CodingAgent | `CodingAgent` | ollama / llama3 |
| MemoryAgent | `MemoryAgent` | ollama / llama3 |
| LiteratureAgent | `LiteratureAgent` | ollama / llama3 |
| ExperimentAgent | `ExperimentAgent` | ollama / llama3 |
| EvaluationAgent | `EvaluationAgent` | ollama / llama3 |
| ResearchLoopAgent | `ResearchLoopAgent` | ollama / llama3 |

Override any agent's model in `llm_config.yaml` — no source changes required.

```yaml
agents:
  CodingAgent: {provider: ollama, model: qwen2.5-coder}
```

---

*Version: 1.0 · 9 agents · all routed via `resolve_llm()`*