# Tools Reference

Reference for all **61 typed tools** in the Autonomous ML Research Engineer. Every tool follows the `Tool[InputType, OutputType]` ABC (`async execute`, `async validate`, `ToolError`). Inputs and outputs are Pydantic v2 models.

> See [System Design](system_design.md) for the tool interface contract and [Agents](agents.md) for how tools are wired into agents.

---

## Tool interface

```python
class Tool(ABC, Generic[InputType, OutputType]):
    @abstractmethod
    async def execute(self, input: InputType) -> OutputType: ...
    async def validate(self, input: InputType) -> bool: ...
    async def __call__(self, input: InputType) -> OutputType: ...
```

---

## Phase 1 — Paper Analysis

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `ArxivTool` | `ArxivInput → ArxivOutput` | Fetch paper metadata + PDF from arXiv by ID/URL. |
| `PDFTool` | `PDFInput → PDFOutput` | Extract text, figures, tables from a PDF (PyMuPDF). Accepts path/bytes/URL. |
| `PaperParserTool` | `ParserInput → ParserOutput` | Parse raw PDF content into sections, figures, tables, equations. |
| `StorageTool` | `StorageInput → StorageOutput` | Persist papers to SQLite (`papers` table). |

---

## Phase 2 — Repository Analysis

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `RepositoryScannerTool` | `RepoScanInput → RepoScanOutput` | Walk a repo, classify files, detect project type. |
| `ASTAnalysisTool` | `ASTInput → ASTOutput` | Parse Python AST: classes, functions, complexity, imports. |
| `DependencyGraphTool` | `DependencyInput → DependencyOutput` | Build file-level + module-level dependency graphs. |
| `TrainingPipelineTool` | `TrainingPipelineInput → TrainingPipelineOutput` | Detect training loops, optimizers, dataloaders, schedulers. |
| `ConfigAnalysisTool` | `ConfigInput → ConfigOutput` | Parse YAML/JSON/TOML configs; extract hyperparameters. |
| `KnowledgeGraphTool` | `KnowledgeGraphInput → KnowledgeGraphOutput` | Build a repo knowledge graph (entities + relations). |
| `DocumentationTool` | `DocumentationInput → DocumentationOutput` | Generate markdown docs (architecture, README, diagrams). |

---

## Phase 3 — Experiment Planning

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `CompatibilityAnalysisTool` | `CompatibilityInput → CompatibilityOutput` | 7 dimensions: architecture, API, data, compute, training, inference, deployment. |
| `ImplementationPlannerTool` | `ImplementationPlannerInput → ImplementationPlannerOutput` | Ordered implementation steps with targets per complexity level. |
| `ImpactAnalysisTool` | `ImpactAnalysisInput → ImpactAnalysisOutput` | 7 impact dimensions with confidence levels. |
| `ExperimentDesignTool` | `ExperimentDesignInput → ExperimentDesignOutput` | 5 groups: baseline, MVE, ablation, stress, scaling. |
| `ValidationPlannerTool` | `ValidationPlannerInput → ValidationPlannerOutput` | 6 suites: unit, integration, numerical-equivalence, regression, performance, checkpoint-compat. |
| `RiskAssessmentTool` | `RiskAssessmentInput → RiskAssessmentOutput` | 7 risk categories with mitigations. |
| `ComputeEstimatorTool` | `ComputeEstimatorInput → ComputeEstimatorOutput` | GPU hours, memory, storage, cloud cost. |
| `ResultPredictionTool` | `ResultPredictionInput → ResultPredictionOutput` | Best/likely/worst scenarios + failure modes. |

---

## Phase 4 — Code Implementation

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `CodeGenerationTool` | `CodeGenerationInput → CodeGenerationOutput` | Generate `CodeChange` list from plans/tasks/repo context. |
| `PatchGenerationTool` | `PatchGenerationInput → PatchGenerationOutput` | Create unified-diff `GeneratedPatch` list. |
| `SelfReviewTool` | `SelfReviewInput → SelfReviewOutput` | Automated review: architecture, style, performance checks → `ReviewResult`. |
| `TestGenerationTool` | `TestGenerationInput → TestGenerationOutput` | Generate `TestSuite` list (unit, integration, regression, edge cases). |
| `MigrationPlannerTool` | `MigrationPlannerInput → MigrationPlannerOutput` | Plan data + checkpoint migrations. |
| `RollbackPlannerTool` | `RollbackPlannerInput → RollbackPlannerOutput` | Plan rollback steps (data + checkpoint recovery). |
| `PatchApplicationTool` | `PatchApplicationInput → PatchApplicationOutput` | Apply patches (dry-run or actual). Explicit, approval-gated. |
| `ImplementationReportTool` | `ImplementationReportInput → ImplementationReportOutput` | Generate markdown reports (impl, change summary, patch review, migration, test, rollback). |

---

## Phase 5 — Research Memory

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `MemoryStorageTool` | `MemoryStorageInput \| MemoryQueryInput → MemoryStorageOutput \| MemoryQueryOutput` | SQLite persistence for memories; query by ID/type/tags/text. |
| `ChromaVectorStore` | (`VectorStore` impl) | Vector embeddings (`all-mpnet-base-v2`) for semantic search. |
| `EmbeddingStrategy` | (`EmbeddingConfig`) | Embed text for vector storage. |
| `QueryProcessor` | (`SemanticQuery`) | Process natural-language queries into structured search. |
| `MemoryKnowledgeGraph` / `MemoryGraphTool` | `MemoryGraphInput → MemoryGraphOutput` | Typed, weighted graph; 10 relationship types; `GraphStats`. |
| `RelationshipDetector` | `RelationshipDetectorInput → RelationshipDetectorOutput` | Auto-detect relationships between new and existing memories. |
| `DirectLookupStrategy` | (`RetrievalQuery`) | Retrieval by exact ID. |
| `SemanticSearchStrategy` | (`RetrievalQuery`) | Vector similarity search. |
| `GraphTraversalStrategy` | (`RetrievalQuery`) | Traverse the knowledge graph. |
| `TagBasedFilterStrategy` | (`RetrievalQuery`) | Filter by tags. |
| `TemporalQueryStrategy` | (`RetrievalQuery`) | Time-based retrieval. |
| `HybridSearchStrategy` | (`RetrievalQuery`) | Blend semantic + graph + tag signals. |
| `MemoryQueryTool` | `MemoryQueryToolInput → MemoryQueryToolOutput` | Agent-facing query tool. |
| `MemoryWriteTool` | `MemoryWriteToolInput → MemoryWriteToolOutput` | Agent-facing write tool. |
| `MemoryRecallTool` | `MemoryRecallToolInput → MemoryRecallToolOutput` | Agent-facing recall (context) tool. |

See [Memory System](memory_system.md) for the full deep dive.

---

## Phase 6 — Literature Intelligence

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `PaperSearchTool` | `PaperSearchInput → PaperSearchOutput` | Multi-source search (local + arXiv + Semantic Scholar), dedup, rank. |
| `PaperComparisonTool` | `PaperComparisonInput → PaperComparisonOutput` | 7 dimensions: similarity, difference, consensus, conflict, rankings. |
| `LiteratureReviewTool` | `LiteratureReviewInput → LiteratureReviewOutput` | Structured review: sections, timeline, gaps, recommendations. |
| `PaperRelationshipTool` | `PaperRelationshipInput → PaperRelationshipOutput` | Citation, extension, similarity, contradiction detection. |
| `TrendAnalysisTool` | `TrendAnalysisInput → TrendAnalysisOutput` | Rising/stable/declining trends; emerging/hot/declining topics. |
| `PaperRecommendationTool` | `PaperRecommendationInput → PaperRecommendationOutput` | Impact/novelty/implementability/relevance scoring + ranking. |
| `RelevanceScoringTool` | `RelevanceScoringInput → RelevanceScoringOutput` | 6 dimensions: architecture, training, inference, evaluation, data, feasibility. |

---

## Phase 7 — Experiment Execution

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `ExperimentRunnerTool` | `ExperimentRunnerInput → ExperimentRunnerOutput` | Launch subprocess with allowlist, dry-run, timeout, status tracking. |
| `MonitoringTool` | `MonitoringInput → MonitoringOutput` | Analyze output, scan checkpoints, detect anomalies. |
| `MetricCollectorTool` | `MetricCollectorInput → MetricCollectorOutput` | Parse metrics from logs/JSON/CSV; build `MetricSeries` + summaries. |
| `ArtifactCollectorTool` | `ArtifactCollectorInput → ArtifactCollectorOutput` | Glob-based artifact discovery, copy with checksum, size capping. |
| `FailureDetectorTool` | `FailureDetectorInput → FailureDetectorOutput` | Rule-based failure classification, root cause, recommendations. |
| `ExperimentStorageTool` | `ExperimentStorageInput \| ExperimentQueryInput → ExperimentStorageOutput \| ExperimentQueryOutput` | SQLite persistence; query by ID/paper/repo/status/type/text. |

### Experiment runner safety

- **Dry-run default** — commands echoed, not executed.
- **Command allowlist:** `python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`.
- **Timeouts** + working-directory confinement.

---

## Phase 8 — Evaluation

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `ExperimentComparisonTool` | `ExperimentComparisonInput → ExperimentComparisonOutput` | Metric deltas, winner selection, shared/unique metrics, findings. |
| `TrainingDynamicsTool` | `TrainingDynamicsInput → TrainingDynamicsOutput` | Over/underfit, convergence, instability, divergence, plateau detection. |
| `StatisticalSignificanceTool` | `StatisticalSignificanceInput → StatisticalSignificanceOutput` | Welch t-test, Cohen's d, 95% CIs (pure Python, no SciPy). |
| `NextExperimentTool` | `NextExperimentInput → NextExperimentOutput` | Rule-based recommendations + paper query. |
| `EvaluationStorageTool` | `EvaluationStorageInput \| EvaluationQueryInput → EvaluationStorageOutput \| EvaluationQueryOutput` | SQLite persistence; query by ID/paper/repo/experiment/text. |

---

## Phase 9 — Autonomous Loop

| Tool | Input → Output | Key logic |
|------|----------------|-----------|
| `LoopStorageTool` | `LoopStorageInput \| LoopQueryInput \| IterationStorageInput \| IterationQueryInput → ...` | SQLite persistence for loops + iterations; query by ID/loop/paper/status/text. |
| `StoppingConditionChecker` | `StoppingCheckInput → StoppingCheckOutput` | 4 conditions: target_achieved, max_iterations_reached, budget_exceeded, no_improvement. |
| `ReportGeneratorTool` | `ReportInput → ReportOutput` | Generate `research_report.md` + `.json` (executive summary, iteration history, findings, conclusions, config). |

---

## Phase 10 — LLM Layer

| Component | Type | Key logic |
|-----------|------|-----------|
| `LLMProvider` | ABC | Provider-agnostic interface; `complete`, `stream`, `validate`. |
| `OllamaCloudProvider` | `LLMProvider` | OpenAI-compatible Chat Completions over httpx; env fallbacks; streaming. |
| `ProviderFactory` | — | Builds/caches providers from config; `${VAR}` env expansion; registry. |
| `ModelRouter` | — | Resolves provider+model per agent; `_BoundProvider` pins model on every request. |

See [LLM Integration](llm_integration.md) for the full guide.

---

## Caching & rate limiting

| Component | File | Purpose |
|-----------|------|---------|
| `CacheBase` / `SimpleCache` / `FileCache` | `tools/base_cache.py` | In-memory + file caching for tool results. |
| `TokenBucketRateLimiter` / `SlidingWindowRateLimiter` | `tools/base_cache.py` | Rate limiting for external APIs. |
| `AsyncRateLimiter` | `tools/rate_limiter.py` | Async-aware rate limiter. |
| `generate_key` | `tools/base_cache.py` | Cache key generation. |

---

## Adding a new tool

1. Subclass `Tool[YourInput, YourOutput]` (Pydantic models for I/O).
2. Implement `async execute()` and optionally `async validate()`.
3. Export from `tools/__init__.py`.
4. Wire into the relevant agent constructor.
5. Add tests under `tests/`.

```python
from research_engineer.tools.base import Tool, ToolError

class MyInput(BaseModel):
    query: str

class MyOutput(BaseModel):
    result: str

class MyTool(Tool[MyInput, MyOutput]):
    async def execute(self, input: MyInput) -> MyOutput:
        return MyOutput(result=f"processed: {input.query}")
```

---

*Version: 2.0 · 61 tools · all typed via `Tool[Input, Output]`*