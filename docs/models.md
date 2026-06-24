# Models Reference

Reference for all **186 Pydantic v2 models** across 18 modules in the Autonomous ML Research Engineer. All enums use `StrEnum`.

> See [System Design](system_design.md) for enum values and [Tools](tools.md) for I/O contracts.

---

## models/paper.py (2 classes)

| Model | Fields |
|-------|--------|
| `Author` | `name`, `affiliation?` |
| `Paper` | `paper_id`, `title`, `authors: list[Author]`, `abstract`, `url`, `published`, `content_full?` |

## models/summary.py (1 class)

| Model | Fields |
|-------|--------|
| `ResearchSummary` | 14 fields: `executive_summary`, `problem_statement`, `core_contributions`, `model_architecture`, `training_methodology`, `dataset_information`, `evaluation_methodology`, `key_results`, `limitations`, `reproduction_challenges`, `engineering_complexity`, `implementation_difficulty`, `compute_requirements`, `hardware_requirements` |

## models/plan.py (3 classes)

| Model | Fields |
|-------|--------|
| `ComplexityMetrics` | `code_complexity`, `data_requirements`, `compute_requirements`, `inference_complexity` |
| `FileRequirement` | `path`, `purpose`, `complexity`, `estimated_lines` |
| `EngineeringReport` | `complexity_analysis`, `step_by_step_implementation`, `files_required`, `development_effort`, `dependencies`, `pytorch_modules`, `test_coverage_requirements`, `benchmark_targets` |

## models/storage.py (1 class)

| Model | Fields |
|-------|--------|
| `StoredPaper` | `id?`, `paper_id`, `title`, `authors_json`, `summary_json`, `plan_json`, `timestamp` |

## models/repo.py (10 classes)

| Model | Purpose |
|-------|---------|
| `Repository` | Repository metadata. |
| `RepositorySummary` | Analysis result (name, type, architecture, important_files, training_pipeline, knowledge_graph, implementation_targets, configuration_analysis). |
| `RepositoryType` (enum) | Repository classification. |
| `ArchitectureOverview` | High-level structure. |
| `FileImportance` | File importance ranking. |
| `ImplementationTarget` | Where to insert changes. |
| `KnowledgeGraph` | Repo knowledge graph. |
| `ConfigurationAnalysis` | Parsed config. |
| `TrainingPipelineInfo` | Detected training pipeline. |
| `DependencyInfo` | Dependency graph data. |

## models/ast_models.py (12 classes)

AST analysis models: `ASTClass`, `ASTFunction`, `ASTImport`, `ASTModule`, `ComplexityMetrics`, `ASTAnalysisResult`, etc.

---

## models/planner.py (27 classes) — Phase 3

**Enums:** `CompatibilityLevel`, `RiskLevel`, `ConfidenceLevel`, `DifficultyLevel`, `ExperimentType`, `ValidationTestType`

**Core models:**

| Model | Purpose |
|-------|---------|
| `CompatibilityDimension` | One of 7 compatibility dimensions. |
| `CompatibilityReport` | Full 7-dimension report. |
| `ImplementationTarget` | Target file + insertion point + complexity. |
| `ImplementationStep` | Ordered step with dependencies. |
| `ImplementationPlan` | Ordered steps. |
| `ImpactDimension` | One of 7 impact dimensions. |
| `ImpactReport` | Full 7-dimension impact. |
| `MetricDefinition` | Metric name, type, target. |
| `Experiment` | Single experiment config. |
| `ExperimentGroup` | Group of related experiments. |
| `ExperimentMatrix` | 5 groups (baseline, MVE, ablation, stress, scaling). |
| `TestCase` | Single test case. |
| `TestSuite` | Suite of test cases. |
| `ValidationPlan` | 6 test suites. |
| `RiskItem` | Risk + mitigation. |
| `RiskAssessment` | 7 risk categories. |
| `ComputeEstimate` | GPU hours, memory, storage, cost. |
| `ScenarioOutcome` | Best/likely/worst outcome. |
| `FailureMode` | Predicted failure mode. |
| `ResultPrediction` | Scenarios + failure modes. |
| `PlanResult` | Top-level plan result (all of the above). |

---

## models/coding.py (18 classes) — Phase 4

**Enums:** `ChangeType`, `PatchStatus`, `ReviewStatus`, `RiskLevel`, `ComplexityLevel`, `TestType`

**Core models:** `CodeChange`, `GeneratedPatch`, `TestSpecification`, `TestSuite`, `ReviewComment`, `ReviewResult`, `MigrationStep`, `MigrationPlan`, `RollbackStep`, `RollbackPlan`, `ImplementationRequest`, `ImplementationResult`

---

## models/memory.py (21 classes) — Phase 5

**Enums:** `MemoryType` (29 values), `InsightType` (13 values), `FailureMode`, `ExecutionOutcome`, `RelationshipType` (10 values)

**Core models:** `MemoryBase`, `PaperMemory`, `RepositoryMemory`, `ExperimentPlanMemory`, `PatchMemory`, `ArchitectureDecisionMemory`, `ResearchInsightMemory`, `FailedApproachMemory`, `SuccessfulApproachMemory`, `MemoryRelationship`, `MemoryAccessLog`, `MemoryVersion`, `MemoryStats`, `MemoryFilters`, `MemoryResult`, `MemoryRecommendation`

See [Memory System](memory_system.md) for details.

---

## models/literature.py (38 classes) — Phase 6

**Enums:** `SearchSource`, `ReviewDepth`, `PaperRelationType`, `TrendDirection`, `RelevanceLevel`

**Core models:** `PaperSummary`, `SearchResult`, `PaperSearchInput/Output`, `ComparisonDimension`, `ComparisonMatrix`, `SimilarityPair`, `DifferencePair`, `ConflictItem`, `PaperRanking`, `PaperComparisonInput/Output`, `ReviewSection`, `TimelineEntry`, `LiteratureReview`, `LiteratureReviewInput/Output`, `PaperRelationship`, `PaperRelationshipInput/Output`, `ResearchTrend`, `TopicEntry`, `TrendAnalysisInput/Output`, `RecommendationCriteria`, `PaperRecommendation`, `PaperRecommendationInput/Output`, `RelevanceScore`, `RelevanceDimension`, `RelevanceScoringInput/Output`, `LiteratureResult`

---

## models/experiment.py (30 classes) — Phase 7

**Enums:** `ExperimentType`, `ExperimentStatus`, `MetricType`, `ArtifactType`, `FailureSeverity`

**Core models:** `StatusTransition`, `ExperimentRun`, `ExperimentRunnerInput/Output`, `MonitoringInput/Output`, `MetricPattern`, `MetricReading`, `MetricSeries`, `MetricCollectorInput/Output`, `ArtifactPattern`, `ExperimentArtifact`, `ArtifactCollectorInput/Output`, `AnomalyIndicator`, `FailureDetectorInput/Output`, `ExperimentRecord`, `ExperimentStorageInput/Output`, `ExperimentQueryInput/Output`, `ExperimentResult`, `ExperimentConfig`

---

## models/evaluation.py (22 classes) — Phase 8

**Enums:** `DynamicsPatternType`, `RecommendationPriority`

**Core models:** `MetricDelta`, `ExperimentComparisonInput/Output`, `DynamicsPattern`, `TrainingDynamicsInput/Output`, `SignificanceResult`, `StatisticalSignificanceInput/Output`, `ExperimentRecommendation`, `PaperSuggestion`, `NextExperimentInput/Output`, `EvaluationRecord`, `EvaluationStorageInput/Output`, `EvaluationQueryInput/Output`, `EvaluationResult`, `EvaluationConfig`

---

## models/loop.py (22 classes) — Phase 9

**Enums:** `LoopStatus`, `IterationPhase`, `StoppingCondition`, `ApprovalGate`

**Core models:** `LoopConfig`, `ApprovalRequest`, `LoopState`, `LoopIteration`, `IterationStorageInput/Output`, `IterationQueryInput/Output`, `LoopRecord`, `LoopStorageInput/Output`, `LoopQueryInput/Output`, `StoppingCheckInput/Output`, `ReportInput/Output`, `LoopResult`

### LoopConfig fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `goal` | `str` | required | High-level research goal. |
| `repo_path` | `str` | required | Repository path. |
| `max_iterations` | `int` | 5 | Max iterations (1–100). |
| `target_metric_name` | `str?` | None | Metric to optimize. |
| `target_metric_value` | `float?` | None | Target value. |
| `higher_is_better` | `bool` | False | True for accuracy, False for loss. |
| `budget_hours` | `float?` | None | GPU-hour budget. |
| `budget_cost` | `float?` | None | USD budget. |
| `approval_mode` | `bool` | False | Enable human-approval gates. |
| `skip_literature_after_first` | `bool` | True | Skip literature discovery after iteration 1. |
| `stagnation_window` | `int` | 3 | Iterations without improvement before stopping. |
| `improvement_threshold` | `float` | 0.01 | Minimum relative improvement. |
| `dry_run` | `bool` | True | Dry-run experiments. |
| `stop_on_error` | `bool` | True | Stop loop on iteration error. |

---

## LLM layer models (Phase 10)

| Model | Fields |
|-------|--------|
| `LLMRole` (enum) | `SYSTEM`, `USER`, `ASSISTANT`, `TOOL` |
| `LLMMessage` | `role`, `content`, `name?`, `tool_call_id?` |
| `LLMRequest` | `messages`, `model?`, `temperature`, `max_tokens?`, `top_p`, `stop?`, `stream`, `extra` |
| `LLMUsage` | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `LLMResponse` | `content`, `model`, `provider`, `usage`, `finish_reason?`, `raw` |
| `ProviderError` | `message`, `provider?`, `cause?` |

---

*Version: 2.0 · 186 Pydantic models · 18 modules*