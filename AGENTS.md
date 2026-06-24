# AGENTS.md - Autonomous ML Research Engineer

## Quick Commands

| Command | Description |
|---------|-------------|
| `research-engineer analyze <paper>` | Analyze a paper (arXiv ID, URL, or PDF) |
| `research-engineer analyze-repo <path>` | Analyze a repository |
| `research-engineer plan <paper> <repo>` | Plan experiment integration |
| `research-engineer implement --paper\|--task\|--plan <repo>` | Implement code changes |
| `research-engineer memory <query>` | Search memories |
| `research-engineer memory --stats` | Show memory statistics |
| `research-engineer memory --related <id>` | Show related memories |
| `research-engineer get <paper_id>` | Retrieve stored paper |
| `research-engineer search <query>` | Search stored papers |
| `research-engineer history [--limit N]` | Show analysis history |
| `research-engineer cache-status` | Show cache statistics |
| `research-engineer literature search <query>` | Search papers (local, arXiv, Semantic Scholar) |
| `research-engineer literature review <topic>` | Generate literature review |
| `research-engineer literature compare <papers>` | Compare papers across dimensions |
| `research-engineer literature relationships <papers>` | Detect paper relationships |
| `research-engineer literature trends <topic>` | Analyze research trends |
| `research-engineer literature recommend <topic>` | Recommend papers for implementation |
| `research-engineer literature relevance <paper> <repo>` | Score paper-repo relevance |
| `research-engineer literature discover <topic>` | Full literature intelligence workflow |
| `research-engineer experiment run --command <cmd> --repo <path>` | Run an experiment (dry-run default) |
| `research-engineer experiment monitor <id>` | Show experiment monitoring summary |
| `research-engineer experiment list [--status <s>] [--limit N]` | List experiments |
| `research-engineer experiment get <id>` | Get experiment details |
| `research-engineer experiment search <query>` | Search experiment history |
| `research-engineer experiment cancel <id>` | Cancel a running experiment |
| `research-engineer experiment history <paper_id>` | Show experiments for a paper |
| `research-engineer evaluate run <experiment>` | Evaluate a single experiment (dynamics) |
| `research-engineer evaluate compare <ids>` | Compare experiments |
| `research-engineer evaluate analyze <ids>` | Full evaluation workflow |
| `research-engineer evaluate dynamics <id>` | Show training dynamics |
| `research-engineer evaluate significance <ids>` | Statistical significance |
| `research-engineer evaluate next <ids>` | Recommend next experiments |
| `research-engineer evaluate list [--paper\|--repo\|--experiment]` | List evaluations |
| `research-engineer evaluate get <id>` | Get evaluation details |
| `research-engineer evaluate search <query>` | Search evaluation history |
| `research-engineer loop run <goal> --repo <path>` | Start an autonomous research loop |
| `research-engineer loop list [--status <s>]` | List research loops |
| `research-engineer loop get <loop_id>` | Get loop details |
| `research-engineer loop iterations <loop_id>` | List iterations of a loop |
| `research-engineer loop iteration <iteration_id>` | Get iteration details |
| `research-engineer loop search <query>` | Search loop history |
| `research-engineer loop report <loop_id>` | Generate research report |
| `research-engineer task <goal> --repo <path>` | Terminal-first autonomous coding (Phase 11) |
| `research-engineer research <goal>` | Autonomous research workflow (Phase 15) |
| `research-engineer memory build --repo <path>` | Build repository memory (Phase 12) |
| `research-engineer memory query <query> --repo <path>` | Query repository memory |
| `research-engineer memory symbol-graph <symbol> --repo <path>` | Explore symbol graph |
| `research-engineer llm status` | Show per-agent provider/model routing |
| `research-engineer llm config [--config <path>]` | Dump resolved LLM YAML config |
| `uv run pytest` | Run test suite |
| `uv run mypy .` | Type check |
| `uv run ruff check .` | Lint |

## Input Detection

- **arXiv ID**: `^\d{4}\.\d{5}$` (e.g., `2503.12345`)
- **arXiv URL**: `^https?://arxiv.org/(abs|pdf)/\d{4}\.\d{5}(:?v\d+)?$`
- **PDF file**: `.*\.pdf$`

## Architecture

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
              ↓
      MonitoringTool (analyze output, scan checkpoints)
              ↓
      MetricCollectorTool (parse metrics from logs/json/csv)
              ↓
      ArtifactCollectorTool (copy checkpoints, logs, plots, configs)
              ↓
      FailureDetectorTool (classify outcome, detect anomalies)
              ↓
      ExperimentStorageTool (persist to SQLite + output files)
              ↓
      MemoryAgent.store_success() / MemoryAgent.store_failure()
              ↓
      MemoryKnowledgeGraph.add_relationship() (auto graph update)
              ↓
      ExperimentResult → markdown + JSON output files
```

### Phase 8: Evaluation

```
CLI → EvaluationAgent
              ↓
      ExperimentComparisonTool (compare runs)
              ↓
      TrainingDynamicsTool (over/underfit, convergence, instability)
              ↓
      StatisticalSignificanceTool (t-tests, effect size, CIs)
              ↓
      NextExperimentTool (recommend next runs + paper recommendations)
              ↓
      EvaluationStorageTool (persist to SQLite + output files)
              ↓
      MemoryAgent.store_insight() / store_success() / store_failure()
              ↓
      MemoryKnowledgeGraph.add_relationship() (auto graph update)
              ↓
      EvaluationResult → markdown + JSON output files
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
              ↓
      LoopResult → SQLite (research_loops + loop_iterations tables)
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

### Phase 11: Terminal-First Autonomous Coding

```
CLI → TaskAgent
      ↓
TerminalTool (7 operations: run_command, read_file, write_file,
              search_code, apply_patch, git_status, git_diff)
      ↓
Analyze → Plan → Implement → Diff → (Optional) Test
      ↓
TaskResult → output/tasks/<task_id>/
```

### Phase 12: Repository Memory

```
CLI → RepositoryMemory
      ↓
RepositoryIndexer (AST parsing → symbols + chunks)
SymbolGraph (deps, callers, callees, related, tests)
HashingEmbedder (offline lightweight embeddings)
InMemoryVectorBackend (no heavy deps)
      ↓
HybridRetriever (semantic + graph + metadata)
      ↓
RepositoryMemoryStore (SQLite-backed, incremental refresh)
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

## Source Structure

```
src/research_engineer/
├── __init__.py              # Package root (exports agents)
├── __main__.py              # Entry point
├── cli/
│   ├── __init__.py           # Typer app + all CLI commands
│   └── main.py               # (empty, app in __init__)
├── agents/
│   ├── __init__.py           # Exports all agents
│   ├── research_agent.py     # Phase 1: paper analysis
│   ├── repository_agent.py   # Phase 2: repo analysis
│   ├── experiment_planner_agent.py  # Phase 3: experiment planning
│   ├── coding_agent.py       # Phase 4: code implementation
│   ├── memory_agent.py       # Phase 5: research memory
│   ├── literature_agent.py    # Phase 6: literature intelligence
│   ├── experiment_agent.py    # Phase 7: experiment execution
│   ├── evaluation_agent.py    # Phase 8: evaluation & conclusions
│   ├── research_loop_agent.py  # Phase 9: autonomous research loop
│   ├── _llm_support.py         # Phase 10: resolve_llm() helper for agents
│   ├── task_agent.py           # Phase 11: terminal-first autonomous coding
│   ├── self_repair.py          # Phase 14: SelfRepairFramework, FailureAnalyzer, RepairStrategist
│   ├── delegation.py           # Phase 13: DelegationFramework, AgentRole, AgentCapability
│   ├── _adapters.py            # Phase 13: ArchitectAgent, ReviewerAgent, TestAgent adapters
│   ├── research_stages.py      # Phase 15: 7 research stage agents
│   ├── research_workflow.py    # Phase 15: ResearchWorkflowFramework, ResearchConfig
│   └── research_orchestrator.py # Phase 15: ResearchOrchestrator
├── memory/                     # Phase 12: repository memory
│   ├── __init__.py
│   ├── models.py
│   ├── indexer.py
│   ├── symbol_graph.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── storage.py
│   └── repository_memory.py
├── llm/                       # Phase 10: provider-agnostic LLM layer
│   ├── __init__.py            # Public exports
│   ├── base.py                # LLMProvider ABC, LLMRequest/Response/Message/Role, ProviderError
│   ├── ollama_provider.py     # OllamaCloudProvider (OpenAI-compatible over httpx)
│   ├── factory.py             # ProviderFactory, load_config, env expansion, registry
│   └── router.py              # ModelRouter, _BoundProvider, get_router
├── models/
│   ├── __init__.py           # Re-exports all models
│   ├── paper.py              # Author, Paper
│   ├── summary.py            # ResearchSummary
│   ├── plan.py               # ComplexityMetrics, FileRequirement, EngineeringReport
│   ├── repo.py               # Repository, RepositorySummary, RepositoryType, etc.
│   ├── storage.py            # StoredPaper
│   ├── planner.py            # All Phase 3 models (27 classes)
│   ├── coding.py             # All Phase 4 models (15 classes)
│   ├── memory.py             # All Phase 5 models (20+ classes)
│   ├── literature.py          # All Phase 6 models (30+ classes)
│   ├── experiment.py          # All Phase 7 models (30+ classes)
│   ├── evaluation.py          # All Phase 8 models (30+ classes)
│   ├── loop.py                # All Phase 9 models (25+ classes)
│   ├── ast_models.py         # AST analysis models
│   ├── task.py               # Phase 11: TaskConfig, TaskResult (with delegation fields)
│   ├── delegation.py         # Phase 13: AgentRole, AgentCapability, DelegationResult
│   ├── repair.py             # Phase 14: RepairConfig, RepairResult, FailureReport
│   └── research.py           # Phase 15: ResearchConfig, WorkflowResult, ResearchHypothesis
├── tools/
│   ├── __init__.py           # Re-exports all tools + I/O models
│   ├── base.py               # Tool[InputType, OutputType] ABC, ToolError
│   ├── base_cache.py         # CacheBase, SimpleCache, FileCache, rate limiters
│   ├── arxiv.py              # ArxivTool (paper fetch)
│   ├── pdf.py                # PDFTool (PDF extraction)
│   ├── parser.py             # PaperParserTool (content parsing)
│   ├── storage.py            # StorageTool, StorageInput (SQLite)
│   ├── scanner.py            # RepositoryScannerTool
│   ├── ast_analyzer.py       # ASTAnalysisTool
│   ├── dependency_graph.py   # DependencyGraphTool
│   ├── training_pipeline.py  # TrainingPipelineTool
│   ├── config_analyzer.py    # ConfigAnalysisTool
│   ├── knowledge_graph.py    # KnowledgeGraphTool
│   ├── documentation.py      # DocumentationTool
│   ├── llm_analyzer.py       # LLMAnalysisTool (optional, may fail import)
│   ├── rate_limiter.py       # AsyncRateLimiter
│   ├── compatibility.py      # CompatibilityAnalysisTool (Phase 3)
│   ├── implementation_planner.py  # ImplementationPlannerTool (Phase 3)
│   ├── impact_analysis.py    # ImpactAnalysisTool (Phase 3)
│   ├── experiment_design.py  # ExperimentDesignTool (Phase 3)
│   ├── validation_planner.py # ValidationPlannerTool (Phase 3)
│   ├── risk_assessment.py    # RiskAssessmentTool (Phase 3)
│   ├── compute_estimator.py  # ComputeEstimatorTool (Phase 3)
│   ├── result_prediction.py  # ResultPredictionTool (Phase 3)
│   ├── code_generation.py    # CodeGenerationTool (Phase 4)
│   ├── patch_generation.py   # PatchGenerationTool (Phase 4)
│   ├── self_review.py        # SelfReviewTool (Phase 4)
│   ├── test_generation.py    # TestGenerationTool (Phase 4)
│   ├── migration_planner.py  # MigrationPlannerTool (Phase 4)
│   ├── rollback_planner.py   # RollbackPlannerTool (Phase 4)
│   ├── patch_application.py  # PatchApplicationTool (Phase 4)
│   ├── implementation_report.py  # ImplementationReportTool (Phase 4)
│   ├── memory_storage.py     # MemoryStorageTool (Phase 5)
│   ├── vector_store.py       # VectorStore, ChromaVectorStore (Phase 5)
│   ├── embedding_strategy.py # EmbeddingStrategy (Phase 5)
│   ├── memory_query.py       # QueryProcessor, SemanticQuery (Phase 5)
│   ├── memory_graph.py       # MemoryKnowledgeGraph, MemoryGraphTool (Phase 5)
│   ├── relationship_detector.py  # RelationshipDetector (Phase 5)
│   ├── retrieval_strategies.py   # RetrievalStrategy, 6 strategies (Phase 5)
│   ├── memory_tools.py       # MemoryQueryTool, MemoryWriteTool, MemoryGraphTool, MemoryRecallTool (Phase 5)
│   ├── paper_search.py       # PaperSearchTool (Phase 6)
│   ├── paper_comparison.py   # PaperComparisonTool (Phase 6)
│   ├── literature_review.py  # LiteratureReviewTool (Phase 6)
│   ├── paper_relationship.py # PaperRelationshipTool (Phase 6)
│   ├── trend_analysis.py     # TrendAnalysisTool (Phase 6)
│   ├── paper_recommendation.py # PaperRecommendationTool (Phase 6)
│   ├── relevance_scoring.py  # RelevanceScoringTool (Phase 6)
│   ├── experiment_runner.py   # ExperimentRunnerTool (Phase 7)
│   ├── monitoring.py          # MonitoringTool (Phase 7)
│   ├── metric_collector.py    # MetricCollectorTool (Phase 7)
│   ├── artifact_collector.py  # ArtifactCollectorTool (Phase 7)
│   ├── failure_detector.py    # FailureDetectorTool (Phase 7)
│   ├── experiment_storage.py  # ExperimentStorageTool (Phase 7)
│   ├── _stats.py              # Pure-Python statistics helpers (Phase 8)
│   ├── experiment_comparison.py # ExperimentComparisonTool (Phase 8)
│   ├── training_dynamics.py   # TrainingDynamicsTool (Phase 8)
│   ├── statistical_significance.py # StatisticalSignificanceTool (Phase 8)
│   ├── next_experiment.py     # NextExperimentTool (Phase 8)
│   ├── evaluation_storage.py  # EvaluationStorageTool (Phase 8)
│   ├── loop_storage.py        # LoopStorageTool (Phase 9)
│   ├── stopping_condition.py  # StoppingConditionChecker (Phase 9)
│   ├── report_generator.py    # ReportGeneratorTool (Phase 9)
│   └── data/                     # Runtime data dir (SQLite DB, etc.)
```

### Tests

```
tests/
├── test_agent.py              # ResearchAgent tests
├── test_agents.py             # All agents tests
├── test_cli.py                # CLI command tests
├── test_integration.py        # End-to-end integration tests
├── test_integration_phases.py # Multi-phase integration tests
├── test_models.py             # Core model tests (Paper, EngineeringReport, etc.)
├── test_phase4.py             # Phase 4 tests (CodingAgent + tools)
├── test_planner_agent.py      # Phase 3 agent tests
├── test_planner_models.py     # Phase 3 model tests (27 classes)
├── test_planner_tools.py      # Phase 3 tool tests (8 tools)
├── test_tools.py              # Core tool tests
├── test_memory_models.py      # Phase 5 memory model tests
├── test_relationship_detector.py  # Phase 5 relationship detection tests
├── test_memory_graph.py       # Phase 5 knowledge graph tests
├── test_retrieval_strategies.py   # Phase 5 retrieval strategy tests
├── test_memory_tools.py      # Phase 5 formal memory tool tests
├── test_memory_agent_integration.py # Phase 5 agent integration tests
├── test_memory_cli.py         # Phase 5 CLI command tests
├── test_literature_models.py  # Phase 6 model tests
├── test_literature_tools.py    # Phase 6 tool tests
├── test_literature_agent.py    # Phase 6 agent tests
├── test_literature_cli.py      # Phase 6 CLI command tests
├── test_experiment_models.py  # Phase 7 model tests
├── test_experiment_tools.py    # Phase 7 tool tests
├── test_experiment_agent.py    # Phase 7 agent tests
├── test_experiment_cli.py      # Phase 7 CLI command tests
├── test_evaluation_models.py   # Phase 8 model tests
├── test_evaluation_tools.py    # Phase 8 tool tests
├── test_evaluation_agent.py    # Phase 8 agent tests
├── test_evaluation_cli.py      # Phase 8 CLI command tests
├── test_loop_models.py         # Phase 9 model tests
├── test_loop_tools.py          # Phase 9 tool tests
├── test_loop_agent.py          # Phase 9 agent tests
├── test_loop_cli.py            # Phase 9 CLI command tests
├── test_llm.py                # Phase 10 LLM layer tests (provider, factory, router, agent wiring)
├── test_terminal_tool.py       # Phase 11 TerminalTool tests
├── test_task_agent.py          # Phase 11 TaskAgent tests
├── test_task_cli.py            # Phase 11 CLI command tests
├── test_repo_memory.py         # Phase 12 repository memory tests
├── test_delegation.py          # Phase 13 delegation framework tests
├── test_self_repair.py         # Phase 14 self-repair framework tests
└── test_research_workflow.py   # Phase 15 research workflow tests
```

## Phase 3 Models

All in `models/planner.py` (27 classes):

**Enums**: `CompatibilityLevel`, `RiskLevel`, `ConfidenceLevel`, `DifficultyLevel`, `ExperimentType`, `ValidationTestType`

**Core models**: `CompatibilityDimension`, `CompatibilityReport`, `ImplementationTarget`, `ImplementationStep`, `ImplementationPlan`, `ImpactDimension`, `ImpactReport`, `MetricDefinition`, `Experiment`, `ExperimentGroup`, `ExperimentMatrix`, `TestCase`, `TestSuite`, `ValidationPlan`, `RiskItem`, `RiskAssessment`, `ComputeEstimate`, `ScenarioOutcome`, `FailureMode`, `ResultPrediction`, `PlanResult`

## Phase 3 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| CompatibilityAnalysisTool | CompatibilityInput → CompatibilityOutput | 7 dimensions (architecture, API, data, compute, training, inference, deployment) |
| ImplementationPlannerTool | ImplementationPlannerInput → ImplementationPlannerOutput | Ordered steps with targets per complexity level |
| ImpactAnalysisTool | ImpactAnalysisInput → ImpactAnalysisOutput | 7 impact dimensions with confidence levels |
| ExperimentDesignTool | ExperimentDesignInput → ExperimentDesignOutput | 5 experiment groups (baseline, MVE, ablation, stress, scaling) |
| ValidationPlannerTool | ValidationPlannerInput → ValidationPlannerOutput | 6 test suites (unit, integration, numerical, regression, performance, checkpoint) |
| RiskAssessmentTool | RiskAssessmentInput → RiskAssessmentOutput | 7 risk categories with mitigations |
| ComputeEstimatorTool | ComputeEstimatorInput → ComputeEstimatorOutput | GPU hours, memory, storage, cloud cost |
| ResultPredictionTool | ResultPredictionInput → ResultPredictionOutput | Best/likely/worst scenarios + failure modes |

## Phase 3 CLI Output

```bash
research-engineer plan 2503.12345 ./my_repo
research-engineer plan paper.pdf ./DeepSeek --output-format json
research-engineer plan https://arxiv.org/abs/2503.12345 ./repo --output-dir output/
```

Generates 9 files in `output/plans/<paper_id>_<repo>/`:
1. `compatibility_analysis.md`
2. `implementation_plan.md`
3. `experiment_matrix.md`
4. `validation_strategy.md`
5. `risk_assessment.md`
6. `cost_estimation.md`
7. `expected_results.md`
8. `engineering_report.md`
9. `plan_result.json`

## Phase 4 Models

All in `models/coding.py` (15 classes):

**Enums**: `ChangeType`, `TestType`, `ReviewStatus`, `PatchStatus`, `PatchRiskLevel`, `ComplexityLevel`

**Core models**: `CodeChange`, `GeneratedPatch`, `TestSpecification`, `TestSuite`, `ReviewComment`, `ReviewResult`, `MigrationStep`, `MigrationPlan`, `RollbackStep`, `RollbackPlan`, `ImplementationRequest`, `ImplementationResult`

## Phase 5 Models

All in `models/memory.py` (20+ classes):

**Enums**: `MemoryType`, `InsightType`, `FailureMode`, `ExecutionOutcome`, `RelationshipType`

**Core models**: `MemoryBase`, `PaperMemory`, `RepositoryMemory`, `ExperimentPlanMemory`, `PatchMemory`, `ArchitectureDecisionMemory`, `ResearchInsightMemory`, `FailedApproachMemory`, `SuccessfulApproachMemory`, `MemoryRelationship`, `MemoryStats`, `MemoryFilters`, `MemoryResult`, `MemoryRecommendation`

## Phase 6 Models

All in `models/literature.py` (30+ classes):

**Enums**: `SearchSource`, `ReviewDepth`, `PaperRelationType`, `TrendDirection`, `RelevanceLevel`

**Core models**: `PaperSummary`, `SearchResult`, `PaperSearchInput`, `PaperSearchOutput`, `ComparisonDimension`, `ComparisonMatrix`, `SimilarityPair`, `DifferencePair`, `ConflictItem`, `PaperRanking`, `PaperComparisonInput`, `PaperComparisonOutput`, `ReviewSection`, `TimelineEntry`, `LiteratureReview`, `LiteratureReviewInput`, `LiteratureReviewOutput`, `PaperRelationship`, `PaperRelationshipInput`, `PaperRelationshipOutput`, `ResearchTrend`, `TopicEntry`, `TrendAnalysisInput`, `TrendAnalysisOutput`, `RecommendationCriteria`, `PaperRecommendation`, `PaperRecommendationInput`, `PaperRecommendationOutput`, `RelevanceScore`, `RelevanceDimension`, `RelevanceScoringInput`, `RelevanceScoringOutput`, `LiteratureResult`

## Phase 4 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| CodeGenerationTool | CodeGenerationInput → CodeGenerationOutput | Generate code changes from plans or tasks |
| PatchGenerationTool | PatchGenerationInput → PatchGenerationOutput | Create unified diff patches |
| SelfReviewTool | SelfReviewInput → SelfReviewOutput | Automated code review with architecture/style checks |
| TestGenerationTool | TestGenerationInput → TestGenerationOutput | Generate test suites for changes |
| MigrationPlannerTool | MigrationPlannerInput → MigrationPlannerOutput | Plan data/checkpoint migrations |
| RollbackPlannerTool | RollbackPlannerInput → RollbackPlannerOutput | Plan rollback procedures |
| PatchApplicationTool | PatchApplicationInput → PatchApplicationOutput | Apply patches (dry-run or actual) |
| ImplementationReportTool | ImplementationReportInput → ImplementationReportOutput | Generate implementation reports |

## Phase 6 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| PaperSearchTool | PaperSearchInput → PaperSearchOutput | Multi-source search (local, arXiv, Semantic Scholar), dedup, rank |
| PaperComparisonTool | PaperComparisonInput → PaperComparisonOutput | 7 comparison dimensions, similarity/difference/consensus/conflict |
| PaperRelationshipTool | PaperRelationshipInput → PaperRelationshipOutput | Citation, extension, similarity, contradiction detection |
| TrendAnalysisTool | TrendAnalysisInput → TrendAnalysisOutput | Rising/stable/declining trends, emerging/hot/declining topics |
| LiteratureReviewTool | LiteratureReviewInput → LiteratureReviewOutput | Structured review: sections, timeline, gaps, recommendations |
| PaperRecommendationTool | PaperRecommendationInput → PaperRecommendationOutput | Impact/novelty/implementability/relevance scoring + ranking |
| RelevanceScoringTool | RelevanceScoringInput → RelevanceScoringOutput | 6 dimensions: architecture, training, inference, evaluation, data, feasibility |

## Phase 7 Models

All in `models/experiment.py` (30+ classes):

**Enums**: `ExperimentType`, `ExperimentStatus`, `MetricType`, `ArtifactType`, `FailureSeverity`

**Core models**: `StatusTransition`, `ExperimentRun`, `ExperimentRunnerInput`, `ExperimentRunnerOutput`, `MonitoringInput`, `MonitoringOutput`, `MetricPattern`, `MetricReading`, `MetricSeries`, `MetricCollectorInput`, `MetricCollectorOutput`, `ArtifactPattern`, `ExperimentArtifact`, `ArtifactCollectorInput`, `ArtifactCollectorOutput`, `AnomalyIndicator`, `FailureDetectorInput`, `FailureDetectorOutput`, `ExperimentRecord`, `ExperimentStorageInput`, `ExperimentStorageOutput`, `ExperimentQueryInput`, `ExperimentQueryOutput`, `ExperimentResult`, `ExperimentConfig`

## Phase 7 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| ExperimentRunnerTool | ExperimentRunnerInput → ExperimentRunnerOutput | Launch subprocess with allowlist, dry-run, timeout, status tracking |
| MonitoringTool | MonitoringInput → MonitoringOutput | Analyze output, scan checkpoints, detect anomalies |
| MetricCollectorTool | MetricCollectorInput → MetricCollectorOutput | Parse metrics from logs, JSON, CSV; build series + summaries |
| ArtifactCollectorTool | ArtifactCollectorInput → ArtifactCollectorOutput | Glob-based artifact discovery, copy with checksum, size capping |
| FailureDetectorTool | FailureDetectorInput → FailureDetectorOutput | Rule-based failure classification, root cause, recommendations |
| ExperimentStorageTool | ExperimentStorageInput \| ExperimentQueryInput → ExperimentStorageOutput \| ExperimentQueryOutput | SQLite persistence, query by ID/paper/repo/status/type/text |

## Phase 8 Models

All in `models/evaluation.py` (30+ classes):

**Enums**: `DynamicsPatternType`, `RecommendationPriority`

**Core models**: `MetricDelta`, `ExperimentComparisonInput`, `ExperimentComparisonOutput`, `DynamicsPattern`, `TrainingDynamicsInput`, `TrainingDynamicsOutput`, `SignificanceResult`, `StatisticalSignificanceInput`, `StatisticalSignificanceOutput`, `ExperimentRecommendation`, `PaperSuggestion`, `NextExperimentInput`, `NextExperimentOutput`, `EvaluationRecord`, `EvaluationStorageInput`, `EvaluationStorageOutput`, `EvaluationQueryInput`, `EvaluationQueryOutput`, `EvaluationResult`, `EvaluationConfig`

## Phase 8 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| ExperimentComparisonTool | ExperimentComparisonInput → ExperimentComparisonOutput | Metric deltas, winner selection, shared/unique metrics, findings |
| TrainingDynamicsTool | TrainingDynamicsInput → TrainingDynamicsOutput | Over/underfit, convergence, instability, divergence, healthy detection |
| StatisticalSignificanceTool | StatisticalSignificanceInput → StatisticalSignificanceOutput | Welch t-test, Cohen's d, 95% CIs (pure Python, no SciPy) |
| NextExperimentTool | NextExperimentInput → NextExperimentOutput | Rule-based recommendations + paper query |
| EvaluationStorageTool | EvaluationStorageInput \| EvaluationQueryInput → EvaluationStorageOutput \| EvaluationQueryOutput | SQLite persistence, query by ID/paper/repo/experiment/text |

## Phase 6 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| PaperSearchTool | PaperSearchInput → PaperSearchOutput | Multi-source search (local, arXiv, Semantic Scholar), dedup, rank |
| PaperComparisonTool | PaperComparisonInput → PaperComparisonOutput | 7 comparison dimensions, similarity/difference/consensus/conflict |
| PaperRelationshipTool | PaperRelationshipInput → PaperRelationshipOutput | Citation, extension, similarity, contradiction detection |
| TrendAnalysisTool | TrendAnalysisInput → TrendAnalysisOutput | Rising/stable/declining trends, emerging/hot/declining topics |
| LiteratureReviewTool | LiteratureReviewInput → LiteratureReviewOutput | Structured review: sections, timeline, gaps, recommendations |
| PaperRecommendationTool | PaperRecommendationInput → PaperRecommendationOutput | Impact/novelty/implementability/relevance scoring + ranking |
| RelevanceScoringTool | RelevanceScoringInput → RelevanceScoringOutput | 6 dimensions: architecture, training, inference, evaluation, data, feasibility |

## Phase 9 Models

All in `models/loop.py` (25+ classes):

**Enums**: `LoopStatus`, `IterationPhase`, `StoppingCondition`, `ApprovalGate`

**Core models**: `LoopConfig`, `ApprovalRequest`, `LoopState`, `LoopIteration`, `IterationStorageInput`, `IterationStorageOutput`, `IterationQueryInput`, `IterationQueryOutput`, `LoopRecord`, `LoopStorageInput`, `LoopStorageOutput`, `LoopQueryInput`, `LoopQueryOutput`, `StoppingCheckInput`, `StoppingCheckOutput`, `ReportInput`, `ReportOutput`, `LoopResult`

## Phase 9 Tools

| Tool | Input → Output | Key Logic |
|------|---------------|-----------|
| LoopStorageTool | LoopStorageInput \| LoopQueryInput \| IterationStorageInput \| IterationQueryInput → LoopStorageOutput \| LoopQueryOutput \| IterationStorageOutput \| IterationQueryOutput | SQLite persistence for loops + iterations, query by ID/loop/paper/status/text |
| StoppingConditionChecker | StoppingCheckInput → StoppingCheckOutput | 4 conditions: target_achieved, max_iterations_reached, budget_exceeded, no_improvement |
| ReportGeneratorTool | ReportInput → ReportOutput | Generate research_report.md + research_report.json with executive summary, iteration history, findings, conclusions |

## Phase 9 CLI Output

```bash
research-engineer loop run "Improve training stability" --repo ./my_repo --max-iterations 3 --dry-run
research-engineer loop run "Reduce loss" --repo ./repo --target-metric loss --target-value 0.01
research-engineer loop run "Boost accuracy" --repo ./repo --approval --max-iterations 5
research-engineer loop list --status stopped
research-engineer loop get loop_abc123
research-engineer loop iterations loop_abc123
research-engineer loop report loop_abc123 --output-dir ./reports
```

Generates files in `output/loops/<loop_id>/`:
1. `research_report.md` (executive summary, methodology, iteration history, findings, failures, conclusions, config)
2. `research_report.json` (full LoopResult serialized)

## Phase 10: Provider-Agnostic LLM Layer

### Phase 10 Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `LLMProvider` ABC | `llm/base.py` | Provider-agnostic interface; `LLMRequest`/`LLMResponse`/`LLMMessage`/`LLMRole`/`LLMUsage`/`ProviderError` |
| `OllamaCloudProvider` | `llm/ollama_provider.py` | Ollama Cloud (OpenAI-compatible) over httpx; env-fallback settings |
| `ProviderFactory` | `llm/factory.py` | Builds/caches providers from config; `${VAR}` env expansion; provider registry |
| `ModelRouter` | `llm/router.py` | Resolves provider+model per agent; `_BoundProvider` pins model on every request |
| `resolve_llm()` | `agents/_llm_support.py` | Helper agents use to obtain a provider (explicit > router > disabled) |
| Config | `llm_config.yaml` | Default provider + per-agent model assignments |
| Docs | `docs/llm_integration.md` | Integration guide |

### Phase 10 CLI Output

```bash
research-engineer llm status               # per-agent provider/model routing
research-engineer llm status --format json
research-engineer llm config                # dump resolved YAML config
research-engineer llm config --config path/to/llm_config.yaml
```

### Phase 10 Resolution Rules

1. Explicit `LLMProvider` passed to an agent constructor wins.
2. `llm_enabled=False` (e.g. `RepositoryAgent` default) → no provider attached.
3. Otherwise `ModelRouter.for_agent(agent_name)` resolves from `llm_config.yaml`.

### Phase 10 Environment Variables

| Variable | Used by | Default |
|----------|---------|---------|
| `RE_LLM_CONFIG` | factory | `llm_config.yaml` at repo root |
| `OLLAMA_BASE_URL` | OllamaCloudProvider | `https://api.olama.cloud` |
| `OLLAMA_API_KEY` | OllamaCloudProvider | (none) |
| `OLLAMA_MODEL` / `OLLAMA_DEFAULT_MODEL` | OllamaCloudProvider | `glm-5.2:cloud` |
| `OLLAMA_TIMEOUT` | OllamaCloudProvider | `60` |

## Storage

- **Database**: `data/research_engineer.db` (SQLite)
- **Papers table**: `paper_id`, `title`, `authors_json`, `summary_json`, `plan_json`
- **Plans table**: `plan_id`, `paper_id`, `repo_path`, `compatibility_json`, `implementation_plan_json`, `impact_json`, `experiment_json`, `validation_json`, `risk_json`, `compute_json`, `prediction_json`
- **Memories table**: `memory_id`, `memory_type`, `content_json`, `embedding_key`, `tags`, `confidence_score`, `accessed_count`
- **Memory relationships table**: `relationship_id`, `source_memory_id`, `target_memory_id`, `relationship_type`, `confidence`
- **Vector store**: `data/vector_store/` (ChromaDB)
- **Phase 1 output**: `output/<paper_id>_summary.json`, `output/<paper_id>_plan.json`
- **Phase 3 output**: `output/plans/<paper_id>_<repo>/` (8 .md + plan_result.json)
- **Phase 4 output**: `output/<implementation_id>/` (reports + patches + tests)
- **Phase 5 output**: Memory system (SQLite + ChromaDB)
- **Phase 6 output**: `output/literature/<topic_slug>_<timestamp>/` (search results, comparison, review, relationships, trends, recommendations, relevance, literature_result.json)
- **Phase 7 DB table**: `experiments` (experiment_id, paper_id, plan_id, patch_id, repo_path, command_json, experiment_type, status, metrics_json, failure_mode, etc.)
- **Phase 7 output**: `output/experiments/<experiment_id>/` (experiment_result.json, experiment_summary.md, metrics.json, metrics_summary.md, run_log.txt, failure_report.md, artifacts/)
- **Phase 8 DB table**: `evaluations` (evaluation_id, experiment_ids_json, paper_id, repo_path, comparison_json, dynamics_json, significance_json, next_experiments_json, summary, conclusions_json, memory_ids_json, tags_json)
- **Phase 8 output**: `output/evaluations/<evaluation_id>/` (evaluation_result.json, evaluation_summary.md, comparison.md, dynamics.md, significance.md, next_experiments.md)
- **Phase 9 DB tables**: `research_loops` (loop_id, goal, config_json, status, iteration_count, best_metric_value, primary_metric_name, stopping_condition, stopping_reason, memory_ids_json), `loop_iterations` (iteration_id, loop_id, iteration_number, phase, paper_id, paper_title, plan_id, implementation_id, experiment_id, evaluation_id, metrics_json, primary_metric_name, primary_metric_value, best_metric_value, improvement, decision, memory_ids_json, error, status)
- **Phase 9 output**: `output/loops/<loop_id>/` (research_report.md, research_report.json)
- **Phase 10 config**: `llm_config.yaml` (provider + per-agent model assignments, `${VAR}` env expansion)
- **Phase 11 output**: `output/tasks/<task_id>/` (task_result.json, task_summary.md, diff.patch)
- **Phase 12 storage**: `data/repo_memory/` (SQLite symbol index + vectors)
- **Phase 15 output**: `output/research/<workflow_id>/` (research_report.md, research_report.json)

## Tool Interface Pattern

All tools follow `Tool[Input, Output]` ABC:
- `async execute(input: InputType) -> OutputType`
- `async validate(input: InputType) -> bool`
- `ToolError` on failure

## Constraints

- **No LLM in Phase 1-3**: Rule-based extraction only (no API costs)
- **Phase 4 allows code generation**: Generates patches and tests, not direct code modification
- **Patch-first philosophy**: Phase 4 generates patches for review, does not directly modify code by default
- **No experiment execution**: Phase 3 plans only, never runs experiments
- **Async-first**: All tools and agents use async/await
- **Pydantic v2**: All I/O models are typed Pydantic models
- **Python 3.12+**: `requires-python = ">=3.12"`
- **StrEnum**: All enums use `StrEnum` (not `str, Enum`)
- **Type checking**: `disallow_untyped_defs = true`, `strict_optional = true`
- **Line length**: 88 chars (ruff)
- **Tests**: Must pass before PR (target >90% coverage)
- **Repository-agnostic**: Never hardcode assumptions about specific repos
- **Paper-agnostic**: Works for any ML paper (attention, MoE, diffusion, etc.)
- **No direct model calls**: Agents obtain LLM providers via `resolve_llm()` (Phase 10); model switching is config-only

## Key Dependencies

**Core**: pydantic>=2.0, typer>=0.12, httpx>=0.26, pymupdf>=1.23, arxiv>=2.0

**Dev**: pytest>=8.0, pytest-asyncio>=0.23, ruff>=0.6, mypy>=1.11, pytest-cov>=4.1.0

## Test Status

- **Total Tests**: 878 (29 Phase 10, 60 Phase 11, 51 Phase 12, 31 Phase 13, 31 Phase 14, 39 Phase 15)
- **Coverage**: All phases (1–15)
- **Status**: All passing
