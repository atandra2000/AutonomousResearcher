# Storage Schema

Complete reference for the Autonomous ML Research Engineer's persistent storage: SQLite tables, ChromaDB vector store, knowledge graph, and the output directory layout.

> See [Memory System](memory_system.md) for how memory tables are used and [Architecture](architecture.md) for the data-flow diagram.

---

## SQLite database

**Location:** `data/research_engineer.db`

**10 tables** across all phases:

### papers (Phase 1)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Autoincrement. |
| `paper_id` | TEXT UNIQUE | arXiv ID or derived ID. |
| `title` | TEXT | Paper title. |
| `authors_json` | TEXT | JSON array of authors. |
| `summary_json` | TEXT | JSON `ResearchSummary`. |
| `plan_json` | TEXT | JSON `EngineeringReport`. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |

### plans (Phase 3)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Autoincrement. |
| `plan_id` | TEXT UNIQUE | Plan identifier. |
| `paper_id` | TEXT | FK → papers. |
| `repo_path` | TEXT | Repository path. |
| `compatibility_json` | TEXT | JSON `CompatibilityReport`. |
| `implementation_plan_json` | TEXT | JSON `ImplementationPlan`. |
| `impact_json` | TEXT | JSON `ImpactReport`. |
| `experiment_matrix_json` | TEXT | JSON `ExperimentMatrix`. |
| `validation_plan_json` | TEXT | JSON `ValidationPlan`. |
| `risk_assessment_json` | TEXT | JSON `RiskAssessment`. |
| `compute_estimate_json` | TEXT | JSON `ComputeEstimate`. |
| `result_prediction_json` | TEXT | JSON `ResultPrediction`. |
| `engineering_report_md` | TEXT | Consolidated markdown. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |

### memories (Phase 5)

| Column | Type | Notes |
|--------|------|-------|
| `memory_id` | TEXT PK | UUID-based. |
| `memory_type` | TEXT | `MemoryType` enum value. |
| `content_json` | TEXT | JSON memory content. |
| `embedding_key` | TEXT | ChromaDB embedding key. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |
| `accessed_count` | INTEGER | Default 0. |
| `last_accessed_at` | TIMESTAMP | Nullable. |
| `tags` | TEXT | Comma-separated tags. |
| `confidence_score` | REAL | Nullable. |
| `is_archived` | BOOLEAN | Default 0. |

### memory_relationships (Phase 5)

| Column | Type | Notes |
|--------|------|-------|
| `relationship_id` | TEXT PK | UUID-based. |
| `source_memory_id` | TEXT | FK → memories. |
| `target_memory_id` | TEXT | FK → memories. |
| `relationship_type` | TEXT | `RelationshipType` enum value. |
| `confidence` | REAL | Default 1.0. |
| `metadata_json` | TEXT | Nullable. |
| `created_at` | TIMESTAMP | Default current. |
| `validated` | BOOLEAN | Default 0. |

### memory_access_log (Phase 5)

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Autoincrement. |
| `memory_id` | TEXT | FK → memories. |
| `access_type` | TEXT | read/write/archive. |
| `accessed_by` | TEXT | Nullable. |
| `context` | TEXT | Nullable. |
| `accessed_at` | TIMESTAMP | Default current. |

### memory_versions (Phase 5)

| Column | Type | Notes |
|--------|------|-------|
| `version_id` | TEXT PK | UUID-based. |
| `memory_id` | TEXT | FK → memories. |
| `version_number` | INTEGER | Monotonic. |
| `content_json` | TEXT | JSON content snapshot. |
| `change_summary` | TEXT | Nullable. |
| `created_at` | TIMESTAMP | Default current. |

### experiments (Phase 7)

| Column | Type | Notes |
|--------|------|-------|
| `experiment_id` | TEXT PK | UUID-based. |
| `paper_id` | TEXT | Nullable. |
| `plan_id` | TEXT | Nullable. |
| `patch_id` | TEXT | Nullable. |
| `implementation_id` | TEXT | Nullable. |
| `repo_path` | TEXT | Required. |
| `command_json` | TEXT | JSON command. |
| `experiment_type` | TEXT | `ExperimentType` value. |
| `status` | TEXT | `ExperimentStatus` value. |
| `start_time` | TIMESTAMP | Required. |
| `end_time` | TIMESTAMP | Nullable. |
| `duration_seconds` | REAL | Nullable. |
| `exit_code` | INTEGER | Nullable. |
| `metrics_json` | TEXT | Nullable. |
| `failure_mode` | TEXT | Nullable. |
| `failure_severity` | TEXT | Nullable. |
| `root_cause` | TEXT | Nullable. |
| `output_dir` | TEXT | Nullable. |
| `memory_id` | TEXT | Nullable (linked memory). |
| `tags` | TEXT | Nullable. |
| `notes` | TEXT | Nullable. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |

### evaluations (Phase 8)

| Column | Type | Notes |
|--------|------|-------|
| `evaluation_id` | TEXT PK | UUID-based. |
| `experiment_ids_json` | TEXT | JSON array. |
| `paper_id` | TEXT | Nullable. |
| `repo_path` | TEXT | Nullable. |
| `comparison_json` | TEXT | Nullable. |
| `dynamics_json` | TEXT | Nullable. |
| `significance_json` | TEXT | Nullable. |
| `next_experiments_json` | TEXT | Nullable. |
| `summary` | TEXT | Nullable. |
| `conclusions_json` | TEXT | Nullable. |
| `memory_ids_json` | TEXT | Nullable. |
| `tags_json` | TEXT | Nullable. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |

### research_loops (Phase 9)

| Column | Type | Notes |
|--------|------|-------|
| `loop_id` | TEXT PK | UUID-based. |
| `goal` | TEXT | Research goal. |
| `config_json` | TEXT | JSON `LoopConfig`. |
| `status` | TEXT | `LoopStatus` value. |
| `iteration_count` | INTEGER | Default 0. |
| `best_metric_value` | REAL | Nullable. |
| `primary_metric_name` | TEXT | Nullable. |
| `stopping_condition` | TEXT | Nullable. |
| `stopping_reason` | TEXT | Nullable. |
| `memory_ids_json` | TEXT | Nullable. |
| `created_at` | TIMESTAMP | Default current. |
| `updated_at` | TIMESTAMP | Nullable. |

### loop_iterations (Phase 9)

| Column | Type | Notes |
|--------|------|-------|
| `iteration_id` | TEXT PK | UUID-based. |
| `loop_id` | TEXT | FK → research_loops. |
| `iteration_number` | INTEGER | 1-based. |
| `phase` | TEXT | `IterationPhase` value. |
| `paper_id` | TEXT | Nullable. |
| `paper_title` | TEXT | Nullable. |
| `plan_id` | TEXT | Nullable. |
| `implementation_id` | TEXT | Nullable. |
| `experiment_id` | TEXT | Nullable. |
| `evaluation_id` | TEXT | Nullable. |
| `metrics_json` | TEXT | Nullable. |
| `primary_metric_name` | TEXT | Nullable. |
| `primary_metric_value` | REAL | Nullable. |
| `best_metric_value` | REAL | Nullable. |
| `improvement` | REAL | Nullable. |
| `decision` | TEXT | Nullable. |
| `memory_ids_json` | TEXT | Nullable. |
| `error` | TEXT | Nullable. |
| `status` | TEXT | `LoopStatus` value. |
| `timestamp` | TIMESTAMP | Nullable. |
| `created_at` | TIMESTAMP | Default current. |

---

## ChromaDB vector store

**Location:** `data/vector_store/`

| Property | Value |
|----------|-------|
| Embedding model | `sentence-transformers/all-mpnet-base-v2` |
| Managed by | `EmbeddingStrategy` / `ChromaVectorStore` |
| Used for | Semantic memory search (`SemanticSearchStrategy`, `HybridSearchStrategy`) |
| Fallback | `None` if ChromaDB unavailable (graceful degradation) |

---

## Knowledge graph

The knowledge graph is persisted as `memory_relationships` rows in SQLite and held in-memory by `MemoryKnowledgeGraph` for traversal.

| Property | Value |
|----------|-------|
| Node type | Memories |
| Edge type | `MemoryRelationship` (typed + weighted by confidence) |
| Relationship types | 10 (`CITES`, `IMPLEMENTS`, `EXTENDS`, `SIMILAR_TO`, `DEPENDS_ON`, `CONFLICTS_WITH`, `VALIDATES`, `FAILED_WITH`, `SUCCEEDED_WITH`, `INSPIRED_BY`) |
| Stats | `GraphStats` (node_count, edge_count, density, avg_degree, connected_components, most_central, relationship_counts) |

---

## Output directory layout

```
output/
├── <paper_id>_summary.json              # Phase 1 — ResearchSummary
├── <paper_id>_plan.json                 # Phase 1 — EngineeringReport
├── plans/
│   └── <paper_id>_<repo>/               # Phase 3 — 9 files
│       ├── compatibility_analysis.md
│       ├── implementation_plan.md
│       ├── experiment_matrix.md
│       ├── validation_strategy.md
│       ├── risk_assessment.md
│       ├── cost_estimation.md
│       ├── expected_results.md
│       ├── engineering_report.md
│       └── plan_result.json
├── <implementation_id>/                # Phase 4 — patches + tests + reports
├── literature/
│   └── <topic_slug>_<timestamp>/         # Phase 6 — literature outputs
│       └── literature_result.json
├── experiments/
│   └── <experiment_id>/                 # Phase 7 — run results + artifacts
│       ├── experiment_result.json
│       ├── experiment_summary.md
│       ├── metrics.json
│       ├── metrics_summary.md
│       ├── run_log.txt
│       ├── failure_report.md
│       └── artifacts/
├── evaluations/
│   └── <evaluation_id>/                  # Phase 8 — evaluation outputs
│       ├── evaluation_result.json
│       ├── evaluation_summary.md
│       ├── comparison.md
│       ├── dynamics.md
│       ├── significance.md
│       └── next_experiments.md
└── loops/
    └── <loop_id>/                        # Phase 9 — loop report
        ├── research_report.md
        └── research_report.json
```

### Runtime data

```
data/
├── research_engineer.db                  # SQLite (10 tables)
└── vector_store/                         # ChromaDB
```

---

## Customizing storage locations

```python
from research_engineer.agents import MemoryAgent
from research_engineer.agents.memory_agent import MemoryConfig

agent = MemoryAgent(config=MemoryConfig(
    db_path="/custom/path/research_engineer.db",
    vector_store_path="/custom/path/vector_store",
))
```

---

*Version: 2.0 · 12+ SQLite tables · ChromaDB · knowledge graph · repository memory*