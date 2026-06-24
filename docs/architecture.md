# Architecture

Technical architecture for the Autonomous ML Research Engineer v1.0 — a ten-phase, agent-based platform that automates the ML research lifecycle.

> **Status:** 10/10 phases complete · 9 agents · 51 tools · 207 Pydantic models · 49 CLI commands · 690 tests.

---

## Executive summary

The platform decomposes ML research work into **ten cooperating phases**, each a self-contained layer with agents, typed tools, and Pydantic models. Phases 1–8 are individual capabilities; **Phase 9** orchestrates them into an autonomous loop; **Phase 10** is the provider-agnostic LLM substrate every agent sits on.

### Design principles

- **Modularity** — each component is independently testable and replaceable.
- **Typed contracts** — every tool I/O is a Pydantic v2 model; every enum is `StrEnum`.
- **Async-first** — all tools and agents use `async`/`await`.
- **Patch-first** — code changes are reviewable unified diffs, never applied silently.
- **Memory-first** — every run writes to SQLite + ChromaDB + a knowledge graph, enabling cross-run learning.
- **Provider-agnostic** — agents never call a model directly; they go through `resolve_llm()`.
- **Repository-agnostic & paper-agnostic** — no hardcoded assumptions about specific repos or topics.

---

## High-level architecture

```mermaid
flowchart TD
    subgraph Input["Inputs"]
        P["📄 arXiv / PDF paper"]
        R["🗂️ ML repository"]
        G["🎯 Research goal"]
    end

    subgraph Agents["Agent Ecosystem (Phases 1–9)"]
        A1["ResearchAgent"]
        A2["RepositoryAgent"]
        A3["ExperimentPlannerAgent"]
        A4["CodingAgent"]
        A5["MemoryAgent"]
        A6["LiteratureAgent"]
        A7["ExperimentAgent"]
        A8["EvaluationAgent"]
        A9["ResearchLoopAgent<br/>(orchestrator)"]
    end

    subgraph LLM["Phase 10 — LLM Layer"]
        CFG["llm_config.yaml"]
        FAC["ProviderFactory"]
        ROUT["ModelRouter"]
        PROV["OllamaCloudProvider"]
    end

    subgraph Store["Persistent State"]
        SQL[("SQLite<br/>8 tables")]
        CHR[("ChromaDB<br/>vector store")]
        KG[("Knowledge graph")]
    end

    P --> A1
    R --> A2
    G --> A9
    A9 --> A1 & A2 & A3 & A4 & A6 & A7 & A8
    A1 & A2 & A3 & A4 & A6 & A7 & A8 --> A5
    A5 <--> SQL
    A5 <--> CHR
    A5 <--> KG
    A9 --> REP["📄 research_report.md / .json"]
    A1 -.-> ROUT
    A2 -.-> ROUT
    A3 -.-> ROUT
    A4 -.-> ROUT
    A5 -.-> ROUT
    A6 -.-> ROUT
    A7 -.-> ROUT
    A8 -.-> ROUT
    A9 -.-> ROUT
    CFG --> FAC --> ROUT --> PROV
```

---

## Component layers

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI Layer  (Typer — 49 commands across 6 sub-apps)             │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer  (9 agents + _llm_support.resolve_llm)            │
│   ResearchAgent · RepositoryAgent · ExperimentPlannerAgent     │
│   CodingAgent · MemoryAgent · LiteratureAgent                   │
│   ExperimentAgent · EvaluationAgent · ResearchLoopAgent        │
├─────────────────────────────────────────────────────────────────┤
│  Tool Layer  (51 typed tools, Tool[Input, Output] ABC)         │
│   Phases 1–9 each contribute 4–8 tools                          │
├─────────────────────────────────────────────────────────────────┤
│  LLM Layer  (Phase 10 — provider-agnostic)                     │
│   LLMProvider ABC · OllamaCloudProvider · ProviderFactory      │
│   ModelRouter · _BoundProvider · resolve_llm                   │
├─────────────────────────────────────────────────────────────────┤
│  Domain Layer  (207 Pydantic models across 13 modules)         │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                          │
│   SQLite · ChromaDB · arXiv API · PyMuPDF · AST · httpx       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase pipeline

```mermaid
flowchart LR
    P1[Phase 1<br/>Paper Analysis] --> P3[Phase 3<br/>Experiment Planning]
    P2[Phase 2<br/>Repo Analysis] --> P3
    P3 --> P4[Phase 4<br/>Code Implementation]
    P4 --> P7[Phase 7<br/>Experiment Execution]
    P7 --> P8[Phase 8<br/>Evaluation]
    P8 --> P9{Phase 9<br/>Loop}
    P9 -->|recall| P5[Phase 5<br/>Memory]
    P9 -->|discover| P6[Phase 6<br/>Literature]
    P6 --> P3
    P5 --> P3
    P9 -->|iterate| P3
    P10[Phase 10<br/>LLM Layer] -.-> P1 & P2 & P3 & P4 & P5 & P6 & P7 & P8 & P9
```

### Phase summary

| Phase | Name | Agent | Tools | Models | Key output |
|-------|------|-------|-------|--------|------------|
| 1 | Paper Analysis | ResearchAgent | 4 | 6 | `output/<paper_id>_summary.json`, `_plan.json` |
| 2 | Repository Analysis | RepositoryAgent | 7 | 22 | `output/<repo>/` docs + knowledge graph |
| 3 | Experiment Planning | ExperimentPlannerAgent | 8 | 27 | `output/plans/<paper>_<repo>/` (8 .md + JSON) |
| 4 | Code Implementation | CodingAgent | 8 | 18 | `output/<impl_id>/` patches + tests + reports |
| 5 | Research Memory | MemoryAgent | 12 | 21 | SQLite memories + ChromaDB + knowledge graph |
| 6 | Literature Intelligence | LiteratureAgent | 7 | 38 | `output/literature/<topic>_<ts>/` |
| 7 | Experiment Execution | ExperimentAgent | 6 | 30 | `output/experiments/<exp_id>/` |
| 8 | Evaluation | EvaluationAgent | 5 | 22 | `output/evaluations/<eval_id>/` |
| 9 | Autonomous Loop | ResearchLoopAgent | 3 | 22 | `output/loops/<loop_id>/research_report.*` |
| 10 | LLM Layer | — | 4 | 6 | `llm_config.yaml` + provider routing |

---

## Phase 1 — Paper Analysis

```mermaid
flowchart LR
    IN["arXiv ID / URL / PDF"] --> DET{Detect type}
    DET -->|arXiv| AX["ArxivTool"]
    DET -->|PDF| PD["PDFTool"]
    AX & PD --> PR["PaperParserTool"]
    PR --> SUM["ResearchSummary<br/>(14 fields)"]
    PR --> PLN["EngineeringReport"]
    SUM & PLN --> ST["StorageTool<br/>(SQLite)"]
```

- **Input detection:** regex on arXiv ID (`^\d{4}\.\d{5}$`), arXiv URL, or `.pdf` path.
- **Rule-based extraction** — no LLM cost for Phases 1–3.
- **Output:** 14-field `ResearchSummary` + `EngineeringReport` (complexity, files, effort, dependencies).

---

## Phase 2 — Repository Analysis

```mermaid
flowchart LR
    R["Repository path"] --> SC["ScannerTool"]
    SC --> AST["ASTAnalysisTool"]
    SC --> DEP["DependencyGraphTool"]
    SC --> TR["TrainingPipelineTool"]
    SC --> CFG["ConfigAnalysisTool"]
    SC --> KG["KnowledgeGraphTool"]
    AST & DEP & TR & CFG & KG --> DOC["DocumentationTool"]
    DOC --> OUT["output/<repo>/"]
```

Produces architecture overview, file-importance rankings, dependency graph, training-pipeline extraction, config analysis (YAML/JSON/TOML), knowledge graph, and generated markdown docs.

---

## Phase 3 — Experiment Planning

```mermaid
flowchart LR
    PAP["ResearchSummary"] & REP["RepositorySummary"] --> COMP["CompatibilityAnalysisTool<br/>(7 dims)"]
    COMP --> IMPL["ImplementationPlannerTool"]
    IMPL --> IMP["ImpactAnalysisTool<br/>(7 dims)"]
    IMP --> EXP["ExperimentDesignTool<br/>(5 groups)"]
    EXP --> VAL["ValidationPlannerTool<br/>(6 suites)"]
    VAL --> RSK["RiskAssessmentTool<br/>(7 categories)"]
    RSK --> CMP["ComputeEstimatorTool"]
    CMP --> PRD["ResultPredictionTool"]
    PRD --> OUT["9 files in output/plans/<paper>_<repo>/"]
```

**7 compatibility dimensions:** architecture, API, data, compute, training, inference, deployment.
**5 experiment groups:** baseline, minimum-viable, ablation, stress, scaling.
**6 validation suites:** unit, integration, numerical-equivalence, regression, performance, checkpoint-compatibility.

---

## Phase 4 — Code Implementation

```mermaid
flowchart LR
    IN["Plan or task"] --> CG["CodeGenerationTool"]
    CG --> PG["PatchGenerationTool<br/>(unified diff)"]
    PG --> SR["SelfReviewTool"]
    PG --> TG["TestGenerationTool"]
    PG --> MP["MigrationPlannerTool"]
    PG --> RP["RollbackPlannerTool"]
    SR & TG & MP & RP --> IR["ImplementationReportTool"]
    IR --> OUT["output/<impl_id>/"]
```

**Patch-first philosophy:** patches are generated for review, never applied by default. Application is a separate, explicit, approval-gated step (`PatchApplicationTool`).

---

## Phase 5 — Research Memory

```mermaid
flowchart TD
    MEM["9 memory types"] --> SQL[("SQLite<br/>memories + relationships<br/>+ access_log + versions")]
    MEM --> CHR[("ChromaDB<br/>all-mpnet-base-v2")]
    MEM --> KG[("Knowledge graph<br/>10 relationship types")]
    RET["6 retrieval strategies"] --> SQL & CHR & KG
    DET["RelationshipDetector"] --> SQL
```

See [Memory System](memory_system.md) for the full deep dive.

---

## Phase 6 — Literature Intelligence

```mermaid
flowchart LR
    Q["Topic / query"] --> PS["PaperSearchTool<br/>(local + arXiv + Semantic Scholar)"]
    PS --> PC["PaperComparisonTool<br/>(7 dims)"]
    PS --> LR["LiteratureReviewTool"]
    PS --> PR["PaperRelationshipTool"]
    PS --> TA["TrendAnalysisTool"]
    PS --> RC["PaperRecommendationTool"]
    PS --> RS["RelevanceScoringTool<br/>(6 dims)"]
    PC & LR & PR & TA & RC & RS --> OUT["output/literature/<topic>_<ts>/"]
```

---

## Phase 7 — Experiment Execution

```mermaid
flowchart LR
    CMD["Command + repo"] --> ER["ExperimentRunnerTool<br/>(allowlist + dry-run + timeout)"]
    ER --> MN["MonitoringTool"]
    ER --> MC["MetricCollectorTool<br/>(logs/JSON/CSV)"]
    ER --> AC["ArtifactCollectorTool<br/>(checksums)"]
    MN & MC & AC --> FD["FailureDetectorTool"]
    FD & MC & AC --> ES["ExperimentStorageTool<br/>(SQLite)"]
```

**Safety:** dry-run default, command allowlist (`python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`), timeouts, working-directory confinement.

---

## Phase 8 — Evaluation

```mermaid
flowchart LR
    EXP["Experiment records"] --> EC["ExperimentComparisonTool"]
    EXP --> TD["TrainingDynamicsTool<br/>(over/underfit, convergence, instability)"]
    EXP --> SS["StatisticalSignificanceTool<br/>(Welch t-test, Cohen's d, CIs)"]
    EC & TD & SS --> NE["NextExperimentTool"]
    NE --> ES["EvaluationStorageTool<br/>(SQLite)"]
```

Statistical tests are **pure Python** (no SciPy dependency): Welch's t-test, Cohen's d, 95% confidence intervals.

---

## Phase 9 — Autonomous Research Loop

```mermaid
flowchart TD
    START([Start loop]) --> RECALL[1. MemoryAgent.get_context<br/>recall past insights]
    RECALL --> DISC[2. LiteratureAgent.discover<br/>first iteration only]
    DISC --> PLAN[3. ExperimentPlannerAgent.plan]
    PLAN --> IMPL[4. CodingAgent.implement]
    IMPL --> RUN[5. ExperimentAgent.run<br/>dry-run default]
    RUN --> EVAL[6. EvaluationAgent.analyze]
    EVAL --> STORE[7. LoopStorageTool<br/>persist iteration]
    STORE --> MEM[8. MemoryAgent.store<br/>success/failure/insight]
    MEM --> GRAPH[9. KnowledgeGraph<br/>add_node/add_relationship]
    GRAPH --> STOP{10. StoppingConditionChecker}
    STOP -->|target_achieved| END([Stop])
    STOP -->|max_iterations| END
    STOP -->|budget_exceeded| END
    STOP -->|no_improvement| END
    STOP -->|continue| APPROVAL{11. Approval gate?}
    APPROVAL -->|approved| RECALL
    APPROVAL -->|rejected| END
    END --> REPORT[ReportGeneratorTool<br/>research_report.md + .json]
```

**Loop state machine:** `created → running → iterating → evaluated → stopped` (with `awaiting_approval` when approval gates are enabled).

**Stopping conditions:** `target_achieved`, `max_iterations_reached`, `budget_exceeded`, `no_improvement`.

**Approval gates:** `plan`, `implementation`, `next_iteration`.

---

## Phase 10 — Provider-Agnostic LLM Layer

```mermaid
flowchart TD
    CFG["llm_config.yaml"] --> FAC["ProviderFactory<br/>builds + caches providers<br/>\${VAR} env expansion"]
    FAC --> ROUT["ModelRouter<br/>for_agent(name) → _BoundProvider"]
    ROUT --> PROV["OllamaCloudProvider<br/>POST /v1/chat/completions"]
    AGENTS["9 agents<br/>resolve_llm(agent_name, llm)"] --> ROUT
    PROV --> RESP["LLMResponse<br/>content + usage + model"]
```

**Resolution rules:**
1. Explicit `LLMProvider` passed to an agent constructor wins.
2. `llm_enabled=False` → no provider attached.
3. Otherwise `ModelRouter.for_agent(agent_name)` resolves from `llm_config.yaml`.

See [LLM Integration](llm_integration.md) for the full guide.

---

## Data flow

```mermaid
flowchart LR
    subgraph Write["Persistent writes"]
        P1["Phase 1"] --> DB[("SQLite")]
        P3["Phase 3"] --> DB
        P5["Phase 5"] --> DB
        P5 --> CHR[("ChromaDB")]
        P5 --> KG[("Knowledge graph")]
        P7["Phase 7"] --> DB
        P8["Phase 8"] --> DB
        P9["Phase 9"] --> DB
    end
    subgraph Read["Cross-run reads"]
        LOOP["Phase 9 loop"] --> MEM["MemoryAgent"]
        MEM --> DB
        MEM --> CHR
        MEM --> KG
    end
```

Every agent that stores a memory also calls `MemoryKnowledgeGraph.add_relationship()`, so the graph stays consistent without manual wiring. The loop's first step is always `MemoryAgent.get_context()` — it recalls relevant past insights before planning the next iteration.

---

## Storage overview

The platform persists to **three backends**:

| Backend | Purpose | Location |
|---------|---------|----------|
| **SQLite** | Structured records (papers, plans, memories, relationships, experiments, evaluations, loops, iterations) | `data/research_engineer.db` |
| **ChromaDB** | Vector embeddings for semantic memory search | `data/vector_store/` |
| **Knowledge graph** | Typed, weighted relationships between memories | in-memory + SQLite relationships table |

See [Storage Schema](storage_schema.md) for every table and column.

---

## Project structure

```
src/research_engineer/
├── agents/      # 9 agents + _llm_support.py
├── llm/          # Phase 10: base, ollama_provider, factory, router
├── models/       # 207 Pydantic models across 13 modules
├── tools/        # 51 typed tools
└── cli/          # 49 Typer commands
tests/            # 37 test files, 690 tests
llm_config.yaml   # provider + per-agent model config
docs/             # this documentation set
```

See [System Design](system_design.md) for the full file tree.

---

## Testing strategy

```mermaid
flowchart BT
    UNIT["Unit tests<br/>(models, tools, agents)"] --> INT["Integration tests<br/>(multi-phase)"]
    INT --> E2E["End-to-end tests<br/>(CLI, full pipelines)"]
```

- **690 tests** across 37 files.
- Every phase has dedicated model, tool, agent, and CLI test files.
- `test_integration.py` and `test_integration_phases.py` cover end-to-end pipelines.
- `test_llm.py` (29 tests) covers the LLM layer with a mock httpx transport.

```bash
uv run pytest -q          # 690 passed
uv run mypy src/research_engineer/llm   # clean
uv run ruff check .       # lint
```

---

## Dependencies

**Core:** `pydantic>=2.0`, `typer>=0.12`, `httpx>=0.26`, `pymupdf>=1.23`, `arxiv>=2.0`, `numpy`, `chromadb`, `sentence-transformers`.

**Dev:** `pytest>=8.0`, `pytest-asyncio>=0.23`, `ruff>=0.6`, `mypy>=1.11`, `pytest-cov`.

**Python:** ≥ 3.12. **Build:** hatchling. **Package manager:** uv (recommended).

---

*Version: 1.0 · Phase 10 complete · 690 tests passing*