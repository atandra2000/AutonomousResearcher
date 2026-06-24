# Roadmap

Versioned roadmap for the Autonomous ML Research Engineer, from the v1.0 baseline through v2.0 and beyond.

> **Current state (v2.0):** 15/15 phases complete · 23 agents · 61 tools · 186 models · 878 tests.

---

## v2.0 (current — complete)

All fifteen phases are production-ready:

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Paper Analysis | ✅ Complete | arXiv/PDF → structured `ResearchSummary` + `EngineeringReport` |
| 2 — Repository Analysis | ✅ Complete | AST scan, dependency graph, training pipeline, config analysis, knowledge graph |
| 3 — Experiment Planning | ✅ Complete | 7-dim compatibility, 5-group experiment matrix, 7-category risk, GPU-hour estimate |
| 4 — Code Implementation | ✅ Complete | Patch-first: code generation → unified diffs → self-review → tests → migration |
| 5 — Research Memory | ✅ Complete | 9 memory types, 10 relationship types, SQLite + ChromaDB + knowledge graph |
| 6 — Literature Intelligence | ✅ Complete | Multi-source search, 7-dim comparison, structured reviews, trend analysis |
| 7 — Experiment Execution | ✅ Complete | Sandboxed runner with allowlist + dry-run default + failure detection |
| 8 — Evaluation | ✅ Complete | Experiment comparison, training dynamics, Welch t-tests + Cohen's d (pure Python) |
| 9 — Autonomous Loop | ✅ Complete | State-machine orchestrator with stopping conditions + approval gates |
| 10 — LLM Layer | ✅ Complete | Provider-agnostic LLM layer (Ollama Cloud), per-agent model routing |
| 11 — Terminal-First Coding | ✅ Complete | `TaskAgent` with `TerminalTool`: analyze → plan → implement → diff → test |
| 12 — Repository Memory | ✅ Complete | AST-based symbol indexing, semantic chunking, symbol graph, hybrid retrieval |
| 13 — Multi-Agent Delegation | ✅ Complete | `DelegationFramework` with role/capability routing, `ArchitectAgent`, `ReviewerAgent`, `TestAgent` |
| 14 — Autonomous Self-Repair | ✅ Complete | `SelfRepairFramework` with `FailureAnalyzer`, `RepairStrategist`, 4 termination conditions |
| 15 — Research Workflows | ✅ Complete | `ResearchOrchestrator`: literature → synthesis → hypotheses → experiments → analysis → report |

---

## v2.1 — Multi-provider LLM support

**Goal:** Run the platform on any OpenAI-compatible provider, not just Ollama Cloud.

- [ ] `OpenAIProvider` (GPT-4o, o1, etc.)
- [ ] `AnthropicProvider` (Claude 3.5 Sonnet, Opus)
- [ ] `LocalOllamaProvider` (local Ollama daemon, `http://localhost:11434`)
- [ ] Provider health checks + automatic failover
- [ ] Cost tracking per agent (token usage → USD)
- [ ] Streaming-first agent outputs (chunked `LLMResponse`)

**No agent changes required** — all providers implement the existing `LLMProvider` ABC.

---

## v2.2 — Structured tool-calling

**Goal:** Let agents use LLM tool-calling for richer code generation and analysis.

- [ ] `LLMRequest.tools` field for function declarations
- [ ] `LLMResponse.tool_calls` parsing
- [ ] `CodingAgent` uses tool-calling for multi-file edits
- [ ] `EvaluationAgent` uses tool-calling for natural-language metric interpretation
- [ ] Structured output schemas (JSON Schema → Pydantic)

---

## v2.3 — Web UI dashboard

**Goal:** Visualize loops, memory, and the knowledge graph in a browser.

- [ ] FastAPI app serving loop state + iteration history
- [ ] Live loop monitoring (WebSocket streaming)
- [ ] Knowledge-graph visualization (D3.js / vis.js)
- [ ] Memory browser (search, filter, inspect)
- [ ] Experiment metric charts (Plotly)
- [ ] Approval-gate UI (approve/reject from the browser)

---

## v2.4 — Distributed execution

**Goal:** Run experiments across multiple nodes / GPUs.

- [ ] `DistributedExperimentRunnerTool` (Ray / Slurm backend)
- [ ] Multi-repo experiment matrices (one plan → N repos)
- [ ] Parallel experiment scheduling
- [ ] Aggregate metric collection across nodes
- [ ] Distributed artifact storage (S3 / GCS)

---

## v3.0 — Self-improving meta-loop

**Goal:** The platform proposes its own research goals from memory trends.

- [ ] `MetaLoopAgent` that analyzes the knowledge graph for promising directions
- [ ] Trend-driven goal generation (from `TrendAnalysisTool` output)
- [ ] Cross-repo insight synthesis
- [ ] Automated hypothesis → experiment → evaluation cycles
- [ ] Research report generation in paper format (LaTeX export)

---

## Backlog (unscheduled)

- PostgreSQL storage backend
- Redis caching layer
- Fine-tuned embedding model for ML papers
- Multi-modal paper analysis (figures, tables, equations)
- arXiv subscription / alerting (new papers in tracked topics)
- Integration with Weights & Biases / MLflow
- Hugging Face Hub dataset/model linking
- GitHub PR automation (apply approved patches as PRs)

---

## Versioning policy

This project follows [semantic versioning](https://semver.org/):

- **MAJOR** — breaking changes to agent/tool/model interfaces.
- **MINOR** — new phases, agents, tools, or providers (backwards-compatible).
- **PATCH** — bug fixes, test additions, doc improvements.

---

*Last updated: v2.0 · 15/15 phases complete*
