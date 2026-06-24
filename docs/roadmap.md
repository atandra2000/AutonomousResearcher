# Roadmap

Versioned roadmap for the Autonomous ML Research Engineer, from the v1.0 baseline through v2.0.

> **Current state (v1.0):** 10/10 phases complete · 9 agents · 51 tools · 207 models · 690 tests.

---

## v1.0 (current — complete)

All ten phases are production-ready:

| Phase | Status |
|-------|--------|
| 1 — Paper Analysis | ✅ Complete |
| 2 — Repository Analysis | ✅ Complete |
| 3 — Experiment Planning | ✅ Complete |
| 4 — Code Implementation | ✅ Complete (patch-first) |
| 5 — Research Memory | ✅ Complete (SQLite + ChromaDB + knowledge graph) |
| 6 — Literature Intelligence | ✅ Complete |
| 7 — Experiment Execution | ✅ Complete (sandboxed) |
| 8 — Evaluation | ✅ Complete (pure-Python stats) |
| 9 — Autonomous Loop | ✅ Complete (stopping conditions + approval gates) |
| 10 — LLM Layer | ✅ Complete (Ollama Cloud, per-agent routing) |

---

## v1.1 — Multi-provider LLM support

**Goal:** Run the platform on any OpenAI-compatible provider, not just Ollama Cloud.

- [ ] `OpenAIProvider` (GPT-4o, o1, etc.)
- [ ] `AnthropicProvider` (Claude 3.5 Sonnet, Opus)
- [ ] `LocalOllamaProvider` (local Ollama daemon, `http://localhost:11434`)
- [ ] Provider health checks + automatic failover
- [ ] Cost tracking per agent (token usage → USD)
- [ ] Streaming-first agent outputs (chunked `LLMResponse`)

**No agent changes required** — all providers implement the existing `LLMProvider` ABC.

---

## v1.2 — Structured tool-calling

**Goal:** Let agents use LLM tool-calling for richer code generation and analysis.

- [ ] `LLMRequest.tools` field for function declarations
- [ ] `LLMResponse.tool_calls` parsing
- [ ] `CodingAgent` uses tool-calling for multi-file edits
- [ ] `EvaluationAgent` uses tool-calling for natural-language metric interpretation
- [ ] Structured output schemas (JSON Schema → Pydantic)

---

## v1.3 — Web UI dashboard

**Goal:** Visualize loops, memory, and the knowledge graph in a browser.

- [ ] FastAPI app serving loop state + iteration history
- [ ] Live loop monitoring (WebSocket streaming)
- [ ] Knowledge-graph visualization (D3.js / vis.js)
- [ ] Memory browser (search, filter, inspect)
- [ ] Experiment metric charts (Plotly)
- [ ] Approval-gate UI (approve/reject from the browser)

---

## v1.4 — Distributed execution

**Goal:** Run experiments across multiple nodes / GPUs.

- [ ] `DistributedExperimentRunnerTool` (Ray / Slurm backend)
- [ ] Multi-repo experiment matrices (one plan → N repos)
- [ ] Parallel experiment scheduling
- [ ] Aggregate metric collection across nodes
- [ ] Distributed artifact storage (S3 / GCS)

---

## v2.0 — Self-improving meta-loop

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

*Last updated: v1.0 · 10/10 phases complete*