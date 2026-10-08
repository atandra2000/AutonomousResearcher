# Roadmap

Versioned roadmap for the Autonomous ML Research Engineer, from the v1.0 baseline through v2.0 and beyond.

> **Current state (v2.1):** 15/15 phases complete · 20 agents · 61 tools · 251 models · 1451 tests.

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

## v2.1 — Multi-provider LLM support (complete)

**Goal:** Run the platform on any OpenAI-compatible provider, not just Ollama Cloud.

- [x] `OpenAIProvider` (GPT-4o, o1, etc.)
- [x] `AnthropicProvider` (Messages-API wire, model pinned at config time)
- [x] `LocalOllamaProvider` (local Ollama daemon, `http://localhost:11434`)
- [x] Provider health checks + automatic failover
- [x] Cost tracking per agent (token usage → USD)
- [x] Streaming-first agent outputs (chunked `LLMResponse`)

**No agent changes required** — all providers implement the existing `LLMProvider` ABC.

---

## E1 — Production Agent Runtime (complete)

**Goal:** A generic, async-first `AgentRuntime` that becomes the central
orchestration layer for autonomous agents.

- [x] `AgentRuntime` with deterministic state machine (`CREATED → RUNNING → TERMINATED`)
- [x] `plan → act → observe → evaluate` loop via injected async callables
- [x] Typed models: `AgentState`, `AgentContext`, `AgentExecution`, `AgentPolicy`, `AgentBudget`
- [x] Serializable execution state (JSON) for future checkpointing (E2)
- [x] Budgets: max steps, tool calls, runtime, cost, tokens
- [x] Termination reasons: success, budget exceeded, timeout, cancelled, error, no-progress
- [x] Recoverable vs fatal error handling with retry
- [x] Cooperative cancellation
- [x] Observability integration (structured `agent_runtime` events)
- [x] `AgentAdapter` for running existing agents unchanged through the runtime
- [x] 34 unit/integration tests

**Extension points (E2+):** checkpoint persistence, eval framework, deployment.

---

## v2.2 — Structured tool-calling

**Goal:** Let agents use LLM tool-calling for richer code generation and analysis.

- [x] Typed function declarations and tool-call/result contracts
- [x] Provider tool-call parsing and ReAct execution loop
- [x] Coding and benchmark agents route tools through the policy gateway
- [x] Evaluation agents can use LLM-backed typed analysis
- [x] Pydantic schemas at agent/tool boundaries

---

## v2.3 — CLI-native ergonomics

- [x] Project-aware chat startup (`chat --repo`) and `/repo` switching
- [x] Interactive `/task` over the production `TaskAgent`
- [x] Safe task modes: dry-run default; explicit apply/tests/delegation
- [x] Bounded multi-turn LLM context and `/clear`
- [x] Atomic local session state and `chat --resume` for UI continuity
- [ ] Streaming token output in the chat session (`/research` stages + LLM turns)
- [ ] Mid-task resume from an `AgentRuntime` checkpoint
- [ ] Knowledge-graph + memory visualization rendered as terminal graphs
- [ ] Experiment metric sparklines in the loop monitor
- [ ] Approval-gate prompts inline in the chat session

### Completion path to a production Codex-like CLI

1. **Unify execution:** model TaskAgent stages as `AgentRuntime` steps so coding
   turns gain durable checkpoints, resume, cancellation, budgets, and one event
   stream instead of a parallel orchestration path.
2. **Make safety interactive:** connect `ToolGateway` approval requests to the
   chat prompt and show the command, workspace, risk, and diff before consent.
3. **Persist sessions:** store the active repository, LLM messages, task IDs,
   and runtime execution ID; add `chat --resume <session-id>`.
4. **Stream truthful progress:** render runtime/tool/stage events and token
   deltas without claiming completion until persisted results are available.
5. **Prove the product path:** add PTY-level smoke tests for interrupt, resume,
   approval denial, failed tests, and a successful dry-run task; benchmark this
   same path in P1/P2.

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

## Benchmark execution path (CLI-dedicated) ✅ CURRENT

Benchmarks run **directly through the production runtime path** — the
same `AgentRuntime → ToolGateway → SafetyController → checkpointing`
stack an autonomous CLI run uses. There is no separate serving tier:
`BenchmarkRunner` builds the safety chain per case, executes via
`AgentRuntime`, persists the finished run context as JSON, and grades
the persisted payload. The former queue/worker/API service layer was
removed in the CLI-dedicated rework; the benchmark machinery now lives
in `research_engineer.benchmark`.
