# Autonomous Agentic Implementation Plan

> **Goal:** Transform the Autonomous ML Research Engineer from a sequential
> pipeline orchestrator into a fully autonomous, industry-grade, deployable
> agentic research system.
>
> **Status:** In progress — **A1 (tool-calling protocol) is complete.** The
> remaining workstreams are queued and will be implemented one at a time in
> subsequent sessions.

---

## 1. Executive summary

The platform is architecturally strong: 15 phases, 23 agents, 63 tools,
pydantic v2 typed contracts, and a passing test suite. The core gap is that it
is a **sequential pipeline orchestrator**, not an **agentic system**. Agents
call the LLM once per stage, there is no tool-calling protocol, no ReAct loop,
no real feedback between iterations, and no production-grade observability or
cost accounting.

This plan closes that gap in **four tiers** across **seven workstreams**,
each with concrete file-level changes, effort estimates, and dependencies.

| Tier | Theme | Workstreams |
|------|-------|-------------|
| **Tier 1** | Agentic core | A1 tool-calling protocol · A2 ReAct loop · A3 multi-provider · A4 truncation |
| **Tier 2** | Feedback loop | B1 `_derive_next_command()` · B2 memory-driven iteration · B3 research-output evaluation |
| **Tier 3** | Real grounding | C1 real experiment execution · C2 parallelization · C3 artifact management |
| **Tier 4** | Production | D1 cost accounting · D2 observability · D3 human-in-the-loop UX · D4 better memory |

---

## 2. Current-state findings (the gaps)

### 2.1 LLM layer (`src/research_engineer/llm/`)

- `LLMProvider` ABC exposes only `complete()` / `stream()` — **no function-calling**.
- Only `OllamaCloudProvider` is implemented; no OpenAI / Anthropic / local Ollama.
- No truncation handling: `finish_reason='length'` is not retried with a larger budget.
- No cost / token accounting surfaced to callers or persisted.
- No observability hooks (no structured logging of prompts, responses, latency, cost).

### 2.2 Orchestration (`agents/research_workflow.py`, `agents/loop_agent.py`)

- `ResearchWorkflowFramework._run_stage()` calls each agent **once** — no retry, no reflection, no tool loop.
- `ResearchLoopAgent._derive_next_command()` is a **stub returning `None`** — the loop cannot actually decide what to do next.
- `_estimate_cost_hours()` is **hardcoded to `0.5`**.
- Orchestration is strictly sequential; no parallel stage execution.

### 2.3 Memory (`memory/`)

- Uses `HashingEmbedder` (offline, weak) + `InMemoryVectorBackend` — no real semantic retrieval.
- No persistent vector store for production.

### 2.4 Experiments (`tools/`, `agents/experiment_agent.py`)

- Experiments default to **dry-run**; no real training execution.
- No parallel experiment scheduling.
- No artifact management beyond copying files.

### 2.5 Reporting (`agents/research_stages.py`)

- Report generator uses a **template footer**, not reasoned synthesis from results.

---

## 3. Tier 1 — Agentic core

### 3.1 Workstream A1 — Tool-calling protocol ✅ COMPLETE

**Goal:** Give the LLM layer a first-class function-calling protocol so agents
can request tool invocations and feed results back into the conversation.

**Files changed (done):**

| File | Change |
|------|--------|
| `llm/base.py` | Added `ToolDefinition`, `ToolCall`, `ToolResult` models; `tool_calls` on `LLMMessage` and `LLMResponse`; `LLMProvider.complete_with_tools()` (default raises `NotImplementedError`) |
| `llm/ollama_provider.py` | Implemented `complete_with_tools()` (builds OpenAI `tools` array, POSTs, parses tool calls) + `_parse_tool_calls()` helper |
| `llm/router.py` | `_BoundProvider.complete_with_tools()` delegates with model pinning + retry |
| `llm/resilience.py` | `complete_with_retry(..., tools=...)` routes through `complete_with_tools`; extracted `_call_once` helper |
| `llm/__init__.py` | Exported `ToolDefinition`, `ToolCall`, `ToolResult` |
| `tests/test_llm.py` | +11 tests (models, Ollama round-trip, default `NotImplementedError`, bound-provider delegation) |

**Validation:** full suite **912 passed** (was 901), ruff clean, no new mypy errors.

**Effort:** ~0.5 day. **Dependency:** none (foundation for A2).

---

### 3.2 Workstream A2 — ReAct agent loop

**Goal:** Add a reusable ReAct (Reason + Act) loop that lets any agent iterate:
call the model → execute requested tools → feed results back → repeat until the
model produces a final answer or a step budget is exhausted.

**New file:** `src/research_engineer/llm/react_loop.py`

**Proposed API:**

```python
class ReActLoopConfig(BaseModel):
    max_steps: int = 8
    max_tool_calls_per_step: int = 4
    stop_on_final: bool = True
    final_marker: str = "FINAL_ANSWER:"
    on_step: Callable[[ReActStep], None] | None = None  # observability hook

class ReActStep(BaseModel):
    step: int
    messages: list[LLMMessage]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    response: LLMResponse

async def run_react_loop(
    provider: LLMProvider,
    messages: list[LLMMessage],
    tools: list[ToolDefinition],
    executor: Callable[[ToolCall], Awaitable[ToolResult]],
    config: ReActLoopConfig | None = None,
) -> ReActResult
```

**Behavior:**
1. Append the assistant message (with `tool_calls`) to the conversation.
2. Execute each requested tool via the injected `executor`.
3. Append `tool`-role `LLMMessage`s carrying each `ToolResult`.
4. Re-call the model; repeat until `FINAL_ANSWER:` appears, no tool calls are
   requested, or `max_steps` is reached.
5. Return the final `LLMResponse` plus the full step trace (for observability).

**Files to change:**
- `llm/react_loop.py` (new) — the loop + models.
- `llm/__init__.py` — export `run_react_loop`, `ReActLoopConfig`, `ReActStep`, `ReActResult`.
- `tests/test_react_loop.py` (new) — fake provider + fake executor; assert multi-step tool round-trips, final-answer termination, step-budget exhaustion, error propagation.

**Effort:** 1 day. **Dependency:** A1 (complete).

---

### 3.3 Workstream A3 — Multi-provider support

**Goal:** Run on any OpenAI-compatible provider plus Anthropic, with health
checks and failover.

**New files:**
- `llm/openai_provider.py` — `OpenAIProvider` (GPT-4o, o1, etc.), OpenAI-compatible.
- `llm/anthropic_provider.py` — `AnthropicProvider` (Claude 3.5/3.7 Sonnet, Opus), native Anthropic Messages API.
- `llm/local_ollama_provider.py` — `LocalOllamaProvider` (`http://localhost:11434`).

**Files to change:**
- `llm/factory.py` — register the new provider types in `_PROVIDER_REGISTRY`; add `health_check` + `failover` ordering.
- `llm/base.py` — add `async def health() -> bool` to `LLMProvider` (default `True`).
- `llm/router.py` — add failover: if the primary provider is unhealthy, fall back to the next configured provider.
- `llm_config.yaml` — document the new provider blocks.
- `tests/test_providers.py` (new) — mock-transport tests for each provider; failover test.

**Effort:** 2 days. **Dependency:** A1 (tool-calling parity across providers).

---

### 3.4 Workstream A4 — Truncation handling

**Goal:** Handle `finish_reason='length'` gracefully — retry with a larger
`max_tokens` budget (once), then surface a clear signal.

**Files to change:**
- `llm/resilience.py` — already partially handles empty-content + `length`
  escalation in `complete_with_retry`. Extend to also handle the case where
  content is **non-empty but truncated** (retry with doubled budget once).
- `llm/base.py` — add `truncated: bool` convenience property on `LLMResponse`
  (`finish_reason == 'length'`).
- `tests/test_resilience.py` — add truncation-retry tests.

**Effort:** 0.5 day. **Dependency:** none.


---

## 4. Tier 2 — Feedback loop

### 4.1 Workstream B1 — Implement `_derive_next_command()`

**Goal:** Replace the stub in `ResearchLoopAgent` with a real decision function
that inspects the last iteration's evaluation and memory context to decide the
next action.

**Files to change:**
- `agents/loop_agent.py` — implement `_derive_next_command()`:
  - If last iteration improved the target metric → continue in the same direction.
  - If it regressed → propose a corrective experiment (change hyperparameters, architecture, or data).
  - If no improvement over N iterations → trigger a literature re-discovery or stop.
  - Return a structured `NextCommand` (pydantic model) instead of `None`.
- `models/loop.py` — add `NextCommand` model.
- `tests/test_loop_agent.py` — unit tests for each decision branch.

**Effort:** 1 day. **Dependency:** none.

---

### 4.2 Workstream B2 — Memory-driven iteration

**Goal:** Make each loop iteration actually use prior memory (insights,
successes, failures) to inform planning and implementation.

**Files to change:**
- `agents/loop_agent.py` — inject `MemoryAgent.get_context()` results into the
  planner prompt and the coding-agent context on every iteration.
- `agents/experiment_planner_agent.py` — accept a `memory_context` parameter and
  surface it in the plan.
- `agents/coding_agent.py` — accept `memory_context` and include it in the
  implementation prompt.
- `tests/test_loop_agent.py` — assert memory context flows into planner/coder.

**Effort:** 1 day. **Dependency:** B1.

---

### 4.3 Workstream B3 — Research-output evaluation

**Goal:** Evaluate the *quality of the research output itself* (not just
experiment metrics) — e.g. whether the report is coherent, the hypothesis is
tested, and the conclusions follow from the data.

**Files to change:**
- `agents/evaluation_agent.py` — add a `evaluate_research_output()` method that
  scores the generated report against the hypothesis and results.
- `models/evaluation.py` — add `ResearchOutputEvaluation` model.
- `agents/research_stages.py` — wire the report generator to consume the
  evaluation and produce reasoned synthesis instead of a template footer.
- `tests/test_evaluation_agent.py` — unit tests.

**Effort:** 1.5 days. **Dependency:** none.


---

## 5. Tier 3 — Real grounding

### 5.1 Workstream C1 — Real experiment execution

**Goal:** Move from dry-run-only to real, sandboxed training execution with
proper failure detection and resource limits.

**Files to change:**
- `tools/experiment_runner.py` — add a `real_run` mode that actually launches
  the training subprocess (with the existing allowlist), enforces time/memory
  limits, and streams output.
- `agents/experiment_agent.py` — add a `--real` flag / config to opt into real
  execution; keep dry-run as the safe default.
- `tools/failure_detector.py` — extend to detect OOM, NaN loss, and timeouts.
- `tests/test_experiment_runner.py` — integration tests with a tiny fake training script.

**Effort:** 2 days. **Dependency:** none.

---

### 5.2 Workstream C2 — Parallelization

**Goal:** Run independent stages and experiments concurrently.

**Files to change:**
- `agents/research_workflow.py` — allow stages that have no data dependency to
  run concurrently via `asyncio.gather`.
- `tools/experiment_runner.py` — add a `run_many()` that schedules independent
  experiments in parallel with a concurrency cap.
- `models/experiment.py` — add `parallel_group` / `concurrency` fields.
- `tests/test_research_workflow.py` — assert concurrent stage execution.

**Effort:** 1.5 days. **Dependency:** C1.

---

### 5.3 Workstream C3 — Artifact management

**Goal:** Centralize artifact collection, versioning, and retrieval.

**Files to change:**
- `tools/artifact_collector.py` — add versioned artifact storage (hash-based
  dedup), metadata sidecars, and a retrieval API.
- `models/experiment.py` — add `ArtifactManifest` model.
- `tools/experiment_storage.py` — persist artifact manifests alongside results.
- `tests/test_artifact_collector.py` — unit tests.

**Effort:** 1 day. **Dependency:** C1.


---

## 6. Tier 4 — Production

### 6.1 Workstream D1 — Cost accounting

**Goal:** Track token usage and estimate USD cost per agent, per loop, per run.

**Files to change:**
- `llm/base.py` — add `cost_usd` to `LLMUsage` (computed from a per-model
  price table).
- `llm/factory.py` — load a `pricing` section from `llm_config.yaml`.
- `llm/router.py` — accumulate usage/cost per agent in a `UsageTracker`.
- `agents/loop_agent.py` — replace the hardcoded `_estimate_cost_hours()` with a
  real estimate from tracked usage.
- `tests/test_llm.py` / `tests/test_loop_agent.py` — cost-accounting tests.

**Effort:** 1 day. **Dependency:** A3 (multi-provider pricing).

---

### 6.2 Workstream D2 — Observability

**Goal:** Structured, queryable logs of every LLM call, tool call, and stage.

**Files to change:**
- `llm/base.py` — add an optional `on_complete` callback / event hook.
- `llm/resilience.py` — emit structured log records (prompt hash, latency,
  tokens, cost, finish_reason).
- `agents/research_workflow.py` — log stage start/end with durations.
- New `observability/` module (or extend `tools/`) — a lightweight event sink
  (JSONL + optional SQLite table).
- `tests/test_observability.py` — assert events are emitted.

**Effort:** 1.5 days. **Dependency:** D1 (cost in logs).

---

### 6.3 Workstream D3 — Human-in-the-loop UX

**Goal:** Better approval gates and interactive review of plans, patches, and
experiments.

**Files to change:**
- `agents/loop_agent.py` — richer approval prompts (show plan diff, expected
  cost, risk) before each gate.
- `cli/` — add interactive review commands (`review plan`, `review patch`,
  `approve experiment`).
- `models/loop.py` — extend `ApprovalRequest` with context fields.
- `tests/test_cli.py` — CLI review-flow tests.

**Effort:** 1.5 days. **Dependency:** B1, D1.

---

### 6.4 Workstream D4 — Better memory

**Goal:** Replace the weak offline embedder with a real semantic vector store.

**Files to change:**
- `memory/embeddings.py` — add a `SentenceTransformerEmbedder` (optional, gated
  behind a config flag) alongside `HashingEmbedder`.
- `memory/vector_backend.py` — add a persistent `ChromaDBBackend` alongside
  `InMemoryVectorBackend`.
- `memory/retriever.py` — support hybrid retrieval across the new backend.
- `llm_config.yaml` / `config` — document the embedding/vector-store settings.
- `tests/test_memory_*.py` — backend-agnostic tests.

**Effort:** 2 days. **Dependency:** none.


---

## 7. Dependency graph

```
A1 (done) ──► A2 ──► A3 ──► D1
                │
                └──► B1 ──► B2 ──► D3
A4 (independent)
B3 (independent)
C1 ──► C2 ──► C3
D2 (depends on D1)
D4 (independent)
```

**Critical path:** A1 → A2 → A3 → D1 → D2.

---

## 8. Effort summary

| Workstream | Tier | Effort | Dependency |
|------------|------|--------|------------|
| A1 Tool-calling protocol | 1 | 0.5 day | — (done) |
| A2 ReAct loop | 1 | 1 day | A1 |
| A3 Multi-provider | 1 | 2 days | A1 |
| A4 Truncation handling | 1 | 0.5 day | — |
| B1 `_derive_next_command()` | 2 | 1 day | — |
| B2 Memory-driven iteration | 2 | 1 day | B1 |
| B3 Research-output evaluation | 2 | 1.5 days | — |
| C1 Real experiment execution | 3 | 2 days | — |
| C2 Parallelization | 3 | 1.5 days | C1 |
| C3 Artifact management | 3 | 1 day | C1 |
| D1 Cost accounting | 4 | 1 day | A3 |
| D2 Observability | 4 | 1.5 days | D1 |
| D3 Human-in-the-loop UX | 4 | 1.5 days | B1, D1 |
| D4 Better memory | 4 | 2 days | — |

**Total:** ~19.5 days of focused implementation.

---

## 9. Validation gates (per workstream)

Every workstream must pass, in order, before being declared complete:

1. `uv run python -m pytest` — full suite green, **never below 912 passing**.
2. `uv run ruff check .` — clean.
3. `uv run mypy .` — no new errors introduced by the change.
4. New tests added for every new public API and behavior branch.

---

## 10. Suggested implementation order (one at a time)

1. **A2** ReAct loop (builds directly on A1).
2. **A4** Truncation handling (small, independent).
3. **A3** Multi-provider support (unlocks D1).
4. **B1** `_derive_next_command()` (unlocks the feedback loop).
5. **B2** Memory-driven iteration.
6. **B3** Research-output evaluation.
7. **C1** Real experiment execution.
8. **C2** Parallelization.
9. **C3** Artifact management.
10. **D1** Cost accounting.
11. **D2** Observability.
12. **D3** Human-in-the-loop UX.
13. **D4** Better memory.

---

*Last updated: A1 complete · 912 tests passing · next up: A2 ReAct loop.*

