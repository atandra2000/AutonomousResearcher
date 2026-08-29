# P2 - Real LLM Research Benchmark

P2 adds the **LLM-backed tier** to the existing E1-E9 + P1 architecture.
No new evaluation framework was introduced: the tier reuses the E4
`EvalSuite`/`EvalTask` model, the P1 `BenchmarkRunner` production path
(API -> store -> worker -> `AgentRuntime` -> `ToolGateway` ->
`SafetyController` -> checkpointing -> evaluation -> E6 telemetry -> E8
failure mining), and the Phase 10 provider abstraction.

## Tiers

| Tier | Suite | Mode | Agent kind | Purpose |
|------|-------|------|------------|---------|
| Deterministic regression | `evals/research_benchmark/v1/suite.yaml` | `deterministic_sandbox` / `transient_recovery` / `policy_guardrail` | `bench_tool`, `bench_flaky`, `bench_loop`, `bench_risky` | Infrastructure, recovery and safety validation without credentials |
| LLM-backed research | `evals/research_benchmark/v2/suite.yaml` | `llm_agent` | `llm_react` | Real autonomous-research capability under fixed budgets |

Results from the tiers are **never mixed**: reports split every metric by
mode family.

## Real-agent execution path

`llm_react` (`src/research_engineer/benchmark/llm_agent.py`) runs one model
call per runtime step through `LLMProvider.complete_with_tools` (router
bound provider = retry/backoff + USD cost stamping). Every tool request the
model makes is dispatched via `AgentRuntime.call_tool`, i.e. the full E3
gateway policy/approval/sandbox chain and E5 safety controls - the model
cannot bypass them; denials are surfaced back as tool errors.

Fail-closed guarantees:

* Provider unreachable -> step raises -> run terminates ERROR (never
  fabricated results).
* Credentials only via environment expansion in `llm_config.yaml`
  (`${OLLAMA_API_KEY}` etc.); keys are never logged or persisted.
* No runtime attached -> tool dispatch refused (`RuntimeError`).

## Configuration

Provider/model selection is configuration-driven
(`llm_config.yaml` / `OLLAMA_*` env vars); per-case scalar overrides:

```
llm_provider, llm_model, llm_temperature, llm_max_tokens_per_call,
max_steps, max_tool_calls, max_tokens, max_cost_usd,
max_recoverable_errors
```

## Evaluation integrity

* Tasks are self-contained (evidence lives in the goal text); no answers
  are embedded and nothing beyond the goal enters agent prompts.
* Objective success is gated exclusively by deterministic graders
  (`termination`, `recovery`, `budget`, `tool_usage`,
  `output_json_field(op>=)`, `output_contains`, `output_regex`);
  structural markers are case-defined format contracts, not content
  answers; numeric checks grade derivable arithmetic (seed means, dollar
  math).
* The LLM judge (`llm_quality`) is always `required=false`: it reports
  research quality separately and cannot inflate pass/fail.
* Case revisions are recorded per report plus SHA-256 fingerprints of the
  suite revision set and of the LLM configuration; regression comparisons
  refuse to silently compare different configurations.
* Judge failures score 0.0 (fail closed).

## Metrics reported

Per tier: autonomous completion rate, objective task success rate, mean
weighted score, human/safety-intervention rates, gateway-denial rate,
median tokens/cost/latency, termination distribution. Composite headline:
autonomous successful completion of real research tasks under fixed
cost/token budgets. Variance section repeats selected cases (default 8)
and flags unstable successes.

## Execution

```bash
# Both tiers + variance repeats (default)
uv run research-engineer benchmark run --tier all \
    --output-dir artifacts/p2_benchmark --repeats 2

# Deterministic only / LLM only
uv run research-engineer benchmark run --tier deterministic
uv run research-engineer benchmark run --tier llm --repeats 2

# Regression comparison against a previous composite report
uv run research-engineer benchmark run --tier llm \
    --compare artifacts/p2_benchmark/p2_report.json
```

Outputs: `p2_report.json` + `p2_report.md` plus raw per-tier
`benchmark_report.json/.md`. E8 mining stays read-only: patterns are
reported, never applied automatically.

## Files added/changed (P2)

Added:
- `src/research_engineer/benchmark/llm_agent.py` - `llm_react` agent kind
- `src/research_engineer/benchmark/llm_judge.py` - rubric judge wiring
- `src/research_engineer/benchmark/p2_benchmark.py` - orchestration/reporting
- `evals/research_benchmark/v2/suite.yaml` - 20-case LLM suite
- `tests/test_llm_benchmark.py`

Changed:
- `benchmark/benchmark.py` - validation for `llm_agent` mode / `llm_react`
  kind / optional `llm_quality` criteria
- `benchmark/benchmark_runner.py` - optional `extra_graders`/`factories`,
  unknown-grader fail-closed criterion
- `benchmark/agents.py` - token/cost/tool-call budget override plumbing
- `eval/graders.py` - comparison ops for `output_json_field`
- `llm/resilience.py` - tool-call responses are final (no empty-content retry)
- `llm/ollama_provider.py`, `llm/openai_provider.py` - OpenAI-wire shape
  for assistant `tool_calls` on multi-turn conversations
- `cli/__init__.py` - `research-engineer benchmark run`
