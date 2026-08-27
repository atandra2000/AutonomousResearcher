# P2 Session Compressed State (working notes — safe to delete)

## Task
P2 Real LLM Research Benchmark using existing E1–E9+P1 stack. Deliverables:
versioned v2 suite (done), llm_react agent factory (done), execution command
(done: `research-engineer benchmark run`), reports (JSON+MD done), failure
analysis, reproducibility metadata, regression comparison, final 13-section
report. Do NOT fabricate results; production path stays fail-closed; E8 is
read-only.

## Stack facts
- Python 3.12, uv, pydantic v2, StrEnum, line length 88, ruff+mypy strict.
- Run commands: `uv run python -m pytest`, `uv run ruff check .`,
  `uv run mypy .`. Long jobs: `nohup ... &` then poll with <25s sleeps
  (tool aborts >~28s). Bash `timeout` unavailable on macOS.
- Provider resolved via Phase 10 router (`llm_config.yaml`). Credentials
  env-only (`OLLAMA_API_KEY`). NEVER print/log secrets.

## Current model/config
- `llm_config.yaml`: ALL glm-5.2:cloud → **glm-5.3-flash** (22 refs,
  global default_model too). User requested this switch mid-session.
- Router agent names: `BenchmarkLLM` (agent kind binding),
  `EvaluationAgent` (judge).

## Files created
- `src/research_engineer/service/llm_agent.py`: KIND_LLM_REACT=`llm_react`;
  LLMReActAdapter(RuntimeAwareAdapter): 1 model call per runtime step;
  tools = research_note_write/research_note_list/echo_probe only; empty
  assistant content placeholder fix; resolve_llm_provider → router bound
  provider (cost stamping + retry); overrides llm_provider/llm_model/
  llm_temperature/llm_max_tokens_per_call (default 2048).
- `src/research_engineer/service/llm_judge.py`: make_judge_score_fn
  (SCORE regex, fail-closed 0.0, empty rubric→0), build_llm_quality_grader,
  max_tokens=1024.
- `src/research_engineer/service/p2_benchmark.py`: run_p2(output_dir, *,
  suite_path, p1_regression, repeats, variance_cases(8 default), label,
  previous_report, factories, extra_graders, case_filter) -> P2Report;
  P2Report.verdict thresholds; suite/config fingerprints;
  compare_reports (CONFIG MISMATCH note); tier_metrics_for(mode);
  build_p2_factories(provider=None). DEFAULT_VARIANCE_CASES=8 ids.
- `evals/research_benchmark/v2/suite.yaml`: 20 cases x 8 categories,
  mode=llm_agent, revision r1, budgets {steps 8, tools 16, tokens 24-28k,
  cost 0.5-0.6, runtime 300s}, timeout 420s; objective criteria required +
  optional llm_quality rubrics (block-style YAML).
- `tests/test_llm_benchmark.py`: 18 tests (suite contract, adapter flows,
  denials, fail-closed, budget overrides, grading, judge, fingerprints,
  comparison, e2e production stack w/ ScriptedProvider).
- `docs/benchmark_p2.md`: documentation.

## Files modified
- `service/benchmark.py`: modes+=llm_agent; kinds+=llm_react;
  LLM_JUDGED_GRADERS={"llm_quality"} (non-required only, llm_agent-mode);
  DEFAULT_SUITE_V2_PATH.
- `service/benchmark_runner.py`: BenchmarkRunner(extra_graders=,
  factories=); _grade_case(extra_graders) + unknown-grader fail-closed.
- `service/agents.py`: build honors max_tool_calls/max_tokens/max_cost_usd/
  max_recoverable_errors overrides.
- `eval/graders.py`: OutputJSONFieldGrader op eq|gte|lte|gt|lt.
- `llm/resilience.py`: response WITH tool_calls is final (no re-try when
  content empty).
- `llm/ollama_provider.py`,`openai_provider.py`: `_wire_message` — tool_calls
  serialized to OpenAI wire shape (arguments JSON STRING) else Ollama
  HTTP 400 "invalid tool call arguments" on turn>=2. REAL bug fixed.
- `cli/__init__.py`: `benchmark run` sub-app.

## Validation baseline
- Full pytest: **1424 passed** (pre-final minor edits). 18 llm_bench tests pass.
- Repo ruff: ~311 pre-existing errors unchanged; my new files CLEAN.
- mypy: HEAD 854/58 vs mine 854/57 → zero regressions; touched files clean.

## Runs executed
1. Deterministic P1 via CLI: artifacts/p2_benchmark/ → completion 100%.
2. Smoke (glm-5.2, post-fixes): artifacts/p2_smoke → 3/3 success.
3. FULL RUN on glm-5.3-flash COMPLETED:
   artifacts/p2_final_run/{v2_full_suite.json, v2_variance_r2.json,
   deterministic_regression/p1_deterministic_regression.json,
   p2_report.json, p2_report.md}
   Printed headline: VERDICT READY; autonomous_completion_rate=0.8056;
   objective_task_success_rate=0.8056; llm_tokens_total=148719;
   llm_cost_total_usd=0.16169; wall_clock_minutes=15.42; llm_agent
   tier cases_total=36 (BUG). Raw result.json count 76.

## OPEN BUGS in run_p2 — ALL RESOLVED (this session)
A. Mode split — fixed: `{o.mode for o in all_outcomes}`.
B. Variance labels — fixed: repeat=1 per attempt, relabel to rep via
   model_copy before merge.
C. Judge zeros — TWO stacked root causes:
   1. extra_graders skipped configure() -> rubric "" -> fail-closed 0.0.
      Fix: per-criterion deepcopy(shared) + configure in _grade_case.
   2. Remaining zeros were swallowed judge exceptions; made score_fn
      propagate so LLMPromptGrader records "scoring function failed: <r>"
      (fail closed WITH cause). max_tokens 1024->2048 (flash reasoning
      truncation at 1024).
D. StatisticsError when a mode family is all guardrail anti-cases:
   tier_metrics_for mean over empty effective set -> guarded (0.0).
E. Aggregation extracted to build_p2_report() — run_p2 delegates;
   offline re-aggregation from persisted stage JSONs possible without
   re-executing agents (used for the final report).

## FINAL RUN (authoritative) — artifacts/p2_final_run/p2_report.json
Model/provider: ollama/glm-5.3-flash (agent + judge), llm_config sha256
a411775c... Repeats: full suite r1 (20 LLM cases) + variance subset r2
(8 cases) + P1 deterministic tier. VERDICT: READY WITH RISKS.
- llm_agent tier: n=28, completion 0.7857, success 0.7857, mws 0.8907,
  median tokens/task 3984, median cost $0.004174, latency med 11.8s,
  terminations all success, zero human/safety/gateway interventions.
- deterministic_sandbox n=28: 1.0/1.0/1.0. transient_recovery n=10:
  1.0/1.0/1.0 (safety_int 0.6 by design). policy_guardrail n=2: 0% by
  design (safety_terminated, fail-closed verified).
- Research quality (judge): mean 0.693 across 20 judged cases; spread
  0.4-1.0; two unparseable-judge events (litdisc_01, design_02 at 1024
  cap) now auditable in criterion detail.
- Failures (6 of 28 attempts): impl_01_attention_module_plan x2
  (deterministic, weighted 0.5217 both attempts -> genuine capability
  gap), design_02_data_scaling_study, debug_03_oom_step1200,
  abl_02_next_component_choice, e2e_03_build_decision_memorandum.
  E8 taxonomy labels cluster on evaluation + implementation.
- Variance: all 8 repeated cases success_consistent=True; max score
  spread 1.74%; no flakiness detected.

## Validation FINAL
- pytest: **1425 passed** / 0 failed (8m52s) incl. 21 llm_benchmark tests
  (+3 new regressions: anti-case-only tier, judge cause surfacing,
  per-criterion config).
- ruff: output byte-identical to HEAD baseline (no new findings).
- mypy service/: only pre-existing config.py errors (file untouched).

## Preferred next step after A/B/C fixes
Delete artifacts/p2_final_run and re-run full via existing script:
/tmp/p2_full.py calls run_p2('artifacts/p2_final_run', p1_regression=True,
repeats=2, label='p2-final-glm52'). ~15 min wall. Then verify report has
both tiers + non-zero judge scores; demo --compare regression machinery;
run pytest/ruff/mypy deltas; write final 13-section deliverable report to
user (verdict likely READY WITH RISKS: judge fragility + single model arm
+ structural v1 cases documented).

## Misc gotchas
- YAML flow mapping cannot contain block scalars (`rubric: >-`).
- v2 graders use adapter-reported notes_written (op gte) + markers; not
  v1 sentence-chunk mechanics.
- _StoredExecutionView needs payload keys termination/reason/output/context.
- Runtime stagnation guard needs strictly rising evaluator scores.
- Keep own files ruff/mypy clean; ignore pre-existing repo-wide noise.

