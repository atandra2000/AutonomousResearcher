# P4 — Production Readiness Closure Report

> **Historical record (2026-08):** This closure report predates the CLI-dedicated
> rework: the Docker deployment verification and the serving tier it exercised were
> removed in `c31c536`, and `scripts/p4_rescore.py` / `scripts/pilot_benchmark.py`
> are retained for provenance only. The grader-hardening and E8 findings remain
> valid.

**Date:** 2026-08-27 · **Scope:** E4 grader hardening, DeepSeek candidate
re-evaluation, production ImprovementStore, safety-chain regression test,
CI stabilization, Docker deployment verification.

---

## 1. Grader issues found / fixed

Audit of the 20-case P2 benchmark surfaced three brittle criteria plus a
judge-failure accounting defect:

| Case | Verdict | Fix |
|------|---------|-----|
| `design_02_data_scaling_study` | **Grader failure** (exact string `"50000"` rejected the format-valid `"$50,000"`/`"50,000"` renderings the agent actually produced) | Replaced by the format-tolerant `output_number` grader (new, registered in `DETERMINISTIC_GRADERS`); suite case revision `r1 → r2` |
| `e2e_03_build_decision_memorandum` | **Grader failure** (same class: exact string `"8000"` vs `"$8,000/month"`) | Same `output_number` replacement; revision `r1 → r2` |
| `impl_01_attention_module_plan` | **Genuine agent failure** (outputs lack explicit batch/sequence dims under any reasonable reading; re-grading with a tolerant regex still fails) | Exact substring `"(batch, seq"` replaced by case/spacing-tolerant `output_regex` `\(\s*b(?:atch)?\s*,\s*s(?:eq)?\b` — intent preserved, agent still fails on merit; revision `r1 → r2` |
| LLM-judge malformed replies | **Benchmark-specification problem** (judge malfunction was scored as a genuine 0) | New `EvalResult.error_kind` field; judge path emits `error_kind="judge_error"` on unparseable/failed judge calls (score still fails closed at 0.0, never fabricated); benchmark runner **excludes** judge-error criteria from the weighted score and renormalizes weights |

New deterministic graders in `src/research_engineer/eval/graders.py`:
`output_number` (format-tolerant numeric occurrence) and `output_regex`
(tolerant pattern match), both registered in `DETERMINISTIC_GRADERS`.

## 2. Corrected P2/P3 results (re-scored persisted artifacts — no LLM re-run)

`scripts/p4_rescore.py` re-graded all persisted `v2_*.json` reports from
their stored worker payloads (17 reports; originals untouched; corrected
copies + `corrected_summary.json` under `artifacts/p4_rescore/`).

**P2 final run (GLM-5.3-Flash frozen baseline):** 18/20 → **20/20** after
correcting the two grader failures.

**P3-Short (2 repeats × 20 cases, corrected graders):**

| Arm | Corrected success |
|-----|-------------------|
| glm (frozen baseline) | 18/20 = **0.900** (rep2) |
| kimi | 17/20 = **0.850** (rep2) |
| **deepseek (ds)** | **38/40 = 0.950** (rep1+rep2) |

Remaining failures under corrected graders are genuine: `impl_01` fails on
all arms except DeepSeek; glm also misses `debug_03`; kimi misses
`debug_02`/`expa_01`.

## 3. DeepSeek E8 status

The corrected-grader E8 gate recomputation
(`artifacts/p4_rescore/e8_gate/deepseek_candidate_gate.json`) built both
arms purely from corrected measured outcomes:

- baseline `success_rate` 0.848 / quality 0.910
- candidate `success_rate` **0.950** / quality **0.965**
- verdict **PASS** (no violations; hard safety metrics unchanged at 0)

Candidate **`ollama/deepseek-v4-pro:0813` still dominates** the frozen
GLM-5.3-Flash baseline. It was registered via the standard E8 pipeline and
left in **APPROVED-CANDIDATE** state (`promotion_executed: false`).
Production configuration is untouched; promotion awaits explicit human
approval.

## 4. ImprovementStore production status

The JSON store is unsafe for the deployed multi-process service (concurrent
operators can lose read-modify-write updates on active-version pointers;
the decisions log can interleave). Added
`src/research_engineer/improve/pg_store.py`:

- `PostgresImprovementStore` — same interface as the JSON store, backed by
  the existing E7 PostgreSQL infrastructure (lazy `psycopg` import),
  tables auto-created;
- concurrency protection: `pg_advisory_xact_lock` serializes the
  active-pointer read-modify-write across processes; decisions are an
  append-only BIGSERIAL log; baseline/candidate rows are idempotent
  upserts;
- `build_improvement_store(config)` factory: PostgreSQL when
  `postgres_dsn` is configured (same signal as the run store), JSON store
  otherwise — **development/tests keep the local store**.

Validated against a live PostgreSQL 16 container: full store contract +
4-thread concurrent-pointer test pass
(`tests/test_improvement_pg_store.py`; PG variants gate on
`RE_TEST_PG_DSN`).

## 5. Production safety verification

The existing enforcement was verified (worker `require_safety_chain=True`,
set by compose `RE_SERVICE_ENFORCE_SAFETY=1`) and pinned by a new
integration regression test `tests/test_service_safety_chain.py`:

1. The production assembly (`build_default_safety_chain` + the exact
   `serve._worker_main` wiring) routes runs through ToolGateway + E5
   SafetyController (proven via persisted gateway `tool_call_log` and
   `safety_state`).
2. Fail-closed: an enforcing worker missing **either** component refuses
   to execute agent code — the run fails with an explicit "fail-closed"
   error naming the missing pieces, no artifacts produced.
3. Dev flexibility preserved: `enforce_safety_chain` defaults to false.

## 6. CI / test stabilization

- **Network isolation:** `tests/conftest.py` adds a `network` marker +
  session connectivity probe; live-network tests are skipped when arXiv is
  unreachable (mechanism in place and documented in
  `docs/contributing.md`).
- **mypy baseline:** `configs/mypy-baseline.txt` records all 596 legacy
  error signatures (file+message, line numbers normalized);
  `scripts/ci_mypy.sh` fails **only on NEW errors** (verified: exit 0;
  the legacy 857-error debt is not touched).
- **No hangs:** `pytest-timeout` (300 s per test, thread method) added to
  `pyproject.toml`; full suite is deterministic.
- Suite count grew 1427 → **1462 passing** (new P4 tests), 2 skipped
  (PG-gated), **0 failed** in 12m14s.

## 7. Docker / pilot result

Docker was available and healthy (engine 29.7.2):

- `docker compose build` — success (api + worker images).
- `docker compose up` — postgres (healthy), otel-collector, api (healthy),
  worker all up; `/ready` → `{"ready": true, store/queue/artifacts true}`.
- `scripts/pilot_benchmark.py --mode docker` — **verdict PASS**:
  API → PostgreSQL → Worker → Runtime → Gateway → Safety → Checkpoint →
  Evaluation → Telemetry all exercised
  (`artifacts/p4_pilot/pilot_report.json`).
- **Real crash → recovery → resume:** the mid-flight worker container was
  SIGKILLed (exit 137) and respawned; the fresh worker waited out the
  stale lease, took the run over from the checkpoint store
  (`claim_count = 2`, new `worker_id`) and drove it to completion.

Two Docker-environment facts surfaced and are handled/documented in
`scripts/pilot_benchmark.py::_crash_worker`: (a) in-container
`os.kill(1, SIGKILL)` cannot kill PID 1 (PID-namespace init is immune to
signals from inside its own namespace), and (b) this engine treats
`docker kill` as a manual stop, so `restart: unless-stopped` does not
re-launch; the pilot performs the supervisor restart step explicitly —
identical semantics to local mode's kill-then-spawn.

## 8. Files changed

```
M  docs/contributing.md                    CI practices documentation
M  evals/research_benchmark/v2/suite.yaml  r2 revisions (design_02/impl_01/e2e_03)
M  pyproject.toml                          pytest-timeout + network marker
M  scripts/p3/aggregate_p3.py              judge-error-aware quality accounting
M  scripts/pilot_benchmark.py              working docker crash+respawn mechanism
M  src/research_engineer/eval/graders.py   output_number/output_regex + JUDGE_ERROR
M  src/research_engineer/eval/models.py    EvalResult.error_kind
M  src/research_engineer/improve/__init__.py  export PG store + factory
M  src/research_engineer/service/benchmark_runner.py  error_kind propagation + judge-error exclusion
M  src/research_engineer/service/llm_judge.py         explicit JUDGE_ERROR marking
M  uv.lock                                 pytest-timeout
A  configs/mypy-baseline.txt               596-signature legacy mypy baseline
A  scripts/ci_mypy.sh                      new-errors-only mypy gate
A  scripts/p4_rescore.py                   persisted-artifact rescorer + corrected E8 gate
A  src/research_engineer/improve/pg_store.py  PostgreSQL ImprovementStore + factory
A  tests/conftest.py                       network marker + offline skip
A  tests/test_improvement_pg_store.py      store contract + concurrency tests
A  tests/test_service_safety_chain.py      production safety-chain regression
A  docs/p4_closure_report.md               this report
A  artifacts/p4_rescore/                   corrected reports + summary + E8 gate
A  artifacts/p4_pilot/pilot_report.json    PASS pilot report
```

## 9. Validation results

| Check | Result |
|---|---|
| `uv run python -m pytest` (full) | **1462 passed, 2 skipped, 0 failed** (12m14s) |
| `uv run ruff check .` (touched files) | All checks passed |
| `bash scripts/ci_mypy.sh` | no new errors vs baseline (exit 0) |
| Focused: E4/E8/service/safety/PG-store/P3 | 101 + 22 + 19 + 4 passed, 2 skipped |
| PG store vs live PostgreSQL 16 | contract + concurrency: 3 passed |
| Docker compose build/up/ready | success |
| Pilot `--mode docker` | **PASS** (crash→recovery→resume, claim_count 2) |

## 10. Remaining risks

1. **DeepSeek promotion is pending human decision** (by design): the
   candidate sits in APPROVED-CANDIDATE; production still runs
   `glm-5.3-flash`.
2. **Legacy mypy debt (857 errors)** remains; the baseline gate prevents
   growth but does not reduce it.
3. **LLM-judge dependency:** judge errors are now visible and excluded
   from scores, but a high judge-error rate would shrink the effective
   evaluation surface; monitor `judge_error` counts in future runs.
4. **Docker restart policy semantics** vary by engine: on this engine
   `docker kill` does not trigger `restart: unless-stopped`. Real
   crashes (OOM, panic) do trigger it; verify on the target engine.
5. **P3-short sample size** (2 repeats × 20 cases) supports the gate
   decision but is not a statistically deep comparison.
6. `impl_01` remains a genuinely hard case (3/4 arms fail it) — model
   selection should not assume uniform per-case competence.

## 11. FINAL VERDICT: **READY**

All P2/P3/E9-identified production-critical gaps are closed: graders are
semantic and judge-failure-safe, the DeepSeek candidate is re-validated and
correctly gated, the production ImprovementStore is PostgreSQL-backed with
concurrency protection, the safety-chain invariant is regression-tested at
the integration level, CI is stabilized (network isolation, mypy baseline,
test timeouts), and the full Docker deployment path — including a real
worker crash with checkpoint takeover and resume — is verified end to end.

