# P3 — Agent Capability Optimization Experiment

**Objective:** determine how much autonomous research performance can be
improved over the frozen P2 baseline without materially degrading cost,
latency, safety, or reliability. Uses ONLY existing infrastructure
(E4/E5/E6/E8/P2) — no new evaluation framework.

## Frozen baseline

Authoritative inputs recorded in `artifacts/p3/baseline/baseline_manifest.json`
(by `scripts/p3/freeze_baseline.py`, which **fails non-zero** if the live
environment no longer reproduces the P2 final-run fingerprints):

| Item | Value |
|---|---|
| Provider / model | `ollama` / `glm-5.3-flash` |
| Judge | `EvaluationAgent → glm-5.3-flash` (constant across ALL arms) |
| `llm_config.yaml` sha256 | `a411775c…15425` |
| v2 suite content sha256 | `f652c7e2…b2736` (20 cases × 8 categories, revision r1) |

Reproduction proof: `artifacts/p3/baseline_repro/` re-executes the identical
configuration through the production path.

## Arms (one variable each)

| Arm | Variable | Mechanism |
|---|---|---|
| `baseline_repro` | none | byte-identical suite snapshot |
| `exp_a1_kimi_k27_code` | model | per-case `llm_model` override |
| `exp_a2_minimax_m3_cloud` | model | per-case `llm_model` override |
| `exp_a3_gpt_oss_120b` | model | per-case `llm_model` override |
| `exp_b_plan_first_strategy` | planning strategy | `llm_strategy=plan_first` prompt appendix |
| `exp_c_reasoning_budget` | token budget | per-call cap 2048→4096 + cumulative cap 48000 |
| `exp_d_stagnation_window6` | stagnation policy | `stagnation_window` 3→6 |
| `exp_e_impl_strategy` | impl-task strategy | `llm_strategy=impl_focus` on implementation cases only |

Guarantees enforced by code (`service/p3_experiment.py`):

* override keys are restricted to an explicit allowlist;
* every selector-matched case must actually change (typo fail-closed);
* derived suites re-validate against the full benchmark contract;
* snapshots are deterministic (byte-identical on re-derivation).

## Execution

```bash
uv run python scripts/p3/freeze_baseline.py     # Phase 1 - freeze + verify
bash scripts/p3/run_all_p3.sh                   # Phase 2 - all arms, sequential
uv run python scripts/p3/aggregate_p3.py        # deltas + CIs + failure analysis
uv run python scripts/p3/e8_gate.py --candidate <arm> \
    --component model_provider_config --changes '{"model_provider.model": "..."}'
uv run python scripts/p3/crash_recovery_validation.py
```

Each arm = full v2 suite + variance repeats (`repeats=3`) over a subset that
includes every known P2 failure case plus the judge-event case (n≥2 attempts
per case, per arm). Arms run through `run_p2` (API → store → worker →
AgentRuntime → ToolGateway → SafetyController → checkpointing → E4 grading →
E6 telemetry); judge binding stays frozen so model/policy effects are never
confounded with evaluator changes.

## Statistical discipline

* Success/completion rates get Wilson 95% CIs (`n` measured attempts).
* Judge quality means use t-based 95% CIs and EXCLUDE unparseable-judge
  events (counted separately — evaluator failures are never optimized
  against and cannot inflate quality).
* No improvement claim from a tiny sample: rate deltas whose 95% CIs overlap
  are reported as inconclusive, not as wins.
* Cost/token/latency medians compared against configurable regression-gate
  fractions in the E8 gate.

## Decision criteria

A configuration is recommended only if it improves autonomous task success
or judged quality meaningfully while safety/human interventions stay at
zero-or-equal, recovery reliability holds, and cost/token/latency growth
stays within gate limits. Nothing is promoted automatically; the E8 pipeline
stops at the gate verdict and records `promotion_executed=false`.
