# P3 Final Report — Agent Capability Optimization Experiment

> Status: IN PROGRESS — this report is finalized when all arms complete.
> Everything below is generated from measured runs; nothing fabricated.

## 1. Frozen baseline

See `artifacts/p3/baseline/baseline_manifest.json` and §Baseline in
`configs/p3/README.md`. Reproduction verified pre-experiment
(`freeze_baseline.py` compares live fingerprints against the P2 final-run
report and refuses to proceed otherwise); a fresh reproduction run lands in
`artifacts/p3/baseline_repro/`.

## 2. Experiments performed

Eight single-variable arms declared under `configs/p3/arms/*.yaml`
(see README table). Each executed through the unchanged P2 production path.

## 3. Model/provider comparison

Filled by `artifacts/p3/aggregate_comparison.md`.

## 4–6. Success / quality / cost / statistics

Filled by `aggregate_comparison.{json,md}` (Wilson CIs on rates, t-CIs on
judge quality excluding unparseable events, medians for cost/tokens/latency,
per-arm deltas vs the measured baseline reproduction).

## 7. Failure analysis

`artifacts/p3/failure_analysis.md` — trace-grounded cause classification
over every known P2 failure case, re-run n≥2 times in every arm.

## 8–9. Best configuration + E8 gate

`artifacts/p3/e8_gate/<arm>_gate.json` — gate verdict only;
**no automatic promotion**.

## 10. Files changed

* Added: `src/research_engineer/service/p3_experiment.py`,
  `configs/p3/{README.md,arms/*.yaml}`, `scripts/p3/*`,
  `tests/test_p3_experiment.py`, `docs/p3_report.md`.
* Modified: `service/agents.py` (+stagnation-window override plumbing),
  `service/llm_agent.py` (+`llm_strategy` prompt hook).
* Untouched: `evals/research_benchmark/v2/suite.yaml`, `llm_config.yaml`,
  `artifacts/p2_final_run/**`.

## 11. Tests/validation

`uv run python -m pytest && uv run ruff check . && uv run mypy .`
plus crash/recovery validation (`artifacts/p3/crash_recovery/report.json`).

## 12. Recommendation

TBD after evidence.
