#!/usr/bin/env python
"""P3 - E8 integration: baseline -> candidate -> gate. NO promotion.

Runs the existing E8 ImprovementPipeline over the P3 measured evidence:
the frozen baseline arm is registered as the production baseline (an E4
EvalReport built honestly from the measured benchmark outcomes), the chosen
candidate configuration becomes an allowlisted proposal, the candidate's
measured outcomes are supplied as its offline evaluation, and the standard
RegressionGate decides PASS/FAIL.

Promotion is deliberately NOT executed: approval/promote steps are left to
a human operator. This script reports the gate decision only.

    uv run python scripts/p3/e8_gate.py --candidate exp_a1_kimi_k27_code \
        --component model_provider_config --changes '{"model_provider.model": "kimi-k2.7-code"}' \
        --changes-alias model_provider.model=llm_model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

from research_engineer.eval.models import (
    CaseMetrics,
    EvalReport,
    EvalResult,
    EvalStatus,
    SuiteMetrics,
)
from research_engineer.improve import (
    CandidateComponent,
    ImprovementPipeline,
    ImprovementStore,
)
from research_engineer.improve.models import ImprovementProposal

ROOT = Path("artifacts/p3")


def build_eval_report(arm: str, label: str) -> EvalReport:
    """Rebuild the arm's measured llm_agent outcomes as an E4 report.

    Every field below is computed from persisted BenchmarkRunner outcomes;
    nothing is synthesized. The report_id embeds the arm id so provenance
    stays auditable in the ImprovementStore artifacts.
    """
    d = ROOT / arm
    results: list[EvalResult] = []
    tokens_total = 0
    latencies: list[float] = []
    terminations: dict[str, int] = {}
    for path in sorted(d.glob("v2_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for o in data.get("outcomes", []):
            if o.get("mode") != "llm_agent":
                continue
            reason_key = str(o.get("termination_reason") or "unknown")
            short = reason_key.split(" ")[0][:40]
            terminations[short] = terminations.get(short, 0) + 1
            tokens_total += int(o.get("tokens") or 0)
            latency = float(o.get("latency_seconds") or 0.0)
            latencies.append(latency)
            metrics = CaseMetrics(
                steps=int(o.get("steps") or 0),
                tool_calls=int(o.get("tool_calls") or 0),
                tokens=int(o.get("tokens") or 0),
                cost_usd=float(o.get("cost_usd") or 0.0),
                latency_seconds=latency,
                recoverable_errors=int(o.get("recoverable_errors") or 0),
                fatal_errors=int(o.get("fatal_errors") or 0),
                recovered=bool(
                    o.get("recoverable_errors") and o.get("graded_success")
                ),
                terminated=True,
                termination_reason=short,
            )
            results.append(EvalResult(
                run_id=f"{arm}:{path.stem}:{o.get('case_id')}:"
                       f"{o.get('repeat')}:{o.get('submitted_at', '')}",
                case_id=str(o.get("case_id")),
                status=(EvalStatus.COMPLETED if o.get("graded_success")
                        else EvalStatus.FAILED),
                success=bool(o.get("graded_success")),
                completion=bool(o.get("runtime_success")),
                weighted_score=min(1.0, max(0.0, float(
                    o.get("weighted_score") or 0.0))),
                metrics=metrics,
            ))
    n = len(results)
    if n == 0:
        raise SystemExit(f"no llm_agent outcomes under {d}")
    successes = sum(1 for r in results if r.success)
    completions = sum(1 for r in results if r.completion)
    sorted_lat = sorted(latencies)
    p95 = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))]
    aggregate = SuiteMetrics(
        cases_total=n,
        success_rate=successes / n,
        completion_rate=completions / n,
        avg_steps=statistics.mean([r.metrics.steps for r in results]),
        avg_tool_calls=statistics.mean(
            [r.metrics.tool_calls for r in results]),
        total_tokens=tokens_total,
        total_cost_usd=round(
            sum(r.metrics.cost_usd for r in results), 6),
        avg_latency_seconds=statistics.mean(latencies),
        p95_latency_seconds=p95,
        failure_rate=1 - successes / n,
        recovery_rate=(
            sum(1 for r in results if r.metrics.recovered)
            / max(1, sum(1 for r in results
                         if r.metrics.recoverable_errors > 0))
        ) if any(r.metrics.recoverable_errors > 0 for r in results) else 0.0,
        human_intervention_rate=sum(
            r.metrics.human_interventions for r in results) / n,
        termination_reasons=terminations,
    )
    return EvalReport(
        report_id=f"p3e8_{arm}_{label}",
        suite_id="research_benchmark_v2_llm",
        suite_version="2",
        label=label,
        results=results,
        aggregate=aggregate,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True,
                        help="candidate arm id (measured outcomes)")
    parser.add_argument(
        "--component", default="model_provider_config",
        choices=[c.value for c in CandidateComponent],
    )
    parser.add_argument(
        "--changes", required=True,
        help='JSON config patch inside the component allowlist, '
             'e.g. \'{"model_provider.model": "kimi-k2.7-code"}\'',
    )
    parser.add_argument("--store", default=str(ROOT / "e8_gate"))
    args = parser.parse_args()

    baseline_report = build_eval_report("baseline_repro", "frozen-baseline")
    candidate_report = build_eval_report(args.candidate, args.candidate)
    changes = json.loads(args.changes)

    pipe = ImprovementPipeline(store=ImprovementStore(root=args.store))
    snapshot = {
        "arm": "baseline_repro",
        "provider": "ollama",
        "model": "glm-5.3-flash",
        "judge": "glm-5.3-flash",
        "suite_sha256": None,
    }
    manifest = ROOT / "baseline" / "baseline_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        snapshot.update({
            "llm_config_sha256": data.get("llm_config_sha256"),
            "suite_content_sha256":
                data.get("suite_fingerprint", {}).get("content_sha256"),
        })
    baseline = pipe.register_baseline(
        "p3-frozen-p2-baseline", baseline_report, snapshot,
        suite_version="2",
    )
    proposal = ImprovementProposal(
        title=f"P3 candidate: {args.candidate}",
        description=(
            f"Promote-eligible configuration from measured P3 arm "
            f"{args.candidate}; single variable vs frozen baseline."
        ),
        component=CandidateComponent(args.component),
        changes=changes,
        evidence=[
            f"p3/arm={args.candidate}",
            "p3/baseline=baseline_repro",
        ],
        source="p3_experiment",
    )
    candidate = pipe.create_candidate(proposal, baseline.baseline_id)

    async def run_eval(_cand: Any) -> EvalReport:
        # The candidate's offline evaluation IS the measured P3 arm run
        # (same suite fingerprint, graders, judge); it is supplied here as
        # the E4 evaluation input exactly as produced by the benchmark.
        return candidate_report

    evaluated = await pipe.evaluate_candidate(candidate.candidate_id,
                                              run_eval=run_eval)
    ev = evaluated.evaluation
    assert ev is not None
    result = {
        "candidate_id": candidate.candidate_id,
        "status": evaluated.status.value,
        "gate_passed": ev.verdict.passed,
        "violations": ev.verdict.violations,
        "warnings": ev.verdict.warnings,
        "recommended_action": (
            ev.verdict.recommended().value if ev.verdict.passed else "none"
        ),
        "deltas": {k: round(v, 6) for k, v in ev.deltas.items()},
        "promotion_executed": False,
        "note": (
            "Regression gate decision only; promotion requires explicit "
            "human approval through the E8 pipeline (pipe.approve/promote)."
        ),
        "baseline_metrics": ev.baseline.model_dump(),
        "candidate_metrics": ev.candidate.model_dump(),
        "provenance": {
            "baseline_report_id": baseline_report.report_id,
            "candidate_report_id": candidate_report.report_id,
            "built_from_measured_outcomes": True,
        },
    }
    out_path = ROOT / "e8_gate" / f"{args.candidate}_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"candidate={candidate.candidate_id}")
    print(f"status={evaluated.status.value} gate_passed={ev.verdict.passed}")
    for v in ev.verdict.violations or ["(no violations)"]:
        print(f"  violation: {v}")
    print(f"wrote {out_path}")
    print("promotion NOT executed (explicit human approval required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
