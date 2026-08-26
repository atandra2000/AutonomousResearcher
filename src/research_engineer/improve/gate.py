"""E8 - Regression gate.

Extracts comparable :class:`~research_engineer.improve.models.MetricSnapshot`
values from E4 :class:`~research_engineer.eval.models.EvalReport` aggregates
and applies a configurable :class:`~research_engineer.improve.models.RegressionGate`.

A candidate passes only when *all* configured constraints hold. Regressions
on hard safety metrics are unconditional rejects.
"""

from __future__ import annotations

from research_engineer.eval.models import EvalReport
from research_engineer.improve.models import (
    HARD_SAFETY_METRICS,
    GateVerdict,
    MetricSnapshot,
    RegressionGate,
)

#: Guard against float noise when comparing "no regression" constraints.
_EPSILON = 1e-9


def snapshot_from_report(report: EvalReport) -> MetricSnapshot:
    """Build a :class:`MetricSnapshot` from an E4 report aggregate."""
    agg = report.aggregate
    quality = (
        sum(r.weighted_score for r in report.results) / len(report.results)
        if report.results
        else 0.0
    )
    safety = float(agg.termination_reasons.get("safety_terminated", 0))
    return MetricSnapshot(
        cases_total=agg.cases_total,
        success_rate=agg.success_rate,
        completion_rate=agg.completion_rate,
        quality_score=quality,
        failure_rate=agg.failure_rate,
        safety_interventions=safety,
        human_interventions=agg.human_intervention_rate,
        total_cost_usd=agg.total_cost_usd,
        total_tokens=float(agg.total_tokens),
        avg_latency_seconds=agg.avg_latency_seconds,
        p95_latency_seconds=agg.p95_latency_seconds,
        termination_reasons={
            k: float(v) for k, v in sorted(agg.termination_reasons.items())
        },
    )


def _check_hard_safety(
    b: MetricSnapshot, c: MetricSnapshot, deltas: dict[str, float],
    gate: RegressionGate,
) -> list[str]:
    """Hard safety metrics reject on ANY increase (or absolute-cap breach)."""
    out: list[str] = []
    for name in sorted(HARD_SAFETY_METRICS):
        if deltas.get(name, 0.0) > _EPSILON:
            out.append(
                f"HARD SAFETY: {name} regressed "
                f"({getattr(b, name):.4g} -> {getattr(c, name):.4g})"
            )
        # The absolute cap applies regardless of the baseline level.
        cap = gate.max_safety_interventions_abs
        if cap is not None and getattr(c, name) > cap + _EPSILON:
            out.append(
                f"{name}={getattr(c, name):.4g} exceeds absolute cap {cap:.4g}"
            )
    return out


def _check_floors(
    b: MetricSnapshot, c: MetricSnapshot, deltas: dict[str, float],
    gate: RegressionGate,
) -> list[str]:
    """Success / quality / failure-rate / human-intervention floors."""
    out: list[str] = []
    if gate.require_success_gte_baseline and deltas["success_rate"] < -_EPSILON:
        out.append(
            f"success_rate {b.success_rate:.3f} -> {c.success_rate:.3f} "
            "(must be >= baseline)"
        )
    if deltas["quality_score"] < -gate.quality_tolerance - _EPSILON:
        out.append(
            f"quality_score {b.quality_score:.3f} -> {c.quality_score:.3f} "
            f"(tolerance {gate.quality_tolerance:.3g})"
        )
    if deltas["failure_rate"] > gate.max_failure_rate_increase + _EPSILON:
        out.append(
            f"failure_rate increased by {deltas['failure_rate']:.4g} "
            f"(max allowed {gate.max_failure_rate_increase:.4g})"
        )
    if (
        deltas["human_interventions"]
        > gate.max_human_intervention_increase + _EPSILON
    ):
        out.append(
            f"human interventions {b.human_interventions:.3f} -> "
            f"{c.human_interventions:.3f}"
        )
    return out


def _check_resources(
    b: MetricSnapshot, c: MetricSnapshot, gate: RegressionGate,
) -> tuple[list[str], list[str]]:
    """Fractional cost/token/latency budgets; returns (violations, warnings).

    A zero baseline means the metric was unobservable in the baseline suite
    (not that the resource is free), so a relative fraction is meaningless —
    that case records a warning instead of rejecting every candidate.
    """
    violations: list[str] = []
    warnings: list[str] = []

    def rel_check(name: str, base: float, cand: float, frac: float,
                  fmt: str) -> None:
        if base <= 0.0:
            warnings.append(f"{name}: zero baseline; budget check skipped")
            return
        slack = max(_EPSILON * 100, abs(base) * frac)
        if cand - base > slack:
            violations.append(fmt)

    rel_check(
        "cost", b.total_cost_usd, c.total_cost_usd,
        gate.max_cost_increase_fraction,
        f"cost ${b.total_cost_usd:.4f} -> ${c.total_cost_usd:.4f} "
        f"exceeds +{gate.max_cost_increase_fraction:.0%} budget",
    )
    rel_check(
        "tokens", b.total_tokens, c.total_tokens,
        gate.max_token_increase_fraction,
        f"tokens {b.total_tokens:.0f} -> {c.total_tokens:.0f} "
        f"exceeds +{gate.max_token_increase_fraction:.0%} budget",
    )
    rel_check(
        "avg latency", b.avg_latency_seconds, c.avg_latency_seconds,
        gate.max_latency_increase_fraction,
        f"avg latency {b.avg_latency_seconds:.4g}s -> "
        f"{c.avg_latency_seconds:.4g}s exceeds "
        f"+{gate.max_latency_increase_fraction:.0%} budget",
    )
    return violations, warnings


def _termination_warnings(b: MetricSnapshot, c: MetricSnapshot) -> list[str]:
    """Informational termination-reason shifts between the two snapshots."""
    out: list[str] = []
    all_reasons = set(b.termination_reasons) | set(c.termination_reasons)
    for reason in sorted(all_reasons):
        delta = c.termination_reasons.get(reason, 0.0) - (
            b.termination_reasons.get(reason, 0.0)
        )
        if abs(delta) > _EPSILON:
            out.append(f"termination '{reason}' delta {delta:+.0f}")
    return out


def evaluate_gate(
    baseline: MetricSnapshot,
    candidate: MetricSnapshot,
    gate: RegressionGate | None = None,
) -> GateVerdict:
    """Apply every gate constraint; returns all violations at once."""
    gate = gate or RegressionGate()
    deltas = baseline.deltas(candidate)

    violations = [
        *_check_hard_safety(baseline, candidate, deltas, gate),
        *_check_floors(baseline, candidate, deltas, gate),
    ]
    res_violations, warnings = _check_resources(baseline, candidate, gate)
    violations.extend(res_violations)

    # Extra no-regression metrics (tightening only; hard safety already done).
    for name in gate.no_regression_metrics:
        if name in HARD_SAFETY_METRICS:
            continue
        if deltas.get(name, 0.0) < -_EPSILON:
            violations.append(f"{name} regressed ({name} delta {deltas[name]:+.4g})")

    warnings.extend(_termination_warnings(baseline, candidate))
    return GateVerdict(passed=not violations, violations=violations,
                       warnings=warnings)


__all__ = ["evaluate_gate", "snapshot_from_report"]
