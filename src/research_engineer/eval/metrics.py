"""E4 - Metric aggregation and regression comparison.

Pure functions over lists of :class:`~research_engineer.eval.models.EvalResult`:
aggregate :class:`~research_engineer.eval.models.SuiteMetrics`, build an
class:`~research_engineer.eval.models.EvalReport`, and compare two reports
(regression support) so a new agent/runtime version can be evaluated against
a baseline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_engineer.eval.models import EvalReport, EvalResult, SuiteMetrics


def aggregate(results: list[EvalResult]) -> SuiteMetrics:
    """Compute suite-level metrics from per-case results."""
    total = len(results)
    metrics = SuiteMetrics(cases_total=total)
    if total == 0:
        return metrics

    succeeded = [r for r in results if r.success]
    completed = [r for r in results if r.completion]
    errored = [
        r for r in results
        if r.metrics.recoverable_errors > 0 or r.metrics.fatal_errors > 0
    ]
    recovered = [r for r in errored if r.success]

    latencies = sorted(r.metrics.latency_seconds for r in results)

    metrics.success_rate = len(succeeded) / total
    metrics.completion_rate = len(completed) / total
    metrics.avg_steps = sum(r.metrics.steps for r in results) / total
    metrics.avg_tool_calls = sum(r.metrics.tool_calls for r in results) / total
    metrics.total_tokens = sum(r.metrics.tokens for r in results)
    metrics.total_cost_usd = sum(r.metrics.cost_usd for r in results)
    metrics.avg_latency_seconds = sum(latencies) / total
    p95_index = min(total - 1, int(round(0.95 * (total - 1))))
    metrics.p95_latency_seconds = latencies[p95_index] if latencies else 0.0
    metrics.failure_rate = 1.0 - metrics.success_rate
    metrics.recovery_rate = (
        len(recovered) / len(errored) if errored else 1.0
    )
    metrics.human_intervention_rate = (
        sum(1 for r in results if r.metrics.human_interventions > 0) / total
    )
    for result in results:
        reason = result.metrics.termination_reason or "none"
        metrics.termination_reasons[reason] = (
            metrics.termination_reasons.get(reason, 0) + 1
        )
    return metrics


def build_report(
    results: list[EvalResult],
    *,
    suite_id: str,
    suite_version: str = "1",
    label: str = "",
) -> EvalReport:
    """Build a full :class:`EvalReport` from results."""
    report = EvalReport(
        suite_id=suite_id,
        suite_version=suite_version,
        label=label,
        results=results,
        aggregate=aggregate(results),
    )
    return report


class CaseDiff(BaseModel):
    """Outcome change of one case between baseline and candidate."""

    case_id: str
    baseline_success: bool
    candidate_success: bool
    baseline_score: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = Field(
        default="unchanged",
        description="improved | regressed | unchanged | missing",
    )


class RegressionComparison(BaseModel):
    """Structured comparison between a baseline and a candidate report."""

    baseline_label: str = Field(default="")
    candidate_label: str = Field(default="")
    metric_deltas: dict[str, float] = Field(
        default_factory=dict, description="candidate - baseline per metric"
    )
    case_diffs: list[CaseDiff] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    is_regression_free: bool = Field(default=True)


def compare_reports(baseline: EvalReport, candidate: EvalReport) -> RegressionComparison:
    """Compare a candidate report against a baseline report.

    ``success_rate`` and case-level flips are the primary regression
    signals; cost/latency/token increases above the baseline are also
    surfaced as regressions.
    """
    deltas: dict[str, float] = {}
    base_agg = baseline.aggregate.model_dump()
    cand_agg = candidate.aggregate.model_dump()
    for key, base_value in base_agg.items():
        if isinstance(base_value, (int, float)):
            deltas[key] = float(cand_agg[key]) - float(base_value)

    case_diffs, regressions, improvements = _diff_cases(baseline, candidate)
    resource_regs, extra_improvements = _diff_resources(deltas)
    regressions.extend(resource_regs)
    improvements.extend(extra_improvements)

    return RegressionComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        metric_deltas=deltas,
        case_diffs=case_diffs,
        regressions=regressions,
        improvements=improvements,
        is_regression_free=not regressions,
    )


def _diff_cases(
    baseline: EvalReport, candidate: EvalReport,
) -> tuple[list[CaseDiff], list[str], list[str]]:
    """Per-case comparison; returns diffs, regressions, improvements."""
    diffs: list[CaseDiff] = []
    regressions: list[str] = []
    improvements: list[str] = []
    base_by_case = {r.case_id: r for r in baseline.results}
    cand_by_case = {r.case_id: r for r in candidate.results}
    for case_id in sorted(set(base_by_case) | set(cand_by_case)):
        base_res = base_by_case.get(case_id)
        cand_res = cand_by_case.get(case_id)
        if base_res is None or cand_res is None:
            diffs.append(CaseDiff(
                case_id=case_id,
                baseline_success=bool(base_res.success) if base_res else False,
                candidate_success=bool(cand_res.success) if cand_res else False,
                baseline_score=base_res.weighted_score if base_res else 0.0,
                candidate_score=cand_res.weighted_score if cand_res else 0.0,
                status="missing",
            ))
            regressions.append(f"case {case_id}: missing from one report")
            continue
        flipped_down = base_res.success and not cand_res.success
        flipped_up = not base_res.success and cand_res.success
        status = (
            "regressed" if flipped_down
            else "improved" if flipped_up
            else "unchanged"
        )
        diffs.append(CaseDiff(
            case_id=case_id,
            baseline_success=base_res.success,
            candidate_success=cand_res.success,
            baseline_score=base_res.weighted_score,
            candidate_score=cand_res.weighted_score,
            status=status,
        ))
        delta = cand_res.weighted_score - base_res.weighted_score
        if flipped_down:
            regressions.append(f"case {case_id}: success -> failure")
        elif flipped_up:
            improvements.append(f"case {case_id}: failure -> success")
        elif delta < -1e-9:
            regressions.append(
                f"case {case_id}: score {base_res.weighted_score:.3f} "
                f"-> {cand_res.weighted_score:.3f}"
            )
    return diffs, regressions, improvements


def _diff_resources(
    deltas: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Resource-level regression/improvement signals from metric deltas."""
    regressions: list[str] = []
    improvements: list[str] = []
    if deltas.get("total_tokens", 0.0) > 0:
        regressions.append("total_tokens increased")
    if deltas.get("avg_latency_seconds", 0.0) > 0:
        regressions.append("avg_latency_seconds increased")
    if deltas.get("human_intervention_rate", 0.0) > 0:
        regressions.append("human_intervention_rate increased")
    if deltas.get("success_rate", 0.0) < 0:
        regressions.append("success_rate decreased")
    elif deltas.get("success_rate", 0.0) > 0:
        improvements.append("success_rate increased")
    return regressions, improvements


def save_report(report: EvalReport, path: object) -> None:
    """Serialize a report to JSON at ``path``."""
    from pathlib import Path

    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def load_report(path: object) -> EvalReport:
    """Load a previously saved :class:`EvalReport` from JSON."""
    from pathlib import Path

    data = Path(str(path)).read_text(encoding="utf-8")
    report: EvalReport = EvalReport.model_validate_json(data)
    return report


__all__ = [
    "aggregate",
    "build_report",
    "compare_reports",
    "save_report",
    "load_report",
    "CaseDiff",
    "RegressionComparison",
]
