"""P2 - End-to-end LLM-backed benchmark orchestration and reporting.

Composes the P1 runner with the LLM tier:

* Full-suite v2 execution through the production stack (``llm_react`` kind),
  plus a variance subset run ``--repeats`` times.
* Optional P1 deterministic regression re-run for same-report comparison.
* Mode-split metrics: deterministic results never mix with LLM-backed
  results in aggregates.
* Reproducibility fingerprints (suite/case/config hashes, provider/model
  binding) recorded in the report; credentials are never included - only
  SHA-256 digests of configuration inputs.
* Category -> failure-taxonomy aggregation over E8 mined outcomes
  (read-only; mining never mutates agent behavior or promotions).
* Regression comparison against a previous report JSON.

The module is importable so tests can drive pieces without the CLI.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.eval.models import EvalSuite
from research_engineer.service.agents import AgentFactoryRegistry
from research_engineer.service.benchmark import (
    DEFAULT_SUITE_PATH,
    DEFAULT_SUITE_V2_PATH,
    load_benchmark_suite,
)
from research_engineer.service.benchmark_runner import (
    BenchmarkReport,
    BenchmarkRunner,
    CaseOutcome,
)
from research_engineer.service.llm_agent import register_llm_agent_kinds
from research_engineer.service.llm_judge import build_llm_quality_grader

#: Cases re-executed per repeat pass to measure score variance. Chosen to
#: span every category plus the numeric-derivation cases most likely to
#: expose reasoning instability.
DEFAULT_VARIANCE_CASES: tuple[str, ...] = (
    "litdisc_02_moe_routing_compare",
    "hypo_01_dropout_curves",
    "design_01_lora_rank_ablation",
    "impl_01_attention_module_plan",
    "debug_01_nan_loss_timeline",
    "abl_01_seed_mean_attribution",
    "expa_02_significance_reasoning",
    "e2e_01_brief_to_conclusion",
)


# ---------------------------------------------------------------------------
# Fingerprinting (reproducibility metadata)
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of UTF-8 ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def suite_fingerprint(suite: EvalSuite) -> dict[str, Any]:
    """Content hash of a suite revision set (order-independent)."""
    case_items = sorted(
        (
            c.case_id,
            c.revision,
            c.goal,
            json.dumps(
                [crit.model_dump() for crit in c.criteria], sort_keys=True
            ),
        )
        for c in suite.cases
    )
    payload = json.dumps(case_items, sort_keys=True)
    return {
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "case_count": len(suite.cases),
        "case_revisions": {c.case_id: c.revision for c in suite.cases},
        "content_sha256": sha256_text(payload),
    }


def config_fingerprint() -> dict[str, str]:
    """Fingerprint of the LLM configuration driving the benchmark."""
    try:
        from research_engineer.llm.factory import get_factory
        from research_engineer.llm.router import get_router

        router = get_router(get_factory())
        return {
            "provider": router.provider_name_for("BenchmarkLLM") or "",
            "model": router.model_for("BenchmarkLLM") or "",
            "judge_provider": router.provider_name_for("EvaluationAgent")
            or "",
            "judge_model": router.model_for("EvaluationAgent") or "",
            "llm_config_sha256": sha256_text(
                json.dumps(
                    get_factory().config, sort_keys=True, default=str
                )
            ),
        }
    except Exception as exc:  # noqa: BLE001 - fingerprint must not crash runs
        return {"error": f"fingerprint unavailable: {type(exc).__name__}"}


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class TierMetrics(BaseModel):
    """Metrics restricted to one mode family (deterministic | llm_agent)."""

    mode: str
    cases_total: int
    autonomous_completion_rate: float
    task_success_rate: float
    mean_weighted_score: float
    human_intervention_rate: float
    safety_intervention_rate: float
    gateway_denial_rate: float
    median_latency_seconds: float
    median_cost_per_task_usd: float
    median_tokens_per_task: float
    termination_distribution: dict[str, int] = Field(default_factory=dict)


class VarianceStat(BaseModel):
    case_id: str
    attempts: int
    successes: list[int]
    scores: list[float]
    success_consistent: bool
    score_stddev: float
    score_spread: float


class FailureInsight(BaseModel):
    category: str
    label_counts: dict[str, int]
    recurring_patterns: dict[str, Any] = Field(default_factory=dict)


class RegressionComparison(BaseModel):
    compared_against: str
    baseline_autonomous_rate: float | None = None
    candidate_autonomous_rate: float | None = None
    delta_autonomous_rate: float | None = None
    baseline_task_success_rate: float | None = None
    candidate_task_success_rate: float | None = None
    note: str = ""


class P2Report(BaseModel):
    """Composite P2 deliverable: versioned, reproducible, split by tier."""

    report_id: str = Field(
        default_factory=lambda: (
            f"p2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    generated_at: datetime = Field(default_factory=datetime.now)
    label: str = ""
    suites: list[dict[str, Any]] = Field(default_factory=list)
    configuration: dict[str, Any] = Field(default_factory=dict)
    tier_metrics: list[TierMetrics] = Field(default_factory=list)
    headline: dict[str, float] = Field(default_factory=dict)
    llm_quality_by_case: dict[str, float] = Field(default_factory=dict)
    variance_stats: list[VarianceStat] = Field(default_factory=list)
    failure_insights: list[FailureInsight] = Field(default_factory=list)
    regression_comparison: RegressionComparison | None = None
    raw_reports: list[str] = Field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Production-readiness verdict derived from LLM-tier evidence."""
        llm_tiers = [t for t in self.tier_metrics if t.mode == "llm_agent"]
        if not llm_tiers:
            return "NOT READY (no LLM-tier evidence)"
        rate = mean(t.autonomous_completion_rate for t in llm_tiers)
        unstable = [v for v in self.variance_stats
                    if not v.success_consistent]
        if rate >= 0.8 and not unstable:
            return "READY"
        if rate < 0.5 or len(unstable) > max(
            2, len(self.variance_stats) // 2
        ):
            return "NOT READY"
        return "READY WITH RISKS"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _termination_counts(outcomes: list[CaseOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        key = o.termination or "none"
        counts[key] = counts.get(key, 0) + 1
    return counts


def tier_metrics_for(
    outcomes: list[CaseOutcome], mode: str,
) -> TierMetrics | None:
    """Aggregate one mode family into :class:`TierMetrics`.

    Guardrail anti-cases are excluded from rate denominators exactly like
    the P1 aggregate does; rates live on the *effective* set so an
    anti-case that "passes" cannot inflate completion metrics.
    """
    group = [o for o in outcomes if o.mode == mode]
    if not group:
        return None
    effective = [o for o in group if not o.expected_failed_by_design]
    denom = len(effective) or 1
    completed_untouched = [
        o for o in effective
        if o.status == "completed" and o.graded_success
        and o.human_interventions == 0
        and "budget" not in o.termination.lower()
        and o.termination != "timeout"
    ]
    return TierMetrics(
        mode=mode,
        cases_total=len(group),
        autonomous_completion_rate=len(completed_untouched) / denom,
        task_success_rate=(
            sum(1 for o in effective if o.graded_success) / denom
        ),
        # A mode family made entirely of guardrail anti-cases has no
        # effective attempts; report a neutral score instead of raising.
        mean_weighted_score=(
            mean(o.weighted_score for o in effective) if effective else 0.0
        ),
        human_intervention_rate=(
            sum(1 for o in group if o.human_interventions > 0)
            / len(group)
        ),
        safety_intervention_rate=(
            sum(1 for o in group if o.safety_interventions > 0)
            / len(group)
        ),
        gateway_denial_rate=(
            sum(1 for o in group if o.gateway_denials > 0) / len(group)
        ),
        median_latency_seconds=_median([o.latency_seconds for o in group]),
        median_cost_per_task_usd=_median([o.cost_usd for o in group]),
        median_tokens_per_task=_median([float(o.tokens) for o in group]),
        termination_distribution=_termination_counts(group),
    )


def build_p2_factories(
    config_max_steps: int = 12,
    *, provider: Any = None,
) -> AgentFactoryRegistry:
    """Registry carrying the four deterministic kinds plus ``llm_react``."""
    registry = AgentFactoryRegistry(config_max_steps=config_max_steps)
    from research_engineer.service.bench_agents import (
        register_benchmark_kinds,
    )

    register_benchmark_kinds(registry)
    register_llm_agent_kinds(registry, provider=provider)
    return registry


def _extract_llm_quality(report: BenchmarkReport) -> dict[str, float]:
    """Mean judged score per case from optional ``llm_quality`` criteria."""
    per_case: dict[str, list[float]] = {}
    for o in report.outcomes:
        for c in o.criteria:
            if c.grader == "llm_quality":
                per_case.setdefault(o.case_id, []).append(c.score)
    return {k: round(mean(v), 4) for k, v in per_case.items()}


def _variance_from_reports(
    reports: list[BenchmarkReport], wanted: set[str],
) -> list[VarianceStat]:
    by_case: dict[str, list[CaseOutcome]] = {}
    seen_attempts: set[tuple[str, int]] = set()
    for rep in reports:
        for o in rep.outcomes:
            if o.case_id not in wanted or o.mode != "llm_agent":
                continue
            key = (o.case_id, o.repeat)
            if key in seen_attempts:
                continue
            seen_attempts.add(key)
            by_case.setdefault(o.case_id, []).append(o)
    stats: list[VarianceStat] = []
    for case_id in sorted(by_case):
        runs = sorted(by_case[case_id], key=lambda o: o.repeat)
        scores = [o.weighted_score for o in runs]
        stats.append(VarianceStat(
            case_id=case_id,
            attempts=len(runs),
            successes=[int(o.graded_success) for o in runs],
            scores=[round(s, 4) for s in scores],
            success_consistent=(
                len({int(o.graded_success) for o in runs}) <= 1
            ),
            score_stddev=(
                round(pstdev(scores), 6) if len(scores) > 1 else 0.0
            ),
            score_spread=(
                round(max(scores) - min(scores), 4) if scores else 0.0
            ),
        ))
    return stats


def compare_reports(
    previous_json: str | Path, candidate: P2Report,
) -> RegressionComparison:
    """Compare a new P2 report against a stored previous one."""
    data = json.loads(Path(previous_json).read_text(encoding="utf-8"))
    prev_llm = [
        t for t in data.get("tier_metrics", [])
        if t.get("mode") == "llm_agent"
    ]
    cand_llm = [
        t.model_dump() for t in candidate.tier_metrics
        if t.mode == "llm_agent"
    ]

    def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [float(r.get(key, 0.0)) for r in rows]
        return mean(vals) if vals else None

    prev_rate = _mean(prev_llm, "autonomous_completion_rate")
    cand_rate = _mean(cand_llm, "autonomous_completion_rate")
    comparison = RegressionComparison(
        compared_against=str(data.get("report_id", "?")),
        baseline_autonomous_rate=prev_rate,
        candidate_autonomous_rate=cand_rate,
        delta_autonomous_rate=(
            (cand_rate - prev_rate)
            if prev_rate is not None and cand_rate is not None else None
        ),
        baseline_task_success_rate=_mean(prev_llm, "task_success_rate"),
        candidate_task_success_rate=_mean(cand_llm, "task_success_rate"),
    )
    prev_cfg = data.get("configuration") or {}
    same_cfg = bool(prev_cfg) and all(
        prev_cfg.get(k) == v
        for k, v in candidate.configuration.items()
        if k in prev_cfg and k.endswith(("sha256", "model", "provider"))
    )
    comparison.note = (
        "identical configuration fingerprints"
        if same_cfg else
        "CONFIG MISMATCH: baselines differ in provider/model/config "
        "hashes; treat deltas as directional only"
    )
    return comparison


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(report: P2Report) -> str:  # noqa: C901 - report sections
    """Human-readable composite report."""
    lines: list[str] = [
        f"# P2 benchmark report - {report.report_id}",
        "",
        f"- Generated: {report.generated_at.isoformat(timespec='seconds')}",
        f"- Label: {report.label or '(none)'}",
        "",
        "## Headline (LLM tier)",
        "",
    ]
    llm = [t for t in report.tier_metrics if t.mode == "llm_agent"]
    det = [t for t in report.tier_metrics if t.mode != "llm_agent"]
    if llm:
        lines += [
            f"- **Autonomous completion rate**: "
            f"{mean(t.autonomous_completion_rate for t in llm):.1%}",
            f"- Objective task success rate: "
            f"{mean(t.task_success_rate for t in llm):.1%}",
            f"- Mean weighted score: "
            f"{mean(t.mean_weighted_score for t in llm):.3f}",
            f"- Safety-intervention rate: "
            f"{mean(t.safety_intervention_rate for t in llm):.1%}",
            f"- Median tokens/task: "
            f"{_median([t.median_tokens_per_task for t in llm]):.0f}",
            f"- Median cost/task: "
            f"${_median([t.median_cost_per_task_usd for t in llm]):.6f}",
            f"- Median latency/task: "
            f"{_median([t.median_latency_seconds for t in llm]):.3f}s",
        ]
    else:
        lines.append("- No LLM-tier evidence collected.")
    lines += ["", "## Verdict", "", f"- **{report.verdict}**", ""]
    lines += ["## Tier metrics", ""]
    for t in report.tier_metrics:
        lines += [
            f"### Tier `{t.mode}` ({t.cases_total} executions)",
            "",
            f"- Autonomous completion: "
            f"{t.autonomous_completion_rate:.1%}",
            f"- Task success: {t.task_success_rate:.1%}",
            f"- Mean weighted score: {t.mean_weighted_score:.3f}",
            f"- Human interventions: {t.human_intervention_rate:.1%}",
            f"- Safety interventions: {t.safety_intervention_rate:.1%}",
            f"- Gateway denials: {t.gateway_denial_rate:.1%}",
            f"- Median latency: {t.median_latency_seconds:.3f}s",
            f"- Median cost/task: ${t.median_cost_per_task_usd:.6f}",
            f"- Median tokens/task: {t.median_tokens_per_task:.0f}",
        ]
        terms = ", ".join(
            f"`{k}`={v}" for k, v in sorted(
                t.termination_distribution.items())
        )
        lines += [f"- Terminations: {terms or '-'}", ""]
    lines += ["## Deterministic vs LLM tiers (kept separate)", ""]
    for t in det:
        lines.append(
            f"- `{t.mode}`: completion "
            f"{t.autonomous_completion_rate:.1%}, success "
            f"{t.task_success_rate:.1%}"
        )
    lines += ["", "## Research quality (LLM-judged, reported separately)",
              ""]
    if report.llm_quality_by_case:
        for cid, score in sorted(report.llm_quality_by_case.items()):
            lines.append(f"- {cid}: {score:.3f}")
    else:
        lines.append("- No judge scores recorded.")
    lines += ["", "## Variance (repeat analysis)", ""]
    if report.variance_stats:
        for v in report.variance_stats:
            flag = "" if v.success_consistent else " **UNSTABLE**"
            lines.append(
                f"- {v.case_id}: attempts={v.attempts} "
                f"successes={v.successes} stddev={v.score_stddev:.4f} "
                f"spread={v.score_spread:.4f}{flag}"
            )
    else:
        lines.append("- No repeated cases.")
    lines += ["", "## Failure analysis (taxonomy x category)", ""]
    if report.failure_insights:
        for fi in report.failure_insights:
            if fi.label_counts:
                labels = ", ".join(
                    f"`{k}`={v}" for k, v in sorted(
                        fi.label_counts.items())
                )
                lines.append(f"- {fi.category}: {labels}")
            else:
                lines.append(f"- {fi.category}: no failures")
    else:
        lines.append("- No failures recorded.")
    if report.regression_comparison is not None:
        rc = report.regression_comparison
        lines += [
            "", "## Regression comparison", "",
            f"- Against: {rc.compared_against}",
            f"- Baseline autonomous rate: "
            f"{rc.baseline_autonomous_rate!r}",
            f"- Candidate autonomous rate: "
            f"{rc.candidate_autonomous_rate!r}",
            f"- Delta: {rc.delta_autonomous_rate!r}",
            f"- Note: {rc.note}",
        ]
    lines += ["", "## Configuration & reproducibility", ""]
    for key, value in sorted(report.configuration.items()):
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Raw component reports", ""]
    for path in report.raw_reports:
        lines.append(f"- {path}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------


def _judge_available() -> bool:
    """True when an LLM provider resolves for the judge; fail-closed."""
    try:
        from research_engineer.llm.factory import get_factory
        from research_engineer.llm.router import get_router

        provider = get_router(get_factory()).for_agent("EvaluationAgent")
        return provider is not None
    except Exception:  # noqa: BLE001 - availability probing never raises
        return False


def _pattern_touches(pattern: Any, case_ids: set[str]) -> bool:
    """True when an E8 pattern mentions any of ``case_ids``."""
    affected = pattern.get("affected_runs") if isinstance(
        pattern, dict) else None
    if not isinstance(affected, list):
        return False
    run_prefixes = {str(a).split(":")[0] for a in affected}
    return bool(run_prefixes & case_ids)


def build_p2_report(
    *,
    base: str | Path,
    suite_file: str | Path,
    all_outcomes: list[CaseOutcome],
    variance_reports: list[BenchmarkReport],
    raw_paths: list[str],
    label: str,
    variance_cases: tuple[str, ...] = DEFAULT_VARIANCE_CASES,
    repeats: int = 1,
    p1_regression: bool = True,
    judge_enabled: bool = False,
    previous_report: str | Path | None = None,
    elapsed_seconds: float | None = None,
) -> P2Report:
    """Aggregate executed stage reports into a persisted :class:`P2Report`.

    Separated from :func:`run_p2` so a report can be rebuilt offline from
    already-persisted stage artifacts (crash recovery, re-aggregation
    after grading fixes) without re-executing agents or spending budget.
    ``elapsed_seconds=None`` marks the offline path in wall-clock metrics.
    """
    started = time.monotonic()
    base_path = Path(base)
    suite = load_benchmark_suite(suite_file)

    # Aggregate per tier (mode family) — deterministic and LLM-backed
    # results are never mixed.
    modes = sorted({o.mode for o in all_outcomes} - {""})
    tiers = [t for m in modes if (t := tier_metrics_for(all_outcomes, m))]
    llm_quality: dict[str, float] = {}
    for variance_report in variance_reports:
        llm_quality.update(_extract_llm_quality(variance_report))

    # Failure analysis grouped by research category.
    failed = [o for o in all_outcomes
              if not o.graded_success and not o.expected_failed_by_design]
    insights: list[FailureInsight] = []
    src_rep = variance_reports[-1]
    for cat in sorted({o.category for o in failed}):
        labels: dict[str, int] = {}
        cat_cases: set[str] = set()
        for o in failed:
            if o.category != cat:
                continue
            cat_cases.add(o.case_id)
            for lbl in o.failure_categories:
                labels[lbl] = labels.get(lbl, 0) + 1
        patterns = {
            pid: pat
            for pid, pat in (src_rep.patterns or {}).items()
            if _pattern_touches(pat, cat_cases)
        }
        insights.append(FailureInsight(
            category=cat, label_counts=labels, recurring_patterns=patterns,
        ))

    llm_outcomes = [o for o in all_outcomes if o.mode == "llm_agent"]
    headline = {
        "autonomous_completion_rate": (
            mean(t.autonomous_completion_rate for t in tiers
                 if t.mode == "llm_agent") if llm_outcomes else 0.0
        ),
        "objective_task_success_rate": (
            mean(t.task_success_rate for t in tiers
                 if t.mode == "llm_agent") if llm_outcomes else 0.0
        ),
        "research_quality_score_mean": (
            mean(list(llm_quality.values())) if llm_quality else 0.0
        ),
        "llm_tokens_total": float(sum(o.tokens for o in llm_outcomes)),
        "llm_cost_total_usd": float(
            round(sum(o.cost_usd for o in llm_outcomes), 6)
        ),
        "wall_clock_minutes": (
            round(elapsed_seconds / 60.0, 2)
            if elapsed_seconds is not None
            else round((time.monotonic() - started) / 60.0, 2)
        ),
    }

    report = P2Report(
        label=label,
        suites=[suite_fingerprint(s) for s in (suite,)],
        configuration={
            **config_fingerprint(),
            "suite_path": str(suite_file),
            "variance_cases": ",".join(variance_cases),
            "repeats": str(repeats),
            "p1_regression_included": str(p1_regression),
            "judge_enabled": str(judge_enabled),
        },
        tier_metrics=tiers,
        headline=headline,
        llm_quality_by_case=llm_quality,
        variance_stats=_variance_from_reports(
            variance_reports, set(variance_cases)
        ),
        failure_insights=insights,
        raw_reports=[p for p in raw_paths if p],
    )
    if previous_report:
        report.regression_comparison = compare_reports(previous_report, report)

    json_path = base_path / "p2_report.json"
    md_path = base_path / "p2_report.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    report.raw_reports = [*report.raw_reports, str(json_path), str(md_path)]
    return report


async def run_p2(  # noqa: C901 - staged benchmark flow
    output_dir: str | Path,
    *,
    suite_path: str | Path | None = None,
    p1_regression: bool = True,
    repeats: int = 2,
    variance_cases: tuple[str, ...] = DEFAULT_VARIANCE_CASES,
    label: str = "p2-default",
    previous_report: str | Path | None = None,
    factories: AgentFactoryRegistry | None = None,
    extra_graders: dict[str, Any] | None = None,
    case_filter: set[str] | None = None,
) -> P2Report:
    """Execute the full P2 flow and persist JSON + markdown reports.

    Steps: full v2 suite -> variance-subset repeats (attempts 2..N on top
    of each case's full-suite first attempt) -> optional P1 deterministic
    regression tier -> aggregation, fingerprints, failure insights and
    regression comparison. When no judge/provider is available the flow
    still runs and the report simply records absent LLM evidence rather
    than fabricating any.
    """
    started = time.monotonic()
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    registry = factories or build_p2_factories()
    graders = extra_graders if extra_graders is not None else {}
    if extra_graders is None and _judge_available():
        graders = {"llm_quality": build_llm_quality_grader()}

    runner = BenchmarkRunner(base, extra_graders=graders, factories=registry)

    suite_file = Path(suite_path) if suite_path else DEFAULT_SUITE_V2_PATH
    suite = load_benchmark_suite(suite_file)
    all_outcomes: list[CaseOutcome] = []
    raw_paths: list[str] = []

    # 1. Full suite (attempt 1 of every case); ``case_filter`` narrows the
    # executed subset (smoke runs) while the whole suite stays validated.
    full = await runner.run_suite(
        suite_file, repeat=1, report_name="v2_full_suite",
        case_ids=case_filter,
    )
    all_outcomes.extend(full.outcomes)
    raw_paths.extend([full.json_path or "", full.markdown_path or ""])

    # 2. Variance repeats (attempt 2..N for selected cases). Each pass is a
    # single extra attempt (repeat=1) whose outcome rows are relabeled to
    # the attempt number so cross-attempt dedupe/aggregation is correct.
    variance_reports: list[BenchmarkReport] = [full]
    if repeats > 1:
        wanted = [c for c in variance_cases if c in suite.case_ids]
        for rep in range(2, repeats + 1):
            sub = await runner.run_suite(
                suite_file, repeat=1, report_name=f"v2_variance_r{rep}",
                case_ids=set(wanted),
            )
            sub = sub.model_copy(update={
                "outcomes": [o.model_copy(update={"repeat": rep})
                             for o in sub.outcomes],
            })
            all_outcomes.extend(sub.outcomes)
            variance_reports.append(sub)
            raw_paths.extend(
                [sub.json_path or "", sub.markdown_path or ""]
            )

    # 3. Deterministic P1 regression tier (separate runner directory so
    # the tiers never share artifacts or workspace sandboxes).
    if p1_regression:
        p1_runner = BenchmarkRunner(base / "deterministic_regression")
        p1_report = await p1_runner.run_suite(
            str(DEFAULT_SUITE_PATH), repeat=1,
            report_name="p1_deterministic_regression",
        )
        all_outcomes.extend(p1_report.outcomes)
        raw_paths.extend(
            [p1_report.json_path or "", p1_report.markdown_path or ""]
        )

    return build_p2_report(
        base=base,
        suite_file=suite_file,
        all_outcomes=all_outcomes,
        variance_reports=variance_reports,
        raw_paths=raw_paths,
        label=label,
        variance_cases=variance_cases,
        repeats=repeats,
        p1_regression=p1_regression,
        judge_enabled=bool(graders),
        previous_report=previous_report,
        elapsed_seconds=time.monotonic() - started,
    )


__all__ = [
    "DEFAULT_VARIANCE_CASES",
    "FailureInsight",
    "P2Report",
    "RegressionComparison",
    "TierMetrics",
    "VarianceStat",
    "build_p2_factories",
    "compare_reports",
    "config_fingerprint",
    "render_markdown",
    "run_p2",
    "sha256_text",
    "suite_fingerprint",
    "tier_metrics_for",
]






