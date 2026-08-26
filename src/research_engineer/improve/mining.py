"""E8 - Failure & pattern mining over E4 results and E6 telemetry.

Consumes the artifacts that already exist — E4 :class:`EvalReport` /
:class:`EvalResult` plus, when available, the run's
:class:`~research_engineer.runtime.models.AgentContext` (its ``metadata``
carries ``tool_call_log`` and the E5 ``safety_state`` decision history) —
and extracts recurring :class:`FailurePattern` instances.

Evidence rule: a pattern is only emitted when **distinct runs** affected
meet ``MiningConfig.min_affected_runs``. A single anomalous run is never
sufficient evidence for an improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_engineer.eval.models import EvalReport
from research_engineer.improve.models import (
    FailurePattern,
    PatternKind,
    content_hash,
)
from research_engineer.runtime.models import AgentContext

#: Tool-call statuses that count as failed invocations in tool_call_log.
_FAILED_TOOL_STATUSES = frozenset({"denied", "approval_denied", "error", "failed"})
#: Safety triggers recorded by the E5 controller that imply an intervention.
_SAFETY_TRIGGERS = frozenset(
    {
        "policy_violation",
        "risk_escalation",
        "approval_required",
        "approval_denied",
    }
)


@dataclass(frozen=True)
class MiningConfig:
    """Thresholds governing what counts as recurring evidence."""

    #: Distinct runs a problem must affect before it becomes a pattern.
    min_affected_runs: int = 2
    #: weighted_score below this marks a "poor evaluation" case.
    poor_score_threshold: float = 0.5
    #: Per-run cost above this (USD) flags excessive cost; <= 0 disables.
    excessive_cost_usd: float = 0.10
    #: Per-run wall-clock above this (s) flags excessive latency; <= 0 disables.
    excessive_latency_seconds: float = 0.0
    #: Fraction of runs with human interventions flagging approval churn.
    human_approval_rate: float = 0.5
    #: Minimum failed/denied events within one run to call its tool usage failing.
    tool_failure_events_per_run: int = 2
    #: Replan decisions within one run indicating replanning churn.
    replans_per_run: int = 2

    def validate(self) -> None:
        if self.min_affected_runs < 1:
            raise ValueError("min_affected_runs must be >= 1")
        if self.tool_failure_events_per_run < 1:
            raise ValueError("tool_failure_events_per_run must be >= 1")


@dataclass
class RunFeatures:
    """Per-run features extracted once from a result (+ optional context)."""

    run_id: str
    case_id: str
    success: bool
    score: float
    termination_reason: str
    latency_seconds: float
    cost_usd: float
    tokens: int
    recoverable_errors: int = 0
    fatal_errors: int = 0
    human_interventions: int = 0
    #: per-tool failure event counts within this run
    tool_failures: dict[str, int] = field(default_factory=dict)
    #: redundant same-tool/same-outcome repetitions inside one run
    duplicate_tool_events: int = 0
    replan_decisions: int = 0
    safety_trigger_events: dict[str, int] = field(default_factory=dict)

    @property
    def tool_failure_total(self) -> int:
        return sum(self.tool_failures.values())

    @property
    def safety_event_total(self) -> int:
        return sum(self.safety_trigger_events.values())


def _tool_log(ctx: AgentContext) -> list[dict[str, Any]]:
    """Return the run's tool_call_log entries (empty when absent)."""
    log = ctx.metadata.get("tool_call_log")
    if not isinstance(log, list):
        return []
    return [entry for entry in log if isinstance(entry, dict)]


def _features_from_context(features: RunFeatures, ctx: AgentContext) -> None:
    """Enrich features from the run's AgentContext (best-effort)."""
    counts: dict[tuple[str, str], int] = {}
    for entry in _tool_log(ctx):
        tool = str(entry.get("tool", "?"))
        status = str(entry.get("status", "")).lower()
        if status in _FAILED_TOOL_STATUSES:
            features.tool_failures[tool] = (
                features.tool_failures.get(tool, 0) + 1
            )
        key = (tool, status)
        counts[key] = counts.get(key, 0) + 1
    features.duplicate_tool_events = sum(c - 1 for c in counts.values() if c > 1)

    state = ctx.metadata.get("safety_state")
    if isinstance(state, dict):
        records = state.get("decisions") or []
        for record in records:
            if not isinstance(record, dict):
                continue
            action = str(record.get("action", ""))
            trigger = str(record.get("trigger", ""))
            if action == "replan":
                features.replan_decisions += 1
            if trigger in _SAFETY_TRIGGERS:
                features.safety_trigger_events[trigger] = (
                    features.safety_trigger_events.get(trigger, 0) + 1
                )
    if not features.human_interventions:
        features.human_interventions = int(
            ctx.metadata.get("human_interventions", 0) or 0
        )


def extract_features(
    report: EvalReport,
    contexts: dict[str, AgentContext] | None = None,
) -> list[RunFeatures]:
    """Build one :class:`RunFeatures` per result; contexts keyed by run_id."""
    contexts = contexts or {}
    features: list[RunFeatures] = []
    for r in report.results:
        f = RunFeatures(
            run_id=r.run_id,
            case_id=r.case_id,
            success=r.success,
            score=r.weighted_score,
            termination_reason=r.metrics.termination_reason or "",
            latency_seconds=r.metrics.latency_seconds,
            cost_usd=r.metrics.cost_usd,
            tokens=r.metrics.tokens,
            recoverable_errors=r.metrics.recoverable_errors,
            fatal_errors=r.metrics.fatal_errors,
            human_interventions=r.metrics.human_interventions,
        )
        ctx = contexts.get(r.run_id)
        if ctx is not None:
            _features_from_context(f, ctx)
        elif r.metrics.recoverable_errors + r.metrics.fatal_errors >= 2:
            # Report-only fallback: aggregate runtime errors as tool failures.
            failed = r.metrics.recoverable_errors + r.metrics.fatal_errors
            f.tool_failures["(runtime_error)"] = failed
            f.duplicate_tool_events = max(0, failed - 1)
        features.append(f)
    return features


def _pattern_dict(
    kind: PatternKind,
    signature: str,
    description: str,
    runs: list[RunFeatures],
    extra_events: int | None = None,
) -> dict[str, Any]:
    ids = sorted({r.run_id for r in runs})
    total = extra_events if extra_events is not None else len(runs)
    return {
        "pattern_id": f"pat_{content_hash([kind.value, signature])[:12]}",
        "kind": kind,
        "signature": signature,
        "description": description,
        "affected_runs": len(ids),
        "total_events": total,
        "run_ids": ids,
        "example_case_ids": sorted({r.case_id for r in runs})[:5],
    }


def _require(config: MiningConfig, runs: list[RunFeatures]) -> bool:
    """Evidence threshold: >= ``min_affected_runs`` distinct affected runs."""
    return len(runs) >= config.min_affected_runs


def _mine_tool_patterns(
    features: list[RunFeatures], config: MiningConfig,
) -> list[dict[str, Any]]:
    """Tool failures, duplicated calls, replan churn, safety interventions."""
    out: list[dict[str, Any]] = []

    fail_runs = [
        f for f in features
        if f.tool_failure_total >= config.tool_failure_events_per_run
    ]
    if _require(config, fail_runs):
        events = sum(f.tool_failure_total for f in fail_runs)
        tools = sorted({t for f in fail_runs for t in f.tool_failures})
        out.append(_pattern_dict(
            PatternKind.REPEATED_TOOL_FAILURES, ",".join(tools),
            f"{len(fail_runs)} runs hit >= {config.tool_failure_events_per_run} "
            f"failed/denied tool calls ({events} events; tools: {tools})",
            fail_runs, extra_events=events,
        ))

    dup_runs = [f for f in features if f.duplicate_tool_events > 0]
    if _require(config, dup_runs):
        events = sum(f.duplicate_tool_events for f in dup_runs)
        out.append(_pattern_dict(
            PatternKind.UNNECESSARY_TOOL_CALLS, "",
            f"{len(dup_runs)} runs repeated identical tool calls "
            f"({events} redundant events)",
            dup_runs, extra_events=events,
        ))

    replan_runs = [
        f for f in features if f.replan_decisions >= config.replans_per_run
    ]
    if _require(config, replan_runs):
        out.append(_pattern_dict(
            PatternKind.REPEATED_REPLANNING, "",
            f"{len(replan_runs)} runs issued >= {config.replans_per_run} "
            "REPLAN decisions (E5 loop/failure escalation)",
            replan_runs,
            extra_events=sum(f.replan_decisions for f in replan_runs),
        ))

    safety_runs = [f for f in features if f.safety_event_total > 0]
    if _require(config, safety_runs):
        triggers = sorted({
            t for f in safety_runs for t in f.safety_trigger_events
        })
        out.append(_pattern_dict(
            PatternKind.SAFETY_INTERVENTIONS, ",".join(triggers),
            f"{len(safety_runs)} runs triggered safety interventions "
            f"(triggers: {triggers})",
            safety_runs,
            extra_events=sum(f.safety_event_total for f in safety_runs),
        ))
    return out


def _mine_outcome_patterns(
    features: list[RunFeatures], config: MiningConfig,
) -> list[dict[str, Any]]:
    """Termination-driven patterns: budget, no-progress, poor scores."""
    out: list[dict[str, Any]] = []

    budget_runs = [
        f for f in features if f.termination_reason == "budget_exceeded"
    ]
    if _require(config, budget_runs):
        out.append(_pattern_dict(
            PatternKind.BUDGET_EXHAUSTION, "",
            f"{len(budget_runs)} runs terminated on BUDGET_EXCEEDED",
            budget_runs,
        ))

    noprog_runs = [f for f in features if f.termination_reason == "no_progress"]
    if _require(config, noprog_runs):
        out.append(_pattern_dict(
            PatternKind.NO_PROGRESS_TERMINATION, "",
            f"{len(noprog_runs)} runs terminated on NO_PROGRESS",
            noprog_runs,
        ))

    poor_runs = [f for f in features if f.score < config.poor_score_threshold]
    if _require(config, poor_runs):
        out.append(_pattern_dict(
            PatternKind.POOR_EVALUATION_SCORE,
            f"score<{config.poor_score_threshold:g}",
            f"{len(poor_runs)} runs scored below {config.poor_score_threshold:g}",
            poor_runs,
        ))
    return out


def _mine_efficiency_patterns(
    features: list[RunFeatures], config: MiningConfig,
) -> list[dict[str, Any]]:
    """Cost / latency outliers plus human-approval frequency."""
    out: list[dict[str, Any]] = []

    if config.excessive_cost_usd > 0:
        cost_runs = [
            f for f in features if f.cost_usd > config.excessive_cost_usd
        ]
        if _require(config, cost_runs):
            out.append(_pattern_dict(
                PatternKind.EXCESSIVE_COST,
                f"cost>${config.excessive_cost_usd:g}",
                f"{len(cost_runs)} runs exceeded ${config.excessive_cost_usd:g}",
                cost_runs,
            ))

    if config.excessive_latency_seconds > 0:
        late_runs = [
            f for f in features
            if f.latency_seconds > config.excessive_latency_seconds
        ]
        if _require(config, late_runs):
            out.append(_pattern_dict(
                PatternKind.EXCESSIVE_LATENCY,
                f">={config.excessive_latency_seconds:g}s",
                f"{len(late_runs)} runs exceeded "
                f"{config.excessive_latency_seconds:g}s wall clock",
                late_runs,
            ))

    humans = [f for f in features if f.human_interventions > 0]
    if (
        features
        and len(humans) / len(features) >= config.human_approval_rate
        and _require(config, humans)
    ):
        out.append(_pattern_dict(
            PatternKind.HUMAN_APPROVAL_FREQUENCY,
            f"rate>={config.human_approval_rate:.0%}",
            f"{len(humans)}/{len(features)} runs required human intervention",
            humans,
        ))
    return out

def mine_patterns(
    features: list[RunFeatures], config: MiningConfig,
) -> dict[str, dict[str, Any]]:
    """Apply each detector; returns raw pattern dicts keyed by pattern id.

    Every detector requires >= ``config.min_affected_runs`` distinct runs;
    detectors producing empty sets are simply absent from the output.
    """
    config.validate()
    found: dict[str, dict[str, Any]] = {}
    groups = (
        _mine_tool_patterns(features, config)
        + _mine_outcome_patterns(features, config)
        + _mine_efficiency_patterns(features, config)
    )
    for pattern in groups:
        found[pattern["pattern_id"]] = pattern
    return found

    return found


def mine_report(
    report: EvalReport,
    *,
    contexts: dict[str, AgentContext] | None = None,
    config: MiningConfig | None = None,
) -> list[FailurePattern]:
    """Mine :class:`FailurePattern` models from an E4 report.

    Args:
        report: baseline/current evaluation report (E4).
        contexts: optional :class:`AgentContext` per ``run_id`` enriching the
            mining with tool-call logs and E5 safety decision history.
        config: thresholds (defaults flag problems seen in >= 2 runs).

    Returns:
        Failure patterns ordered by kind then signature.
    """
    cfg = config or MiningConfig()
    raw = mine_patterns(extract_features(report, contexts), cfg)
    patterns = [FailurePattern.model_validate(v) for v in raw.values()]
    return sorted(patterns, key=lambda p: (p.kind.value, p.signature))


__all__ = ["FailurePattern", "MiningConfig", "extract_features", "mine_patterns", "mine_report"]
