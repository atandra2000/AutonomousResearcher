"""P1 §5 - End-to-end benchmark runner over the production runtime path.

Every case executes through the real agent runtime stack (``AgentRuntime
-> ToolGateway -> SafetyController -> checkpointing``) exactly as an
autonomous CLI run does, and is graded afterwards from the *persisted
payload* built from the finished run context — the same bytes an operator
would inspect on disk. This is deliberately NOT the scripted E4 factory
path - production agents are what the benchmark measures.

Outputs a versioned report (JSON + markdown): headline
**autonomous task completion rate under fixed budgets**, the full P1 §6
metric set, deterministic failure-taxonomy labels, recurring patterns mined
with E8 (read-only; never mutates agent behavior or promotions).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.benchmark.agents import (
    DEFAULT_AGENT_KIND,
    AgentFactoryRegistry,
)
from research_engineer.benchmark.bench_agents import register_benchmark_kinds
from research_engineer.benchmark.benchmark import load_benchmark_suite
from research_engineer.benchmark.safety import build_default_safety_chain
from research_engineer.eval.graders import GradingRequest, build_grader
from research_engineer.eval.models import EvalSuite
from research_engineer.improve.mining import (
    MiningConfig,
    RunFeatures,
    mine_patterns,
)
from research_engineer.runtime.checkpoint_stores import SQLiteCheckpointStore
from research_engineer.runtime.models import AgentTermination
from research_engineer.runtime.runtime import AgentRuntime

#: P1 failure taxonomy (§11 categories).
FAILURE_TAXONOMY: tuple[str, ...] = (
    "planning", "tool_use", "coding", "experiment", "research_grounding",
    "analysis", "safety", "budget", "infrastructure", "recovery",
    "evaluation",
)


# ---------------------------------------------------------------------------
# Persisted-payload views (grading without deserializing live contexts)
# ---------------------------------------------------------------------------


class _StoredContextView:
    """Duck-typed ``AgentContext`` rebuilt from a worker result payload.

    Exposes exactly the attributes E4 graders read (termination,
    termination_reason, current_step, counters, metadata["tool_call_log"]).
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        summary: dict[str, Any] = (
            payload.get("context")
            or {}
        )
        raw_term = str(payload.get("termination") or "")
        try:
            self.termination: AgentTermination | None = (
                AgentTermination(raw_term) if raw_term else None
            )
        except ValueError:
            self.termination = None
        self.termination_reason = str(payload.get("reason") or "")
        self.output = payload.get("output")
        self.current_step = int(summary.get("current_step", 0) or 0)
        self.tool_calls = int(summary.get("tool_calls", 0) or 0)
        self.tokens = int(summary.get("tokens", 0) or 0)
        self.cost_usd = float(summary.get("cost_usd", 0.0) or 0.0)
        self.recoverable_errors = int(
            summary.get("recoverable_errors", 0) or 0
        )
        fatal = summary.get("fatal_errors", [])
        self.fatal_error_count = (
            len(fatal) if isinstance(fatal, list) else int(fatal or 0)
        )
        self.duration_seconds = float(
            summary.get("duration_seconds", 0.0) or 0.0
        )
        self.metadata: dict[str, Any] = {
            "human_interventions": int(
                summary.get("human_interventions", 0) or 0
            ),
            "tool_call_log": [
                e for e in (summary.get("tool_call_log") or [])
                if isinstance(e, dict)
            ],
            "safety_state": summary.get("safety_state") or {},
        }

    def is_success(self) -> bool:
        return self.termination == AgentTermination.SUCCESS


class _StoredExecutionView:
    """Duck-typed ``AgentExecution`` around :class:`_StoredContextView`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.context = _StoredContextView(payload)

    @property
    def output(self) -> Any:
        return self.context.output


# ---------------------------------------------------------------------------
# Outcome models
# ---------------------------------------------------------------------------


class CriterionOutcome(BaseModel):
    grader: str
    weight: float
    required: bool
    passed: bool
    score: float
    detail: str = ""
    # "" = valid verdict; "judge_error" = evaluator malfunction that must
    # never be counted as an agent failure (excluded from aggregation).
    error_kind: str = ""


class CaseOutcome(BaseModel):
    """One benchmark case execution through the production stack."""

    case_id: str
    repeat: int
    category: str
    mode: str
    agent_kind: str
    submitted_at: datetime
    latency_seconds: float
    status: str                     # "completed" | "failed" | "cancelled"
    runtime_success: bool           # AgentTermination.SUCCESS
    graded_success: bool            # required criteria + runtime success
    weighted_score: float
    criteria: list[CriterionOutcome]
    steps: int
    tool_calls: int
    tokens: int
    cost_usd: float
    recoverable_errors: int
    fatal_errors: int
    recovered: bool                 # had transient failures and finished ok
    human_interventions: int
    safety_interventions: int       # non-continue deterministic decisions
    gateway_denials: int            # approval/policy denials seen
    termination: str
    termination_reason: str
    failure_categories: list[str] = Field(default_factory=list)
    artifacts_added: int = 0
    budget_max_steps: int = 0
    expected_failed_by_design: bool = False


async def _grade_case(
    suite_case: Any,
    execution_view: _StoredExecutionView,
    harness_error: str = "",
    extra_graders: dict[str, Any] | None = None,
) -> tuple[list[CriterionOutcome], bool, float]:
    """Async twin of E4's ``_grade`` against the stored-payload view."""

    # The real request type derives ``context``/``output`` properties from
    # ``execution`` - a hand-rolled shim is exactly what breaks here.
    req = GradingRequest(
        suite_case,
        execution_view,  # type: ignore[arg-type]  # persisted-payload view
        human_interventions=execution_view.context.metadata[
            "human_interventions"
        ],
        harness_error=harness_error,
    )

    outcomes: list[CriterionOutcome] = []
    for criterion in suite_case.criteria:
        shared = (extra_graders or {}).get(criterion.grader)
        try:
            if shared is not None:
                # Per-criterion configuration (rubric/threshold/patterns)
                # must not leak between cases sharing one grader instance,
                # so each grade gets its own configured copy.
                grader = deepcopy(shared)
                if hasattr(grader, "configure"):
                    grader.configure(criterion.config)
                result = await grader.grade(req)
            else:
                result = await build_grader(
                    criterion.grader, criterion.config
                ).grade(req)
        except KeyError as exc:
            # Unknown grader names fail closed as a failed criterion
            # instead of aborting the whole benchmark run.
            outcomes.append(CriterionOutcome(
                grader=criterion.grader,
                weight=criterion.weight,
                required=criterion.required,
                passed=False,
                score=0.0,
                detail=f"unknown grader: {exc}",
            ))
            continue
        outcomes.append(CriterionOutcome(
            grader=criterion.grader,
            weight=criterion.weight,
            required=criterion.required,
            passed=result.passed,
            score=result.score,
            detail=result.detail,
            error_kind=getattr(result, "error_kind", "") or "",
        ))
    required_ok = all(o.passed for o in outcomes if o.required)
    runtime_success = execution_view.context.is_success()
    graded_success = required_ok and (
        runtime_success if suite_case.require_runtime_success else True
    )
    # Judge errors are evaluator malfunctions, not agent failures: their
    # weight is excluded (and the remaining weights renormalized) so a
    # broken judge can neither deflate nor inflate the weighted score.
    graded = [o for o in outcomes if o.error_kind != "judge_error"]
    total_weight = sum(o.weight for o in graded) or 1.0
    weighted = sum(o.score * o.weight for o in graded) / total_weight
    return outcomes, graded_success, round(weighted, 4)


def classify_failure(outcome: CaseOutcome) -> list[str]:
    """Deterministic P1 §7 failure-taxonomy labeling (multi-label).

    Rules are evidence-based and ordered; a case may carry several labels.
    Research-grounding / coding / analysis / experiment labels attach only
    when future (LLM-mode) suites emit their signature markers - the
    deterministic v1 stack cannot legitimately trigger them.
    """
    labels: set[str] = set()
    reason = f"{outcome.termination_reason} {outcome.termination}".lower()
    incomplete = outcome.status != "completed"

    if incomplete and ("budget" in reason or "max steps" in reason):
        labels.add("budget")
    elif incomplete and "timeout" in reason:
        labels.update({"budget", "infrastructure"})
    elif incomplete and ("safety" in reason or "replan_limit" in reason):
        labels.add("safety")
    elif incomplete:
        # Any other non-success ending: classify as recovery failure,
        # plus infrastructure when transient errors went unrecovered.
        labels.add("recovery")
        if outcome.recoverable_errors > 0 or outcome.fatal_errors > 0:
            labels.add("infrastructure")

    if outcome.gateway_denials > 0:
        labels.update({"tool_use", "safety"})

    if incomplete and outcome.steps == 0:
        labels.add("planning")

    if outcome.recoverable_errors > 0 and not outcome.recovered:
        labels.add("recovery")

    # Failed required criteria on a nominally successful run => the graded
    # contract disagrees with runtime success (evaluation-layer mismatch).
    failed_required = [
        c.grader for c in outcome.criteria
        if c.required and not c.passed
    ]
    if failed_required and outcome.runtime_success:
        labels.add("evaluation")

    return sorted(labels, key=FAILURE_TAXONOMY.index)


# ---------------------------------------------------------------------------
# Aggregation + reporting models
# ---------------------------------------------------------------------------


class RepeatSpread(BaseModel):
    """Cross-repeat stability for one case (P1 §6 reproducibility)."""

    case_id: str
    successes: list[int]
    scores: list[float]
    consistent_success: bool
    score_stddev: float


class BenchmarkReport(BaseModel):
    suite_id: str
    suite_version: str
    generated_at: datetime
    repeat_count: int
    cases_total: int
    outcomes: list[CaseOutcome]

    #: P1 §6 metrics (keys documented in ``_aggregate``).
    metrics: dict[str, float]
    termination_distribution: dict[str, int]
    failure_summary: dict[str, int]
    patterns: dict[str, Any] = Field(default_factory=dict)
    repeat_spreads: list[RepeatSpread] = Field(default_factory=list)
    markdown_path: str | None = None
    json_path: str | None = None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def _aggregate(outcomes: list[CaseOutcome]) -> tuple[dict[str, float],
                                                     Counter]:
    """Compute the full P1 §6 metric battery."""
    n = len(outcomes)
    # Guardrail anti-cases exist to be stopped, so they are excluded from
    # rate denominators - completing them was never a possible outcome.
    effective = [o for o in outcomes if not o.expected_failed_by_design]
    n_eff = len(effective) if effective else n
    completed_untouched = [
        o for o in outcomes
        if o.status == "completed" and o.graded_success
        and o.human_interventions == 0
        and "budget" not in o.termination.lower()
        and o.termination != "timeout"
    ]
    # Numerator AND denominator both live on the effective set so an
    # anti-case that grades PASS on enforcement criteria cannot inflate
    # the success rates.
    successes = [o for o in effective if o.graded_success]
    latencies = [o.latency_seconds for o in outcomes]
    with_recoverables = [o for o in outcomes if o.recoverable_errors > 0]
    successful_tools = 0
    total_tools = 0
    for o in outcomes:
        # Deterministic kinds issue two gateway calls per step (write +
        # list); a goal-compliant run's calls are all attributable, others
        # contribute only their step-bounded share to the denominator.
        total_tools += o.tool_calls
        successful_tools += min(o.tool_calls, o.steps * 2) \
            if o.graded_success else 0

    metrics: dict[str, float] = {
        # Headline (P1: most important metric).
        "autonomous_completion_rate":
            (len(completed_untouched) / n_eff) if n_eff else 0.0,
        "task_success_rate": (len(successes) / n_eff) if n_eff else 0.0,
        "research_quality_score":
            mean([o.weighted_score for o in effective])
            if effective else 0.0,
        "by_design_exclusions": float(n - len(effective)),
        "human_intervention_rate":
            (sum(1 for o in outcomes if o.human_interventions > 0) / n)
            if n else 0.0,
        "safety_intervention_rate":
            (sum(1 for o in outcomes if o.safety_interventions > 0) / n)
            if n else 0.0,
        "gateway_denial_rate":
            (sum(1 for o in outcomes if o.gateway_denials > 0) / n)
            if n else 0.0,
        "recovery_success_rate":
            (sum(1 for o in with_recoverables if o.recovered)
             / len(with_recoverables)) if with_recoverables else 1.0,
        "median_latency_seconds": median(latencies) if latencies else 0.0,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "median_cost_per_task_usd":
            median([o.cost_usd for o in outcomes]) if n else 0.0,
        "median_tokens_per_task":
            median([float(o.tokens) for o in outcomes]) if n else 0.0,
        "total_tokens": float(sum(o.tokens for o in outcomes)),
        "total_cost_usd": float(sum(o.cost_usd for o in outcomes)),
        "total_safety_interventions":
            float(sum(o.safety_interventions for o in outcomes)),
        "total_gateway_denials":
            float(sum(o.gateway_denials for o in outcomes)),
        # Tool efficiency: share of performed calls attributable to
        # successful (goal-compliant) executions.
        "tool_efficiency":
            (successful_tools / total_tools) if total_tools else 1.0,
        # Experiment efficiency: goal-completions per executed step.
        "experiment_efficiency":
            (len(successes) / max(1, sum(o.steps for o in outcomes))),
        "mean_budget_utilization":
            mean([
                (o.steps / o.budget_max_steps)
                for o in outcomes if o.budget_max_steps
            ]) if n else 0.0,
    }
    terminations = Counter(o.termination or "none" for o in outcomes)
    return metrics, terminations


def _mine_feature_view(outcomes: list[CaseOutcome]) -> list[RunFeatures]:
    """Map outcomes onto E8 RunFeatures (report-only path).

    ponytail: context-level tool-failure detail is dropped (summary lacks
    per-tool fail counts); upgrade when reports need tool-pattern mining.
    """
    features: list[RunFeatures] = []
    for o in outcomes:
        f = RunFeatures(
            run_id=f"{o.case_id}:r{o.repeat}",
            case_id=o.case_id,
            success=o.graded_success,
            score=o.weighted_score,
            termination_reason=o.termination_reason or o.termination,
            latency_seconds=o.latency_seconds,
            cost_usd=o.cost_usd,
            tokens=o.tokens,
            recoverable_errors=o.recoverable_errors,
            fatal_errors=o.fatal_errors,
            human_interventions=o.human_interventions,
        )
        blips = o.recoverable_errors + o.fatal_errors
        if blips >= 2:
            f.tool_failures["(runtime_error)"] = blips
            f.duplicate_tool_events = blips - 1
        features.append(f)
    return features


def render_markdown(report: BenchmarkReport) -> str:
    """Human-readable markdown rendering of the benchmark report."""
    m = report.metrics
    lines: list[str] = [
        f"# Benchmark report — {report.suite_id} "
        f"(v{report.suite_version})",
        "",
        f"- Generated: {report.generated_at.isoformat(timespec='seconds')}",
        f"- Cases executed: {report.cases_total} "
        f"(x{report.repeat_count} repeats)",
        *( [f"- Guardrail anti-cases excluded from rates (by design): "
            f"{int(m['by_design_exclusions'])}"]
           if m.get("by_design_exclusions") else [] ),
        "",
        "## Headline",
        "",
        f"- **Autonomous task completion rate**: "
        f"{m['autonomous_completion_rate']:.1%}",
        f"- Task success rate: {m['task_success_rate']:.1%}",
        f"- Research quality score: {m['research_quality_score']:.3f}",
        "",
        "## Safety & autonomy",
        "",
        f"- Human-intervention rate: {m['human_intervention_rate']:.1%}",
        f"- Safety-intervention rate: {m['safety_intervention_rate']:.1%}",
        f"- Gateway denial rate: {m['gateway_denial_rate']:.1%}",
        f"- Recovery success rate: {m['recovery_success_rate']:.1%}",
        "",
        "## Cost & efficiency",
        "",
        f"- Median latency: {m['median_latency_seconds']:.3f}s "
        f"(p95 {m['p95_latency_seconds']:.3f}s)",
        f"- Median cost/task: ${m['median_cost_per_task_usd']:.6f}",
        f"- Median tokens/task: {m['median_tokens_per_task']:.0f}",
        f"- Tool efficiency: {m['tool_efficiency']:.1%}",
        f"- Experiment efficiency: "
        f"{m['experiment_efficiency']:.3f} completions/step",
        f"- Mean budget utilization: "
        f"{m['mean_budget_utilization']:.1%}",
        "",
        "## Termination distribution",
        "",
    ]
    for term, count in sorted(
        report.termination_distribution.items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"- `{term}`: {count}")
    lines += ["", "## Failure taxonomy (labeled outcomes)", ""]
    if not report.failure_summary:
        lines.append("- No failures recorded.")
    else:
        for label, count in sorted(
            report.failure_summary.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"- `{label}`: {count}")
    lines += ["", "## Cross-repeat reproducibility", ""]
    inconsistent = [s for s in report.repeat_spreads
                    if not s.consistent_success]
    lines.append(
        f"- Cases with unstable success across repeats: "
        f"{len(inconsistent)}"
    )
    worst = max(
        (s.score_stddev for s in report.repeat_spreads), default=0.0
    )
    lines.append(f"- Worst per-case score stddev: {worst:.4f}")
    lines += ["", "## Mined patterns (E8, read-only)", ""]
    if not report.patterns:
        lines.append("- No recurring patterns above thresholds.")
    else:
        for pid, pat in report.patterns.items():
            lines.append(f"- `{pid}`: {json.dumps(pat, default=str)[:200]}")
    lines += ["", "## Per-case results", "",
              "| case | cat | mode | rep | status | success | score | "
              "steps | tools | safety | labels |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in report.outcomes:
        note = " (by design)" if o.expected_failed_by_design else ""
        lines.append(
            f"| {o.case_id}{note} | {o.category} | {o.mode} | r{o.repeat} "
            f"| {o.status} | {'yes' if o.graded_success else 'NO'} "
            f"| {o.weighted_score:.3f} | {o.steps} | {o.tool_calls} "
            f"| {o.safety_interventions} "
            f"| {','.join(o.failure_categories) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Execute a benchmark suite over the production runtime path.

    P2 extensions (both optional, backward compatible):

    * ``extra_graders``: additional grader instances keyed by criterion
      name (e.g. the ``llm_quality`` LLM-judge); deterministic graders stay
      the default resolution path.
    * ``factories``: a pre-populated ``AgentFactoryRegistry``; when omitted,
      the P1 default registry is built and benchmark kinds are registered.
    """

    def __init__(
        self,
        base_dir: Path | str,
        *,
        extra_graders: dict[str, Any] | None = None,
        factories: AgentFactoryRegistry | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self._extra_graders = extra_graders or {}
        self._factories = factories

    async def run_suite(
        self,
        suite_path: str | Path | None = None,
        *,
        repeat: int = 1,
        report_name: str = "benchmark_report",
        case_ids: set[str] | None = None,
    ) -> BenchmarkReport:
        """Run cases ``repeat`` times; persist JSON+markdown.

        ``case_ids`` subsets execution (pilot/smoke) while the *whole* suite
        is still loaded and contract-validated first.
        """
        suite = load_benchmark_suite(suite_path)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        if case_ids is not None:
            unknown = case_ids - {c.case_id for c in suite.cases}
            if unknown:
                raise ValueError(f"unknown case_ids: {sorted(unknown)}")
            suite = suite.model_copy(update={
                "cases": [c for c in suite.cases if c.case_id in case_ids],
            })

        outcomes: list[CaseOutcome] = []
        for rep in range(1, repeat + 1):
            rep_dir = self.base_dir / f"repeat_{rep}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            outcomes.extend(await self._run_once(suite, rep, rep_dir))

        metrics, terminations = _aggregate(outcomes)
        spreads = self._spreads(suite, outcomes)
        labeled = [o for o in outcomes if not o.graded_success]
        failure_summary = Counter(
            lbl for o in labeled for lbl in o.failure_categories
        )
        patterns = mine_patterns(
            _mine_feature_view(outcomes), MiningConfig(min_affected_runs=2)
        )

        report = BenchmarkReport(
            suite_id=suite.suite_id,
            suite_version=suite.version,
            generated_at=datetime.now(),
            repeat_count=repeat,
            cases_total=len(outcomes),
            outcomes=outcomes,
            metrics={k: float(v) for k, v in metrics.items()},
            termination_distribution=dict(terminations),
            failure_summary={
                k: int(v) for k, v in failure_summary.items()
            },
            patterns=patterns,
            repeat_spreads=spreads,
        )

        json_path = self.base_dir / f"{report_name}.json"
        md_path = self.base_dir / f"{report_name}.md"
        json_path.write_text(
            json.dumps(report.model_dump(mode="json"),
                       indent=2, default=str),
            encoding="utf-8",
        )
        md_path.write_text(render_markdown(report), encoding="utf-8")
        report.json_path = str(json_path)
        report.markdown_path = str(md_path)
        return report

    # -- internals ------------------------------------------------------

    async def _run_once(
        self, suite: EvalSuite, repeat: int, rep_dir: Path
    ) -> list[CaseOutcome]:
        checkpoint_store = SQLiteCheckpointStore(
            str(rep_dir / "checkpoints.db")
        )
        if self._factories is not None:
            factories = self._factories
        else:
            factories = AgentFactoryRegistry(config_max_steps=12)
            register_benchmark_kinds(factories)

        results: list[CaseOutcome] = []
        for case in suite.cases:
            case_dir = rep_dir / case.case_id
            workspace = case_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            gateway, controller = build_default_safety_chain(workspace)
            notes_before = _notes_snapshot(workspace)
            overrides = dict(case.metadata.get("agent_overrides") or {})
            payload, submitted_at = await self._execute_case(
                case, factories, checkpoint_store, gateway, controller,
                overrides,
            )
            (case_dir / "result.json").write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            added = len(
                _notes_snapshot(workspace) - notes_before
            )
            results.append(await self._outcome_for(
                case, repeat, payload, submitted_at, added,
                extra_graders=self._extra_graders or None,
            ))
        return results

    async def _execute_case(
        self,
        case: Any,
        factories: AgentFactoryRegistry,
        checkpoint_store: Any,
        gateway: Any,
        controller: Any,
        overrides: dict[str, Any],
    ) -> tuple[dict[str, Any], datetime]:
        """Run one case directly through ``AgentRuntime``; return payload.

        Mirrors the production execution recipe: build the agent from the
        registry, attach the runtime so every tool call crosses the
        gateway + safety chain, run to a terminal state under the case
        deadline, and build the same JSON payload graders consume.
        """
        submitted_at = datetime.now()
        deadline = float(case.timeout_seconds or 120.0)
        try:
            adapter, policy = await factories.build(
                str(case.metadata.get("agent_kind", DEFAULT_AGENT_KIND)),
                overrides,
            )
            runtime = AgentRuntime(
                planner=adapter.planner,
                actor=adapter.actor,
                observer=adapter.observer,
                evaluator=adapter.evaluator,
                policy=policy,
                checkpoint_store=checkpoint_store,
                tool_gateway=gateway,
                safety_controller=controller,
            )
            attach = getattr(adapter, "attach_runtime", None)
            if attach is not None:
                attach(runtime)
            execution = await asyncio.wait_for(
                runtime.run(
                    case.goal,
                    metadata={"case_id": case.case_id},
                ),
                timeout=deadline,
            )
        except TimeoutError:
            return (
                _failure_payload(
                    "timeout", f"case exceeded {deadline:g}s deadline",
                    submitted_at,
                ),
                submitted_at,
            )
        except Exception as exc:  # noqa: BLE001 - failures become outcomes
            return (
                _failure_payload(
                    "error", f"{type(exc).__name__}: {exc}", submitted_at,
                ),
                submitted_at,
            )
        return (
            _payload_from_context(execution.context, submitted_at),
            submitted_at,
        )

    async def _outcome_for(
        self, case: Any, repeat: int,
        payload: dict[str, Any], submitted_at: datetime,
        artifacts_added: int,
        *,
        extra_graders: dict[str, Any] | None = None,
    ) -> CaseOutcome:
        view = _StoredExecutionView(payload)
        criteria, graded_success, weighted = await _grade_case(
            case, view, extra_graders=extra_graders
        )
        ctx = view.context
        safety_state: dict[str, Any] = (
            ctx.metadata.get("safety_state") or {}
        )
        decisions: list[Any] = safety_state.get("decisions") or []
        interventions = sum(
            1 for d in decisions
            if isinstance(d, dict)
            and str(d.get("action", "")).lower()
            not in ("", "none", "continue")
        )
        log_entries: list[dict[str, Any]] = [
            e for e in (ctx.metadata.get("tool_call_log") or [])
            if isinstance(e, dict)
        ]
        denials = sum(
            1 for e in log_entries
            if "denied" in str(e.get("status", ""))
            or "violation" in str(e.get("status", ""))
        )
        outcome = CaseOutcome(
            case_id=case.case_id,
            repeat=repeat,
            category=str(case.metadata.get("category")),
            mode=str(case.metadata.get("mode")),
            agent_kind=str(case.metadata.get("agent_kind")),
            submitted_at=submitted_at,
            latency_seconds=float(ctx.duration_seconds),
            status=_status_from_termination(
                str(payload.get("termination") or "")),
            runtime_success=ctx.is_success(),
            graded_success=graded_success,
            weighted_score=weighted,
            criteria=criteria,
            steps=ctx.current_step,
            tool_calls=ctx.tool_calls,
            tokens=ctx.tokens,
            cost_usd=ctx.cost_usd,
            recoverable_errors=ctx.recoverable_errors,
            fatal_errors=ctx.fatal_error_count,
            recovered=(
                ctx.recoverable_errors > 0 and ctx.is_success()
            ),
            human_interventions=int(
                ctx.metadata.get("human_interventions", 0)
            ),
            safety_interventions=interventions,
            gateway_denials=denials,
            termination=(
                ctx.termination.value if ctx.termination else "none"
            ),
            termination_reason=ctx.termination_reason,
            budget_max_steps=int(case.budget.max_steps or 0),
            expected_failed_by_design=bool(
                case.metadata.get("expected_failed_by_design")
            ),
        )
        outcome.artifacts_added = artifacts_added
        if not graded_success:
            outcome.failure_categories = classify_failure(outcome)
        return outcome

    @staticmethod
    def _spreads(suite: EvalSuite, outcomes: list[CaseOutcome],
                 ) -> list[RepeatSpread]:
        by_case: dict[str, list[CaseOutcome]] = {}
        for o in outcomes:
            by_case.setdefault(o.case_id, []).append(o)
        spreads: list[RepeatSpread] = []
        for case in suite.cases:
            group = by_case.get(case.case_id, [])
            if len(group) < 2:
                continue
            spread = RepeatSpread(
                case_id=case.case_id,
                successes=[int(o.graded_success) for o in group],
                scores=[o.weighted_score for o in group],
                consistent_success=(
                    len({int(o.graded_success) for o in group}) <= 1
                ),
                score_stddev=(
                    round(pstdev([o.weighted_score for o in group]), 6)
                    if len(group) > 1 else 0.0
                ),
            )
            spreads.append(spread)
        return spreads


def _status_from_termination(termination: str) -> str:
    """Map an ``AgentTermination`` value to the run-status vocabulary."""
    if termination == "cancelled":
        return "cancelled"
    if termination in ("error", "safety_terminated", "timeout"):
        return "failed"
    return "completed"


def _failure_payload(
    termination: str, reason: str, submitted_at: datetime,
) -> dict[str, Any]:
    """Payload for a case that never produced a finished run context."""
    return {
        "output": None,
        "termination": termination,
        "reason": reason,
        "steps": 0,
        "execution_id": "",
        "submitted_at": submitted_at.isoformat(),
        "context": {},
    }


def _context_summary(context: Any) -> dict[str, Any]:
    """Compact, JSON-safe snapshot of a finished ``AgentContext``.

    Downstream consumers (benchmark grading/metrics, E8 mining) rebuild
    their analysis inputs from this payload without live runtime state:
    budget counters, the E4/E8 ``tool_call_log``, E5 safety decisions, and
    per-step outcomes.
    """
    metadata = getattr(context, "metadata", {}) or {}
    safety_state = metadata.get("safety_state")
    if isinstance(safety_state, dict):
        decisions = safety_state.get("decisions")
        if isinstance(decisions, list):
            # keep last 50 decisions; full history stays in the checkpoint
            safety_state = {**safety_state, "decisions": decisions[-50:]}
    return {
        "execution_id": getattr(context, "execution_id", ""),
        "state": str(
            getattr(getattr(context, "state", None), "value",
                    getattr(context, "state", "")) or ""
        ),
        "current_step": getattr(context, "current_step", 0),
        "tool_calls": getattr(context, "tool_calls", 0),
        "tokens": getattr(context, "tokens", 0),
        "cost_usd": getattr(context, "cost_usd", 0.0),
        "recoverable_errors": getattr(context, "recoverable_errors", 0),
        "fatal_errors": sum(
            1 for s in getattr(context, "steps", []) or []
            if getattr(s, "error", None) is not None
        ),
        "stagnation_count": getattr(context, "stagnation_count", 0),
        "best_score": getattr(context, "best_score", 0.0),
        "duration_seconds": getattr(context, "duration_seconds", 0.0),
        "human_interventions": int(
            metadata.get("human_interventions", 0) or 0
        ),
        "tool_call_log": [
            entry
            for entry in (metadata.get("tool_call_log") or [])
            if isinstance(entry, dict)
        ],
        "safety_state": (
            safety_state if isinstance(safety_state, dict) else None
        ),
        "steps": [
            {
                "step": getattr(s, "step", 0),
                "score": getattr(s, "score", 0.0),
                "error": (
                    {
                        "message": s.error.message,
                        "type": s.error.error_type,
                        "recoverable": s.error.recoverable,
                        "phase": (
                            s.error.phase.value
                            if s.error.phase is not None else ""
                        ),
                    }
                    if s.error is not None
                    else None
                ),
                "tool_calls": getattr(s, "tool_calls", 0),
                "tokens": getattr(s, "tokens", 0),
                "cost_usd": getattr(s, "cost_usd", 0.0),
                "duration_seconds": getattr(s, "duration_seconds", 0.0),
            }
            for s in getattr(context, "steps", []) or []
        ],
    }


def _payload_from_context(
    context: Any, submitted_at: datetime,
) -> dict[str, Any]:
    """Build the persisted result payload from a finished run context.

    The same JSON shape graders and metrics consume through
    :class:`_StoredExecutionView`.
    """
    termination = str(
        getattr(context.termination, "value", context.termination) or ""
    )
    return {
        "output": getattr(context, "output", None),
        "termination": termination,
        "reason": getattr(context, "termination_reason", None),
        "steps": len(getattr(context, "steps", []) or []),
        "execution_id": getattr(context, "execution_id", ""),
        "submitted_at": submitted_at.isoformat(),
        "context": _context_summary(context),
    }


def _notes_snapshot(artifacts_root: Path) -> set[str]:
    """Names of research notes currently on disk under ``root``."""
    notes_dir = Path(artifacts_root) / "sandbox" / "notes"
    if not notes_dir.is_dir():
        return set()
    return {p.name for p in notes_dir.iterdir() if p.is_file()}


__all__ = [
    "FAILURE_TAXONOMY",
    "BenchmarkReport",
    "BenchmarkRunner",
    "CaseOutcome",
    "classify_failure",
    "render_markdown",
]
