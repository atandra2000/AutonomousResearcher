"""E8 - Continuous Agent Improvement models.

Typed Pydantic v2 models for the controlled improvement pipeline that
converts agent failures and E4 evaluation results into measurable
improvement candidates.

Safety model (mirrors E5's "advisory vs mandatory" split): candidates are
*data/configuration artifacts* produced from mined failure patterns. They can
only tune declared configuration surfaces (prompts, planning parameters,
autonomy thresholds, tool-selection strategy, model/provider config) and are
never allowed to modify source code or weaken hard safety limits. Production
promotion always requires explicit human approval; every promotion is
reversible via rollback.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Hashing helpers (deterministic candidate identity)
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Stable JSON serialization used for content hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(payload: Any) -> str:
    """SHA-256 hex digest over the canonical JSON of ``payload``."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CandidateComponent(StrEnum):
    """Declared agent-configuration surface a candidate may retune."""

    SYSTEM_PROMPT = "system_prompt"
    PLANNING_PARAMS = "planning_params"
    AUTONOMY_THRESHOLDS = "autonomy_thresholds"
    SAFETY_POLICY_CONFIG = "safety_policy_config"
    TOOL_SELECTION_STRATEGY = "tool_selection_strategy"
    MODEL_PROVIDER_CONFIG = "model_provider_config"


class CandidateStatus(StrEnum):
    """Explicit improvement-candidate lifecycle states.

    Allowed transitions are enumerated by
    :data:`research_engineer.improve.pipeline.LIFECYCLE_TRANSITIONS`;
    nothing else is legal.
    """

    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    PASSED = "passed"
    FAILED = "failed"
    APPROVED = "approved"
    CANARY = "canary"
    PROMOTED = "promoted"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (CandidateStatus.PROMOTED, CandidateStatus.REJECTED)


class PatternKind(StrEnum):
    """Recurring failure/problem classes the miner can surface."""

    REPEATED_TOOL_FAILURES = "repeated_tool_failures"
    UNNECESSARY_TOOL_CALLS = "unnecessary_tool_calls"
    REPEATED_REPLANNING = "repeated_replanning"
    SAFETY_INTERVENTIONS = "safety_interventions"
    BUDGET_EXHAUSTION = "budget_exhaustion"
    NO_PROGRESS_TERMINATION = "no_progress_termination"
    POOR_EVALUATION_SCORE = "poor_evaluation_score"
    EXCESSIVE_LATENCY = "excessive_latency"
    EXCESSIVE_COST = "excessive_cost"
    HUMAN_APPROVAL_FREQUENCY = "human_approval_frequency"


class PromotionAction(StrEnum):
    """Recorded lifecycle decision actions (audit trail)."""

    RECOMMENDED = "recommended"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANARY_STARTED = "canary_started"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Mining artifacts
# ---------------------------------------------------------------------------


class FailurePattern(BaseModel):
    """A recurring problem observed across *multiple* runs.

    ``affected_runs`` counts distinct runs (not events) so a single
    anomalous run can never satisfy the evidence bar by itself.
    """

    pattern_id: str = Field(
        ..., description="Deterministic hash of kind + signature",
    )
    kind: PatternKind
    signature: str = Field(
        default="", description="Sub-classifier, e.g. failing tool name",
    )
    description: str = Field(default="", description="Human-readable summary")
    affected_runs: int = Field(default=0, ge=1)
    total_events: int = Field(default=0, ge=0)
    run_ids: list[str] = Field(default_factory=list)
    example_case_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Proposals / candidates
# ---------------------------------------------------------------------------


class ImprovementProposal(BaseModel):
    """A data/configuration-only change suggestion.

    Generated pluggably (deterministic rule engine or an LLM), but always a
    plain artifact: it names one :class:`CandidateComponent` plus a JSON
    dict of changes. It carries no executable code.
    """

    title: str = Field(..., min_length=1)
    description: str = Field(default="")
    component: CandidateComponent
    changes: dict[str, Any] = Field(..., description="JSON configuration patch")
    evidence: list[str] = Field(
        default_factory=list,
        description="Pattern ids / observations motivating this proposal",
    )
    source: str = Field(default="rule_based", description="Proposal generator name")
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def proposal_key(self) -> str:
        """Content identity shared by identical proposals."""
        return canonical_json([self.component.value, self.changes])


class ImprovementCandidate(BaseModel):
    """Versioned, immutable record of one improvement candidate.

    Frozen: state changes happen only through
    :meth:`ImprovementPipeline` transitions, which persist updated copies.
    The id/hash are pure functions of content + parentage, so an identical
    candidate is always reproducible with the same identifier.

    ``changes`` semantics per component:

    * ``system_prompt`` -> ``{"append": str}`` (prompt template patch)
    * ``planning_params`` / ``autonomy_thresholds`` /
      ``safety_policy_config`` -> dotted config keys to values, validated
      against guardrails at creation time
    * ``tool_selection_strategy`` / ``model_provider_config`` -> free-form
      JSON chosen keys validated against the component allow-list
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str = Field(..., description='Deterministic: "cand_" + hash[:12]')
    version: str = Field(..., description="Config-hash prefix (content identity)")
    component: CandidateComponent
    changes: dict[str, Any]
    config_hash: str = Field(..., description="SHA-256 of component+changes")
    parent_baseline_id: str = Field(
        ..., description="Baseline this candidate was derived from",
    )
    parent_config_hash: str = Field(
        ..., description="Configuration snapshot hash of the parent baseline",
    )
    suite_id: str = Field(..., description="Evaluation suite the candidate must pass")
    suite_version: str = Field(default="1")
    proposal_source: str = Field(default="rule_based")
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: CandidateStatus = Field(default=CandidateStatus.PROPOSED)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    evaluation: CandidateEvaluation | None = None
    decision: PromotionDecision | None = None

    def evolve(self, **updates: Any) -> ImprovementCandidate:
        """Return an updated copy (the only sanctioned mutation path)."""
        updates.setdefault("updated_at", datetime.now())
        return self.model_copy(update=updates)


class PromotionDecision(BaseModel):
    """Immutable audit record for one lifecycle decision."""

    action: PromotionAction
    decided_by: str = Field(
        default="system",
        description="Human approver id, or 'system' for gate outcomes",
    )
    reason: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.now)
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation comparison
# ---------------------------------------------------------------------------


class MetricSnapshot(BaseModel):
    """Comparable scalar metrics extracted from one :class:`EvalReport`.

    Every metric the regression gate needs lives here so comparisons never
    re-read raw reports; see
    :func:`research_engineer.improve.gate.snapshot_from_report`.
    """

    cases_total: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean weighted grader score",
    )
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_interventions: float = Field(
        default=0.0, ge=0.0,
        description="Runs terminated with safety_terminated (hard metric)",
    )
    human_interventions: float = Field(
        default=0.0, ge=0.0, description="Fraction of runs needing a human",
    )
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_tokens: float = Field(default=0.0, ge=0.0)
    avg_latency_seconds: float = Field(default=0.0, ge=0.0)
    p95_latency_seconds: float = Field(default=0.0, ge=0.0)
    termination_reasons: dict[str, float] = Field(default_factory=dict)

    def deltas(self, other: MetricSnapshot) -> dict[str, float]:
        """Element-wise ``other - self`` for scalar fields."""
        mine = self.model_dump(exclude={"termination_reasons"})
        theirs = other.model_dump(exclude={"termination_reasons"})
        return {k: theirs[k] - mine[k] for k in mine}


class GateVerdict(BaseModel):
    """Outcome of the configurable regression gate."""

    passed: bool
    violations: list[str] = Field(
        default_factory=list, description="Blocking violations (empty => passed)",
    )
    warnings: list[str] = Field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.passed

    def recommended(self) -> PromotionAction:
        """Map verdict onto the diagram's reject/recommend branches."""
        if self.passed:
            return PromotionAction.RECOMMENDED
        return PromotionAction.REJECTED


class Baseline(BaseModel):
    """Versioned baseline: production configuration + its eval metrics."""

    baseline_id: str = Field(
        ..., description='Deterministic: "base_" + hash(label|suite|config)[:12]',
    )
    label: str = Field(..., description="Human version label, e.g. 'v1.2'")
    suite_id: str
    suite_version: str = Field(default="1")
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict, description="Production agent configuration",
    )
    config_hash: str = Field(..., description="SHA-256 of config_snapshot")
    metrics: MetricSnapshot
    report_id: str = Field(default="", description="Source EvalReport.report_id")
    created_at: datetime = Field(default_factory=datetime.now)


class CandidateEvaluation(BaseModel):
    """Paired baseline-vs-candidate evaluation result."""

    evaluated_at: datetime = Field(default_factory=datetime.now)
    suite_id: str
    suite_version: str
    seed: int = Field(default=0, description="EvalRunner seed used")
    baseline_report_id: str = Field(default="")
    candidate_report_id: str = Field(default="")
    baseline: MetricSnapshot
    candidate: MetricSnapshot
    deltas: dict[str, float] = Field(
        default_factory=dict, description="candidate - baseline per metric",
    )
    verdict: GateVerdict


# ---------------------------------------------------------------------------
# Regression gate configuration
# ---------------------------------------------------------------------------

#: Metrics where ANY increase is an automatic hard reject. A regression in a
#: hard safety metric must reject the candidate regardless of other gains;
#: E3 policy enforcement and E5 safety policies stay authoritative outside
#: this pipeline.
HARD_SAFETY_METRICS: frozenset[str] = frozenset({"safety_interventions"})


class RegressionGate(BaseModel):
    """Configurable promotion criteria applied to baseline vs candidate.

    A candidate qualifies only when *all* constraints hold simultaneously
    (one improved metric is never sufficient). Hard safety regressions are
    unconditional rejects regardless of configuration.
    """

    require_success_gte_baseline: bool = Field(
        default=True, description="success_rate >= baseline success_rate",
    )
    quality_tolerance: float = Field(
        default=1e-6, ge=0.0, description="Allowed quality score decrease",
    )
    max_failure_rate_increase: float = Field(
        default=0.0, ge=0.0, description="failure_rate - baseline must be <=",
    )
    max_human_intervention_increase: float = Field(
        default=0.0, ge=0.0, description="human intervention rate slack",
    )
    max_cost_increase_fraction: float = Field(
        default=0.05, ge=0.0, description="Relative cost increase allowance",
    )
    max_token_increase_fraction: float = Field(
        default=0.10, ge=0.0, description="Relative token increase allowance",
    )
    max_latency_increase_fraction: float = Field(
        default=0.25, ge=0.0, description="Relative latency increase allowance",
    )
    #: Additional absolute-count cap on safety interventions (None = no extra
    #: cap beyond "no regression"; may tighten but never loosen protection).
    max_safety_interventions_abs: float | None = Field(default=None, ge=0.0)
    #: Any increase on these snapshot metrics auto-rejects. Extendable, but
    #: HARD_SAFETY_METRICS members can never be removed.
    no_regression_metrics: tuple[str, ...] = ("quality_score", "success_rate")


def build_default_gate() -> RegressionGate:
    """Gate shipped as default: improves/neutral everywhere, safer or equal."""
    return RegressionGate()


__all__ = [
    "HARD_SAFETY_METRICS",
    "Baseline",
    "CandidateComponent",
    "CandidateEvaluation",
    "CandidateStatus",
    "FailurePattern",
    "GateVerdict",
    "ImprovementCandidate",
    "ImprovementProposal",
    "MetricSnapshot",
    "PatternKind",
    "PromotionAction",
    "PromotionDecision",
    "RegressionGate",
    "build_default_gate",
    "canonical_json",
    "content_hash",
]
