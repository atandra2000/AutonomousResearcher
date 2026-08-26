"""E5 - Autonomy & Safety Controls models.

Typed Pydantic v2 models for the deterministic safety/autonomy layer that
wraps :class:`~research_engineer.runtime.runtime.AgentRuntime`. The layer
observes every completed step, evaluates progress and risk, and returns a
:class:`ControlDecision` telling the runtime whether to CONTINUE, REPLAN,
PAUSE_FOR_APPROVAL, or TERMINATE.

Design constraint: these controls are *deterministic*. An LLM evaluator may
attach advisory notes to a decision but can never change a decision's
action while ``mandatory`` is True (hard limits, deny policies, approval
requirements).

Risk levels are reused from the E3 ToolGateway — no second taxonomy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.gateway.models import RiskLevel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControlAction(StrEnum):
    """What the runtime should do after observing a step."""

    CONTINUE = "continue"
    REPLAN = "replan"
    PAUSE_FOR_APPROVAL = "pause_for_approval"
    TERMINATE = "terminate"


class ControlTrigger(StrEnum):
    """Deterministic, machine-readable reason for a control decision."""

    NONE = "none"
    LOOP_DETECTED = "loop_detected"
    DUPLICATE_TOOL_CALL = "duplicate_tool_call"
    NO_PROGRESS = "no_progress"
    DIMINISHING_RETURNS = "diminishing_returns"
    FAILURE_ESCALATION = "failure_escalation"
    RISK_ESCALATION = "risk_escalation"
    POLICY_VIOLATION = "policy_violation"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    REPLAN_LIMIT = "replan_limit"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class ProgressSignal(BaseModel):
    """Progress/health snapshot for one completed runtime step.

    Derived from existing runtime progress tracking where possible
    (:attr:`AgentContext.stagnation_count`, ``best_score``) plus the
    safety state maintained by the controller.
    """

    step: int = Field(default=0, ge=0)
    score: float | None = Field(
        default=None, description="Evaluation score of this step, if any"
    )
    previous_best: float | None = Field(
        default=None, description="Best score before this step"
    )
    stagnation_count: int = Field(
        default=0, ge=0, description="Consecutive steps without progress"
    )
    consecutive_failures: int = Field(
        default=0, ge=0, description="Consecutive steps that errored"
    )
    steps_taken: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    replans: int = Field(
        default=0, ge=0, description="Replans requested so far this run"
    )
    #: Fraction of each budget consumed, keyed by budget dimension.
    budget_usage: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension fraction of budget consumed in [0, inf)",
    )
    observation_signature: str = Field(
        default="", description="Stable hash of this step's outputs"
    )
    score_improvements: list[float] = Field(
        default_factory=list,
        description=(
            "Recent improvements of the best score, most recent last "
            "(drives diminishing-returns detection)"
        ),
    )



class RiskAssessment(BaseModel):
    """Aggregated risk view of recent tool activity.

    Fed by recorded :class:`~research_engineer.gateway.models.ToolExecutionResult`
    metadata from the E3 ToolGateway (risk levels per tool policy, statuses).
    """

    highest_risk_seen: RiskLevel | None = Field(
        default=None, description="Highest risk level invoked recently"
    )
    high_risk_calls: int = Field(
        default=0, ge=0, description="Recent calls at HIGH risk level"
    )
    critical_calls: int = Field(
        default=0, ge=0, description="Recent calls at CRITICAL risk level"
    )
    policy_failures: int = Field(
        default=0, ge=0, description="Gateway policy/security failures"
    )
    details: list[str] = Field(
        default_factory=list, description="Human-readable risk notes"
    )

    @property
    def has_policy_failure(self) -> bool:
        return self.policy_failures > 0


class ToolCallRecord(BaseModel):
    """One recorded tool invocation (for duplicate/risk analysis)."""

    tool_name: str
    args_hash: str = Field(description="Stable hash of normalized arguments")
    result_signature: str = Field(
        default="", description="Stable hash of status+output"
    )
    status: str = Field(default="")
    is_policy_failure: bool = Field(default=False)
    risk_level: str = Field(
        default=RiskLevel.LOW.value, description="Tool's configured RiskLevel value"
    )


class ControlDecision(BaseModel):
    """The outcome of a safety/policy evaluation of one step."""

    action: ControlAction = Field(default=ControlAction.CONTINUE)
    trigger: ControlTrigger = Field(default=ControlTrigger.NONE)
    #: Machine-readable reason, e.g. ``"loop_detected.cycle_len=2"``.
    reason_code: str = Field(default="")
    reason: str = Field(default="", description="Human-readable explanation")
    #: True when the action may not be overridden by any advisor or caller.
    mandatory: bool = Field(
        default=True,
        description=(
            "True for hard limits/deny/approval requirements; an LLM "
            "advisor must never flip a mandatory decision"
        ),
    )
    warnings: list[str] = Field(default_factory=list)
    advisory_note: str = Field(
        default="",
        description=(
            "Optional supplementary reasoning from an LLM advisor; "
            "advisory only, never authoritative"
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.action == ControlAction.TERMINATE

    @property
    def continues(self) -> bool:
        return self.action == ControlAction.CONTINUE


def continue_decision(**kwargs: Any) -> ControlDecision:
    """Shorthand for an all-clear decision."""
    kwargs.setdefault("trigger", ControlTrigger.NONE)
    kwargs.setdefault("reason_code", "")
    return ControlDecision(action=ControlAction.CONTINUE, **kwargs)


def terminate_decision(
    trigger: ControlTrigger, reason_code: str, reason: str, **kwargs: Any
) -> ControlDecision:
    """Shorthand for a terminal decision (always mandatory)."""
    return ControlDecision(
        action=ControlAction.TERMINATE,
        trigger=trigger,
        reason_code=reason_code,
        reason=reason,
        mandatory=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Safety state (checkpointable)
# ---------------------------------------------------------------------------

#: History caps keeping checkpoints bounded regardless of run length.
MAX_SIGNATURE_HISTORY = 64
MAX_TOOL_CALL_RECORDS = 128
MAX_DECISION_RECORDS = 32


class DecisionRecord(BaseModel):
    """Compact log of one emitted control decision."""

    step: int
    action: ControlAction
    trigger: ControlTrigger
    reason_code: str


class SafetyState(BaseModel):
    """Checkpointable state required for deterministic autonomy decisions.

    Everything the detectors need across resume lives here; the controller
    persists a JSON snapshot under ``ctx.metadata["safety_state"]`` so E2
    checkpoint/resume restores it verbatim.
    """

    run_id: str = Field(default="", description="Execution id this state belongs to")
    signatures: list[str] = Field(
        default_factory=list,
        description="Trailing step-signature history (most recent last)",
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="Recorded tool invocations (most recent last)",
    )
    identical_counts: dict[str, int] = Field(
        default_factory=dict,
        description='"tool:args_hash" -> occurrences within the window',
    )
    replans: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    replan_requested_for_streak: bool = Field(
        default=False,
        description="A replan was already escalated for the current failure streak",
    )
    decisions: list[DecisionRecord] = Field(
        default_factory=list, description="Most recent control decisions"
    )
    total_decisions: int = Field(default=0, ge=0)
    approved_risk: str | None = Field(
        default=None,
        description=(
            "Highest risk level already human-approved this run; future "
            "calls at that level (or below) do not re-trigger pauses"
        ),
    )

    def push_signature(self, signature: str) -> None:
        """Append a signature, trimming to the maximum history cap."""
        self.signatures.append(signature)
        if len(self.signatures) > MAX_SIGNATURE_HISTORY:
            del self.signatures[: len(self.signatures) - MAX_SIGNATURE_HISTORY]

    def push_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)
        if len(self.tool_calls) > MAX_TOOL_CALL_RECORDS:
            del self.tool_calls[: len(self.tool_calls) - MAX_TOOL_CALL_RECORDS]

    def push_decision(self, record: DecisionRecord) -> None:
        self.decisions.append(record)
        self.total_decisions += 1
        if len(self.decisions) > MAX_DECISION_RECORDS:
            del self.decisions[: len(self.decisions) - MAX_DECISION_RECORDS]


__all__ = [
    "ControlAction",
    "ControlTrigger",
    "ControlDecision",
    "continue_decision",
    "terminate_decision",
    "DecisionRecord",
    "ProgressSignal",
    "RiskAssessment",
    "RiskLevel",
    "SafetyState",
    "ToolCallRecord",
]

