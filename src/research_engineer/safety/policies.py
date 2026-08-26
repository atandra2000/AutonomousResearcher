"""E5 - Autonomy & safety policies.

:class:`AutonomyPolicy` is the typed configuration for every control the
safety layer enforces. :class:`SafetyPolicy` is the policy interface and
:class:`RuleBasedSafetyPolicy` the deterministic reference implementation
covering loop detection, duplicate tool calls, no-progress termination,
diminishing returns, failure escalation, risk escalation (via E3 gateway
risk levels), and budget escalation.

Determinism rule: the rule-based policy is authoritative. Optional LLM
advisors may attach reasoning via :attr:`ControlDecision.advisory_note`
but can never change a decision's action while ``mandatory`` is True.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from research_engineer.safety.detectors import (
    DuplicateToolCallDetector,
    LoopDetector,
)
from research_engineer.safety.models import (
    ControlAction,
    ControlDecision,
    ControlTrigger,
    ProgressSignal,
    RiskAssessment,
    SafetyState,
    continue_decision,
    terminate_decision,
)


class AutonomyPolicy(BaseModel):
    """Typed configuration for all autonomy/safety controls.

    Each control has an ``*_enabled`` flag plus its detection parameters,
    and each non-hard-limit control has a configurable ``*_action``
    (CONTINUE / REPLAN / PAUSE_FOR_APPROVAL / TERMINATE).
    """

    enabled: bool = Field(
        default=True, description="Master switch; False makes controls no-ops"
    )

    # --- Loop / cyclic behaviour ---
    loop_enabled: bool = True
    loop_threshold: int = Field(
        default=3, ge=2, description="Identical consecutive steps before loop"
    )
    loop_window: int = Field(default=12, ge=4)
    loop_max_cycle_len: int = Field(default=2, ge=1)
    loop_action: ControlAction = Field(
        default=ControlAction.REPLAN,
        description="Action on detected looping/cycling",
    )

    # --- Duplicate tool calls ---
    duplicate_enabled: bool = True
    duplicate_max_identical: int = Field(
        default=2, ge=1, description="Allowed repeats of identical tool+args calls"
    )
    duplicate_window: int = Field(default=10, ge=3)
    duplicate_require_same_result: bool = Field(
        default=True,
        description="Only flag repeats that produced identical results",
    )
    duplicate_action: ControlAction = Field(default=ControlAction.REPLAN)

    # --- No progress ---
    no_progress_enabled: bool = True
    no_progress_stagnation_limit: int = Field(
        default=5,
        ge=1,
        description="Consecutive stagnant steps before action "
        "(reuses runtime stagnation tracking)",
    )
    no_progress_action: ControlAction = Field(default=ControlAction.TERMINATE)

    # --- Diminishing returns ---
    diminishing_enabled: bool = True
    diminishing_min_deltas: int = Field(
        default=3, ge=2, description="Improvement deltas required to judge"
    )
    diminishing_threshold: float = Field(
        default=0.01,
        gt=0.0,
        description="Latest improvement must be below this absolute delta",
    )
    diminishing_action: ControlAction = Field(default=ControlAction.TERMINATE)

    # --- Failure escalation ---
    failure_consecutive_limit: int = Field(
        default=3, ge=1, description="Consecutive failed steps before escalation"
    )
    failure_action: ControlAction = Field(default=ControlAction.REPLAN)

    # --- Replanning ---
    max_replans: int = Field(
        default=2, ge=0, description="REPLAN actions allowed before terminal"
    )

    # --- Budget escalation ---
    budget_warning_fraction: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Warn when any budget crosses this fraction",
    )
    budget_on_warning: ControlAction = Field(
        default=ControlAction.CONTINUE,
        description="Deterministic action taken at the warning threshold",
    )

    # --- Risk escalation (E3 gateway RiskLevel) ---
    risk_high_action: ControlAction = Field(
        default=ControlAction.PAUSE_FOR_APPROVAL,
        description="Action when HIGH-risk tools were invoked",
    )
    risk_critical_action: ControlAction = Field(
        default=ControlAction.PAUSE_FOR_APPROVAL,
        description="Action when CRITICAL-risk tools were invoked",
    )
    on_policy_failure: ControlAction = Field(
        default=ControlAction.TERMINATE,
        description="Action when the gateway reports a policy/security failure",
    )


class SafetyPolicy(ABC):
    """Interface for autonomy/safety policies.

    Implementations return a :class:`ControlDecision` for one observed
    step. Decisions must be deterministic given the signal/risk/state
    triple so checkpoint/resume replays behave identically.
    """

    @abstractmethod
    def evaluate(
        self,
        signal: ProgressSignal,
        risk: RiskAssessment,
        state: SafetyState,
    ) -> ControlDecision:
        """Evaluate one step and return the control decision."""


class RuleBasedSafetyPolicy(SafetyPolicy):
    """The deterministic reference implementation of ``SafetyPolicy``.

    Evaluation order encodes severity (most severe first):

    1. Gateway policy/security failures -> configurable (TERMINATE)
    2. Exceeded budgets                 -> TERMINATE (defense in depth)
    3. No progress                      -> configurable (TERMINATE)
    4. Diminishing returns              -> configurable (TERMINATE)
    5. Consecutive failures             -> configurable (REPLAN)
    6. Loop/cycle detection             -> configurable (REPLAN)
    7. Duplicate tool calls             -> configurable (REPLAN)
    8. Budget warning                   -> configurable (CONTINUE+warn)
    9. Risk escalation (CRITICAL/HIGH)  -> configurable (PAUSE)
    10. otherwise                       -> CONTINUE

    Args:
        config: :class:`AutonomyPolicy` with all knobs.
    """

    def __init__(self, config: AutonomyPolicy | None = None) -> None:
        self.config = config or AutonomyPolicy()
        self._loop_detector = LoopDetector(
            threshold=self.config.loop_threshold,
            window=self.config.loop_window,
            max_cycle_len=self.config.loop_max_cycle_len,
        )
        self._duplicate_detector = DuplicateToolCallDetector(
            max_identical=self.config.duplicate_max_identical,
            window=self.config.duplicate_window,
            require_same_result=self.config.duplicate_require_same_result,
        )

    def evaluate(
        self,
        signal: ProgressSignal,
        risk: RiskAssessment,
        state: SafetyState,
    ) -> ControlDecision:
        cfg = self.config
        if not cfg.enabled:
            return continue_decision()

        # 1-2. Gateway policy failures and exhausted budgets (hard limits).
        hard = self._check_hard_limits(signal, risk)
        if hard is not None:
            return hard

        # 3-4. No progress and diminishing returns.
        stalled = self._check_stall(signal)
        if stalled is not None:
            return stalled

        # 5-7. Failure escalation, looping, duplicate tool calls.
        behavioural = self._check_behaviour(signal, state)
        if behavioural is not None:
            return behavioural

        # 8. Budget warning.
        warned = sorted(
            dim
            for dim, frac in signal.budget_usage.items()
            if frac >= cfg.budget_warning_fraction
        )
        if warned:
            decision = _configured(
                cfg.budget_on_warning,
                ControlTrigger.BUDGET_WARNING,
                "budget_warning." + "+".join(warned),
                f"Approaching budget limit(s): {', '.join(warned)}",
            )
            decision.warnings.append(
                "budget_usage:"
                + ",".join(f"{d}={signal.budget_usage[d]:.0%}" for d in warned)
            )
            return decision

        # 9. Risk escalation.
        return self._check_risk(risk)

    def _check_hard_limits(
        self, signal: ProgressSignal, risk: RiskAssessment
    ) -> ControlDecision | None:
        """Gateway policy failures and exhausted budgets (steps 1-2)."""
        cfg = self.config
        if risk.has_policy_failure:
            return _configured(
                cfg.on_policy_failure,
                ControlTrigger.POLICY_VIOLATION,
                "policy_violation",
                f"ToolGateway reported {risk.policy_failures} "
                "policy/security failure(s)",
            )
        exceeded = sorted(
            dim for dim, frac in signal.budget_usage.items() if frac >= 1.0
        )
        if exceeded:
            return terminate_decision(
                ControlTrigger.BUDGET_EXCEEDED,
                "budget_exceeded." + "+".join(exceeded),
                f"Budget(s) exhausted: {', '.join(exceeded)}",
            )
        return None

    def _check_stall(self, signal: ProgressSignal) -> ControlDecision | None:
        """No-progress and diminishing-returns controls (steps 3-4)."""
        cfg = self.config
        if (
            cfg.no_progress_enabled
            and signal.stagnation_count >= cfg.no_progress_stagnation_limit
        ):
            return _configured(
                cfg.no_progress_action,
                ControlTrigger.NO_PROGRESS,
                f"no_progress.stagnation={signal.stagnation_count}",
                f"No progress for {signal.stagnation_count} steps",
            )
        if cfg.diminishing_enabled:
            verdict = self._check_diminishing(signal)
            if verdict is not None:
                code, reason = verdict
                return _configured(
                    cfg.diminishing_action,
                    ControlTrigger.DIMINISHING_RETURNS,
                    code,
                    reason,
                )
        return None

    def _check_behaviour(
        self, signal: ProgressSignal, state: SafetyState
    ) -> ControlDecision | None:
        """Failure escalation, looping, duplicate calls (steps 5-7)."""
        cfg = self.config
        if signal.consecutive_failures >= cfg.failure_consecutive_limit:
            if state.replan_requested_for_streak:
                return terminate_decision(
                    ControlTrigger.FAILURE_ESCALATION,
                    "failure_escalation.exhausted",
                    f"Failures persisted after escalation "
                    f"({signal.consecutive_failures} consecutive)",
                )
            return _configured(
                cfg.failure_action,
                ControlTrigger.FAILURE_ESCALATION,
                f"failure_escalation.consecutive={signal.consecutive_failures}",
                f"{signal.consecutive_failures} consecutive failed steps",
            )
        if cfg.loop_enabled:
            detected, cycle_len = self._loop_detector.detect(state)
            if detected:
                return _configured(
                    cfg.loop_action,
                    ControlTrigger.LOOP_DETECTED,
                    f"loop_detected.cycle_len={cycle_len}",
                    (
                        f"Repeated state detected (cycle length {cycle_len}, "
                        f"threshold {cfg.loop_threshold})"
                    ),
                    metadata={"cycle_len": cycle_len},
                )
        if cfg.duplicate_enabled:
            detected, key = self._duplicate_detector.detect(state)
            if detected:
                return _configured(
                    cfg.duplicate_action,
                    ControlTrigger.DUPLICATE_TOOL_CALL,
                    f"duplicate_tool_call.{key}",
                    f"Tool '{key}' called with identical arguments/results more "
                    f"than {cfg.duplicate_max_identical} times without progress",
                )
        return None

    def _check_risk(self, risk: RiskAssessment) -> ControlDecision:
        """Risk escalation for HIGH/CRITICAL gateway tools (step 9)."""
        cfg = self.config
        if risk.critical_calls > 0 or risk.high_risk_calls > 0:
            is_critical = risk.critical_calls > 0
            action = cfg.risk_critical_action if is_critical else cfg.risk_high_action
            decision = _configured(
                action,
                ControlTrigger.RISK_ESCALATION,
                "risk_escalation.critical" if is_critical else "risk_escalation.high",
                (
                    f"{risk.critical_calls} CRITICAL-risk call(s)"
                    if is_critical
                    else f"{risk.high_risk_calls} HIGH-risk call(s)"
                ),
            )
            decision.metadata["highest_risk"] = (
                risk.highest_risk_seen.value if risk.highest_risk_seen else None
            )
            return decision
        return continue_decision()

    def _check_diminishing(
        self, signal: ProgressSignal
    ) -> tuple[str, str] | None:
        """Detect strictly-decreasing improvements converging on zero."""
        cfg = self.config
        history = signal.score_improvements
        if len(history) < cfg.diminishing_min_deltas:
            return None
        recent = history[-cfg.diminishing_min_deltas :]
        strictly_decreasing = all(
            recent[i] > recent[i + 1] >= 0 for i in range(len(recent) - 1)
        )
        if not strictly_decreasing:
            return None
        if recent[-1] > cfg.diminishing_threshold:
            return None
        code = (
            f"diminishing_returns.latest={recent[-1]:.6g}"
            f"_min_deltas={cfg.diminishing_min_deltas}"
        )
        reason = "Improvements shrinking: " + ", ".join(f"{d:.6g}" for d in recent)
        return code, reason


def _configured(
    action: ControlAction,
    trigger: ControlTrigger,
    reason_code: str,
    reason: str,
    *,
    metadata: dict[str, object] | None = None,
) -> ControlDecision:
    """Build a decision for a configured non-hard-limit action."""
    kwargs: dict[str, object] = {}
    if metadata:
        kwargs["metadata"] = metadata
    return ControlDecision(
        action=action,
        trigger=trigger,
        reason_code=reason_code,
        reason=reason,
        mandatory=True,
        **kwargs,
    )


__all__ = [
    "AutonomyPolicy",
    "SafetyPolicy",
    "RuleBasedSafetyPolicy",
]
