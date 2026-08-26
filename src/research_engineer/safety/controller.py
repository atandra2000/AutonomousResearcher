"""E5 - Safety controller: the runtime-facing autonomy control plane.

:class:`SafetyController` sits between the E1
:class:`~research_engineer.runtime.runtime.AgentRuntime` loop and the
deterministic :class:`~research_engineer.safety.policies.SafetyPolicy`:

    execute step -> observe result -> evaluate progress + risk
        -> policy decision (CONTINUE / REPLAN / PAUSE_FOR_APPROVAL /
           TERMINATE)

Responsibilities:

* maintain the checkpointable
  :class:`~research_engineer.safety.models.SafetyState` (persisted under
  ``ctx.metadata["safety_state"]`` so E2 checkpoint/resume restores it
  verbatim);
* record tool invocations routed through the E3 ToolGateway, including
  each tool's configured
  :class:`~research_engineer.gateway.models.RiskLevel`;
* build :class:`~research_engineer.safety.models.ProgressSignal` /
  :class:`~research_engineer.safety.models.RiskAssessment` and run the policy;
* resolve PAUSE_FOR_APPROVAL through the configured
  :class:`~research_engineer.safety.approval.PauseApprovalGate`
  (fail closed when no gate is wired);
* emit structured observability events on the existing event bus
  (kind ``agent_runtime``, event ``safety_decision``) — no second
  telemetry system.

The controller never mutates its own policy at runtime: policies are
injected once at construction and treated as read-only.
"""

from __future__ import annotations

import logging
from typing import Any

from research_engineer.gateway.models import RiskLevel
from research_engineer.safety.approval import PauseRequest
from research_engineer.safety.detectors import step_signature, tool_call_key
from research_engineer.safety.models import (
    ControlAction,
    ControlDecision,
    ControlTrigger,
    DecisionRecord,
    ProgressSignal,
    RiskAssessment,
    SafetyState,
    ToolCallRecord,
    continue_decision,
)
from research_engineer.safety.policies import (
    AutonomyPolicy,
    RuleBasedSafetyPolicy,
    SafetyPolicy,
)

logger = logging.getLogger(__name__)

#: Metadata key under which the serialized safety state is checkpointed.
SAFETY_STATE_METADATA_KEY = "safety_state"


def _epsilon() -> float:
    return 1e-12


def _budget_usage(ctx: Any, budget: Any) -> dict[str, float]:
    """Fractions of each budget dimension consumed (empty when unbounded)."""
    usage: dict[str, float] = {}
    if budget is None:
        return usage
    if getattr(budget, "max_steps", None):
        usage["steps"] = ctx.current_step / budget.max_steps
    if getattr(budget, "max_tool_calls", None):
        usage["tool_calls"] = ctx.tool_calls / budget.max_tool_calls
    if getattr(budget, "max_tokens", None):
        usage["tokens"] = ctx.tokens / budget.max_tokens
    if getattr(budget, "max_cost_usd", None):
        usage["cost_usd"] = ctx.cost_usd / budget.max_cost_usd
    return usage


class SafetyController:
    """Deterministic autonomy/safety control plane.

    Args:
        policy: :class:`AutonomyPolicy` configuration (read-only after
            construction; the agent cannot modify its own policies).
        safety_policy: Optional custom :class:`SafetyPolicy`; defaults to
            the deterministic :class:`RuleBasedSafetyPolicy`.
        approval_gate: Optional gate consulted on PAUSE_FOR_APPROVAL.
            When omitted, pauses fail closed to TERMINATE/APPROVAL_REQUIRED.
        llm_advisor: Optional async ``(signal, risk) -> str`` hook whose
            output is attached as an advisory note only. It can NEVER
            change a decision's action.
        event_bus: Optional event bus; defaults to the process-wide bus.
    """

    def __init__(
        self,
        policy: AutonomyPolicy | None = None,
        *,
        safety_policy: SafetyPolicy | None = None,
        approval_gate: Any | None = None,
        llm_advisor: Any | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.policy = policy or AutonomyPolicy()
        self._safety_policy = safety_policy or RuleBasedSafetyPolicy(self.policy)
        self._approval_gate = approval_gate
        self._llm_advisor = llm_advisor
        self._event_bus = event_bus
        self._states: dict[str, SafetyState] = {}

    # ------------------------------------------------------------------
    # State management (checkpointable across resume)
    # ------------------------------------------------------------------

    def get_state(self, ctx: Any) -> SafetyState:
        """Return the state for this run, restoring from a checkpoint.

        Restoration happens lazily from ``ctx.metadata`` so a context
        resumed via E2 keeps its full detector history.
        """
        exec_id = ctx.execution_id
        cached = self._states.get(exec_id)
        if cached is not None:
            return cached
        raw = ctx.metadata.get(SAFETY_STATE_METADATA_KEY)
        if isinstance(raw, dict):
            try:
                validated = SafetyState.model_validate(raw)
                if not isinstance(validated, SafetyState):
                    raise ValueError("not a SafetyState")
                state = validated
                if not state.run_id:
                    state.run_id = exec_id
                self._states[exec_id] = state
                return state
            except Exception:  # noqa: BLE001 - corrupt state starts fresh
                logger.warning(
                    "Corrupt safety state in metadata for %s; starting fresh",
                    exec_id,
                )
        state = SafetyState(run_id=exec_id)
        self._states[exec_id] = state
        return state

    def _persist(self, ctx: Any, state: SafetyState) -> None:
        """Snapshot the state into the context for E2 checkpointing."""
        ctx.metadata[SAFETY_STATE_METADATA_KEY] = state.model_dump(mode="json")

    def reset_run(self, execution_id: str) -> None:
        """Drop in-memory state for a finished run."""
        self._states.pop(execution_id, None)

    # ------------------------------------------------------------------
    # Step observation -> control decision
    # ------------------------------------------------------------------

    async def observe_step(
        self,
        ctx: Any,
        step: Any,
        *,
        budget: Any = None,
    ) -> ControlDecision:
        """Observe one completed step and produce the control decision.

        Args:
            ctx: The live runtime context.
            step: The just-completed step.
            budget: Optional runtime ``AgentBudget`` used for budget-usage
                escalation signals.
        """
        state = self.get_state(ctx)

        # --- Update the deterministic behaviour history ---
        signature = step_signature(step)
        state.push_signature(signature)
        if step.error is not None:
            state.consecutive_failures += 1
        else:
            state.consecutive_failures = 0
            state.replan_requested_for_streak = False

        signal = self._build_signal(ctx, step, state, budget)
        risk = self._assess_risk(state)

        base = self._safety_policy.evaluate(signal, risk, state)
        decision = await self._resolve(base, signal, risk, ctx, state)

        # Advisory note: LLM reasoning may annotate but never override.
        if self._llm_advisor is not None and decision.continues:
            note = await self._safe_advise(signal, risk)
            if note:
                decision.advisory_note = note

        self._record(ctx, state, decision)
        return decision

    def _build_signal(
        self,
        ctx: Any,
        step: Any,
        state: SafetyState,
        budget: Any,
    ) -> ProgressSignal:
        """Derive the progress signal from the context + safety state."""
        improvements: list[float] = []
        running_best: float | None = None
        for s in ctx.steps:
            if s.score is None:
                continue
            if running_best is None:
                running_best = s.score
            elif abs(s.score - running_best) > _epsilon():
                improvements.append(abs(s.score - running_best))
                running_best = s.score
        return ProgressSignal(
            step=step.step,
            score=step.score,
            previous_best=running_best,
            stagnation_count=ctx.stagnation_count,
            consecutive_failures=state.consecutive_failures,
            steps_taken=ctx.current_step,
            tool_calls=ctx.tool_calls,
            tokens=ctx.tokens,
            cost_usd=round(ctx.cost_usd, 6),
            replans=state.replans,
            budget_usage=_budget_usage(ctx, budget),
            observation_signature=step_signature(step),
            score_improvements=improvements,
        )

    @staticmethod
    def _assess_risk(state: SafetyState) -> RiskAssessment:
        """Aggregate risk levels and policy failures from tool records.

        Calls at (or below) the highest human-approved risk level are
        suppressed from escalation counting — an explicit approval covers
        future calls at that level. Policy/security failures always count.
        """
        order = [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]
        approved_index: int | None = None
        if state.approved_risk is not None:
            try:
                approved_index = order.index(RiskLevel(state.approved_risk))
            except ValueError:
                approved_index = None

        highest: RiskLevel | None = None
        high = critical = failures = 0
        details: list[str] = []
        for record in state.tool_calls[-32:]:
            try:
                level = RiskLevel(record.risk_level)
            except ValueError:  # unknown level — treat conservatively as LOW
                level = RiskLevel.LOW
            if (
                approved_index is not None
                and not record.is_policy_failure
                and order.index(level) <= approved_index
            ):
                continue  # covered by a prior human approval
            if highest is None or order.index(level) > order.index(highest):
                highest = level
            if level == RiskLevel.HIGH:
                high += 1
            elif level == RiskLevel.CRITICAL:
                critical += 1
            if record.is_policy_failure:
                failures += 1
                details.append(f"policy failure: {record.tool_name}")
        return RiskAssessment(
            highest_risk_seen=highest,
            high_risk_calls=high,
            critical_calls=critical,
            policy_failures=failures,
            details=details,
        )

    async def _resolve(
        self,
        decision: ControlDecision,
        signal: ProgressSignal,
        risk: RiskAssessment,
        ctx: Any,
        state: SafetyState,
    ) -> ControlDecision:
        """Post-process a policy decision before it reaches the runtime."""
        cfg = self.policy

        # REPLAN: count replans deterministically; exhaust -> terminate.
        if decision.action == ControlAction.REPLAN:
            if state.replans >= cfg.max_replans:
                return ControlDecision(
                    action=ControlAction.TERMINATE,
                    trigger=ControlTrigger.REPLAN_LIMIT,
                    reason_code=f"replan_limit.replans={state.replans}",
                    reason=(
                        f"Replan limit reached ({cfg.max_replans}); last trigger: "
                        f"{decision.reason}"
                    ),
                    metadata={"last_reason_code": decision.reason_code},
                )
            state.replans += 1
            if decision.trigger == ControlTrigger.FAILURE_ESCALATION:
                state.replan_requested_for_streak = True
            ctx.metadata["request_replan"] = True

        # PAUSE_FOR_APPROVAL: consult the gate; fail closed without one.
        if decision.action == ControlAction.PAUSE_FOR_APPROVAL:
            request = PauseRequest(
                run_id=ctx.execution_id,
                step=signal.step,
                trigger=decision.trigger,
                reason_code=decision.reason_code,
                reason=decision.reason,
                metadata=dict(decision.metadata),
            )
            if self._approval_gate is None:
                return ControlDecision(
                    action=ControlAction.TERMINATE,
                    trigger=ControlTrigger.APPROVAL_REQUIRED,
                    reason_code=f"approval_required.{decision.reason_code}",
                    reason=(
                        "Policy requires human approval and no approval gate is "
                        f"configured: {decision.reason}"
                    ),
                    metadata={"pause_request": request.model_dump(mode="json")},
                )
            approved = bool(await self._approval_gate.approve_pause(request))
            if not approved:
                return ControlDecision(
                    action=ControlAction.TERMINATE,
                    trigger=ControlTrigger.APPROVAL_DENIED,
                    reason_code=f"approval_denied.{decision.reason_code}",
                    reason=f"Human approval denied: {decision.reason}",
                    metadata={"pause_request": request.model_dump(mode="json")},
                )
            decision = continue_decision(
                reason_code=f"approval_granted.{decision.reason_code}",
                reason=f"Human approval granted after pause: {decision.reason}",
            )
            # Record the approved level so equivalent calls don't re-pause.
            highest = decision.metadata.get("highest_risk") or None
            order = [lvl.value for lvl in RiskLevel]
            if highest in order and (
                state.approved_risk is None
                or order.index(highest) > order.index(state.approved_risk)
            ):
                state.approved_risk = highest
            ctx.metadata["human_interventions"] = (
                int(ctx.metadata.get("human_interventions", 0)) + 1
            )
        return decision

    def _record(
        self, ctx: Any, state: SafetyState, decision: ControlDecision
    ) -> None:
        """Log the decision, persist state, and emit an observability event."""
        state.push_decision(
            DecisionRecord(
                step=ctx.current_step,
                action=decision.action,
                trigger=decision.trigger,
                reason_code=decision.reason_code,
            )
        )
        self._persist(ctx, state)
        try:
            bus = self._event_bus
            if bus is None:
                from research_engineer.observability import get_event_bus

                bus = get_event_bus()
            bus.emit(
                {
                    "kind": "agent_runtime",
                    "event": "safety_decision",
                    "execution_id": ctx.execution_id,
                    "step": ctx.current_step,
                    "action": decision.action.value,
                    "trigger": decision.trigger.value,
                    "reason_code": decision.reason_code,
                    "mandatory": decision.mandatory,
                    "warnings": list(decision.warnings),
                }
            )
        except Exception:  # noqa: BLE001 - observability must not break the run
            logger.debug("Failed to emit safety_decision event", exc_info=True)

    # ------------------------------------------------------------------
    # Tool-call recording (fed by AgentRuntime.call_tool / E3 gateway)
    # ------------------------------------------------------------------

    def record_tool_call(
        self,
        ctx: Any,
        tool_name: str,
        tool_input: Any,
        result: Any,
        *,
        risk_level: RiskLevel | str | None = None,
    ) -> None:
        """Record a gateway-dispatched tool invocation for later analysis.

        ``result`` may be a gateway ``ToolExecutionResult`` (its
        ``status``/``output`` are used) or any object; failures here are
        logged but never propagate — recording must never break dispatch.
        """
        try:
            state = self.get_state(ctx)
            key = tool_call_key(tool_name, tool_input)
            args_hash = key.rsplit(":", 1)[-1]
            status = getattr(result, "status", "")
            status_value = getattr(status, "value", status) or ""
            output = getattr(result, "output", None)
            from research_engineer.safety.detectors import _stable_hash

            record = ToolCallRecord(
                tool_name=tool_name,
                args_hash=args_hash,
                result_signature=_stable_hash(output),
                status=str(status_value),
                is_policy_failure=bool(getattr(result, "is_policy_failure", False)),
                risk_level=risk_level.value
                if isinstance(risk_level, RiskLevel)
                else str(risk_level or RiskLevel.LOW.value),
            )
            state.push_tool_call(record)
            counts_key = f"{tool_name}:{args_hash}"
            state.identical_counts[counts_key] = (
                state.identical_counts.get(counts_key, 0) + 1
            )
            # Keep the counts map bounded alongside its window.
            if len(state.identical_counts) > 256:
                for stale in sorted(state.identical_counts)[:64]:
                    del state.identical_counts[stale]
            self._persist(ctx, state)
        except Exception:  # noqa: BLE001 - best effort by design
            logger.debug("Failed to record tool call", exc_info=True)

    async def _safe_advise(self, signal: ProgressSignal, risk: RiskAssessment) -> str:
        """Best-effort LLM advisory note (never authoritative)."""
        if self._llm_advisor is None:
            return ""
        try:
            note = await self._llm_advisor(signal, risk)
            return str(note or "")
        except Exception:  # noqa: BLE001 - advisory must never break control
            logger.debug("LLM advisor failed", exc_info=True)
            return ""


__all__ = [
    "SAFETY_STATE_METADATA_KEY",
    "SafetyController",
]


