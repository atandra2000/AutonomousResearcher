"""Tests for E5 - Autonomy & Safety Controls.

Covers the deterministic safety/autonomy layer: loop/cycle detection,
duplicate tool calls, no-progress termination, diminishing returns,
failure escalation, risk escalation with the E3 ToolGateway, approval
gating, budget warnings, deterministic termination, replanning, E2
checkpoint/resume of safety state, and false-positive avoidance.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from research_engineer.gateway import (
    RiskLevel,
    ToolGateway,
    ToolGatewayConfig,
)
from research_engineer.observability import NullSink, get_event_bus, reset_event_bus
from research_engineer.runtime import (
    AgentBudget,
    AgentContext,
    AgentPolicy,
    AgentRuntime,
    AgentStep,
    AgentTermination,
    InMemoryCheckpointStore,
)
from research_engineer.safety import (
    AutonomyPolicy,
    CallbackPauseGate,
    ControlAction,
    ControlTrigger,
    DuplicateToolCallDetector,
    LoopDetector,
    PauseRequest,
    SafetyController,
    SafetyState,
)
from research_engineer.tools.base import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_planner(ctx: AgentContext) -> Any:
    return {"goal": ctx.goal}


async def _noop_actor(ctx: AgentContext, plan: Any) -> Any:
    return {"action": "a"}


async def _noop_observer(ctx: AgentContext, action: Any) -> Any:
    return {"obs": action}


async def _done_evaluator(ctx: AgentContext, observation: Any) -> Any:
    return {"done": True, "output": observation}


def _controller(policy: AutonomyPolicy | None = None) -> SafetyController:
    return SafetyController(policy or AutonomyPolicy())


def _runtime(
    controller: SafetyController | None,
    *,
    planner: Any = None,
    actor: Any = None,
    observer: Any = None,
    evaluator: Any = None,
    runtime_policy: AgentPolicy | None = None,
    **kwargs: Any,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner or _noop_planner,
        actor=actor or _noop_actor,
        observer=observer or _noop_observer,
        evaluator=evaluator or _done_evaluator,
        policy=runtime_policy or AgentPolicy(stagnation_window=100),
        safety_controller=controller,
        **kwargs,
    )


class EchoTool(Tool[Any, Any]):
    """A trivial tool that echoes its input back."""

    async def execute(self, input: Any) -> Any:
        return {"echo": input}

    async def validate(self, input: Any) -> bool:
        return True


def _fake_result(*, output: Any = {"ok": 1}, status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(status=status, output=output, is_policy_failure=False)


@pytest.fixture(autouse=True)
def _bus() -> Any:
    reset_event_bus()
    bus = get_event_bus()
    bus.add_sink(NullSink())
    yield bus
    reset_event_bus()


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------


def _tc(name: str, args_hash: str, result_sig: str) -> Any:
    from research_engineer.safety.models import ToolCallRecord

    return ToolCallRecord(
        tool_name=name,
        args_hash=args_hash,
        result_signature=result_sig,
        status="success",
    )


class TestLoopDetector:
    def test_repeated_identical_state(self) -> None:
        state = SafetyState()
        for _ in range(3):
            state.push_signature("same")
        detected, cycle = LoopDetector().detect(state)
        assert detected
        assert cycle == 1

    def test_two_step_cycle(self) -> None:
        state = SafetyState()
        for sig in ["a", "b"] * 2:
            state.push_signature(sig)
        detected, cycle = LoopDetector().detect(state)
        assert detected
        assert cycle == 2

    def test_broken_cycle_not_flagged(self) -> None:
        state = SafetyState()
        for sig in ["a", "b", "a", "c", "a", "d", "a", "e"]:
            state.push_signature(sig)
        detected, _ = LoopDetector().detect(state)
        assert not detected

    def test_too_short_history_not_flagged(self) -> None:
        state = SafetyState()
        state.push_signature("a")
        state.push_signature("b")
        assert not LoopDetector().detect(state)[0]


class TestDuplicateToolCallDetector:
    def test_repeated_identical_calls_detected(self) -> None:
        state = SafetyState()
        for _ in range(3):
            state.push_tool_call(_tc("tool", "h1", "r1"))
        detected, key = DuplicateToolCallDetector(max_identical=2).detect(state)
        assert detected
        assert key == "tool"

    def test_different_results_not_flagged(self) -> None:
        """Same args but different outputs means progress -> no false positive."""
        state = SafetyState()
        for i in range(3):
            state.push_tool_call(_tc("tool", "h1", f"r{i}"))
        detected, _ = DuplicateToolCallDetector(max_identical=2).detect(state)
        assert not detected

    def test_different_args_not_flagged(self) -> None:
        state = SafetyState()
        for i in range(3):
            state.push_tool_call(_tc("tool", f"h{i}", "r1"))
        assert not DuplicateToolCallDetector(max_identical=2).detect(state)[0]

    def test_interleaved_calls_not_flagged(self) -> None:
        state = SafetyState()
        for i in range(3):
            state.push_tool_call(_tc(f"tool{i}", "h", "r"))


# ---------------------------------------------------------------------------
# Controller-level tests
# ---------------------------------------------------------------------------


def _step(n: int, eval_value: Any = None) -> AgentStep:
    return AgentStep(step=n, evaluation=eval_value)


def _observe(
    ctx: AgentContext,
    controller: SafetyController,
    step: AgentStep,
    budget: Any = None,
) -> Any:
    ctx.steps.append(step)
    ctx.current_step += 1
    return asyncio.run(controller.observe_step(ctx, step, budget=budget))


class TestControllerControls:
    def test_disabled_policy_always_continues(self) -> None:
        controller = SafetyController(AutonomyPolicy(enabled=False))
        ctx = AgentContext(goal="g")
        for i in range(6):
            decision = _observe(ctx, controller, _step(i + 1, eval_value={"x": 1}))
            assert decision.action == ControlAction.CONTINUE

    def test_duplicate_tool_calls_trigger_replan(self) -> None:
        policy = AutonomyPolicy(loop_enabled=False)
        controller = SafetyController(policy)
        ctx = AgentContext(goal="g")
        for i in range(2):
            controller.record_tool_call(ctx, "search", {"q": 1}, _fake_result())
        # Two identical calls are still within the limit.
        decision = _observe(ctx, controller, _step(1))
        assert decision.action == ControlAction.CONTINUE

        controller.record_tool_call(ctx, "search", {"q": 1}, _fake_result())
        decision = _observe(ctx, controller, _step(2))
        assert decision.action == ControlAction.REPLAN
        assert decision.trigger == ControlTrigger.DUPLICATE_TOOL_CALL
        assert "duplicate_tool_call.search" in decision.reason_code

    def test_duplicate_with_new_results_avoids_false_positive(self) -> None:
        """Identical args but fresh results are legitimate progress."""
        policy = AutonomyPolicy(loop_enabled=False)
        controller = SafetyController(policy)
        ctx = AgentContext(goal="g")
        for i in range(4):
            controller.record_tool_call(
                ctx, "search", {"q": 1}, _fake_result(output={"n": i})
            )
            decision = _observe(ctx, controller, _step(i + 1))
            assert decision.action == ControlAction.CONTINUE

    def test_cyclic_runtime_behavior_detected(self) -> None:
        controller = SafetyController(AutonomyPolicy(duplicate_enabled=False))

        async def evaluator(c: AgentContext, obs: Any) -> Any:
            return {"done": False, "output": c.current_step % 2}

        rt = _runtime(controller, evaluator=evaluator)
        execution = asyncio.run(rt.run("loop forever"))
        assert execution.termination == AgentTermination.SAFETY_TERMINATED
        assert "replan_limit" in execution.reason
        assert execution.context.metadata.get("request_replan") is True
        assert execution.context.current_step > 0

    def test_no_progress_termination(self) -> None:
        policy = AutonomyPolicy(no_progress_stagnation_limit=3)
        controller = SafetyController(policy)

        async def evaluator(c: AgentContext, obs: Any) -> float:
            return 0.5  # never improves

        rt = _runtime(controller, evaluator=evaluator)
        execution = asyncio.run(rt.run("stuck"))
        assert execution.termination == AgentTermination.NO_PROGRESS
        assert "no_progress.stagnation=" in execution.reason

    def test_diminishing_returns_termination(self) -> None:
        scores = [0.0, 1.0, 1.4, 1.45]
        policy = AutonomyPolicy(diminishing_threshold=0.1, diminishing_min_deltas=3)
        controller = SafetyController(policy)

        async def evaluator(c: AgentContext, obs: Any) -> float:
            return scores[min(c.current_step, len(scores) - 1)]

        rt = _runtime(controller, evaluator=evaluator)
        execution = asyncio.run(rt.run("converging"))
        assert execution.termination == AgentTermination.NO_PROGRESS
        assert "diminishing_returns" in execution.reason

    def test_failure_escalation_replan_then_terminate(self) -> None:
        controller = SafetyController(AutonomyPolicy(failure_consecutive_limit=2))

        async def failing_actor(c: AgentContext, plan: Any) -> Any:
            raise ConnectionError("transient network down")

        rt = _runtime(controller, actor=failing_actor)
        execution = asyncio.run(rt.run("failing"))
        assert execution.termination == AgentTermination.SAFETY_TERMINATED
        assert "failure_escalation.exhausted" in execution.reason
        # A replan was escalated once before terminating.
        state = controller.get_state(execution.context)
        assert state.replans >= 1


class TestBudgetAndRisk:
    def test_budget_warning_continues_with_warnings(self) -> None:
        controller = SafetyController(
            AutonomyPolicy(loop_enabled=False, duplicate_enabled=False)
        )
        ctx = AgentContext(goal="g")
        ctx.tokens = 85
        budget = AgentBudget(max_tokens=100)

        decision = _observe(ctx, controller, _step(1), budget=budget)
        assert decision.action == ControlAction.CONTINUE
        assert decision.trigger == ControlTrigger.BUDGET_WARNING
        assert any("tokens" in w for w in decision.warnings)

    def test_budget_warning_configurable_action(self) -> None:
        controller = SafetyController(
            AutonomyPolicy(loop_enabled=False, budget_on_warning=ControlAction.PAUSE_FOR_APPROVAL)
        )
        ctx = AgentContext(goal="g")
        ctx.tokens = 85

        async def approve(request: PauseRequest) -> bool:
            return True

        controller._approval_gate = CallbackPauseGate(approve)
        decision = _observe(ctx, controller, _step(1), budget=AgentBudget(max_tokens=100))
        assert decision.action == ControlAction.CONTINUE
        assert "approval_granted" in decision.reason_code
        assert ctx.metadata["human_interventions"] == 1

    def test_high_risk_tool_triggers_approval_required_without_gate(self) -> None:
        controller = SafetyController(
            AutonomyPolicy(loop_enabled=False, duplicate_enabled=False)
        )
        ctx = AgentContext(goal="g")
        gateway_result = _fake_result()
        gateway_result.is_policy_failure = False
        controller.record_tool_call(
            ctx, "deploy", {"env": "prod"}, gateway_result,
            risk_level=RiskLevel.HIGH,
        )
        decision = _observe(ctx, controller, _step(1))
        # Fail closed: PAUSE becomes TERMINATE/APPROVAL_REQUIRED.
        assert decision.action == ControlAction.TERMINATE
        assert decision.trigger == ControlTrigger.APPROVAL_REQUIRED

    def test_high_risk_with_approving_gate_continues(self) -> None:
        controller = SafetyController(
            AutonomyPolicy(loop_enabled=False, duplicate_enabled=False),
            approval_gate=CallbackPauseGate(lambda r: _async_true()),
        )
        ctx = AgentContext(goal="g")
        controller.record_tool_call(
            ctx, "deploy", {"env": "prod"}, _fake_result(),
            risk_level=RiskLevel.HIGH,
        )
        first = _observe(ctx, controller, _step(1))
        assert first.action == ControlAction.CONTINUE
        assert "approval_granted" in first.reason_code
        # A second observation does not re-pause (approval covers the level).
        second = _observe(ctx, controller, _step(2))
        assert second.action == ControlAction.CONTINUE

    def test_policy_failure_is_hard_limit(self) -> None:
        controller = SafetyController(AutonomyPolicy())
        ctx = AgentContext(goal="g")
        denied = SimpleNamespace(status="denied", output=None, is_policy_failure=True)
        controller.record_tool_call(ctx, "rm_rf", {}, denied)
        decision = _observe(ctx, controller, _step(1))
        assert decision.action == ControlAction.TERMINATE
        assert decision.trigger == ControlTrigger.POLICY_VIOLATION
        assert decision.mandatory


async def _async_true() -> bool:
    return True


class TestGatewayInteraction:
    def test_runtime_gateway_risk_escalation(self) -> None:
        gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
        gw.register_tool(EchoTool(), name="risky", risk_level=RiskLevel.HIGH)
        controller = SafetyController(AutonomyPolicy())

        holder: dict[str, Any] = {}

        async def actor(c: AgentContext, plan: Any) -> Any:
            result = await holder["rt"].call_tool("risky", {"x": 1})
            return {"called": result.status.value}

        rt = _runtime(controller, actor=actor, tool_gateway=gw)
        holder["rt"] = rt
        execution = asyncio.run(rt.run("use risky tool"))
        assert execution.termination == AgentTermination.APPROVAL_REQUIRED
        assert "risk_escalation" in execution.reason

    def test_policy_denied_tool_terminates_run(self) -> None:
        gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))  # default deny
        controller = SafetyController(AutonomyPolicy())

        holder: dict[str, Any] = {}

        async def actor(c: AgentContext, plan: Any) -> Any:
            await holder["rt"].call_tool("unregistered", {})
            return {}

        rt = _runtime(controller, actor=actor, tool_gateway=gw)
        holder["rt"] = rt
        execution = asyncio.run(rt.run("try unknown tool"))
        assert execution.termination == AgentTermination.SAFETY_TERMINATED
        assert "policy_violation" in execution.reason

    def test_duplicate_calls_via_gateway_detected(self) -> None:
        gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
        gw.register_tool(EchoTool(), name="echo", risk_level=RiskLevel.LOW)
        controller = SafetyController(
            AutonomyPolicy(no_progress_enabled=False, loop_enabled=False)
        )

        async def never_done(c: AgentContext, obs: Any) -> Any:
            return {"done": False}

        holder: dict[str, Any] = {}

        async def actor(c: AgentContext, plan: Any) -> Any:
            await holder["rt"].call_tool("echo", {"q": 1})
            return {}

        rt = _runtime(controller, actor=actor, evaluator=never_done,
                      tool_gateway=gw)
        holder["rt"] = rt
        execution = asyncio.run(rt.run("spam echo"))
        assert execution.termination == AgentTermination.SAFETY_TERMINATED
        assert "replan_limit" in execution.reason
        triggers = [
            d.trigger for d in controller.get_state(execution.context).decisions
        ]
        assert ControlTrigger.DUPLICATE_TOOL_CALL in triggers


class TestDeterminismAndLifecycle:
    def test_deterministic_termination_reasons(self) -> None:
        async def evaluator(c: AgentContext, obs: Any) -> Any:
            return {"done": False, "output": c.current_step % 2}

        runs = []
        for _ in range(2):
            controller = SafetyController(
                AutonomyPolicy(duplicate_enabled=False)
            )
            rt = _runtime(controller, evaluator=evaluator)
            execution = asyncio.run(rt.run("loop"))
            runs.append((execution.termination.value, execution.reason))
        assert runs[0] == runs[1]
        # Machine-readable reason: "safety:<code>: <text>"
        assert runs[0][1].startswith("safety:replan_limit.")

    def test_successful_execution_unaffected(self) -> None:
        """A normal progressing agent is untouched by safety controls."""
        scores = [1.0, 2.0, 3.0]

        async def evaluator(c: AgentContext, obs: Any) -> Any:
            if c.current_step >= len(scores):
                return {"done": True, "output": "ok", "score": scores[-1]}
            return scores[c.current_step]

        controller = SafetyController()
        rt = _runtime(controller, evaluator=evaluator)
        execution = asyncio.run(rt.run("healthy run"))
        assert execution.termination == AgentTermination.SUCCESS
        state = controller.get_state(execution.context)
        triggers = [d.trigger for d in state.decisions]
        assert all(t == ControlTrigger.NONE for t in triggers)

    def test_checkpoint_resume_preserves_safety_state(self) -> None:
        store = InMemoryCheckpointStore()
        gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
        gw.register_tool(EchoTool(), name="echo", risk_level=RiskLevel.LOW)

        async def never_done(c: AgentContext, obs: Any) -> Any:
            return {"done": False}

        holder: dict[str, Any] = {}

        async def actor(c: AgentContext, plan: Any) -> Any:
            await holder["rt"].call_tool("echo", {"q": c.metadata.get("phase", 1)})
            return {}

        policy = AutonomyPolicy(no_progress_enabled=False, loop_enabled=False)
        controller = SafetyController(policy)
        rt = _runtime(
            controller,
            actor=actor,
            evaluator=never_done,
            tool_gateway=gw,
            checkpoint_store=store,
        )
        holder["rt"] = rt

        def stop_after_two(step: AgentStep) -> None:
            if step.step >= 2:
                rt.cancel()

        rt._on_step = stop_after_two
        asyncio.run(rt.run("resumable run", metadata={"phase": 1}))
        checkpoint = asyncio.run(store.list())[0]

        # Resume with the same controller configuration and repeat phase-2
        # tool calls; the duplicate detector must see history from before.
        controller2 = SafetyController(policy)
        rt2 = _runtime(
            controller2,
            actor=actor,
            evaluator=never_done,
            tool_gateway=gw,
            checkpoint_store=store,
        )
        holder["rt"] = rt2

        async def drive() -> Any:
            restored = await rt2.resume(checkpoint.run_id)
            state = controller2.get_state(restored)
            assert len(state.tool_calls) == 2  # history survived E2
            return await rt2.run(goal="", context=restored)

        execution = asyncio.run(drive())
        assert "replan_limit" in execution.reason
        triggers = [d.trigger for d in controller2.get_state(execution.context).decisions]
        assert ControlTrigger.DUPLICATE_TOOL_CALL in triggers
