"""Tests for E1 - Production Agent Runtime."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from research_engineer.llm import ProviderError
from research_engineer.observability import NullSink, get_event_bus, reset_event_bus
from research_engineer.runtime import (
    AgentAdapter,
    AgentBudget,
    AgentContext,
    AgentError,
    AgentExecution,
    AgentPhase,
    AgentPolicy,
    AgentRuntime,
    AgentState,
    AgentStep,
    AgentTermination,
    classify_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_planner(ctx: AgentContext) -> Any:
    return {"goal": ctx.goal}


async def _noop_actor(ctx: AgentContext, plan: Any) -> Any:
    return {"action": "done"}


async def _noop_observer(ctx: AgentContext, action: Any) -> Any:
    return action


async def _done_evaluator(ctx: AgentContext, observation: Any) -> Any:
    return {"done": True, "output": observation}


def _runtime(
    *,
    planner: Any = None,
    actor: Any = None,
    observer: Any = None,
    evaluator: Any = None,
    policy: AgentPolicy | None = None,
    **kwargs: Any,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner or _noop_planner,
        actor=actor or _noop_actor,
        observer=observer or _noop_observer,
        evaluator=evaluator or _done_evaluator,
        policy=policy,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestRuntimeModels:
    def test_enum_values(self) -> None:
        assert AgentState.CREATED == "created"
        assert AgentState.RUNNING == "running"
        assert AgentState.TERMINATED == "terminated"
        assert AgentTermination.SUCCESS == "success"
        assert AgentTermination.BUDGET_EXCEEDED == "budget_exceeded"
        assert AgentTermination.TIMEOUT == "timeout"
        assert AgentTermination.CANCELLED == "cancelled"
        assert AgentTermination.ERROR == "error"
        assert AgentTermination.NO_PROGRESS == "no_progress"
        assert AgentPhase.PLANNING == "planning"
        assert AgentPhase.ACTING == "acting"
        assert AgentPhase.OBSERVING == "observing"
        assert AgentPhase.EVALUATING == "evaluating"

    def test_budget_defaults_unbounded(self) -> None:
        b = AgentBudget()
        assert b.max_steps is None
        assert b.max_tool_calls is None
        assert b.max_runtime_seconds is None
        assert b.max_cost_usd is None
        assert b.max_tokens is None

    def test_policy_defaults(self) -> None:
        p = AgentPolicy()
        assert p.max_recoverable_errors == 3
        assert p.stagnation_window == 3
        assert p.progress_threshold == 0.0
        assert p.budget.max_steps is None

    def test_context_serializable(self) -> None:
        ctx = AgentContext(goal="test goal")
        ctx.state = AgentState.RUNNING
        ctx.steps.append(AgentStep(step=1, score=0.5))
        ctx.current_step = 1
        data = ctx.model_dump_json()
        restored = AgentContext.model_validate_json(data)
        assert restored.goal == "test goal"
        assert restored.state == AgentState.RUNNING
        assert restored.current_step == 1
        assert restored.steps[0].score == 0.5

    def test_context_terminal_helpers(self) -> None:
        ctx = AgentContext(goal="g")
        assert not ctx.is_terminal()
        assert not ctx.is_success()
        ctx.state = AgentState.TERMINATED
        ctx.termination = AgentTermination.SUCCESS
        assert ctx.is_terminal()
        assert ctx.is_success()

    def test_execution_roundtrip(self) -> None:
        ctx = AgentContext(goal="g")
        exec_ = AgentExecution(
            context=ctx,
            termination=AgentTermination.SUCCESS,
            reason="done",
            output="out",
        )
        d = exec_.model_dump()
        assert d["termination"] == "success"
        assert d["output"] == "out"

    def test_error_model(self) -> None:
        err = AgentError(
            message="boom",
            error_type="ValueError",
            recoverable=True,
            step=1,
            phase=AgentPhase.ACTING,
        )
        assert err.recoverable is True
        assert err.phase == AgentPhase.ACTING


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_timeout_is_recoverable(self) -> None:
        assert classify_error(TimeoutError("t")) is True

    def test_connection_error_is_recoverable(self) -> None:
        assert classify_error(ConnectionError("c")) is True

    def test_value_error_is_fatal(self) -> None:
        assert classify_error(ValueError("v")) is False

    def test_permanent_provider_error_is_fatal(self) -> None:
        err = ProviderError("HTTP 400 bad request", provider="x")
        assert classify_error(err) is False

    def test_transient_provider_error_is_recoverable(self) -> None:
        err = ProviderError("HTTP 500 server error", provider="x")
        assert classify_error(err) is True


# ---------------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------------


class TestRuntimeLifecycle:
    @pytest.mark.asyncio
    async def test_success_path(self) -> None:
        result = await _runtime().run("do the thing")
        assert result.termination == AgentTermination.SUCCESS
        assert result.context.is_success()
        assert result.context.state == AgentState.TERMINATED
        assert result.context.current_step == 1
        assert result.context.goal == "do the thing"

    @pytest.mark.asyncio
    async def test_phase_order(self) -> None:
        phases: list[str] = []

        async def planner(ctx: AgentContext) -> Any:
            phases.append("plan")
            return {}

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            phases.append("act")
            return {}

        async def observer(ctx: AgentContext, action: Any) -> Any:
            phases.append("observe")
            return {}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            phases.append("evaluate")
            return {"done": True}

        await _runtime(
            planner=planner, actor=actor, observer=observer, evaluator=evaluator
        ).run("g")
        assert phases == ["plan", "act", "observe", "evaluate"]

    @pytest.mark.asyncio
    async def test_metadata_attached(self) -> None:
        result = await _runtime().run("g", metadata={"repo": "/tmp/x"})
        assert result.context.metadata["repo"] == "/tmp/x"

    @pytest.mark.asyncio
    async def test_on_step_hook(self) -> None:
        seen: list[AgentStep] = []

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": True}

        await _runtime(evaluator=evaluator, on_step=seen.append).run("g")
        assert len(seen) == 1
        assert seen[0].step == 1

    @pytest.mark.asyncio
    async def test_multi_step_loop(self) -> None:
        calls = {"n": 0}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            calls["n"] += 1
            if calls["n"] < 3:
                return {"done": False}
            return {"done": True}

        result = await _runtime(evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.SUCCESS
        assert result.context.current_step == 3


class TestRuntimeBudgets:
    @pytest.mark.asyncio
    async def test_max_steps_budget(self) -> None:
        policy = AgentPolicy(budget=AgentBudget(max_steps=2))

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": False}  # never done -> budget stops the loop

        result = await _runtime(policy=policy, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.BUDGET_EXCEEDED
        assert result.context.current_step == 2

    @pytest.mark.asyncio
    async def test_max_tool_calls_budget(self) -> None:
        policy = AgentPolicy(budget=AgentBudget(max_tool_calls=1))

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            ctx.tool_calls += 1
            return {}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": False}

        result = await _runtime(policy=policy, actor=actor, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.BUDGET_EXCEEDED
        assert result.context.tool_calls == 1

    @pytest.mark.asyncio
    async def test_max_cost_budget(self) -> None:
        policy = AgentPolicy(budget=AgentBudget(max_cost_usd=0.5))

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            ctx.cost_usd = 0.6
            return {}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": False}

        result = await _runtime(policy=policy, actor=actor, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_max_tokens_budget(self) -> None:
        policy = AgentPolicy(budget=AgentBudget(max_tokens=100))

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            ctx.tokens = 150
            return {}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": False}

        result = await _runtime(policy=policy, actor=actor, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_timeout_budget(self) -> None:
        policy = AgentPolicy(budget=AgentBudget(max_runtime_seconds=0.01))

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            await asyncio.sleep(0.05)
            return {}

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return {"done": False}

        result = await _runtime(policy=policy, actor=actor, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.TIMEOUT


class TestRuntimeCancellation:
    @pytest.mark.asyncio
    async def test_cancel(self) -> None:
        runtime = _runtime()

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            # Cancel from within the actor; the runtime checks between phases.
            runtime.cancel()
            return {}

        # Rebuild the runtime with the cancelling actor so the same instance
        # is both running and cancelled.
        runtime = AgentRuntime(
            planner=_noop_planner,
            actor=actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
        )
        result = await runtime.run("g")
        assert result.termination == AgentTermination.CANCELLED
        assert result.context.state == AgentState.TERMINATED


class TestRuntimeErrors:
    @pytest.mark.asyncio
    async def test_fatal_error_terminates(self) -> None:
        async def actor(ctx: AgentContext, plan: Any) -> Any:
            raise ValueError("fatal")

        result = await _runtime(actor=actor).run("g")
        assert result.termination == AgentTermination.ERROR
        assert result.context.steps[0].error is not None
        assert result.context.steps[0].error.recoverable is False

    @pytest.mark.asyncio
    async def test_recoverable_error_retries(self) -> None:
        calls = {"n": 0}

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("transient")
            return {}

        result = await _runtime(actor=actor).run("g")
        assert result.termination == AgentTermination.SUCCESS
        assert calls["n"] == 2
        assert result.context.recoverable_errors == 1

    @pytest.mark.asyncio
    async def test_too_many_recoverable_errors(self) -> None:
        policy = AgentPolicy(max_recoverable_errors=1)

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            raise TimeoutError("transient")

        result = await _runtime(policy=policy, actor=actor).run("g")
        assert result.termination == AgentTermination.ERROR
        assert result.context.recoverable_errors == 2

    @pytest.mark.asyncio
    async def test_custom_error_classifier(self) -> None:
        calls = {"n": 0}

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("custom-recoverable")
            return {}

        runtime = _runtime(
            actor=actor,
            error_classifier=lambda exc: isinstance(exc, ValueError),
        )
        result = await runtime.run("g")
        assert result.termination == AgentTermination.SUCCESS
        assert calls["n"] == 2


class TestRuntimeNoProgress:
    @pytest.mark.asyncio
    async def test_no_progress_termination(self) -> None:
        policy = AgentPolicy(stagnation_window=2, progress_threshold=0.0)

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return 0.5  # constant score -> no progress

        result = await _runtime(policy=policy, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.NO_PROGRESS
        assert result.context.stagnation_count == 2

    @pytest.mark.asyncio
    async def test_progress_resets_stagnation(self) -> None:
        policy = AgentPolicy(stagnation_window=3, progress_threshold=0.0)
        scores = iter([0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            return next(scores)

        result = await _runtime(policy=policy, evaluator=evaluator).run("g")
        assert result.termination == AgentTermination.NO_PROGRESS
        assert result.context.best_score == 0.9


class TestRuntimeObservability:
    @pytest.mark.asyncio
    async def test_emits_events(self) -> None:
        reset_event_bus()
        bus = get_event_bus()
        events: list[dict[str, Any]] = []
        bus.add_sink(NullSink())
        original = bus.emit
        bus.emit = lambda event: (events.append(event), original(event))[1]  # type: ignore[method-assign]

        try:
            await _runtime().run("g")
        finally:
            bus.emit = original  # type: ignore[method-assign]
            reset_event_bus()

        kinds = [e["event"] for e in events if e.get("kind") == "agent_runtime"]
        assert "start" in kinds
        assert "step" in kinds
        assert "end" in kinds
        assert "terminate" in kinds


# ---------------------------------------------------------------------------
# AgentAdapter compatibility
# ---------------------------------------------------------------------------


class TestAgentAdapter:
    @pytest.mark.asyncio
    async def test_adapter_runs_through_runtime(self) -> None:
        invoked: dict[str, Any] = {}

        async def fake_agent(**kwargs: Any) -> Any:
            invoked.update(kwargs)
            return {"result": "ok"}

        adapter = AgentAdapter("FakeAgent", fake_agent)
        planner, actor, observer, evaluator = adapter.bind()
        runtime = AgentRuntime(planner, actor, observer, evaluator)
        result = await runtime.run("goal")
        assert result.termination == AgentTermination.SUCCESS
        assert invoked["goal"] == "goal"
        assert isinstance(invoked["context"], AgentContext)
        assert result.output == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_adapter_agent_name(self) -> None:
        adapter = AgentAdapter("MyAgent", _noop_actor)
        assert adapter.agent_name == "MyAgent"

    @pytest.mark.asyncio
    async def test_adapter_arg_mapper(self) -> None:
        received: dict[str, Any] = {}

        async def fake_agent(**kwargs: Any) -> Any:
            received.update(kwargs)
            return {"ok": True}

        def mapper(goal: str, ctx: AgentContext) -> dict[str, Any]:
            return {"repo_path": ctx.metadata.get("repo_path"), "goal": goal}

        adapter = AgentAdapter("FakeAgent", fake_agent, arg_mapper=mapper)
        planner, actor, observer, evaluator = adapter.bind()
        runtime = AgentRuntime(planner, actor, observer, evaluator)
        result = await runtime.run("g", metadata={"repo_path": "/tmp/repo"})
        assert result.termination == AgentTermination.SUCCESS
        assert received["repo_path"] == "/tmp/repo"
        assert received["goal"] == "g"


# ---------------------------------------------------------------------------
# Integration: existing agent through the runtime
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_repository_agent_through_runtime(self, tmp_path) -> None:
        """Run the existing RepositoryAgent through the AgentRuntime."""
        from research_engineer.agents.repository_agent import RepositoryAgent

        repo = tmp_path / "sample_repo"
        repo.mkdir()
        (repo / "train.py").write_text(
            "import torch\n\n"
            "def train(model, data):\n"
            "    return model(data)\n"
        )
        (repo / "config.yaml").write_text("learning_rate: 0.001\nepochs: 10\n")

        agent = RepositoryAgent(llm_enabled=False)

        def mapper(goal: str, ctx: AgentContext) -> dict[str, Any]:
            return {
                "repo_path": ctx.metadata.get("repo_path"),
                "output_dir": str(tmp_path / "out"),
                "enable_llm": False,
            }

        adapter = AgentAdapter("RepositoryAgent", agent.analyze, arg_mapper=mapper)
        planner, actor, observer, evaluator = adapter.bind()
        runtime = AgentRuntime(planner, actor, observer, evaluator)
        result = await runtime.run(
            "Analyze the sample repository",
            metadata={"repo_path": str(repo)},
        )
        assert result.termination == AgentTermination.SUCCESS
        assert result.context.is_success()
        assert isinstance(result.output, dict)
        assert result.output.get("repository_name") == "sample_repo"
