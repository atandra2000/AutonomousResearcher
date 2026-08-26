"""E1 - Agent adapters for runtime compatibility.

Wraps existing agents so they can run through the generic
:class:`~research_engineer.runtime.runtime.AgentRuntime` unchanged. Each
adapter exposes the four phase callables (planner, actor, observer,
evaluator) that the runtime drives, delegating to the wrapped agent's
existing async entry point.

This satisfies the E1 requirement that the runtime be *compatible with
existing agents* without modifying them. The adapter pattern mirrors the
Phase 13 delegation adapters in
:mod:`research_engineer.agents._adapters`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from research_engineer.runtime.models import AgentContext


class AgentAdapter:
    """Wrap an existing agent's async entry point as runtime phase callables.

    The wrapped agent is invoked once per step as the ``actor``; the
    ``planner``, ``observer``, and ``evaluator`` are no-ops that pass the
    goal/context through. This lets a single-shot agent (e.g.
    :class:`~research_engineer.agents.task_agent.TaskAgent`) run inside the
    runtime's lifecycle, budget, and observability machinery.

    Args:
        agent_name: Stable identifier for the wrapped agent.
        invoke: async callable that runs the agent. By default it is called
            with ``goal`` and ``context`` keyword arguments. Pass
            ``arg_mapper`` to adapt the runtime's ``(goal, context)`` to the
            agent's actual signature (e.g. ``RepositoryAgent.analyze`` takes
            ``repo_path``/``output_dir``).
        arg_mapper: optional ``(goal, context) -> dict`` returning the
            keyword arguments for ``invoke``. When omitted, defaults to
            ``{"goal": goal, "context": context}``.
    """

    def __init__(
        self,
        agent_name: str,
        invoke: Callable[..., Awaitable[Any]],
        arg_mapper: Callable[[str, AgentContext], dict[str, Any]] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self._invoke = invoke
        self._arg_mapper = arg_mapper or (
            lambda goal, ctx: {"goal": goal, "context": ctx}
        )

    async def planner(self, ctx: AgentContext) -> Any:
        """Plan phase: pass the goal through as the plan."""
        return {"goal": ctx.goal}

    async def actor(self, ctx: AgentContext, plan: Any) -> Any:
        """Act phase: invoke the wrapped agent."""
        return await self._invoke(**self._arg_mapper(ctx.goal, ctx))

    async def observer(self, ctx: AgentContext, action: Any) -> Any:
        """Observe phase: pass the action through as the observation."""
        return action

    async def evaluator(self, ctx: AgentContext, observation: Any) -> Any:
        """Evaluate phase: mark the run complete with the agent's output."""
        return {"done": True, "output": observation}

    def bind(self) -> tuple[Callable[..., Awaitable[Any]], ...]:
        """Return ``(planner, actor, observer, evaluator)`` for the runtime."""
        return self.planner, self.actor, self.observer, self.evaluator


__all__ = ["AgentAdapter"]
