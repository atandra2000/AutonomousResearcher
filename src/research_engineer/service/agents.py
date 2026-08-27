"""E7 - Agent factories used by the worker to drive the E1 ``AgentRuntime``.

The production default is a deterministic, LLM-free planning agent built on
the existing :class:`~research_engineer.runtime.adapters.AgentAdapter`
pattern: it decomposes the submitted goal into checklist steps and completes
them step by step so the full runtime loop (plan -> act -> observe ->
evaluate), checkpointing (E2), and observability (E6) are exercised without
requiring LLM credentials.

The E3 ToolGateway and E5 SafetyController are deployment-specific: the
worker wires whatever the factory's runtime assembly provides. The default
service stack does not attach a gateway or safety controller, so any run
whose custom factory attempts a tool call fails closed in
``AgentRuntime.call_tool`` rather than bypassing policy enforcement.
Custom factories that need the full safety chain must wire
gateway + controller into their returned adapter/policy themselves.

Custom factories can be registered with
:meth:`AgentFactoryRegistry.register` keyed by an ``agent_kind`` value
supplied in the run request metadata.
"""

from __future__ import annotations

import re
from typing import Any

from research_engineer.runtime.adapters import AgentAdapter
from research_engineer.runtime.models import AgentBudget, AgentPolicy

DEFAULT_AGENT_KIND = "planning_checklist"


def _sentence_chunks(text: str, max_len: int = 80) -> list[str]:
    """Split goal text into short bullet-like chunks."""
    sentences = [s.strip() for s in re.split(r"[.;\n]+", text) if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        if len(sentence) <= max_len:
            chunks.append(sentence)
            continue
        words, buf = sentence.split(), ""
        for word in words:
            candidate = f"{buf} {word}".strip()
            if len(candidate) > max_len:
                if buf:
                    chunks.append(buf)
                buf = word
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
    return chunks or [text[:max_len]]


class PlanningChecklistAdapter(AgentAdapter):
    """Deterministic multi-step adapter over a submitted goal."""

    def __init__(self, step_delay_seconds: float = 0.0) -> None:
        super().__init__(agent_name="service_planning_checklist", invoke=self._act)
        self._step_delay = max(0.0, step_delay_seconds)

    async def _act(self, goal: str, context: Any) -> Any:
        if self._step_delay > 0:
            import asyncio

            await asyncio.sleep(self._step_delay)
        items = self.checklist(context.goal)
        done_count = min(len(context.steps), len(items))
        return {
            "completed_item": items[done_count - 1] if done_count else None,
            "next_item": items[done_count] if done_count < len(items) else None,
            "total_items": len(items),
            "completed": done_count,
        }

    @staticmethod
    def checklist(goal: str) -> list[str]:
        return _sentence_chunks(goal)[:8] or ["analyse goal"]

    async def planner(self, ctx: Any) -> Any:
        return {"checklist": self.checklist(ctx.goal)}

    async def observer(self, ctx: Any, action: Any) -> Any:
        return action

    async def evaluator(self, ctx: Any, observation: Any) -> Any:
        total = observation.get("total_items", 1) if isinstance(observation,
                                                                dict) else 1
        completed = len(ctx.steps)
        score = min(1.0, completed / max(1, total))
        return {"done": completed >= total, "score": score}


AgentFactoryReturn = tuple[AgentAdapter, AgentPolicy]


class AgentFactoryRegistry:
    """Maps ``agent_kind`` metadata values to async agent factories."""

    def __init__(
        self,
        config_max_steps: int = 25,
        config_max_runtime_seconds: float = 600.0,
        config_step_delay_seconds: float = 0.0,
    ) -> None:
        self._factories: dict[str, Any] = {}
        self.config_max_steps = config_max_steps
        self.config_max_runtime_seconds = config_max_runtime_seconds
        self.config_step_delay_seconds = config_step_delay_seconds
        if config_step_delay_seconds > 0:
            self.register(DEFAULT_AGENT_KIND, self._delayed_default_factory)
        else:
            self.register(DEFAULT_AGENT_KIND, self._default_factory)

    def register(
        self,
        kind: str,
        factory: Any,
    ) -> None:
        """Register ``factory(overrides)`` returning ``(adapter, policy)``."""
        self._factories[kind] = factory

    async def build(
        self,
        agent_kind: str,
        overrides: dict[str, Any],
    ) -> AgentFactoryReturn:
        factory = self._factories.get(agent_kind)
        if factory is None:
            raise ValueError(f"Unknown agent_kind {agent_kind!r}")
        adapter, _base_policy = await factory(overrides)
        # P1: apply the configured per-step throttle to every adapter kind
        # (not just the default one) so crash/resume pilots observe a real
        # mid-flight window. Adapters opt in by exposing the attribute.
        if self.config_step_delay_seconds > 0 and hasattr(
            adapter, "step_delay_seconds"
        ):
            adapter.step_delay_seconds = float(self.config_step_delay_seconds)
        budget = AgentBudget(
            max_steps=int(
                overrides.get("max_steps", self.config_max_steps)
            ),
            max_runtime_seconds=float(
                overrides.get(
                    "max_runtime_seconds", self.config_max_runtime_seconds
                )
            ),
            # P2: full budget surface for LLM-backed tiers (token/cost/
            # tool-call budgets are enforced by the runtime between phases).
            max_tool_calls=(
                int(overrides["max_tool_calls"])
                if "max_tool_calls" in overrides else None
            ),
            max_tokens=(
                int(overrides["max_tokens"])
                if "max_tokens" in overrides else None
            ),
            max_cost_usd=(
                float(overrides["max_cost_usd"])
                if "max_cost_usd" in overrides else None
            ),
        )
        policy = AgentPolicy(budget=budget)
        if "max_recoverable_errors" in overrides:
            policy.max_recoverable_errors = int(
                overrides["max_recoverable_errors"]
            )
        return adapter, policy

    @staticmethod
    async def _default_factory(
        overrides: dict[str, Any],
    ) -> AgentFactoryReturn:
        return PlanningChecklistAdapter(), AgentPolicy()

    async def _delayed_default_factory(
        self, overrides: dict[str, Any]
    ) -> AgentFactoryReturn:
        return (
            PlanningChecklistAdapter(self.config_step_delay_seconds),
            AgentPolicy(),
        )


__all__ = [
    "DEFAULT_AGENT_KIND",
    "AgentFactoryRegistry",
    "PlanningChecklistAdapter",
]
