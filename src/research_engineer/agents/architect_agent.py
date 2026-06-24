"""Phase 13 - Architect agent.

Specialized agent that produces an implementation plan grounded in
repository memory context and research findings. Uses the reasoning
model (glm-5.2:cloud by config). Falls back to a rule-based plan when
no LLM provider is configured.

Interface: ``async execute(ctx: SharedTaskContext) -> dict``
"""

from __future__ import annotations

from typing import Any

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.agents._streaming import stream_complete
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.delegation import SharedTaskContext


class ArchitectAgent:
    """Produces an implementation plan for the task goal.

    Reads ``ctx.memory_context`` and ``ctx.research_context`` from the
    shared context, reasons about the goal, and writes
    ``ctx.implementation_plan``. The plan is a concise, ordered list
    of steps suitable for the CodingAgent to follow.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "ArchitectAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Produce an implementation plan and write it to ``ctx``."""
        stream_sink = kwargs.get("stream_sink")
        plan = await self._generate_plan(ctx, stream_sink)
        ctx.implementation_plan = plan
        return {
            "summary": plan[:200],
            "plan": plan,
        }

    async def _generate_plan(
        self,
        ctx: SharedTaskContext,
        stream_sink: Any | None = None,
    ) -> str:
        """Generate a plan via LLM or rule-based fallback."""
        provider = self.llm_provider
        if provider is None:
            return self._rule_based_plan(ctx)
        system = (
            "You are a senior software architect. Produce a concise, "
            "ordered implementation plan (3-7 steps) for the given goal. "
            "Each step: one sentence. Ground the plan in the provided "
            "repository memory context and research findings. Do not "
            "write code."
        )
        user_parts = [f"Goal: {ctx.goal}", f"Repository: {ctx.repo_path}"]
        if ctx.memory_context:
            user_parts.append(f"\n{ctx.memory_context}")
        if ctx.research_context:
            user_parts.append(f"\n## Research Findings\n{ctx.research_context}")
        user = "\n".join(user_parts)
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        if stream_sink is not None:
            resp = await stream_complete(provider, request, sink=stream_sink)
            if resp is not None:
                return resp.content
        try:
            resp = await provider.complete(request)
            return resp.content
        except Exception:
            return self._rule_based_plan(ctx)

    @staticmethod
    def _rule_based_plan(ctx: SharedTaskContext) -> str:
        plan = (
            f"# Implementation Plan\n\n"
            f"Goal: {ctx.goal}\n\n"
            f"1. Locate the relevant module(s) in the repository.\n"
            f"2. Implement the required change with minimal scope.\n"
            f"3. Add or update tests covering the change.\n"
            f"4. Run the test suite to verify.\n"
        )
        if ctx.memory_context:
            plan += f"\n## Repository Memory Context\n\n{ctx.memory_context}\n"
        return plan


__all__ = ["ArchitectAgent"]
