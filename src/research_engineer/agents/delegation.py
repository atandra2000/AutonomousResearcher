"""Phase 13 - Generic multi-agent delegation framework.

A capability-based router that dispatches work to specialized agents
based on their declared roles and capabilities — not hardcoded task
logic. The framework manages a :class:`SharedTaskContext` that flows
through the pipeline, a list of :class:`AgentDescriptor` entries that
declare what each agent can do, and a sequence of
:class:`DelegationStep` records for traceability.

Design principles:
- **Role/capability routing**: Agents are registered with capabilities;
  the coordinator asks the framework to dispatch a capability, and the
  framework finds the right agent.
- **Structured context**: All inter-agent communication flows through
  :class:`SharedTaskContext`; agents never call each other directly.
- **Extensibility**: New agents are added by registering an
  :class:`AgentDescriptor`; no framework code changes needed.
- **Backward compatible**: When delegation is disabled, the TaskAgent
  falls back to its original Phase 11/12 pipeline.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from research_engineer.models.delegation import (
    AgentCapability,
    AgentDescriptor,
    AgentRole,
    DelegationStatus,
    DelegationStep,
    SharedTaskContext,
)


class DelegationFramework:
    """Capability-based agent router and pipeline executor.

    Agents are registered with :meth:`register`. The framework dispatches
    work via :meth:`dispatch`, which finds the agent with the requested
    capability and invokes its ``execute`` method with the shared context.

    A full pipeline is run via :meth:`run_pipeline`, which executes a
    configurable sequence of capabilities. The caller can inject custom
    sequences or use the default pipeline.
    """

    def __init__(self) -> None:
        self._agents: list[AgentDescriptor] = []
        self._by_capability: dict[AgentCapability, AgentDescriptor] = {}
        self._by_role: dict[AgentRole, AgentDescriptor] = {}
        self._steps: list[DelegationStep] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        agent: Any,
        role: AgentRole,
        capabilities: list[AgentCapability],
        agent_name: str | None = None,
    ) -> AgentDescriptor:
        """Register an agent with its role and capabilities."""
        name = agent_name or getattr(agent, "agent_name", agent.__class__.__name__)
        desc = AgentDescriptor(
            agent_name=name,
            role=role,
            capabilities=capabilities,
            agent=agent,
        )
        self._agents.append(desc)
        for cap in capabilities:
            # First registered wins for a capability (deterministic).
            if cap not in self._by_capability:
                self._by_capability[cap] = desc
        self._by_role[role] = desc
        return desc

    def get_by_capability(self, cap: AgentCapability) -> AgentDescriptor | None:
        return self._by_capability.get(cap)

    def get_by_role(self, role: AgentRole) -> AgentDescriptor | None:
        return self._by_role.get(role)

    @property
    def steps(self) -> list[DelegationStep]:
        return self._steps

    @property
    def agents(self) -> list[AgentDescriptor]:
        return list(self._agents)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        capability: AgentCapability,
        ctx: SharedTaskContext,
        **kwargs: Any,
    ) -> DelegationStep:
        """Dispatch a capability to the registered agent.

        Args:
            capability: What to do (e.g. CODE_REVIEW).
            ctx: Shared task context (read + written by the agent).
            **kwargs: Extra arguments passed to the agent's execute method.

        Returns:
            :class:`DelegationStep` recording the invocation.
        """
        desc = self._by_capability.get(capability)
        step_id = f"step_{uuid4().hex[:8]}"
        if desc is None:
            return DelegationStep(
                step_id=step_id,
                role=AgentRole.COORDINATOR,
                capability=capability,
                agent_name="none",
                status=DelegationStatus.SKIPPED,
                summary=f"No agent registered for {capability.value}",
            )

        agent = desc.agent
        step = DelegationStep(
            step_id=step_id,
            role=desc.role,
            capability=capability,
            agent_name=desc.agent_name,
            status=DelegationStatus.RUNNING,
        )
        t0 = time.time()
        try:
            result = await agent.execute(ctx, **kwargs)  # type: ignore[union-attr]
            step.status = DelegationStatus.COMPLETED
            step.finished_at = datetime.now()
            step.duration_seconds = round(time.time() - t0, 3)
            if isinstance(result, dict):
                step.output = result
                step.summary = str(result.get("summary", ""))[:200]
            elif isinstance(result, str):
                step.summary = result[:200]
                step.output = {"result": result}
            else:
                step.output = {"result": str(result)}
                step.summary = str(result)[:200]
        except Exception as e:
            step.status = DelegationStatus.FAILED
            step.finished_at = datetime.now()
            step.duration_seconds = round(time.time() - t0, 3)
            step.error = str(e)
        self._steps.append(step)
        return step

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        ctx: SharedTaskContext,
        capabilities: list[AgentCapability] | None = None,
    ) -> list[DelegationStep]:
        """Run a sequence of capabilities through the pipeline.

        Args:
            ctx: Shared task context (accumulates results).
            capabilities: Ordered list of capabilities to dispatch.
                If None, uses the default pipeline:
                REPOSITORY_ANALYSIS → ARCHITECTURE → CODE_GENERATION →
                CODE_REVIEW → TEST_EXECUTION.

        Returns:
            List of :class:`DelegationStep` records.
        """
        caps = capabilities or [
            AgentCapability.REPOSITORY_ANALYSIS,
            AgentCapability.ARCHITECTURE,
            AgentCapability.CODE_GENERATION,
            AgentCapability.CODE_REVIEW,
            AgentCapability.TEST_EXECUTION,
        ]
        results: list[DelegationStep] = []
        for cap in caps:
            step = await self.dispatch(cap, ctx)
            results.append(step)
            if step.status == DelegationStatus.FAILED:
                break
        return results

    # ------------------------------------------------------------------
    # Repair loop
    # ------------------------------------------------------------------

    async def run_repair_loop(
        self,
        ctx: SharedTaskContext,
        review_cap: AgentCapability = AgentCapability.CODE_REVIEW,
        test_cap: AgentCapability = AgentCapability.TEST_EXECUTION,
        repair_cap: AgentCapability = AgentCapability.CODE_GENERATION,
    ) -> list[DelegationStep]:
        """Run review → test → repair loop until pass or max iterations.

        The loop:
        1. Dispatch review capability.
        2. If review requests changes → dispatch repair → review again.
        3. Dispatch test capability.
        4. If test fails → dispatch repair → test again.
        5. Repeat up to ``ctx.max_repair_iterations``.

        Returns the steps from this loop (not including prior steps).
        """
        loop_steps: list[DelegationStep] = []
        for iteration in range(ctx.max_repair_iterations + 1):
            ctx.repair_iteration = iteration
            # Review.
            review_step = await self.dispatch(review_cap, ctx)
            loop_steps.append(review_step)
            if review_step.status == DelegationStatus.COMPLETED:
                feedback = review_step.output.get("feedback", {})
                if isinstance(feedback, dict):
                    approved = feedback.get("approved", True)
                    if not approved and iteration < ctx.max_repair_iterations:
                        repair_step = await self.dispatch(repair_cap, ctx)
                        loop_steps.append(repair_step)
                        continue
            # Test.
            test_step = await self.dispatch(test_cap, ctx)
            loop_steps.append(test_step)
            if test_step.status == DelegationStatus.COMPLETED:
                if ctx.test_exit_code == 0:
                    break
            if iteration < ctx.max_repair_iterations:
                repair_step = await self.dispatch(repair_cap, ctx)
                loop_steps.append(repair_step)
            else:
                break
        return loop_steps


__all__ = ["DelegationFramework"]
