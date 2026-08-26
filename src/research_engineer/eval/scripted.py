"""E4 - Deterministic scripted agent factory.

Builds :class:`~research_engineer.runtime.runtime.AgentRuntime` instances
whose behaviour is driven by a declarative script carried in the eval
task's ``metadata["script"]``. This lets the harness exercise the real
runtime (state machine, budgets, error recovery, gateways) without LLMs,
while keeping every run fully reproducible.

Script schema::

    script:
      steps: 2                      # loop iterations before done
      final_output: "answer text"   # output when done
      tools:                        # per-step tool calls (in order)
        - {step: 1, name: search, status: allowed}
        - {step: 2, name: shell, status: approval_required}
      error_at_step: 1              # raise a recoverable error here
      tokens_per_step: 100
      cost_per_step_usd: 0.01

Agents built this way log each tool invocation to
``ctx.metadata["tool_call_log"]`` (``{"tool", "status"}`` entries) and
increasing ``ctx.metadata["human_interventions"]`` whenever a call needs
denied/required approval - the same lightweight contract real agents use
so graders can inspect usage without a second telemetry system.
"""

from __future__ import annotations

from typing import Any

from research_engineer.eval.models import EvalTask
from research_engineer.runtime.models import AgentContext, AgentPolicy
from research_engineer.runtime.runtime import AgentRuntime

# Tool-call statuses that imply a human was pulled into the loop.
_INTERVENTION_STATUSES = {"approval_required", "approval_denied", "denied"}


class ScriptedAgentFactory:
    """Build deterministic runtimes from a task's script metadata."""

    def __call__(self, task: EvalTask) -> AgentRuntime:
        return self.build(task)

    def build(self, task: EvalTask) -> AgentRuntime:
        script = dict(task.metadata.get("script", {}))
        total_steps = max(1, int(script.get("steps", 1)))
        final_output = script.get("final_output")
        tool_specs = list(script.get("tools", []))
        error_at_step = int(script.get("error_at_step", 0))
        tokens_per_step = int(script.get("tokens_per_step", 0))
        cost_per_step = float(script.get("cost_per_step_usd", 0.0))

        async def planner(ctx: AgentContext) -> Any:
            return {"step": ctx.current_step + 1, "goal": ctx.goal}

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            step_no = int(plan.get("step", ctx.current_step + 1))
            ctx.tokens += tokens_per_step
            ctx.cost_usd += cost_per_step
            # Emit scripted tool calls for this step.
            log = ctx.metadata.setdefault("tool_call_log", [])
            interventions = int(ctx.metadata.get("human_interventions", 0))
            for spec in tool_specs:
                if int(spec.get("step", step_no)) != step_no:
                    continue
                status = str(spec.get("status", "allowed"))
                log.append({"step": step_no, "tool": spec.get("name", "?"),
                            "status": status})
                if status in _INTERVENTION_STATUSES:
                    interventions += 1
            ctx.metadata["human_interventions"] = interventions
            ctx.tool_calls += sum(
                1 for spec in tool_specs
                if int(spec.get("step", step_no)) == step_no
            )
            if error_at_step and step_no == error_at_step:
                # Recoverable connection error exercises runtime retry logic.
                raise ConnectionError(f"scripted transient failure at {step_no}")
            return {"step": step_no, "action": "done"}

        async def observer(ctx: AgentContext, action: Any) -> Any:
            return action

        async def evaluator(ctx: AgentContext, observation: Any) -> Any:
            if ctx.current_step >= total_steps:
                return {"done": True, "output": final_output}
            return 1.0

        policy = AgentPolicy(budget=task.budget)
        return AgentRuntime(
            planner=planner,
            actor=actor,
            observer=observer,
            evaluator=evaluator,
            policy=policy,
        )


def resolve_agent_factory(name: str) -> ScriptedAgentFactory:
    """Resolve a named agent factory; currently only ``scripted`` ships."""
    if name != "scripted":
        raise KeyError(
            f"unknown agent factory {name!r}; available: 'scripted'"
        )
    return ScriptedAgentFactory()


__all__ = ["ScriptedAgentFactory", "resolve_agent_factory"]
