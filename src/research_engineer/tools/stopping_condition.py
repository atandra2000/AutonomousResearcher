"""Stopping Condition Checker for Phase 9.

Evaluates all four stopping conditions after each iteration and returns
the first matching condition (by priority) or None to continue.
"""

from __future__ import annotations

from research_engineer.models.loop import (
    LoopConfig,
    LoopIteration,
    LoopState,
    StoppingCheckInput,
    StoppingCheckOutput,
    StoppingCondition,
)
from research_engineer.tools.base import Tool


class StoppingConditionChecker(
    Tool[StoppingCheckInput, StoppingCheckOutput]
):
    """Checks whether the research loop should stop."""

    async def execute(
        self, input: StoppingCheckInput
    ) -> StoppingCheckOutput:
        state = input.state
        config = input.config
        history = input.history

        for checker in (
            self._check_target,
            self._check_max_iterations,
            self._check_budget,
            self._check_no_improvement,
        ):
            result = checker(state, config, history)
            if result is not None:
                return result

        return StoppingCheckOutput(
            should_stop=False,
            condition=None,
            reason="",
            metric_value=state.best_metric_value,
            target_value=config.target_metric_value,
        )

    @staticmethod
    def _check_target(
        state: LoopState,
        config: LoopConfig,
        history: list[LoopIteration],
    ) -> StoppingCheckOutput | None:
        """Check if target metric is achieved."""
        if (
            config.target_metric_name
            and config.target_metric_value is not None
            and state.best_metric_value is not None
        ):
            target = config.target_metric_value
            val = state.best_metric_value
            achieved = (
                val >= target if config.higher_is_better else val <= target
            )
            if achieved:
                return StoppingCheckOutput(
                    should_stop=True,
                    condition=StoppingCondition.TARGET_ACHIEVED,
                    reason=(
                        f"Target {config.target_metric_name}={target} "
                        f"achieved (current best: {val:.6f})"
                    ),
                    metric_value=val,
                    target_value=target,
                )
        return None

    @staticmethod
    def _check_max_iterations(
        state: LoopState,
        config: LoopConfig,
        history: list[LoopIteration],
    ) -> StoppingCheckOutput | None:
        """Check if max iterations reached."""
        if state.current_iteration >= config.max_iterations:
            return StoppingCheckOutput(
                should_stop=True,
                condition=StoppingCondition.MAX_ITERATIONS_REACHED,
                reason=f"Max iterations ({config.max_iterations}) reached",
                metric_value=state.best_metric_value,
                target_value=config.target_metric_value,
            )
        return None

    @staticmethod
    def _check_budget(
        state: LoopState,
        config: LoopConfig,
        history: list[LoopIteration],
    ) -> StoppingCheckOutput | None:
        """Check if budget exceeded."""
        if config.budget_hours is not None:
            if state.cumulative_cost_hours >= config.budget_hours:
                return StoppingCheckOutput(
                    should_stop=True,
                    condition=StoppingCondition.BUDGET_EXCEEDED,
                    reason=(
                        f"Budget hours ({config.budget_hours}) exceeded "
                        f"(used: {state.cumulative_cost_hours:.4f})"
                    ),
                    metric_value=state.best_metric_value,
                    target_value=config.target_metric_value,
                )
        if config.budget_cost is not None:
            if state.cumulative_cost_usd >= config.budget_cost:
                return StoppingCheckOutput(
                    should_stop=True,
                    condition=StoppingCondition.BUDGET_EXCEEDED,
                    reason=(
                        f"Budget cost (${config.budget_cost:.2f}) exceeded "
                        f"(spent: ${state.cumulative_cost_usd:.2f})"
                    ),
                    metric_value=state.best_metric_value,
                    target_value=config.target_metric_value,
                )
        return None

    @staticmethod
    def _check_no_improvement(
        state: LoopState,
        config: LoopConfig,
        history: list[LoopIteration],
    ) -> StoppingCheckOutput | None:
        """Check if no improvement over stagnation window."""
        if len(history) < config.stagnation_window:
            return None
        window = history[-config.stagnation_window:]
        window_vals = [
            it.primary_metric_value
            for it in window
            if it.primary_metric_value is not None
        ]
        if len(window_vals) != len(window):
            return None
        prior = history[: -config.stagnation_window]
        prior_vals = [
            it.primary_metric_value
            for it in prior
            if it.primary_metric_value is not None
        ]
        if not prior_vals:
            return None
        if config.higher_is_better:
            best_prior = max(prior_vals)
            best_window = max(window_vals)
        else:
            best_prior = min(prior_vals)
            best_window = min(window_vals)
        improvement = (
            best_window - best_prior
            if config.higher_is_better
            else best_prior - best_window
        )
        if improvement < config.improvement_threshold:
            return StoppingCheckOutput(
                should_stop=True,
                condition=StoppingCondition.NO_IMPROVEMENT,
                reason=(
                    f"No improvement over last "
                    f"{config.stagnation_window} iterations "
                    f"(best prior: {best_prior:.6f}, "
                    f"best window: {best_window:.6f}, "
                    f"improvement: {improvement:.6f})"
                ),
                metric_value=state.best_metric_value,
                target_value=config.target_metric_value,
            )
        return None
