"""Phase 14 - Autonomous self-repair framework.

A generic self-repair engine that replaces the basic review/test repair
loop (Phase 13) with a structured, iterative execution system:

    Failure → Analyze → Strategize → Apply → Validate → (retry or done)

The framework:
1. **Analyzes** the failure using :class:`FailureAnalyzer` to produce a
   structured :class:`FailureReport` (not prompt-based error passing).
2. **Generates repair strategies** using :class:`RepairStrategist`,
   ranked by priority and confidence.
3. **Applies** the best strategy via the registered repair agent
   (typically :class:`CodingAgentAdapter`).
4. **Validates** the fix by running review + tests.
5. **Repeats** until success, budget exhaustion, stagnation detection,
   or no available strategies.

Termination conditions (configurable via :class:`RepairConfig`):
- ``success``: review approved AND tests pass.
- ``budget_exhausted``: max iterations reached.
- ``no_strategies``: strategist generated no actionable strategies.
- ``stagnation``: same failure category repeats N times consecutively.

Integrates with the existing :class:`DelegationFramework`,
:class:`RepositoryMemory`, :class:`TerminalTool`, and all Phase 13
agents. The framework is a drop-in replacement for
:meth:`DelegationFramework.run_repair_loop`.
"""

from __future__ import annotations

import time
from typing import Any

from research_engineer.agents.delegation import DelegationFramework
from research_engineer.agents.failure_analyzer import FailureAnalyzer
from research_engineer.agents.repair_strategist import RepairStrategist
from research_engineer.models.delegation import (
    AgentCapability,
    SharedTaskContext,
)
from research_engineer.models.repair import (
    FailureCategory,
    FailureReport,
    RepairConfig,
    RepairCycle,
    RepairOutcome,
    RepairResult,
    RepairStrategy,
    RepairTerminationReason,
)
from research_engineer.tools.terminal import TerminalInput, TerminalTool


class SelfRepairFramework:
    """Autonomous self-repair engine with structured failure analysis.

    Parameters
    ----------
    delegation:
        The :class:`DelegationFramework` with registered agents (must
        include CODE_REVIEW, TEST_EXECUTION, and REPAIR capabilities).
    terminal:
        :class:`TerminalTool` for git diff operations between cycles.
    config:
        :class:`RepairConfig` controlling budgets and termination.
    failure_analyzer:
        Optional custom :class:`FailureAnalyzer`. If None, a default
        instance is created.
    repair_strategist:
        Optional custom :class:`RepairStrategist`. If None, a default
        instance is created.
    """

    def __init__(
        self,
        delegation: DelegationFramework,
        terminal: TerminalTool | None = None,
        config: RepairConfig | None = None,
        failure_analyzer: FailureAnalyzer | None = None,
        repair_strategist: RepairStrategist | None = None,
    ) -> None:
        self.delegation = delegation
        self.terminal = terminal or TerminalTool()
        self.config = config or RepairConfig()
        self.analyzer = failure_analyzer or FailureAnalyzer()
        self.strategist = repair_strategist or RepairStrategist()
        self._failure_history: list[FailureCategory] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        ctx: SharedTaskContext,
        test_command: str = "uv run pytest",
        timeout_seconds: int = 600,
        stream_sink: Any | None = None,
    ) -> RepairResult:
        """Run the autonomous self-repair loop.

        Args:
            ctx: Shared task context with current failure state.
            test_command: Command for test validation.
            timeout_seconds: Test timeout.
            stream_sink: Optional LLM streaming sink.

        Returns:
            :class:`RepairResult` with all cycle records and outcome.
        """
        start = time.time()
        cycles: list[RepairCycle] = []
        strategies_tried = 0
        unique_failures: set[str] = set()
        termination = RepairTerminationReason.BUDGET_EXHAUSTED
        final_report: FailureReport | None = None

        for cycle_num in range(self.config.max_iterations):
            cycle_start = time.time()
            cycle = RepairCycle(cycle_number=cycle_num)

            # 1. Analyze failure.
            report = await self._analyze_failure(ctx, stream_sink)
            cycle.failure_report = report
            unique_failures.add(report.category.value)
            self._failure_history.append(report.category)
            final_report = report

            # Check stagnation.
            if self._is_stagnated():
                cycle.outcome = RepairOutcome.STAGNATED
                cycle.notes = (
                    f"Stagnation: {report.category.value} repeated "
                    f"{self.config.stagnation_threshold} times."
                )
                termination = RepairTerminationReason.STAGNATION
                cycle.duration_seconds = round(time.time() - cycle_start, 3)
                cycles.append(cycle)
                break

            # 2. Generate strategies.
            strategies = await self._generate_strategies(ctx, report, stream_sink)
            cycle.strategies = strategies

            # Filter by confidence.
            viable = [
                s for s in strategies
                if s.confidence >= self.config.min_confidence_threshold
            ]
            if not viable:
                cycle.outcome = RepairOutcome.FAILURE
                cycle.notes = "No viable strategies above confidence threshold."
                termination = RepairTerminationReason.NO_STRATEGIES
                cycle.duration_seconds = round(time.time() - cycle_start, 3)
                cycles.append(cycle)
                break

            # 3. Apply best strategy.
            best = viable[0]
            cycle.applied_strategy = best
            await self._apply_strategy(ctx, best, stream_sink)
            strategies_tried += 1

            # Re-diff after repair.
            await self._refresh_diff(ctx)

            # 4. Validate: review.
            if self.config.require_review:
                review_passed = await self._validate_review(ctx, stream_sink)
                cycle.review_passed = review_passed
            else:
                cycle.review_passed = True

            # 5. Validate: tests.
            if self.config.require_tests:
                test_passed = await self._validate_tests(
                    ctx, test_command, timeout_seconds
                )
                cycle.test_passed = test_passed
            else:
                cycle.test_passed = True

            # 6. Determine outcome.
            if cycle.review_passed and cycle.test_passed:
                cycle.outcome = RepairOutcome.SUCCESS
                termination = RepairTerminationReason.SUCCESS
                final_report = None
                cycle.duration_seconds = round(time.time() - cycle_start, 3)
                cycles.append(cycle)
                break
            elif cycle.review_passed or cycle.test_passed:
                cycle.outcome = RepairOutcome.PARTIAL
            else:
                cycle.outcome = RepairOutcome.FAILURE

            cycle.duration_seconds = round(time.time() - cycle_start, 3)
            cycle.notes = (
                f"review={'pass' if cycle.review_passed else 'fail'}, "
                f"test={'pass' if cycle.test_passed else 'fail'}"
            )
            cycles.append(cycle)

        successful = (
            termination == RepairTerminationReason.SUCCESS
            or (
                cycles
                and cycles[-1].outcome == RepairOutcome.SUCCESS
            )
        )

        return RepairResult(
            total_cycles=len(cycles),
            successful=successful,
            termination_reason=termination,
            cycles=cycles,
            final_failure_report=final_report,
            total_duration_seconds=round(time.time() - start, 3),
            strategies_tried=strategies_tried,
            unique_failures=len(unique_failures),
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    async def _analyze_failure(
        self, ctx: SharedTaskContext, stream_sink: Any | None
    ) -> FailureReport:
        """Run the failure analyzer and return a structured report."""
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink
        result = await self.analyzer.execute(ctx, **kwargs)
        report_data = result.get("failure_report", {})
        if isinstance(report_data, dict):
            return FailureReport.model_validate(report_data)
        return FailureReport(
            report_id="fr_fallback",
            category=FailureCategory.UNKNOWN,
            root_cause="Failed to produce a failure report.",
        )

    async def _generate_strategies(
        self,
        ctx: SharedTaskContext,
        report: FailureReport,
        stream_sink: Any | None,
    ) -> list[RepairStrategy]:
        """Generate repair strategies from the failure report."""
        kwargs: dict[str, Any] = {"failure_report": report.model_dump()}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink
        result = await self.strategist.execute(ctx, **kwargs)
        strategies_data = result.get("strategies", [])
        strategies: list[RepairStrategy] = []
        for sd in strategies_data:
            if isinstance(sd, dict):
                try:
                    strategies.append(RepairStrategy.model_validate(sd))
                except Exception:
                    pass
        return strategies[: self.config.max_strategies_per_cycle]

    async def _apply_strategy(
        self,
        ctx: SharedTaskContext,
        strategy: RepairStrategy,
        stream_sink: Any | None,
    ) -> None:
        """Apply a repair strategy via the delegation framework."""
        # Inject strategy instructions into the context for the coder.
        ctx.implementation_plan = (
            f"## Repair Strategy: {strategy.action_type.value}\n\n"
            f"{strategy.description}\n\n"
            f"## Instructions\n{strategy.instructions}\n\n"
            f"## Target files\n{', '.join(strategy.target_files)}\n"
            f"## Target symbols\n{', '.join(strategy.target_symbols)}\n"
        )
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink
        await self.delegation.dispatch(
            AgentCapability.REPAIR, ctx, **kwargs
        )

    async def _refresh_diff(self, ctx: SharedTaskContext) -> None:
        """Update ``ctx.diff`` via ``git diff``."""
        out = await self.terminal.execute(
            TerminalInput(
                operation="git_diff",
                repo_path=ctx.repo_path,
            )
        )
        ctx.diff = out.content or out.stdout or ""

    async def _validate_review(
        self, ctx: SharedTaskContext, stream_sink: Any | None
    ) -> bool:
        """Run code review and return whether it passed."""
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink
        step = await self.delegation.dispatch(
            AgentCapability.CODE_REVIEW, ctx, **kwargs
        )
        feedback = step.output.get("feedback", {})
        if isinstance(feedback, dict):
            return feedback.get("approved", True)
        return True

    async def _validate_tests(
        self, ctx: SharedTaskContext, test_command: str, timeout: int
    ) -> bool:
        """Run tests and return whether they passed."""
        await self.delegation.dispatch(
            AgentCapability.TEST_EXECUTION,
            ctx,
            test_command=test_command,
            timeout_seconds=timeout,
        )
        return ctx.test_exit_code == 0

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _is_stagnated(self) -> bool:
        """True if the same failure category repeated N times consecutively."""
        if len(self._failure_history) < self.config.stagnation_threshold:
            return False
        threshold = self.config.stagnation_threshold
        recent = self._failure_history[-threshold:]
        return len(set(recent)) == 1


__all__ = ["SelfRepairFramework"]
