"""E4 - Evaluation runner.

Loads YAML/JSON :class:`~research_engineer.eval.models.EvalSuite`
definitions and executes every case through a real
:class:`~research_engineer.runtime.runtime.AgentRuntime` supplied by a
callable ``AgentFactory``. The harness never bypasses the runtime: goals,
budgets, termination, error recovery, and tool-gateway routing all happen
inside ``AgentRuntime.run``. The runner only *observes* the resulting
``AgentExecution``, applies graders, collects metrics, and assembles an
class:`~research_engineer.eval.models.EvalReport`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # PyYAML

from research_engineer.eval.graders import Grader, GradingRequest, build_grader
from research_engineer.eval.metrics import aggregate
from research_engineer.eval.models import (
    CaseMetrics,
    EvalReport,
    EvalResult,
    EvalStatus,
    EvalSuite,
    EvalTask,
    GraderResult,
)
from research_engineer.runtime.models import AgentExecution, AgentTermination
from research_engineer.runtime.runtime import AgentRuntime

#: Builds a fresh runtime for one case (must not share mutable state).
AgentFactory = Callable[[EvalTask], AgentRuntime]


def load_suite(path: str | Path) -> EvalSuite:
    """Load an :class:`EvalSuite` from a YAML or JSON file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"suite file {source} did not contain an object")
    suite: EvalSuite = EvalSuite.model_validate(data)
    if not suite.cases:
        raise ValueError(f"suite {suite.suite_id} has no cases")
    ids = [c.case_id for c in suite.cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"suite {suite.suite_id} has duplicate case ids")
    return suite


def save_suite(suite: EvalSuite, path: str | Path) -> None:
    """Serialize a suite to JSON at ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(suite.model_dump_json(indent=2), encoding="utf-8")


class EvalRunner:
    """Execute eval suites through AgentRuntime and grade the outcomes.

    Args:
        factory: builds the (fresh) runtime used for each case.
        extra_graders: additional user-supplied grader instances keyed by
            criterion name; merged over the deterministic registry.
        seed: seed folded into run ids for reproducible multi-attempt runs.
        label: label of the evaluated configuration (regression reports).
    """

    def __init__(
        self,
        factory: AgentFactory,
        *,
        extra_graders: dict[str, Grader] | None = None,
        seed: int = 0,
        label: str = "",
    ) -> None:
        self._factory = factory
        self._extra_graders = extra_graders or {}
        self.seed = seed
        self.label = label

    async def run_case(self, task: EvalTask, suite_id: str) -> EvalResult:
        """Run one case end to end and grade it."""
        run_id = task.stable_run_id(suite_id, self.seed)
        result = EvalResult(
            run_id=run_id, case_id=task.case_id, revision=task.revision,
            status=EvalStatus.RUNNING,
        )
        started = time.monotonic()
        try:
            runtime = self._factory(task)
            coro = runtime.run(
                task.goal,
                metadata={
                    "eval_case": task.case_id,
                    "eval_suite": suite_id,
                    "eval_revision": task.revision,
                },
            )
            execution = (
                await asyncio.wait_for(coro, timeout=task.timeout_seconds)
                if task.timeout_seconds
                else await coro
            )
        except TimeoutError:
            result.status = EvalStatus.FAILED
            result.error = f"harness timeout after {task.timeout_seconds}s"
            result.metrics.latency_seconds = time.monotonic() - started
            result.finished_at = datetime.now()
            return result
        except Exception as exc:  # noqa: BLE001 - harness must not crash suites
            result.status = EvalStatus.ERROR
            result.error = f"{type(exc).__name__}: {exc}"
            result.metrics.latency_seconds = time.monotonic() - started
            result.finished_at = datetime.now()
            return result

        elapsed = time.monotonic() - started
        # The runtime converts external cancellation into a CANCELLED
        # termination. When that happens at (approximately) our harness
        # deadline, report the case as a harness timeout failure.
        if (
            task.timeout_seconds
            and elapsed >= task.timeout_seconds
            and execution.termination == AgentTermination.CANCELLED
        ):
            result.status = EvalStatus.FAILED
            result.error = f"harness timeout after {task.timeout_seconds}s"
            result.completion = True
            result.metrics.latency_seconds = elapsed
            result.finished_at = datetime.now()
            return result

        success, weighted, grader_results, metrics = await self._grade(
            task, execution, elapsed
        )
        result.status = EvalStatus.COMPLETED
        result.completion = execution.context.is_terminal()
        result.success = success
        result.weighted_score = weighted
        result.grader_results = grader_results
        result.metrics = metrics
        result.output = execution.output
        result.finished_at = datetime.now()
        return result

    async def _grade(
        self, task: EvalTask, execution: AgentExecution, elapsed: float,
    ) -> tuple[bool, float, list[GraderResult], CaseMetrics]:
        """Collect metrics and apply all configured criteria."""
        ctx = execution.context
        metadata = ctx.metadata or {}
        metrics = CaseMetrics(
            steps=ctx.current_step,
            tool_calls=ctx.tool_calls,
            tokens=ctx.tokens,
            cost_usd=ctx.cost_usd,
            latency_seconds=max(elapsed, ctx.duration_seconds),
            recoverable_errors=ctx.recoverable_errors,
            fatal_errors=sum(1 for s in ctx.steps if s.error is not None),
            recovered=(
                ctx.recoverable_errors > 0
                and ctx.termination == AgentTermination.SUCCESS
            ),
            human_interventions=int(metadata.get("human_interventions", 0)),
            terminated=ctx.is_terminal(),
            termination_reason=(
                ctx.termination.value if ctx.termination else "none"
            ),
        )
        request = GradingRequest(task=task, execution=execution)
        grader_results = []
        for criterion in task.criteria:
            grader = self._resolve_grader(criterion.grader, criterion.config)
            outcome = await _grade_one(grader, request, criterion.weight)
            grader_results.append(outcome)

        runtime_success = ctx.termination == AgentTermination.SUCCESS
        required_ok = all(r.passed for r in grader_results)
        success = required_ok and (
            runtime_success if task.require_runtime_success else True
        )

        total_weight = sum(r.weight for r in grader_results) or 1.0
        weighted = (
            sum(r.score * r.weight for r in grader_results) / total_weight
            if grader_results else (1.0 if success else 0.0)
        )
        return success, weighted, grader_results, metrics

    def _resolve_grader(self, name: str, config: dict[str, Any]) -> Grader:
        grader = self._extra_graders.get(name)
        if grader is not None:
            return grader
        return build_grader(name, config)

    async def run_suite(self, suite: EvalSuite) -> EvalReport:
        """Run every case in the suite and assemble an ``EvalReport``."""
        results: list[EvalResult] = []
        for case in suite.cases:
            results.append(await self.run_case(case, suite.suite_id))
        report = EvalReport(
            suite_id=suite.suite_id,
            suite_version=suite.version,
            label=self.label,
            results=results,
            aggregate=aggregate(results),
        )
        return report


async def _grade_one(
    grader: Grader, request: GradingRequest, weight: float,
) -> GraderResult:
    """Await one grader; any exception becomes a failed result, never raised."""
    try:
        outcome = await grader.grade(request)
    except Exception as exc:  # noqa: BLE001 - graders must not break the run
        return GraderResult(
            grader=getattr(grader, "name", "?"), score=0.0, passed=False,
            detail=f"grader crashed: {exc}",
        )
    outcome.weight = weight
    return outcome


__all__ = ["AgentFactory", "EvalRunner", "load_suite", "save_suite"]
