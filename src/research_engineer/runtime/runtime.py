"""E1 - Production Agent Runtime.

The generic, async-first :class:`AgentRuntime` is the central orchestration
layer for autonomous agents. It owns:

* lifecycle and state transitions (a deterministic state machine)
* ``plan -> act -> observe -> evaluate`` loops
* tool/LLM execution through injected async callables (which may reuse the
  existing :class:`~research_engineer.llm.base.LLMProvider` and
  :class:`~research_engineer.tools.base.Tool` abstractions)
* budgets and termination
* cancellation and error recovery (recoverable vs fatal)
* observability integration (structured events via the event bus)

The runtime is intentionally generic: it knows nothing about any specific
agent. Callers inject the four phase callables and a
:class:`~research_engineer.runtime.models.AgentPolicy`. Existing agents can
be wrapped with :class:`~research_engineer.runtime.adapters.AgentAdapter` so
they run through the runtime unchanged.

State machine::

    CREATED -> RUNNING -> TERMINATED

``RUNNING`` is the only non-terminal state. The precise reason for
termination is captured by
:class:`~research_engineer.runtime.models.AgentTermination` (success, budget
exceeded, timeout, cancelled, error, no-progress).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from research_engineer.runtime.models import (
    AgentContext,
    AgentError,
    AgentExecution,
    AgentPhase,
    AgentPolicy,
    AgentState,
    AgentStep,
    AgentTermination,
)

logger = logging.getLogger(__name__)

#: A phase callable: receives the current context (and, for ``act``, the
#: plan) and returns the phase output. May raise to signal an error.
Planner = Callable[[AgentContext], Awaitable[Any]]
Actor = Callable[[AgentContext, Any], Awaitable[Any]]
Observer = Callable[[AgentContext, Any], Awaitable[Any]]
Evaluator = Callable[[AgentContext, Any], Awaitable[Any]]


def classify_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is a recoverable (retryable) error.

    Recoverable errors are transient failures the runtime may retry:
    ``asyncio.TimeoutError``, ``ConnectionError``, ``OSError``, and
    :class:`~research_engineer.llm.base.ProviderError` instances that are
    not permanent. Everything else is treated as fatal.

    Callers may override this by passing a custom ``error_classifier`` to
    :class:`AgentRuntime`.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    # ProviderError from the LLM layer: recoverable unless permanent.
    try:
        from research_engineer.llm.base import ProviderError
        from research_engineer.llm.resilience import is_permanent_provider_error

        if isinstance(exc, ProviderError):
            return not is_permanent_provider_error(exc)
    except ImportError:  # pragma: no cover - llm layer always present
        pass
    return False


class AgentRuntime:
    """Generic async-first orchestration runtime for autonomous agents.

    Args:
        planner: async ``(ctx) -> plan`` callable.
        actor: async ``(ctx, plan) -> action`` callable.
        observer: async ``(ctx, action) -> observation`` callable.
        evaluator: async ``(ctx, observation) -> evaluation`` callable. The
            evaluation may be a number (used for progress tracking) or any
            object with a ``score`` attribute; otherwise progress tracking
            is disabled for that step.
        policy: :class:`AgentPolicy` controlling budgets and error recovery.
        error_classifier: optional ``(exc) -> bool`` override for
            recoverable-vs-fatal classification.
        on_step: optional sync callback invoked after each completed step
            (observability hook).
        event_bus: optional event bus; defaults to the process-wide bus from
            :func:`research_engineer.observability.get_event_bus`.
    """

    def __init__(
        self,
        planner: Planner,
        actor: Actor,
        observer: Observer,
        evaluator: Evaluator,
        policy: AgentPolicy | None = None,
        error_classifier: Callable[[BaseException], bool] | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.observer = observer
        self.evaluator = evaluator
        self.policy = policy or AgentPolicy()
        self._classify = error_classifier or classify_error
        self._on_step = on_step
        self._event_bus = event_bus
        self._cancel_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        goal: str,
        *,
        metadata: dict[str, Any] | None = None,
        context: AgentContext | None = None,
    ) -> AgentExecution:
        """Run the agent loop to completion and return the execution.

        Args:
            goal: High-level goal for this run.
            metadata: Optional caller metadata attached to the context.
            context: Optional pre-built context (e.g. resumed from a
                checkpoint). When provided, ``goal`` is ignored in favour of
                the context's goal.

        Returns:
            An :class:`AgentExecution` with the final context, termination
            reason, and output.
        """
        ctx = context or AgentContext(goal=goal)
        if metadata:
            ctx.metadata.update(metadata)
        self._cancel_event = asyncio.Event()
        start = time.monotonic()
        ctx.state = AgentState.RUNNING
        ctx.started_at = datetime.now()
        self._emit("start", ctx)

        try:
            while ctx.state == AgentState.RUNNING:
                # Budget/timeout/cancellation checks run before each step.
                if self._check_termination(ctx, start):
                    break
                if not await self._process_step(ctx, start):
                    break
        except asyncio.CancelledError:
            self._terminate(ctx, AgentTermination.CANCELLED, "Run cancelled")
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("AgentRuntime.run failed for %s", ctx.execution_id)
            self._terminate(
                ctx,
                AgentTermination.ERROR,
                f"Unhandled runtime error: {exc}",
            )

        ctx.finished_at = datetime.now()
        ctx.duration_seconds = round(time.monotonic() - start, 6)
        self._emit("end", ctx)
        return AgentExecution(
            context=ctx,
            termination=ctx.termination or AgentTermination.ERROR,
            reason=ctx.termination_reason,
            output=ctx.output,
        )

    async def _process_step(self, ctx: AgentContext, start: float) -> bool:
        """Run one step and post-process it; return False to stop the loop."""
        step = await self._run_step(ctx)
        ctx.steps.append(step)
        ctx.current_step += 1
        ctx.tool_calls += step.tool_calls
        ctx.tokens += step.tokens
        ctx.cost_usd = round(ctx.cost_usd + step.cost_usd, 6)
        if self._on_step is not None:
            self._on_step(step)
        self._emit("step", ctx, step=step)

        # A fatal error during the step terminates the run.
        if ctx.state != AgentState.RUNNING:
            return False

        # Re-check budgets after the step consumed resources.
        if self._check_termination(ctx, start):
            return False

        # Progress tracking from the evaluation score.
        if step.score is not None:
            self._track_progress(ctx, step.score)
            if ctx.state != AgentState.RUNNING:
                return False

        # The evaluator may signal completion by returning a sentinel
        # or by mutating the context; check for a done flag.
        if self._evaluator_says_done(step):
            self._terminate(
                ctx, AgentTermination.SUCCESS, "Evaluator signalled completion"
            )
        return True

    def cancel(self) -> None:
        """Request cancellation of a running execution.

        The runtime checks the cancellation flag between phases and
        terminates with ``CANCELLED``. This is cooperative: it does not
        interrupt an in-flight phase callable.
        """
        if self._cancel_event is not None:
            self._cancel_event.set()

    # ------------------------------------------------------------------
    # Loop internals
    # ------------------------------------------------------------------

    async def _run_step(self, ctx: AgentContext) -> AgentStep:
        """Execute one plan/act/observe/evaluate iteration.

        Always returns the :class:`AgentStep`. On a fatal error the step
        carries the error and the context is transitioned to ``TERMINATED``;
        the caller checks ``ctx.state`` to decide whether to continue.
        """
        step = AgentStep(step=ctx.current_step + 1)
        step.started_at = datetime.now()
        t0 = time.monotonic()

        # --- Plan ---
        ctx.phase = AgentPhase.PLANNING
        try:
            step.plan = await self.planner(ctx)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._handle_error(ctx, step, exc, AgentPhase.PLANNING)
            return self._finish_step(step, t0)

        # --- Act ---
        ctx.phase = AgentPhase.ACTING
        try:
            step.action = await self.actor(ctx, step.plan)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._handle_error(ctx, step, exc, AgentPhase.ACTING)
            return self._finish_step(step, t0)

        # --- Observe ---
        ctx.phase = AgentPhase.OBSERVING
        try:
            step.observation = await self.observer(ctx, step.action)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._handle_error(ctx, step, exc, AgentPhase.OBSERVING)
            return self._finish_step(step, t0)

        # --- Evaluate ---
        ctx.phase = AgentPhase.EVALUATING
        try:
            step.evaluation = await self.evaluator(ctx, step.observation)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._handle_error(ctx, step, exc, AgentPhase.EVALUATING)
            return self._finish_step(step, t0)

        step.score = self._extract_score(step.evaluation)
        return self._finish_step(step, t0)

    @staticmethod
    def _finish_step(step: AgentStep, t0: float) -> AgentStep:
        """Stamp timestamps/duration on a step and return it."""
        step.finished_at = datetime.now()
        step.duration_seconds = round(time.monotonic() - t0, 6)
        return step

    def _handle_error(
        self,
        ctx: AgentContext,
        step: AgentStep,
        exc: BaseException,
        phase: AgentPhase,
    ) -> bool:
        """Classify and record an error; return True to continue, False to stop.

        Recoverable errors are recorded and the step is retried (the loop
        continues). Fatal errors, or exceeding ``max_recoverable_errors``,
        terminate the run with ``ERROR``.
        """
        recoverable = self._classify(exc)
        err = AgentError(
            message=str(exc),
            error_type=type(exc).__name__,
            recoverable=recoverable,
            step=step.step,
            phase=phase,
        )
        step.error = err
        step.finished_at = datetime.now()
        step.duration_seconds = round(
            (step.finished_at - step.started_at).total_seconds(), 6
        )
        self._emit("error", ctx, error=err)

        if not recoverable:
            self._terminate(
                ctx,
                AgentTermination.ERROR,
                f"Fatal error in {phase.value}: {exc}",
            )
            return False

        ctx.recoverable_errors += 1
        if ctx.recoverable_errors > self.policy.max_recoverable_errors:
            self._terminate(
                ctx,
                AgentTermination.ERROR,
                f"Too many recoverable errors ({ctx.recoverable_errors})",
            )
            return False
        logger.warning(
            "Recoverable error in %s step %d (%s); retrying",
            phase.value,
            step.step,
            exc,
        )
        return True

    def _check_termination(self, ctx: AgentContext, start: float) -> bool:
        """Return True when the run should stop (budget/timeout/cancel)."""
        budget = self.policy.budget

        if self._cancel_event is not None and self._cancel_event.is_set():
            self._terminate(ctx, AgentTermination.CANCELLED, "Run cancelled")
            return True

        if budget.max_steps is not None and ctx.current_step >= budget.max_steps:
            self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Max steps reached ({budget.max_steps})",
            )
            return True

        if budget.max_tool_calls is not None and ctx.tool_calls >= budget.max_tool_calls:
            self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Max tool calls reached ({budget.max_tool_calls})",
            )
            return True

        if budget.max_runtime_seconds is not None:
            elapsed = time.monotonic() - start
            if elapsed >= budget.max_runtime_seconds:
                self._terminate(
                    ctx,
                    AgentTermination.TIMEOUT,
                    f"Runtime budget exceeded ({elapsed:.2f}s)",
                )
                return True

        if budget.max_cost_usd is not None and ctx.cost_usd >= budget.max_cost_usd:
            self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Cost budget exceeded (${ctx.cost_usd:.4f})",
            )
            return True

        if budget.max_tokens is not None and ctx.tokens >= budget.max_tokens:
            self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Token budget exceeded ({ctx.tokens})",
            )
            return True

        return False

    def _track_progress(self, ctx: AgentContext, score: float) -> None:
        """Update best-score and stagnation tracking from an evaluation score."""
        if ctx.best_score is None:
            ctx.best_score = score
            ctx.stagnation_count = 0
            return
        delta = abs(score - ctx.best_score)
        if delta > self.policy.progress_threshold:
            ctx.best_score = score
            ctx.stagnation_count = 0
        else:
            ctx.stagnation_count += 1
            if ctx.stagnation_count >= self.policy.stagnation_window:
                self._terminate(
                    ctx,
                    AgentTermination.NO_PROGRESS,
                    f"No progress for {ctx.stagnation_count} steps",
                )

    @staticmethod
    def _extract_score(evaluation: Any) -> float | None:
        """Extract a numeric score from an evaluation output, if possible."""
        if isinstance(evaluation, (int, float)):
            return float(evaluation)
        if hasattr(evaluation, "score") and isinstance(evaluation.score, (int, float)):
            return float(evaluation.score)
        return None

    @staticmethod
    def _evaluator_says_done(step: AgentStep) -> bool:
        """True when the evaluator signalled completion.

        Supports a ``done`` boolean attribute on the evaluation output, or a
        dict with a truthy ``done`` key.
        """
        ev = step.evaluation
        if ev is None:
            return False
        if isinstance(ev, dict):
            return bool(ev.get("done", False))
        if hasattr(ev, "done"):
            return bool(getattr(ev, "done"))
        return False

    def _terminate(
        self, ctx: AgentContext, termination: AgentTermination, reason: str
    ) -> None:
        """Transition the context to a terminal state."""
        ctx.state = AgentState.TERMINATED
        ctx.termination = termination
        ctx.termination_reason = reason
        ctx.output = self._final_output(ctx)
        self._emit("terminate", ctx, termination=termination)

    @staticmethod
    def _final_output(ctx: AgentContext) -> Any:
        """Derive the final output from the last completed step.

        If the evaluation is a ``{"done": ..., "output": X}`` wrapper (as
        produced by :class:`~research_engineer.runtime.adapters.AgentAdapter`),
        the wrapped ``output`` is returned so callers see the agent's actual
        result rather than the runtime's control envelope.
        """
        if not ctx.steps:
            return None
        last = ctx.steps[-1]
        evaluation = last.evaluation
        if isinstance(evaluation, dict) and "output" in evaluation:
            return evaluation["output"]
        return evaluation if evaluation is not None else last.action

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _emit(self, kind: str, ctx: AgentContext, **extra: Any) -> None:
        """Emit a structured ``agent_runtime`` event (best-effort)."""
        try:
            bus = self._event_bus
            if bus is None:
                from research_engineer.observability import get_event_bus

                bus = get_event_bus()
            event: dict[str, Any] = {
                "kind": "agent_runtime",
                "event": kind,
                "execution_id": ctx.execution_id,
                "goal": ctx.goal,
                "state": ctx.state.value,
                "phase": ctx.phase.value if ctx.phase else None,
                "step": ctx.current_step,
                "tool_calls": ctx.tool_calls,
                "tokens": ctx.tokens,
                "cost_usd": ctx.cost_usd,
            }
            event.update(extra)
            bus.emit(event)
        except Exception:  # noqa: BLE001 - observability must not break the run
            logger.debug("Failed to emit agent_runtime event", exc_info=True)


__all__ = [
    "AgentRuntime",
    "classify_error",
    "Planner",
    "Actor",
    "Observer",
    "Evaluator",
]
