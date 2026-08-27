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

from research_engineer.observability.context import (
    CorrelationContext,
    get_correlation,
    new_run_id,
    new_span_id,
    new_trace_id,
    reset_correlation,
    set_correlation,
)
from research_engineer.observability.otel import start_span
from research_engineer.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    CheckpointError,
    CheckpointLockError,
    CheckpointStore,
    CheckpointVersionError,
)
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


#: E5: map safety control triggers to deterministic runtime terminations.
_SAFETY_TERMINATIONS: dict[str, AgentTermination] = {
    "no_progress": AgentTermination.NO_PROGRESS,
    "diminishing_returns": AgentTermination.NO_PROGRESS,
    "budget_exceeded": AgentTermination.BUDGET_EXCEEDED,
    "approval_required": AgentTermination.APPROVAL_REQUIRED,
    "approval_denied": AgentTermination.APPROVAL_REQUIRED,
}


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
        checkpoint_store: optional :class:`CheckpointStore` for durable
            checkpointing (E2). When provided, the runtime checkpoints after
            each completed step and before terminal transitions, and exposes
            :meth:`resume` for crash recovery.
        tool_gateway: optional :class:`ToolGateway` (E3). When provided,
            :meth:`call_tool` routes every tool invocation through the
            gateway's policy/permission/budget/approval/sandbox chain.
        safety_controller: optional
            :class:`~research_engineer.safety.controller.SafetyController`
            (E5). When provided, every completed step is evaluated by the
            deterministic safety/autonomy policy, which may CONTINUE,
            REPLAN, PAUSE_FOR_APPROVAL (resolved via the controller's
            approval gate), or TERMINATE the run. Tool invocations made
            through :meth:`call_tool` are recorded into the controller for
            duplicate-call and risk-escalation analysis.
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
        checkpoint_store: CheckpointStore | None = None,
        tool_gateway: Any | None = None,
        safety_controller: Any | None = None,
    ) -> None:
        self.planner = planner
        self.actor = actor
        self.observer = observer
        self.evaluator = evaluator
        self.policy = policy or AgentPolicy()
        self._classify = error_classifier or classify_error
        self._on_step = on_step
        self._event_bus = event_bus
        self._checkpoint_store = checkpoint_store
        self._tool_gateway = tool_gateway
        self._safety_controller = safety_controller
        self._cancel_event: asyncio.Event | None = None
        self._resume_lock_held = False
        self._active_ctx: AgentContext | None = None
        self._active_step: AgentStep | None = None

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
        self._active_execution_id = ctx.execution_id
        self._active_ctx = ctx
        start = time.monotonic()
        ctx.state = AgentState.RUNNING
        ctx.started_at = datetime.now()

        # E6: seed the trace/correlation hierarchy for this run. A resumed
        # checkpoint keeps its original run_id/trace_id via ``ctx.metadata``.
        run_id = str(ctx.metadata.get("run_id") or new_run_id())
        trace_id = str(ctx.metadata.get("trace_id") or new_trace_id())
        ctx.metadata["run_id"] = run_id
        ctx.metadata["trace_id"] = trace_id
        self._base_correlation = CorrelationContext(
            run_id=run_id,
            execution_id=ctx.execution_id,
            trace_id=trace_id,
            span_id=new_span_id(),
        )
        _corr_token = set_correlation(self._base_correlation)
        self._emit("start", ctx)

        try:
            with start_span(
                "agent.run",
                {
                    "research_engineer.goal": str(goal),
                    "research_engineer.execution_id": ctx.execution_id,
                    "research_engineer.run_id": run_id,
                },
            ):
                while ctx.state == AgentState.RUNNING:
                    # Budget/timeout/cancellation checks run before each step.
                    if await self._check_termination(ctx, start):
                        break
                    if not await self._process_step(ctx, start):
                        break
        except asyncio.CancelledError:
            await self._terminate(ctx, AgentTermination.CANCELLED, "Run cancelled")
        except Exception as exc:  # noqa: BLE001 - top-level safety net
            logger.exception("AgentRuntime.run failed for %s", ctx.execution_id)
            await self._terminate(
                ctx,
                AgentTermination.ERROR,
                f"Unhandled runtime error: {exc}",
            )
        finally:
            # Release any resume lock acquired by :meth:`resume`.
            if self._resume_lock_held and self._checkpoint_store is not None:
                try:
                    await self._checkpoint_store.release_lock(ctx.execution_id)
                except Exception:  # noqa: BLE001 - best-effort
                    logger.debug("Failed to release resume lock", exc_info=True)
                self._resume_lock_held = False
            reset_correlation(_corr_token)

        ctx.finished_at = datetime.now()
        ctx.duration_seconds = round(time.monotonic() - start, 6)
        self._emit(
            "end",
            ctx,
            duration_seconds=ctx.duration_seconds,
            run_id=run_id,
        )
        return AgentExecution(
            context=ctx,
            termination=ctx.termination or AgentTermination.ERROR,
            reason=ctx.termination_reason,
            output=ctx.output,
        )

    async def resume(self, run_id: str) -> AgentContext:
        """Load a checkpoint and return the restored context for continuation.

        This is the crash-recovery entry point. It:

        1. Loads the checkpoint for ``run_id`` from the configured store.
        2. Validates the schema version (raises
           :class:`CheckpointVersionError` on mismatch).
        3. Acquires a resume lock to prevent concurrent resumes of the same
           run (raises :class:`CheckpointLockError` if already held).

        The returned :class:`AgentContext` can be passed to :meth:`run` to
        continue the run without repeating already-completed steps. The lock
        is released automatically when :meth:`run` completes.

        Raises:
            CheckpointNotFoundError: no checkpoint exists for ``run_id``.
            CheckpointCorruptedError: the stored checkpoint is invalid.
            CheckpointVersionError: unsupported schema version.
            CheckpointLockError: another resumer holds the lock.
        """
        if self._checkpoint_store is None:
            raise CheckpointError(
                "No checkpoint store configured; cannot resume run " + run_id
            )
        checkpoint = await self._checkpoint_store.load(run_id)
        if checkpoint.metadata.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointVersionError(
                "Unsupported checkpoint schema version "
                f"{checkpoint.metadata.schema_version} (expected "
                f"{CHECKPOINT_SCHEMA_VERSION})"
            )
        if not await self._checkpoint_store.acquire_lock(run_id):
            raise CheckpointLockError(
                f"Run {run_id} is already being resumed by another process"
            )
        self._resume_lock_held = True
        self._emit("resume", checkpoint.context, run_id=run_id)
        return checkpoint.context

    async def _process_step(self, ctx: AgentContext, start: float) -> bool:
        """Run one step and post-process it; return False to stop the loop."""
        # E6: give each step its own correlation scope (same trace, new
        # span/step ids) so nested LLM/tool/safety events correlate here.
        step_number = ctx.current_step + 1
        base = get_correlation() or getattr(self, "_base_correlation", None)
        step_corr = (
            base.child(step_id=f"{ctx.execution_id}:step:{step_number}")
            if base is not None
            else None
        )
        token = set_correlation(step_corr) if step_corr is not None else None
        try:
            return await self._process_step_inner(ctx, start, step_number, step_corr)
        finally:
            if token is not None:
                reset_correlation(token)

    async def _process_step_inner(
        self,
        ctx: AgentContext,
        start: float,
        step_number: int,
        step_corr: Any | None,
    ) -> bool:
        step = await self._run_step(ctx)
        ctx.steps.append(step)
        ctx.current_step += 1
        ctx.tool_calls += step.tool_calls
        ctx.tokens += step.tokens
        ctx.cost_usd = round(ctx.cost_usd + step.cost_usd, 6)
        if self._on_step is not None:
            self._on_step(step)
        self._emit(
            "step",
            ctx,
            step=step,
            step_number=step_number,
            step_id=getattr(step_corr, "step_id", None),
            score=step.score,
            best_score=ctx.best_score,
            duration_seconds=step.duration_seconds,
            tokens_used=step.tokens,
            cost_used=step.cost_usd,
            tool_calls_used=step.tool_calls,
        )

        # A fatal error during the step terminates the run.
        if ctx.state != AgentState.RUNNING:
            return False

        # Re-check budgets after the step consumed resources.
        if await self._check_termination(ctx, start):
            return False

        # E6: capture the evaluation outcome as its own event.
        self._emit_evaluation(ctx, step_number, step)

        # Progress tracking from the evaluation score.
        if step.score is not None:
            await self._track_progress(ctx, step.score)
            if ctx.state != AgentState.RUNNING:
                return False

        # E5: autonomy/safety evaluation. Deterministic controls are
        # authoritative and may terminate even when the evaluator claims
        # completion (safety trumps success claims).
        if self._safety_controller is not None:
            if not await self._evaluate_safety(ctx, step):
                return False

        # The evaluator may signal completion by returning a sentinel
        # or by mutating the context; check for a done flag.
        if self._evaluator_says_done(step):
            await self._terminate(
                ctx, AgentTermination.SUCCESS, "Evaluator signalled completion"
            )
        else:
            # Safe execution boundary: checkpoint after each completed step
            # so a crash can resume without repeating this work.
            await self._checkpoint(ctx)
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
    # Autonomy & safety controls (E5)
    # ------------------------------------------------------------------

    async def _evaluate_safety(self, ctx: AgentContext, step: AgentStep) -> bool:
        """Run the safety controller on one step; return True to continue."""
        controller = self._safety_controller
        assert controller is not None  # caller checks
        try:
            decision = await controller.observe_step(
                ctx, step, budget=self.policy.budget
            )
        except Exception as exc:  # noqa: BLE001 - safety must not crash the run
            logger.warning(
                "Safety evaluation failed for %s at step %d: %s",
                ctx.execution_id,
                step.step,
                exc,
            )
            return True

        if decision.action is not None and decision.is_terminal:
            termination = _SAFETY_TERMINATIONS.get(
                decision.trigger, AgentTermination.SAFETY_TERMINATED
            )
            reason = f"safety:{decision.reason_code}: {decision.reason}"
            await self._terminate(ctx, termination, reason)
            return False
        return True

    @property
    def safety_controller(self) -> Any | None:
        """The configured E5 safety controller, if any."""
        return self._safety_controller


    # ------------------------------------------------------------------
    # Tool gateway integration (E3)
    # ------------------------------------------------------------------

    @property
    def tool_gateway(self) -> Any | None:
        """The configured :class:`ToolGateway`, if any."""
        return self._tool_gateway

    async def call_tool(
        self,
        tool_name: str,
        input: Any,
        *,
        agent_name: str = "",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool through the configured gateway (E3).

        When a :class:`ToolGateway` is configured, the invocation passes
        through its full policy/permission/budget/approval/sandbox chain
        and returns a :class:`ToolExecutionResult`. When no gateway is
        configured, this raises :class:`RuntimeError` to make it clear that
        tool calls must be routed through a gateway.

        Args:
            tool_name: Stable tool identifier registered with the gateway.
            input: Tool input (a pydantic model or dict).
            agent_name: Optional agent identifier for observability.
            run_id: Optional parent run identifier for observability.
            metadata: Optional caller metadata attached to the call context.

        Returns:
            A :class:`ToolExecutionResult` from the gateway.
        """
        if self._tool_gateway is None:
            raise RuntimeError(
                "No ToolGateway configured on this AgentRuntime; "
                "tool calls must be routed through a gateway."
            )
        # Prefer the E6 correlation run_id; fall back to the legacy
        # active-execution identifier.
        correlation = get_correlation()
        effective_run_id = (
            run_id
            or (getattr(correlation, "run_id", "") if correlation else "")
            or self._current_run_id()
        )
        result = await self._tool_gateway.execute(
            tool_name,
            input,
            agent_name=agent_name,
            run_id=effective_run_id,
            metadata=metadata,
        )
        # E4/E8 telemetry contract: graders and mining read per-call
        # ``{"tool", "status"}`` entries from ``ctx.metadata["tool_call_log"]``
        # (the scripted factory writes the same shape). Production dispatch
        # via a gateway must record it too so offline analysis sees calls.
        if self._active_ctx is not None:
            if self._active_step is not None:
                # Attribute the call to the in-flight step so budget checks
                # and context summaries count gateway-dispatched work.
                self._active_step.tool_calls += 1
            try:
                log = self._active_ctx.metadata.setdefault("tool_call_log", [])
                log.append({
                    "tool": tool_name,
                    "status": str(getattr(result, "status", "unknown")),
                })
            except Exception:  # noqa: BLE001 - logging never breaks dispatch
                logger.debug("tool_call_log append failed", exc_info=True)
        # E5: feed the safety controller (best-effort, never breaks dispatch).
        if self._safety_controller is not None and self._active_ctx is not None:
            risk_level = None
            registry = getattr(self._tool_gateway, "registry", None)
            policy = registry.get(tool_name) if registry is not None else None
            if policy is not None:
                risk_level = policy.risk_level
            try:
                self._safety_controller.record_tool_call(
                    self._active_ctx,
                    tool_name,
                    input,
                    result,
                    risk_level=risk_level,
                )
            except Exception as exc:  # noqa: BLE001 - recording never breaks
                logger.debug("Safety tool-call recording failed: %s", exc)
        return result

    def _current_run_id(self) -> str:
        """Return the execution id of the active run, if any."""
        return getattr(self, "_active_execution_id", "")

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
        self._active_step = step
        t0 = time.monotonic()

        # --- Plan ---
        ctx.phase = AgentPhase.PLANNING
        try:
            step.plan = await self.planner(ctx)
        except Exception as exc:  # noqa: BLE001 - classified below
            await self._handle_error(ctx, step, exc, AgentPhase.PLANNING)
            return self._finish_step(step, t0)

        # --- Act ---
        ctx.phase = AgentPhase.ACTING
        try:
            step.action = await self.actor(ctx, step.plan)
        except Exception as exc:  # noqa: BLE001 - classified below
            await self._handle_error(ctx, step, exc, AgentPhase.ACTING)
            return self._finish_step(step, t0)

        # --- Observe ---
        ctx.phase = AgentPhase.OBSERVING
        try:
            step.observation = await self.observer(ctx, step.action)
        except Exception as exc:  # noqa: BLE001 - classified below
            await self._handle_error(ctx, step, exc, AgentPhase.OBSERVING)
            return self._finish_step(step, t0)

        # --- Evaluate ---
        ctx.phase = AgentPhase.EVALUATING
        try:
            step.evaluation = await self.evaluator(ctx, step.observation)
        except Exception as exc:  # noqa: BLE001 - classified below
            await self._handle_error(ctx, step, exc, AgentPhase.EVALUATING)
            return self._finish_step(step, t0)

        step.score = self._extract_score(step.evaluation)
        return self._finish_step(step, t0)

    @staticmethod
    def _finish_step(step: AgentStep, t0: float) -> AgentStep:
        """Stamp timestamps/duration on a step and return it."""
        step.finished_at = datetime.now()
        step.duration_seconds = round(time.monotonic() - t0, 6)
        return step

    async def _handle_error(
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
            await self._terminate(
                ctx,
                AgentTermination.ERROR,
                f"Fatal error in {phase.value}: {exc}",
            )
            return False

        ctx.recoverable_errors += 1
        if ctx.recoverable_errors > self.policy.max_recoverable_errors:
            await self._terminate(
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

    async def _check_termination(self, ctx: AgentContext, start: float) -> bool:
        """Return True when the run should stop (budget/timeout/cancel)."""
        budget = self.policy.budget

        if self._cancel_event is not None and self._cancel_event.is_set():
            await self._terminate(ctx, AgentTermination.CANCELLED, "Run cancelled")
            return True

        if budget.max_steps is not None and ctx.current_step >= budget.max_steps:
            await self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Max steps reached ({budget.max_steps})",
            )
            return True

        if budget.max_tool_calls is not None and ctx.tool_calls >= budget.max_tool_calls:
            await self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Max tool calls reached ({budget.max_tool_calls})",
            )
            return True

        if budget.max_runtime_seconds is not None:
            elapsed = time.monotonic() - start
            if elapsed >= budget.max_runtime_seconds:
                await self._terminate(
                    ctx,
                    AgentTermination.TIMEOUT,
                    f"Runtime budget exceeded ({elapsed:.2f}s)",
                )
                return True

        if budget.max_cost_usd is not None and ctx.cost_usd >= budget.max_cost_usd:
            await self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Cost budget exceeded (${ctx.cost_usd:.4f})",
            )
            return True

        if budget.max_tokens is not None and ctx.tokens >= budget.max_tokens:
            await self._terminate(
                ctx,
                AgentTermination.BUDGET_EXCEEDED,
                f"Token budget exceeded ({ctx.tokens})",
            )
            return True

        return False

    async def _track_progress(self, ctx: AgentContext, score: float) -> None:
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
                await self._terminate(
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
        if isinstance(evaluation, dict):
            score = evaluation.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                return float(score)
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

    async def _terminate(
        self, ctx: AgentContext, termination: AgentTermination, reason: str
    ) -> None:
        """Transition the context to a terminal state."""
        ctx.state = AgentState.TERMINATED
        ctx.termination = termination
        ctx.termination_reason = reason
        ctx.output = self._final_output(ctx)
        self._emit("terminate", ctx, termination=termination, reason=reason)
        # Persist the terminal state so a resume can observe it.
        await self._checkpoint(ctx)

    async def _checkpoint(self, ctx: AgentContext) -> None:
        """Best-effort persistence of the current context to the store.

        Failures are logged and surfaced as an observability event but never
        propagate, so checkpointing can never break the run loop.
        """
        if self._checkpoint_store is None:
            return
        try:
            checkpoint = Checkpoint.from_context(ctx)
            await self._checkpoint_store.save(checkpoint)
            self._emit("checkpoint", ctx, step=ctx.current_step)
        except Exception as exc:  # noqa: BLE001 - best-effort by design
            logger.warning(
                "Checkpoint failed for %s at step %d: %s",
                ctx.execution_id,
                ctx.current_step,
                exc,
            )
            self._emit("checkpoint_failed", ctx, error=str(exc))

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
                "run_id": ctx.metadata.get("run_id"),
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

    def _emit_evaluation(self, ctx: AgentContext, step_number: int, step: Any) -> None:
        """Emit a structured ``evaluation`` event for one completed step."""
        try:
            evaluation_text = str(step.evaluation)[:500]
        except Exception:  # noqa: BLE001 - best-effort summary only
            evaluation_text = "<unserializable evaluation>"
        self._emit(
            "evaluation",
            ctx,
            step=step_number,
            score=step.score,
            done=self._evaluator_says_done(step),
            duration_seconds=step.duration_seconds,
            evaluation_summary=evaluation_text,
        )


__all__ = [
    "AgentRuntime",
    "classify_error",
    "Planner",
    "Actor",
    "Observer",
    "Evaluator",
]
