"""E7 - Agent Worker: long-running execution separated from request handling.

Workers poll the queue, claim runs exclusively, drive the E1
``AgentRuntime`` (with E2 checkpointing attached), persist results, and
honour cancellation requests raised through the run manager.

Reliability semantics:

* ``queued``  - persisted and enqueued; unclaimed.
* ``running`` - exclusively claimed by one worker (claim transaction).
* ``completed`` / ``failed`` / ``cancelled`` - terminal states persisted.
* Worker crash -> the claimed queue entry is gone, so no other worker picks
  it up immediately; the periodic :meth:`RunManager.recover_stale_runs` scan
  marks abandoned runs ``resumable`` and re-queues them. On re-execution the
  E2 checkpoint is resumed so completed steps are never repeated within the
  runtime loop.

At-least-once boundary (documented, unavoidable without idempotency keys):
if a worker dies *after* the agent finished but *before* the terminal state
is committed, recovery restarts the run from the last checkpoint and the
agent may redo partial side effects. Checkpoints themselves are never
corrupted: writes are full-payload upserts guarded by schema version, and
E2 resume locks prevent two executions overlapping.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from research_engineer.runtime.runtime import AgentRuntime
from research_engineer.service.agents import (
    DEFAULT_AGENT_KIND,
    AgentFactoryRegistry,
)
from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.models import RunRecord, RunStatus
from research_engineer.service.queue import RunQueue
from research_engineer.service.store import (
    BaseRunStore,
    InvalidTransitionError,
)
from research_engineer.service.telemetry import ServiceTelemetry

logger = logging.getLogger(__name__)


class AgentWorker:
    """Long-running executor turning queued runs into runtime executions."""

    def __init__(
        self,
        store: BaseRunStore,
        queue: RunQueue,
        checkpoint_store: Any,
        artifacts: ArtifactStore,
        factories: AgentFactoryRegistry | None = None,
        telemetry: ServiceTelemetry | None = None,
        worker_id: str | None = None,
        concurrency: int = 2,
        poll_interval_seconds: float = 0.5,
        recovery_interval_seconds: float = 30.0,
        stale_run_timeout_seconds: float = 120.0,
    ) -> None:
        self.store = store
        self.queue = queue
        self.checkpoint_store = checkpoint_store
        self.artifacts = artifacts
        self.factories = factories or AgentFactoryRegistry()
        self.telemetry = telemetry or ServiceTelemetry()
        self.worker_id = worker_id or f"worker_{os.uname().nodename}_{uuid.uuid4().hex[:6]}"
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval_seconds
        self.recovery_interval = recovery_interval_seconds
        self.stale_timeout = stale_run_timeout_seconds
        self._tasks: set[asyncio.Task[None]] = set()
        self._stopping = False
        self._cancellable: dict[str, AgentRuntime] = {}

    # ------------------------------------------------------------------
    # Loop control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stopping = True

    @property
    def active_count(self) -> int:
        return len([t for t in self._tasks if not t.done()])

    async def serve_forever(self) -> None:
        """Poll/execute until :meth:`stop` is awaited elsewhere."""
        await self.recover_stale_runs()
        last_recovery = asyncio.get_event_loop().time()
        while not self._stopping:
            try:
                await self.try_claim_and_execute()
                now = asyncio.get_event_loop().time()
                if now - last_recovery >= self.recovery_interval:
                    await self.recover_stale_runs()
                    last_recovery = now
            except Exception:  # noqa: BLE001 - the loop must survive errors
                logger.exception("worker poll cycle failed")
            await asyncio.sleep(self.poll_interval)

    async def try_claim_and_execute(self) -> str | None:
        """Claim at most one run and launch it. Returns the run id or None."""
        if len(self._tasks) >= self.concurrency:
            return None
        run_id = await self.queue.claim(self.worker_id)
        if run_id is None:
            return None
        task = asyncio.create_task(self.execute(run_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run_id

    async def recover_stale_runs(self) -> list[str]:
        from research_engineer.service.manager import RunManager

        manager = RunManager(
            self.store,
            self.queue,
            self.telemetry,
            stale_run_timeout_seconds=self.stale_timeout,
        )
        recovered = await manager.recover_stale_runs()
        await manager.close()
        counts = await self.store.count_by_status()
        self.telemetry.depth_gauges(
            await self.queue.depth(), counts.get("running", 0)
        )
        return recovered

    async def drain(self, timeout: float = 10.0) -> None:
        """Wait for in-flight executions to finish."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, run_id: str) -> None:
        """Execute (or resume) one claimed run to a terminal state."""
        record = await self.store.get_required(run_id)
        if record.status.is_terminal():
            await self.queue.ack(run_id)
            return
        if record.cancel_requested:
            await self._finish_cancelled(record)
            return
        if record.status == RunStatus.RUNNING:
            # Duplicate delivery while another execution may be live.
            # Take over only when the previous heartbeat is older than the
            # staleness threshold (true lost-worker scenario); otherwise
            # acknowledge without executing (duplicate suppression).
            age = (datetime.now() - record.updated_at).total_seconds()
            if age < self.stale_timeout:
                await self.queue.ack(run_id)
                return

        from datetime import datetime

        record.status = RunStatus.RUNNING
        record.worker_id = self.worker_id
        record.claim_count += 1
        if record.started_at is None:
            record.started_at = datetime.now()
            record.error = None
        try:
            await self.store.update(record)
        except InvalidTransitionError:
            # The run was terminalized between our read and this write (e.g.
            # cancel won the race). Never resurrect terminal state; just ack.
            latest = await self.store.get_required(record.run_id)
            logger.info(
                "run %s already terminal (%s); skipping execution",
                record.run_id,
                latest.status.value,
            )
            await self.queue.ack(record.run_id)
            return
        latency_of = getattr(self.queue, "latency_of", None)
        if latency_of is not None:
            queued_for = latency_of(run_id)
            if queued_for is not None:
                self.telemetry.queue_latency(run_id, queued_for)

        heartbeat = asyncio.create_task(self._heartbeat(record.run_id))
        try:
            # Resume (and every later failure) is handled inside the guarded
            # region so an error marks the run FAILED instead of escaping as
            # an orphan task exception with the queue entry never acked.
            resumed_context = await self._maybe_resume(record)
            adapter, policy = await self.factories.build(
                str(record.metadata.get("agent_kind", DEFAULT_AGENT_KIND)),
                dict(record.budget_overrides),
            )
            runtime = AgentRuntime(
                planner=adapter.planner,
                actor=adapter.actor,
                observer=adapter.observer,
                evaluator=adapter.evaluator,
                policy=policy,
                checkpoint_store=self.checkpoint_store,
                on_step=self._make_cancel_hook(run_id),
            )
            self._register_runtime(run_id, runtime)
            context = await self._run_runtime(runtime, record, resumed_context)
        except Exception as exc:  # noqa: BLE001 - failures mark run failed
            logger.exception("run %s failed", run_id)
            from datetime import datetime

            record.status = RunStatus.FAILED
            record.finished_at = datetime.now()
            record.error = f"{type(exc).__name__}: {exc}"[:2000]
            await self.store.update(record)
            self.telemetry.run_failed(run_id, record.error)
            return
        finally:
            heartbeat.cancel()
            self._cancellable.pop(run_id, None)

        await self._persist_outcome(record, context)

    async def _heartbeat(self, run_id: str) -> None:
        """Keep ``updated_at`` fresh so recovery never flags live runs stale."""
        interval = max(self.stale_timeout / 4.0, self.poll_interval)
        try:
            while True:
                await asyncio.sleep(interval)
                record = await self.store.get_required(run_id)
                if record.status != RunStatus.RUNNING:
                    return
                await self.store.update(record)
        except asyncio.CancelledError:
            return

    async def _run_runtime(
        self, runtime: AgentRuntime, record: RunRecord, context: Any | None
    ) -> Any:
        """Run (or resume) the AgentRuntime loop; returns the final context."""

        execution = await runtime.run(
            record.goal,
            metadata={"run_id": record.run_id, "worker": self.worker_id},
            context=context,
        )
        return execution.context

    async def _persist_outcome(self, record: RunRecord, context: Any) -> None:
        """Commit the terminal state, result payload, and artifact."""
        from datetime import datetime

        termination = str(getattr(context.termination, "value",
                                  context.termination))
        result_payload = {
            "output": context.output,
            "termination": termination,
            "reason": context.termination_reason,
            "steps": len(context.steps),
            "execution_id": context.execution_id,
        }
        artifact = self.artifacts.persist_result(record.run_id, result_payload)
        record.result = result_payload
        record.artifacts = [artifact]
        record.termination_reason = context.termination_reason or termination
        record.finished_at = datetime.now()
        if termination == "cancelled":
            record.status = RunStatus.CANCELLED
            self.telemetry.run_cancelled(record.run_id)
        elif termination in ("error", "safety_terminated"):
            record.status = RunStatus.FAILED
            record.error = context.termination_reason or termination
            self.telemetry.run_failed(record.run_id, record.error)
        else:
            record.status = RunStatus.COMPLETED
            self.telemetry.run_completed(record.run_id,
                                         float(context.duration_seconds))
        try:
            await self.checkpoint_store.delete(record.run_id)
        except Exception:  # noqa: BLE001 - terminal cleanup is best-effort
            pass
        try:
            await self.store.update(record)
        except InvalidTransitionError:
            # Lost a race: the store already holds a newer terminal truth
            # (typically a cancel landing while the run finished). The run's
            # own artifact stays on disk, but we must not overwrite state.
            latest = await self.store.get_required(record.run_id)
            logger.info(
                "run %s outcome superseded by terminal status %s; result "
                "kept as artifact only",
                record.run_id,
                latest.status.value,
            )
        await self.queue.ack(record.run_id)

    # ------------------------------------------------------------------
    # Resume & cancellation plumbing
    # ------------------------------------------------------------------

    async def _maybe_resume(self, record: RunRecord) -> Any | None:
        """Load a resumable context when a checkpoint exists.

        Runs are resumed only when (a) they were marked resumable by the
        recovery scan or re-claimed after a crash, and (b) the E2
        checkpoint store still holds a checkpoint for the run id.
        """
        if record.claim_count <= 1:
            return None
        try:
            exists = await self.checkpoint_store.exists(record.run_id)
        except Exception:  # noqa: BLE001 - broken stores must not hide runs
            return None
        if not exists:
            return None
        from research_engineer.runtime.models import AgentPolicy

        probe = AgentRuntime(
            planner=_null_planner,
            actor=_null_actor,
            observer=_null_observer,
            evaluator=_null_evaluator,
            policy=AgentPolicy(),
            checkpoint_store=self.checkpoint_store,
        )
        context = await probe.resume(record.run_id)
        # Release the resume lock immediately. Executor concurrency is
        # already serialized by queue-claim exclusivity plus the stale-run
        # takeover guard, and the probe's lock is never released by its own
        # ``run()`` (it never runs) — leaving it held wedges every later
        # recovery of this run with CheckpointLockError.
        try:
            await self.checkpoint_store.release_lock(record.run_id)
        except Exception:  # noqa: BLE001 - best-effort unlock
            logger.debug(
                "resume-lock release failed for %s", record.run_id, exc_info=True
            )
        return context

    def _register_runtime(self, run_id: str, runtime: AgentRuntime) -> None:
        self._cancellable[run_id] = runtime

    def _request_cancellation(self, run_id: str) -> None:
        runtime = self._cancellable.get(run_id)
        if runtime is not None:
            runtime.cancel()

    def _make_cancel_hook(self, run_id: str) -> Any:
        """Sync per-step callback surfacing cancel requests to the runtime.

        The API/manager only flips ``cancel_requested`` in the store; this
        hook polls it after each completed step and calls
        ``runtime.cancel()``, which terminates with ``CANCELLED``.
        """
        worker = self

        def on_step(step: Any) -> None:
            task = asyncio.ensure_future(worker._poll_cancel(run_id))

            def _swallow(done_task: asyncio.Task[bool]) -> None:
                if done_task.done() and not done_task.cancelled():
                    done_task.exception()

            task.add_done_callback(_swallow)

        return on_step

    async def _poll_cancel(self, run_id: str) -> bool:
        try:
            record = await self.store.get_required(run_id)
        except Exception:  # noqa: BLE001 - vanished records cannot be polled
            return False
        if record.cancel_requested:
            self._request_cancellation(run_id)
            return True
        return False

    async def _finish_cancelled(self, record: RunRecord) -> None:
        from datetime import datetime

        record.status = RunStatus.CANCELLED
        record.finished_at = datetime.now()
        await self.store.update(record)
        await self.queue.ack(record.run_id)
        self.telemetry.run_cancelled(record.run_id)


async def _null_planner(ctx: Any) -> None:
    return None


async def _null_actor(ctx: Any, plan: Any) -> None:
    return None


async def _null_observer(ctx: Any, action: Any) -> None:
    return None


async def _null_evaluator(ctx: Any, observation: Any) -> None:
    return None


__all__ = ["AgentWorker"]
