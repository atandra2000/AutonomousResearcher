"""E7 - Run Manager: request-side orchestration.

The :class:`RunManager` owns the run lifecycle from the API's perspective:
validation, persistence, queueing, cancellation, and resume. It never
executes agents itself — API requests return as soon as the run record is
persisted and the run id is queued.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from research_engineer.service.models import (
    CreateRunRequest,
    RunRecord,
    RunStatus,
)
from research_engineer.service.queue import RunQueue
from research_engineer.service.store import BaseRunStore
from research_engineer.service.telemetry import ServiceTelemetry


class RunManagerError(Exception):
    """Base class for manager-facing errors."""


class ResumeNotAvailableError(RunManagerError):
    """Raised when a run cannot be resumed from a checkpoint."""


class RunManager:
    """Submit/query/cancel/resume runs without blocking on execution."""

    def __init__(
        self,
        store: BaseRunStore,
        queue: RunQueue,
        telemetry: ServiceTelemetry | None = None,
        stale_run_timeout_seconds: float = 120.0,
    ) -> None:
        self._store = store
        self._queue = queue
        self._telemetry = telemetry or ServiceTelemetry()
        self.stale_timeout = stale_run_timeout_seconds

    # ------------------------------------------------------------------
    # Submit / query
    # ------------------------------------------------------------------

    async def submit(self, request: CreateRunRequest) -> RunRecord:
        """Validate, persist, and queue a new run. Never blocks on execution."""
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        overrides: dict[str, object] = {
            k: v for k, v in request.budget_overrides.items()
            if isinstance(v, (str, int, float, bool))
        }
        if request.max_steps is not None:
            overrides["max_steps"] = request.max_steps
        if request.max_runtime_seconds is not None:
            overrides["max_runtime_seconds"] = request.max_runtime_seconds
        # Only scalar metadata survives; complex values are dropped.
        safe_meta = {
            k: v for k, v in (request.metadata or {}).items()
            if isinstance(v, (str, int, float, bool))
        }
        record = RunRecord(
            run_id=run_id,
            goal=request.goal,
            status=RunStatus.QUEUED,
            metadata=safe_meta,
            budget_overrides=dict(overrides),
        )
        await self._store.create(record)
        await self._queue.enqueue(run_id)
        self._telemetry.run_submitted(run_id)
        return record

    async def get(self, run_id: str) -> RunRecord:
        try:
            record: RunRecord = await self._store.get_required(run_id)
        except Exception as exc:  # noqa: BLE001 - normalised for the API layer
            raise RunManagerError(f"Run {run_id} not found") from exc
        return record

    async def counts(self) -> dict[str, int]:
        return await self._store.count_by_status()

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    async def cancel(self, run_id: str) -> tuple[bool, RunRecord]:
        """Request cancellation.

        * queued/resumable -> cancelled immediately
        * running          -> cancel_requested flag set; the worker honours
          it via its shared cancellation registry
        * terminal         -> no-op returning False
        """
        record = await self.get(run_id)
        status = record.status
        if status.is_terminal():
            return False, record
        if status in (RunStatus.QUEUED, RunStatus.RESUMABLE):
            record.status = RunStatus.CANCELLED
            record.finished_at = datetime.now()
            await self._store.update(record)
            await self._queue.ack(run_id)
            self._telemetry.run_cancelled(run_id)
            return True, record
        # running: set the flag and let the worker terminate the runtime.
        record.cancel_requested = True
        await self._store.update(record)
        return True, record

    # ------------------------------------------------------------------
    # Recovery / resume
    # ------------------------------------------------------------------

    async def recover_stale_runs(self) -> list[str]:
        """Mark abandoned ``running`` runs ``resumable`` and requeue them.

        Called periodically by workers (and at startup) so a worker crash
        never permanently loses a run. The checkpoint is left untouched —
        E2 resume locking guarantees two executions cannot overlap.
        """
        recovered: list[str] = []
        stale = await self._store.find_stale_running(self.stale_timeout)
        for record in stale:
            if record.cancel_requested:
                continue
            record.status = RunStatus.RESUMABLE
            record.error = "recovered after lost worker"
            await self._store.update(record)
            await self._queue.release(record.run_id)
            self._telemetry.run_recovered(record.run_id)
            recovered.append(record.run_id)
        return recovered

    async def resume(self, run_id: str) -> RunRecord:
        """Requeue a resumable or failed run."""
        record = await self.get(run_id)
        if record.status == RunStatus.RUNNING:
            raise ResumeNotAvailableError(f"Run {run_id} is currently running")
        if record.status.is_terminal() and record.status != RunStatus.FAILED:
            raise ResumeNotAvailableError(
                f"Run {run_id} is terminal ({record.status.value})"
            )
        # Only FAILED or RESUMABLE runs may be resumed.
        record.status = RunStatus.QUEUED
        record.worker_id = None
        await self._store.update(record)
        await self._queue.enqueue(run_id)
        self._telemetry.run_submitted(run_id)
        return record

    async def close(self) -> None:
        await self._queue.close()


__all__ = ["ResumeNotAvailableError", "RunManager", "RunManagerError"]
