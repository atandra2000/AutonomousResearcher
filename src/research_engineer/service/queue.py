"""E7 - Run queue with reliable claim semantics.

Production backend is PostgreSQL using ``SELECT ... FOR UPDATE SKIP LOCKED``
so multiple workers can poll concurrently without duplicate delivery while a
claim transaction commits. If a worker crashes mid-execution, the claimed
row is gone, and the run is marked ``resumable`` by the recovery scan
(at-least-once across worker crashes — see
:mod:`research_engineer.service.worker` for the exact boundary).

Development/tests use :class:`InMemoryQueue`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class RunQueue(ABC):
    """Abstract work-queue for run ids."""

    @abstractmethod
    async def enqueue(self, run_id: str) -> None:
        """Add a run id to the queue."""

    @abstractmethod
    async def claim(self, worker_id: str) -> str | None:
        """Atomically take one queued run id, or return ``None``."""

    @abstractmethod
    async def ack(self, run_id: str) -> None:
        """Remove the queue entry after successful handling."""

    @abstractmethod
    async def release(self, run_id: str) -> None:
        """Return an unfinished entry to the queue (recovery path)."""

    @abstractmethod
    async def depth(self) -> int: ...

    async def close(self) -> None:
        return None


class InMemoryQueue(RunQueue):
    """Single-process asyncio queue used in development and tests."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._pending: set[str] = set()
        self._enqueued_at: dict[str, datetime] = {}

    def latency_of(self, run_id: str) -> float | None:
        """Seconds between enqueue and now; used by the metrics bridge."""
        t0 = self._enqueued_at.get(run_id)
        if t0 is None:
            return None
        return (datetime.now() - t0).total_seconds()

    def forget(self, run_id: str) -> None:
        self._enqueued_at.pop(run_id, None)

    async def enqueue(self, run_id: str) -> None:
        if run_id not in self._pending:
            self._pending.add(run_id)
            self._enqueued_at[run_id] = datetime.now()
            await self._queue.put(run_id)

    async def claim(self, worker_id: str) -> str | None:
        try:
            run_id = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._pending.discard(run_id)
        return run_id

    async def ack(self, run_id: str) -> None:
        self.forget(run_id)

    async def release(self, run_id: str) -> None:
        self._pending.discard(run_id)
        await self.enqueue(run_id)

    async def depth(self) -> int:
        return self._queue.qsize()


def build_run_queue(config: Any) -> RunQueue:
    """Factory choosing the backend from a :class:`ServiceConfig`."""
    dsn = getattr(config, "postgres_dsn", "")
    if dsn:
        from research_engineer.service.queue_pg import PostgresQueue

        return PostgresQueue(dsn)
    return InMemoryQueue()


__all__ = ["InMemoryQueue", "RunQueue", "build_run_queue"]
