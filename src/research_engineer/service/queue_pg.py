"""E7 - PostgreSQL queue backend (``FOR UPDATE SKIP LOCKED``).

Split out of :mod:`research_engineer.service.queue` so the in-memory dev
module imports with zero database dependencies. Table ``run_queue``::

    run_id TEXT PRIMARY KEY
    enqueued_at TEXT
    claims INTEGER
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from research_engineer.service.queue import RunQueue


class PostgresQueue(RunQueue):
    """PostgreSQL-backed durable queue.

    A claim transaction selects one row ``FOR UPDATE SKIP LOCKED``, deletes
    it, and commits — exactly-once delivery per committed claim. Recovery
    after a worker crash is handled by the run manager's stale-run scan.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> Any:
        with self._lock:
            if self._conn is None:
                import psycopg

                self._conn = psycopg.connect(self._dsn)
                with self._conn.cursor() as cur:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS run_queue ("
                        " run_id TEXT PRIMARY KEY,"
                        " enqueued_at TEXT NOT NULL,"
                        " claims INTEGER NOT NULL DEFAULT 0)"
                    )
                self._conn.commit()
            return self._conn

    async def enqueue(self, run_id: str) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO run_queue (run_id, enqueued_at, claims)"
                " VALUES (%s, %s, 0) ON CONFLICT (run_id) DO NOTHING",
                (run_id, datetime.now().isoformat()),
            )
        conn.commit()

    async def claim(self, worker_id: str) -> str | None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_id FROM run_queue"
                " ORDER BY enqueued_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            row = cur.fetchone()
            if row is None:
                conn.rollback()
                return None
            run_id = str(row[0])
            cur.execute("DELETE FROM run_queue WHERE run_id = %s", (run_id,))
        conn.commit()
        return run_id

    async def ack(self, run_id: str) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM run_queue WHERE run_id = %s", (run_id,))
        conn.commit()

    async def release(self, run_id: str) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO run_queue (run_id, enqueued_at, claims)"
                " VALUES (%s, %s, COALESCE("
                " (SELECT claims FROM run_queue WHERE run_id = %s), 0))"
                " ON CONFLICT (run_id) DO UPDATE SET"
                " enqueued_at = EXCLUDED.enqueued_at",
                (run_id, datetime.now().isoformat(), run_id),
            )
        conn.commit()

    async def depth(self) -> int:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM run_queue")
            r = cur.fetchone()
        return int(r[0]) if r else 0

    async def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                self._conn = None


__all__ = ["PostgresQueue"]
