"""E7 - Persistent run store.

Stores run records (status transitions, termination state, metadata,
artifact references, results) so that state survives process restarts.

Two backends behind one interface:

* :class:`SQLiteRunStore` - stdlib ``sqlite3`` default (development).
* :class:`PostgresRunStore` - production backend used when
  ``RE_POSTGRES_DSN`` is configured; lazily imports ``psycopg`` exactly like
  the E2 :class:`~research_engineer.runtime.checkpoint_stores.PostgresCheckpointStore`.

Every mutation goes through a status-transition guard so that a
terminal run can never be resurrected by a racing worker.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from research_engineer.service.models import RunRecord, RunStatus

#: Environment variable reused from E2 for the PostgreSQL DSN.
POSTGRES_DSN_ENV = "RE_POSTGRES_DSN"

_RUN_RECORD_ADAPTER: TypeAdapter[RunRecord] = TypeAdapter(RunRecord)


class RunNotFoundError(Exception):
    """Raised when a run id is unknown."""


class InvalidTransitionError(Exception):
    """Raised when a status transition violates the lifecycle guard."""


def _now_iso() -> str:
    return datetime.now().isoformat()


class BaseRunStore(ABC):
    """Common transition-guard logic shared by both backends."""

    @staticmethod
    def _validate_transition(current: RunStatus, new: RunStatus) -> None:
        if current.is_terminal() and new != current:
            raise InvalidTransitionError(
                f"Run in terminal status {current} cannot move to {new}"
            )

    @abstractmethod
    async def create(self, record: RunRecord) -> None: ...

    @abstractmethod
    async def get(self, run_id: str) -> RunRecord | None: ...

    @abstractmethod
    async def get_required(self, run_id: str) -> RunRecord:
        """Like :meth:`get` but raises :class:`RunNotFoundError`."""

    @abstractmethod
    async def update(self, record: RunRecord) -> RunRecord: ...

    @abstractmethod
    async def find_stale_running(
        self, older_than_seconds: float
    ) -> list[RunRecord]: ...

    @abstractmethod
    async def count_by_status(self) -> dict[str, int]: ...

    async def close(self) -> None:
        return None

    @staticmethod
    def _record_to_row(record: RunRecord) -> dict[str, object]:
        return {
            "run_id": record.run_id,
            "goal": record.goal,
            "status": record.status.value,
            "created_at": record.created_at.isoformat(),
            "updated_at": _now_iso(),
            "started_at": (
                record.started_at.isoformat() if record.started_at else None
            ),
            "finished_at": (
                record.finished_at.isoformat() if record.finished_at else None
            ),
            "worker_id": record.worker_id,
            "claim_count": record.claim_count,
            "cancel_requested": int(record.cancel_requested),
            "error": record.error,
            "termination_reason": record.termination_reason,
            "payload_json": record.model_dump_json(),
        }


class SQLiteRunStore(BaseRunStore):
    """Default development run store backed by stdlib ``sqlite3``."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    worker_id TEXT,
                    claim_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    termination_reason TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                """
            )

    async def create(self, record: RunRecord) -> None:
        row = self._record_to_row(record)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, goal, status, created_at, updated_at,"
                " started_at, finished_at, worker_id, claim_count,"
                " cancel_requested, error, termination_reason, payload_json)"
                " VALUES (:run_id, :goal, :status, :created_at, :updated_at,"
                " :started_at, :finished_at, :worker_id, :claim_count,"
                " :cancel_requested, :error, :termination_reason,"
                " :payload_json)",
                row,
            )

    async def get(self, run_id: str) -> RunRecord | None:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return cast(RunRecord, _RUN_RECORD_ADAPTER.validate_python(json.loads(str(row[0]))))

    async def get_required(self, run_id: str) -> RunRecord:
        record = await self.get(run_id)
        if record is None:
            raise RunNotFoundError(f"Unknown run {run_id}")
        return record

    async def update(self, record: RunRecord) -> RunRecord:
        record.updated_at = datetime.now()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (record.run_id,)
            )
            existing = cur.fetchone()
            if existing is None:
                raise RunNotFoundError(f"Unknown run {record.run_id}")
            current = RunStatus(existing["status"])
            if record.status != current or not current.is_terminal():
                self._validate_transition(current, record.status)
            conn.execute(
                "UPDATE runs SET status=:status, updated_at=:updated_at,"
                " started_at=:started_at, finished_at=:finished_at,"
                " worker_id=:worker_id, claim_count=:claim_count,"
                " cancel_requested=:cancel_requested, error=:error,"
                " termination_reason=:termination_reason,"
                " payload_json=:payload_json WHERE run_id=:run_id",
                self._record_to_row(record),
            )
        return record

    async def find_stale_running(
        self, older_than_seconds: float
    ) -> list[RunRecord]:
        """Runs stuck in running/resumable without recent heartbeats."""
        out: list[RunRecord] = []
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM runs WHERE status IN"
                " ('running', 'resumable')"
            )
            rows = cur.fetchall()
        cutoff_ts = datetime.now().timestamp() - older_than_seconds
        for row in rows:
            try:
                rec = RunRecord.model_validate(json.loads(row["payload_json"]))
            except Exception:  # noqa: BLE001 - skip corrupted rows
                continue
            ref = rec.updated_at or rec.created_at
            if ref.timestamp() < cutoff_ts:
                out.append(rec)
        return out

    async def count_by_status(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status"
            )
            rows = cur.fetchall()
        return {row[0]: int(row[1]) for row in rows}


class PostgresRunStore(BaseRunStore):
    """Production run store backed by PostgreSQL.

    Requires ``psycopg`` (the same optional driver the E2 checkpoint store
    uses). Tables are created automatically on first use.
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
                        "CREATE TABLE IF NOT EXISTS runs ("
                        " run_id TEXT PRIMARY KEY, goal TEXT NOT NULL,"
                        " status TEXT NOT NULL, created_at TEXT NOT NULL,"
                        " updated_at TEXT NOT NULL, started_at TEXT,"
                        " finished_at TEXT, worker_id TEXT,"
                        " claim_count INTEGER NOT NULL DEFAULT 0,"
                        " cancel_requested INTEGER NOT NULL DEFAULT 0,"
                        " error TEXT, termination_reason TEXT,"
                        " payload_json TEXT NOT NULL)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)"
                    )
                self._conn.commit()
            return self._conn

    async def create(self, record: RunRecord) -> None:
        conn = self._ensure()
        row = self._record_to_row(record)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, goal, status, created_at, updated_at,"
                " started_at, finished_at, worker_id, claim_count,"
                " cancel_requested, error, termination_reason, payload_json)"
                " VALUES (%(run_id)s, %(goal)s, %(status)s, %(created_at)s,"
                " %(updated_at)s, %(started_at)s, %(finished_at)s,"
                " %(worker_id)s, %(claim_count)s, %(cancel_requested)s,"
                " %(error)s, %(termination_reason)s, %(payload_json)s)",
                row,
            )
        conn.commit()

    async def get(self, run_id: str) -> RunRecord | None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM runs WHERE run_id = %s", (run_id,)
            )
            r = cur.fetchone()
        if r is None:
            return None
        return cast(RunRecord, _RUN_RECORD_ADAPTER.validate_python(json.loads(str(r[0]))))

    async def get_required(self, run_id: str) -> RunRecord:
        record = await self.get(run_id)
        if record is None:
            raise RunNotFoundError(f"Unknown run {run_id}")
        return record

    async def update(self, record: RunRecord) -> RunRecord:
        record.updated_at = datetime.now()
        current = await self.get(record.run_id)
        if current is None:
            raise RunNotFoundError(f"Unknown run {record.run_id}")
        if record.status != current.status or not current.status.is_terminal():
            self._validate_transition(current.status, record.status)
        conn = self._ensure()
        row = self._record_to_row(record)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE runs SET status=%(status)s, updated_at=%(updated_at)s,"
                " started_at=%(started_at)s, finished_at=%(finished_at)s,"
                " worker_id=%(worker_id)s, claim_count=%(claim_count)s,"
                " cancel_requested=%(cancel_requested)s, error=%(error)s,"
                " termination_reason=%(termination_reason)s,"
                " payload_json=%(payload_json)s WHERE run_id=%(run_id)s",
                row,
            )
        conn.commit()
        return record

    async def find_stale_running(
        self, older_than_seconds: float
    ) -> list[RunRecord]:
        records: list[RunRecord] = []
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM runs WHERE status IN"
                " ('running', 'resumable')"
            )
            rows = cur.fetchall()
        for row in rows:
            rec = RunRecord.model_validate(json.loads(row[0]))
            age = (datetime.now() - rec.updated_at).total_seconds()
            if age > older_than_seconds:
                records.append(rec)
        return records

    async def count_by_status(self) -> dict[str, int]:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) FROM runs GROUP BY status")
            rows = cur.fetchall()
        return {row[0]: int(row[1]) for row in rows}

    async def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                self._conn = None


def build_run_store(config: Any) -> BaseRunStore:
    """Factory choosing the backend from a :class:`ServiceConfig`."""
    dsn = getattr(config, "postgres_dsn", "")
    if dsn:
        return PostgresRunStore(dsn)
    return SQLiteRunStore(getattr(config, "db_path", Path("data/run_service.db")))


__all__ = [
    "BaseRunStore",
    "InvalidTransitionError",
    "POSTGRES_DSN_ENV",
    "PostgresRunStore",
    "RunNotFoundError",
    "SQLiteRunStore",
    "build_run_store",
]
