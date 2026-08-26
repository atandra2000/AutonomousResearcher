"""E2 - Concrete checkpoint store implementations.

Three backends for the :class:`CheckpointStore` interface:

* :class:`InMemoryCheckpointStore` - a dict-backed store for tests and
  ephemeral use. Provides best-effort in-process resume locking.
* :class:`SQLiteCheckpointStore` - the default production store, using the
  project's stdlib ``sqlite3`` convention (same as the other storage tools).
  Provides cross-process resume locking via a ``resume_locks`` table.
* :class:`PostgresCheckpointStore` - a production-ready PostgreSQL store.
  The ``psycopg`` driver is imported lazily on first use (mirroring the
  project's optional ChromaDB pattern) so the module imports cleanly without
  it; construction raises a clear :class:`ImportError` only when the store is
  actually instantiated. Connection settings are read from ``RE_``-prefixed
  environment variables.

All stores serialize the :class:`Checkpoint` to JSON for storage, so the
on-disk format is identical across backends and the schema can evolve via
:attr:`CheckpointMetadata.schema_version`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from research_engineer.runtime.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointStore,
    CheckpointVersionError,
)

#: Environment variable for the PostgreSQL connection string.
RE_POSTGRES_DSN = "RE_POSTGRES_DSN"
#: Environment variable for the SQLite database path.
RE_CHECKPOINT_DB = "RE_CHECKPOINT_DB"


def _serialize(checkpoint: Checkpoint) -> str:
    """Serialize a checkpoint to a JSON string (with datetime support)."""
    return checkpoint.model_dump_json()


def _deserialize(payload: str) -> Checkpoint:
    """Deserialize a checkpoint from a JSON string.

    Raises :class:`CheckpointCorruptedError` when the payload is not valid
    checkpoint JSON, and :class:`CheckpointVersionError` when the schema
    version is not supported.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CheckpointCorruptedError(f"Invalid checkpoint JSON: {exc}") from exc
    try:
        checkpoint = Checkpoint.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError
        raise CheckpointCorruptedError(f"Invalid checkpoint payload: {exc}") from exc
    if checkpoint.metadata.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointVersionError(
            "Unsupported checkpoint schema version "
            f"{checkpoint.metadata.schema_version} (expected "
            f"{CHECKPOINT_SCHEMA_VERSION})"
        )
    return checkpoint

class InMemoryCheckpointStore(CheckpointStore):
    """Dict-backed checkpoint store for tests and ephemeral use.

    Not durable across processes; provides best-effort in-process resume
    locking via a lock set.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._locks: set[str] = set()
        self._lock_guard = threading.Lock()

    async def save(self, checkpoint: Checkpoint) -> None:
        self._data[checkpoint.metadata.run_id] = _serialize(checkpoint)

    async def load(self, run_id: str) -> Checkpoint:
        payload = self._data.get(run_id)
        if payload is None:
            raise CheckpointNotFoundError(f"No checkpoint for run {run_id}")
        return _deserialize(payload)

    async def delete(self, run_id: str) -> None:
        self._data.pop(run_id, None)
        self._locks.discard(run_id)

    async def exists(self, run_id: str) -> bool:
        return run_id in self._data

    async def list(self) -> list[CheckpointMetadata]:
        out: list[CheckpointMetadata] = []
        for payload in self._data.values():
            try:
                out.append(_deserialize(payload).metadata)
            except CheckpointError:
                continue
        return out

    async def acquire_lock(self, run_id: str) -> bool:
        with self._lock_guard:
            if run_id in self._locks:
                return False
            self._locks.add(run_id)
            return True

    async def release_lock(self, run_id: str) -> None:
        with self._lock_guard:
            self._locks.discard(run_id)


class SQLiteCheckpointStore(CheckpointStore):
    """SQLite-backed checkpoint store (default production backend).

    Uses the project's stdlib ``sqlite3`` convention (same as the other
    storage tools). Two tables:

    * ``checkpoints`` - one row per run id, storing the serialized
      checkpoint plus a denormalized ``schema_version`` column for fast
      version checks.
    * ``resume_locks`` - one row per held resume lock, providing
      cross-process mutual exclusion for resume.

    Args:
        db_path: Path to the SQLite database file. Defaults to the
            ``RE_CHECKPOINT_DB`` env var or ``data/research_engineer.db``.
    """

    _DDL = (
        "CREATE TABLE IF NOT EXISTS checkpoints ("
        "  run_id TEXT PRIMARY KEY,"
        "  schema_version INTEGER NOT NULL,"
        "  payload_json TEXT NOT NULL,"
        "  created_at TEXT NOT NULL"
        ")",
        "CREATE TABLE IF NOT EXISTS resume_locks ("
        "  run_id TEXT PRIMARY KEY,"
        "  locked_at TEXT NOT NULL"
        ")",
    )

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = str(
            db_path or os.environ.get(RE_CHECKPOINT_DB) or "data/research_engineer.db"
        )
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _ensure(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        if not self._initialized:
            for ddl in self._DDL:
                self._conn.execute(ddl)
            self._conn.commit()
            self._initialized = True
        return self._conn

    async def save(self, checkpoint: Checkpoint) -> None:
        payload = _serialize(checkpoint)
        with self._lock:
            conn = self._ensure()
            conn.execute(
                "INSERT INTO checkpoints (run_id, schema_version, payload_json,"
                " created_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(run_id) DO UPDATE SET"
                " schema_version=excluded.schema_version,"
                " payload_json=excluded.payload_json,"
                " created_at=excluded.created_at",
                (
                    checkpoint.metadata.run_id,
                    checkpoint.metadata.schema_version,
                    payload,
                    checkpoint.created_at.isoformat(),
                ),
            )
            conn.commit()

    async def load(self, run_id: str) -> Checkpoint:
        with self._lock:
            conn = self._ensure()
            row = conn.execute(
                "SELECT payload_json FROM checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"No checkpoint for run {run_id}")
        return _deserialize(row[0])

    async def delete(self, run_id: str) -> None:
        with self._lock:
            conn = self._ensure()
            conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM resume_locks WHERE run_id = ?", (run_id,))
            conn.commit()

    async def exists(self, run_id: str) -> bool:
        with self._lock:
            conn = self._ensure()
            row = conn.execute(
                "SELECT 1 FROM checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        return row is not None

    async def list(self) -> list[CheckpointMetadata]:
        with self._lock:
            conn = self._ensure()
            rows = conn.execute(
                "SELECT payload_json FROM checkpoints"
            ).fetchall()
        out: list[CheckpointMetadata] = []
        for (payload,) in rows:
            try:
                out.append(_deserialize(payload).metadata)
            except CheckpointError:
                continue
        return out

    async def acquire_lock(self, run_id: str) -> bool:
        with self._lock:
            conn = self._ensure()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT 1 FROM resume_locks WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is not None:
                    conn.rollback()
                    return False
                conn.execute(
                    "INSERT INTO resume_locks (run_id, locked_at) VALUES (?, ?)",
                    (run_id, datetime.now().isoformat()),
                )
                conn.commit()
                return True
            except sqlite3.OperationalError:
                conn.rollback()
                return False

    async def release_lock(self, run_id: str) -> None:
        with self._lock:
            conn = self._ensure()
            conn.execute("DELETE FROM resume_locks WHERE run_id = ?", (run_id,))
            conn.commit()

    async def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self._initialized = False


class PostgresCheckpointStore(CheckpointStore):
    """Production-ready PostgreSQL checkpoint store.

    The ``psycopg`` driver is imported lazily on first use so the module
    imports cleanly even when it is not installed; construction raises a
    clear :class:`ImportError` only when the store is actually instantiated
    (mirroring the project's optional ChromaDB pattern).

    Connection settings are read from the ``RE_POSTGRES_DSN`` environment
    variable (a libpq connection string), or from individual ``RE_``-prefixed
    variables. This follows the project's ``RE_``-prefixed env-var convention.

    Args:
        dsn: Optional libpq connection string. Defaults to the
            ``RE_POSTGRES_DSN`` env var.
    """

    _DDL = (
        "CREATE TABLE IF NOT EXISTS checkpoints ("
        "  run_id TEXT PRIMARY KEY,"
        "  schema_version INTEGER NOT NULL,"
        "  payload_json TEXT NOT NULL,"
        "  created_at TIMESTAMPTZ NOT NULL"
        ")",
        "CREATE TABLE IF NOT EXISTS resume_locks ("
        "  run_id TEXT PRIMARY KEY,"
        "  locked_at TIMESTAMPTZ NOT NULL"
        ")",
    )

    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "PostgresCheckpointStore requires the 'psycopg' package. "
                "Install it with: uv pip install 'psycopg[binary]'"
            ) from e
        self._dsn = dsn or os.environ.get(RE_POSTGRES_DSN) or ""
        if not self._dsn:
            raise ValueError(
                f"PostgresCheckpointStore requires a DSN via the "
                f"{RE_POSTGRES_DSN} env var or the 'dsn' argument"
            )
        self._conn: Any = None
        self._initialized = False

    def _ensure(self) -> Any:
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self._dsn)
        if not self._initialized:
            with self._conn.cursor() as cur:
                for ddl in self._DDL:
                    cur.execute(ddl)
            self._conn.commit()
            self._initialized = True
        return self._conn

    async def save(self, checkpoint: Checkpoint) -> None:
        payload = _serialize(checkpoint)
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO checkpoints (run_id, schema_version, payload_json,"
                " created_at) VALUES (%s, %s, %s, %s)"
                " ON CONFLICT(run_id) DO UPDATE SET"
                " schema_version=EXCLUDED.schema_version,"
                " payload_json=EXCLUDED.payload_json,"
                " created_at=EXCLUDED.created_at",
                (
                    checkpoint.metadata.run_id,
                    checkpoint.metadata.schema_version,
                    payload,
                    checkpoint.created_at,
                ),
            )
        conn.commit()

    async def load(self, run_id: str) -> Checkpoint:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM checkpoints WHERE run_id = %s", (run_id,)
            )
            row = cur.fetchone()
        if row is None:
            raise CheckpointNotFoundError(f"No checkpoint for run {run_id}")
        return _deserialize(row[0])

    async def delete(self, run_id: str) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoints WHERE run_id = %s", (run_id,))
            cur.execute("DELETE FROM resume_locks WHERE run_id = %s", (run_id,))
        conn.commit()

    async def exists(self, run_id: str) -> bool:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM checkpoints WHERE run_id = %s", (run_id,))
            return cur.fetchone() is not None

    async def list(self) -> list[CheckpointMetadata]:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("SELECT payload_json FROM checkpoints")
            rows = cur.fetchall()
        out: list[CheckpointMetadata] = []
        for (payload,) in rows:
            try:
                out.append(_deserialize(payload).metadata)
            except CheckpointError:
                continue
        return out

    async def acquire_lock(self, run_id: str) -> bool:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO resume_locks (run_id, locked_at) VALUES (%s, %s)"
                " ON CONFLICT(run_id) DO NOTHING",
                (run_id, datetime.now()),
            )
        conn.commit()
        # Re-check whether our insert actually took effect.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM resume_locks WHERE run_id = %s", (run_id,)
            )
            return cur.fetchone() is not None

    async def release_lock(self, run_id: str) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM resume_locks WHERE run_id = %s", (run_id,))
        conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._initialized = False


def is_postgres_available() -> bool:
    """Return True if the ``psycopg`` driver can be imported."""
    try:
        import psycopg  # noqa: F401
    except Exception:
        return False
    return True


__all__ = [
    "InMemoryCheckpointStore",
    "SQLiteCheckpointStore",
    "PostgresCheckpointStore",
    "is_postgres_available",
    "RE_POSTGRES_DSN",
    "RE_CHECKPOINT_DB",
]

