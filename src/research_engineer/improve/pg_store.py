"""E8/P4 - PostgreSQL-backed ImprovementStore for multi-process production.

The JSON :class:`~research_engineer.improve.pipeline.ImprovementStore` is
perfect for development and tests, but a multi-process deployment
(API + N workers, concurrent operators) needs transactional semantics:
JSON files only atomically rename whole files, so two operators performing
read-modify-write on the active-pointer table can silently lose updates,
and the append-only decisions log can interleave partial lines.

This store keeps the exact same interface as the JSON store but persists
to PostgreSQL (the same lazy ``psycopg`` import pattern the runtime's
Postgres checkpoint store uses):

* ``baselines`` / ``candidates``: whole-payload rows keyed by immutable id
  (``INSERT ... ON CONFLICT DO UPDATE`` - candidates are immutable in
  content, status/decision fields evolve inside the payload);
* ``decisions``: append-only log with a BIGSERIAL sequence;
* ``active_pointers``: per-component slots guarded by
  ``pg_advisory_xact_lock`` so concurrent operators serialize on
  read-modify-write instead of racing.

Selection is configuration-driven: :func:`build_improvement_store` returns
the PostgreSQL store when a DSN is configured, otherwise the JSON store.
Development and tests keep the JSON store by default.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from research_engineer.improve.models import (
    Baseline,
    ImprovementCandidate,
    PromotionDecision,
)

#: Advisory-lock key namespace (arbitrary but stable; distinct keys per
#: logical resource keep contention minimal).
_LOCK_ACTIVE_TABLE = 8801


class PostgresImprovementStore:
    """ImprovementStore interface backed by PostgreSQL (production)."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: Any | None = None
        self._lock = threading.Lock()

    # -- connection ------------------------------------------------------
    def _ensure(self) -> Any:
        with self._lock:
            if self._conn is None:
                import psycopg

                self._conn = psycopg.connect(self._dsn)
                with self._conn.cursor() as cur:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS improvement_baselines ("
                        " baseline_id TEXT PRIMARY KEY,"
                        " created_at TEXT NOT NULL,"
                        " payload_json TEXT NOT NULL)"
                    )
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS improvement_candidates ("
                        " candidate_id TEXT PRIMARY KEY,"
                        " created_at TEXT NOT NULL,"
                        " payload_json TEXT NOT NULL)"
                    )
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS improvement_decisions ("
                        " id BIGSERIAL PRIMARY KEY,"
                        " candidate_id TEXT NOT NULL,"
                        " payload_json TEXT NOT NULL)"
                    )
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS active_pointers ("
                        " component TEXT PRIMARY KEY,"
                        " active TEXT NOT NULL DEFAULT '',"
                        " previous TEXT NOT NULL DEFAULT '')"
                    )
                self._conn.commit()
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
                self._conn = None

    # -- baselines ---------------------------------------------------------
    def save_baseline(self, baseline: Baseline) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO improvement_baselines (baseline_id, created_at,"
                " payload_json) VALUES (%s, %s, %s)"
                " ON CONFLICT (baseline_id) DO UPDATE SET payload_json ="
                " EXCLUDED.payload_json",
                (
                    baseline.baseline_id,
                    baseline.created_at.isoformat(),
                    baseline.model_dump_json(),
                ),
            )
        conn.commit()

    def load_baseline(self, baseline_id: str) -> Baseline | None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM improvement_baselines"
                " WHERE baseline_id = %s",
                (baseline_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Baseline.model_validate_json(str(row[0]))

    def list_baselines(self) -> list[Baseline]:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM improvement_baselines"
                " ORDER BY created_at"
            )
            rows = cur.fetchall()
        return [Baseline.model_validate_json(str(r[0])) for r in rows]

    # -- candidates ----------------------------------------------------------
    def save_candidate(self, candidate: ImprovementCandidate) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO improvement_candidates (candidate_id,"
                " created_at, payload_json) VALUES (%s, %s, %s)"
                " ON CONFLICT (candidate_id) DO UPDATE SET payload_json ="
                " EXCLUDED.payload_json",
                (
                    candidate.candidate_id,
                    candidate.created_at.isoformat(),
                    candidate.model_dump_json(),
                ),
            )
        conn.commit()

    def load_candidate(self, candidate_id: str) -> ImprovementCandidate | None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM improvement_candidates"
                " WHERE candidate_id = %s",
                (candidate_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return ImprovementCandidate.model_validate_json(str(row[0]))

    def list_candidates(self) -> list[ImprovementCandidate]:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload_json FROM improvement_candidates"
                " ORDER BY created_at"
            )
            rows = cur.fetchall()
        return [
            ImprovementCandidate.model_validate_json(str(r[0])) for r in rows
        ]

    # -- decisions -------------------------------------------------------------
    def record_decision(
        self, candidate_id: str, decision: PromotionDecision
    ) -> None:
        conn = self._ensure()
        with conn.cursor() as cur:
            # Append-only INSERT: BIGSERIAL ordering gives every concurrent
            # operator a durable, never-interleaved audit record.
            cur.execute(
                "INSERT INTO improvement_decisions (candidate_id,"
                " payload_json) VALUES (%s, %s)",
                (candidate_id, decision.model_dump_json()),
            )
        conn.commit()

    def list_decisions(self) -> list[dict[str, Any]]:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT candidate_id, payload_json FROM improvement_decisions"
                " ORDER BY id"
            )
            rows = cur.fetchall()
        return [
            {"candidate_id": str(r[0]), **json.loads(str(r[1]))} for r in rows
        ]

    # -- active version pointers (advisory-locked read-modify-write) -----
    def _locked_active(self, cur: Any) -> dict[str, dict[str, str]]:
        """Serialize concurrent operators on the active-pointer table."""
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ACTIVE_TABLE,))
        cur.execute("SELECT component, active, previous FROM active_pointers")
        table: dict[str, dict[str, str]] = {}
        for row in cur.fetchall():
            table[str(row[0])] = {
                "active": str(row[1]), "previous": str(row[2]),
            }
        return table

    def set_active(self, component: str, candidate_id: str) -> None:
        conn = self._ensure()
        with conn.transaction():
            with conn.cursor() as cur:
                table = self._locked_active(cur)
                slot = table.setdefault(component, {})
                slot["previous"] = slot.get("active", "")
                slot["active"] = candidate_id
                self._write_pointer(cur, component, slot["active"],
                                    slot["previous"])
        conn.commit()

    def get_active(self, component: str) -> str | None:
        conn = self._ensure()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT active FROM active_pointers WHERE component = %s",
                (component,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return str(row[0]) or None

    def restore_previous(self, component: str) -> str | None:
        """Swap back to the recorded previous pointer (JSON-store parity).

        Returns ``""`` when production reverts to the baseline default,
        ``None`` when nothing was ever promoted for this component.
        """
        conn = self._ensure()
        result: str | None = None
        with conn.transaction():
            with conn.cursor() as cur:
                table = self._locked_active(cur)
                slot = table.get(component)
                if not slot:
                    result = None
                else:
                    current = slot.get("active", "")
                    previous = slot.get("previous", "")
                    if not current or current == previous:
                        result = None
                    else:
                        self._write_pointer(cur, component, previous, current)
                        result = previous
        conn.commit()
        return result

    def revert_to(self, component: str, target: str) -> str | None:
        """Point ``component`` at ``target`` (may be ``""`` = baseline)."""
        conn = self._ensure()
        result: str | None = None
        with conn.transaction():
            with conn.cursor() as cur:
                table = self._locked_active(cur)
                slot = table.get(component)
                if not slot:
                    result = None
                else:
                    current = slot.get("active", "")
                    if current == target:
                        result = None
                    elif not target:
                        cur.execute(
                            "DELETE FROM active_pointers WHERE component = %s",
                            (component,),
                        )
                        result = target
                    else:
                        self._write_pointer(cur, component, target, current)
                        result = target
        conn.commit()
        return result

    @staticmethod
    def _write_pointer(
        cur: Any, component: str, active: str, previous: str
    ) -> None:
        cur.execute(
            "INSERT INTO active_pointers (component, active, previous)"
            " VALUES (%s, %s, %s) ON CONFLICT (component) DO UPDATE SET"
            " active = EXCLUDED.active, previous = EXCLUDED.previous",
            (component, active, previous),
        )


def build_improvement_store(
    config: object,
    *,
    default_root: object = "output/improvements",
) -> Any:
    """Factory: PostgreSQL store in production, JSON store otherwise.

    Selects :class:`PostgresImprovementStore` when the configuration
    carries a ``postgres_dsn`` (the same E7 signal that switches the run
    store), and the filesystem :class:`ImprovementStore` for development
    and tests.
    """
    dsn = str(getattr(config, "postgres_dsn", "") or "")
    if dsn:
        return PostgresImprovementStore(dsn)
    from research_engineer.improve.pipeline import ImprovementStore

    return ImprovementStore(str(default_root))


__all__ = ["PostgresImprovementStore", "build_improvement_store"]
