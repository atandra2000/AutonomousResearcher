"""Lightweight observability: structured event sinks for LLM and stage calls.

Two sink implementations: :class:`JSONLSink` (one JSON object per line) and
:class:`SQLiteSink` (single ``events`` table). The :class:`EventBus` fans
events out to zero or more sinks (best-effort). A process-wide bus is
exposed by :func:`get_event_bus`. Sinks are best-effort: a failing sink
never breaks the caller.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_engineer.llm.base import LLMRequest, LLMResponse


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _prompt_hash(request: LLMRequest) -> str:
    """Stable short hash of the request messages (for log correlation only)."""
    blob = "\n".join(f"{m.role.value}:{m.content}" for m in request.messages)
    if request.model:
        blob += f"\nmodel:{request.model}"
    return f"ph_{abs(hash(blob)) & 0xFFFFFFFFFFFFFFFF:016x}"


class EventSink:
    """Abstract event sink."""

    def emit(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources."""


class JSONLSink(EventSink):
    """Append one JSON object per line to a file path."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"<JSONLSink path={self._path}>"


class NullSink(EventSink):
    """A sink that discards every event (default no-op)."""

    def emit(self, event: dict[str, Any]) -> None:
        return None


class SQLiteSink(EventSink):
    """Write events into a single ``events`` table in a SQLite DB."""

    _DDL = (
        "CREATE TABLE IF NOT EXISTS events ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  ts TEXT NOT NULL,"
        "  kind TEXT NOT NULL,"
        "  payload_json TEXT NOT NULL"
        ")"
    )

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _ensure(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        if not self._initialized:
            self._conn.execute(self._DDL)
            self._conn.commit()
            self._initialized = True
        return self._conn

    def emit(self, event: dict[str, Any]) -> None:
        try:
            with self._lock:
                conn = self._ensure()
                conn.execute(
                    "INSERT INTO events (ts, kind, payload_json) VALUES (?, ?, ?)",
                    (
                        str(event.get("ts", "")),
                        str(event.get("kind", "")),
                        json.dumps(event, default=str),
                    ),
                )
                conn.commit()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self._initialized = False

    def __repr__(self) -> str:
        return f"<SQLiteSink path={self._db_path}>"


class EventBus:
    """Fans events out to zero or more sinks (best-effort)."""

    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self._sinks: list[EventSink] = list(sinks) if sinks else []
        self._lock = threading.Lock()

    def add_sink(self, sink: EventSink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def clear_sinks(self) -> None:
        with self._lock:
            for s in self._sinks:
                try:
                    s.close()
                except Exception:
                    pass
            self._sinks.clear()

    def sinks(self) -> list[EventSink]:
        with self._lock:
            return list(self._sinks)

    def emit(self, event: dict[str, Any]) -> None:
        """Send ``event`` to every sink. Sink failures are swallowed."""
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                pass

    def emit_llm_call(
        self,
        *,
        agent_name: str,
        request: LLMRequest,
        response: LLMResponse,
        latency_seconds: float,
        attempt: int = 1,
    ) -> None:
        """Emit a structured ``llm_call`` event."""
        self.emit(
            {
                "kind": "llm_call",
                "ts": _now_iso(),
                "agent_name": agent_name,
                "prompt_hash": _prompt_hash(request),
                "model": response.model,
                "provider": response.provider,
                "latency_seconds": round(latency_seconds, 6),
                "attempt": attempt,
                "finish_reason": response.finish_reason,
                "truncated": response.truncated,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
                "cost_usd": response.usage.cost_usd,
            }
        )

    def emit_stage(
        self,
        *,
        stage_id: str,
        stage_type: str,
        status: str,
        duration_seconds: float,
        workflow_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Emit a structured ``stage`` event (start/end of a research stage)."""
        self.emit(
            {
                "kind": "stage",
                "ts": _now_iso(),
                "stage_id": stage_id,
                "stage_type": stage_type,
                "status": status,
                "workflow_id": workflow_id,
                "duration_seconds": round(duration_seconds, 6),
                "error": error,
            }
        )


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the process-wide :class:`EventBus` singleton."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Clear and reset the process-wide event bus (tests use this)."""
    global _bus
    with _bus_lock:
        if _bus is not None:
            _bus.clear_sinks()
        _bus = None


__all__ = [
    "EventSink",
    "JSONLSink",
    "SQLiteSink",
    "NullSink",
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
]
