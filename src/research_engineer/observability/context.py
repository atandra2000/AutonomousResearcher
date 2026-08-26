"""E6 - Trace context and stable correlation identifiers.

Every autonomous run carries a coherent correlation hierarchy::

    run_id            one autonomous run (may include several executions)
      execution_id    one runtime execution (checkpoint/resume keeps it)
        trace_id      distributed-trace identifier (OpenTelemetry compatible)
          span_id     one span (step / tool call / LLM call / ...)

The *current* :class:`CorrelationContext` is carried in a ``ContextVar`` so
concurrently executing runs remain fully isolated: asyncio tasks inherit a
copy of the context, so two simultaneous runs never cross-correlate.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Any

_CORRELATION_KEY_ORDER = (
    "trace_id",
    "span_id",
    "parent_span_id",
    "run_id",
    "execution_id",
    "step_id",
    "tool_call_id",
)


def _short(prefix: str, n: int = 16) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:n]}"


def new_trace_id() -> str:
    """Return a new trace identifier (32-hex-body, W3C-like length)."""
    return _short("tr", 32)


def new_span_id() -> str:
    """Return a new span identifier."""
    return _short("sp", 16)


def new_run_id() -> str:
    """Return a new run identifier."""
    return _short("run", 12)


@dataclass(frozen=True)
class CorrelationContext:
    """Immutable correlation identifiers attached to code being observed."""

    run_id: str = ""
    execution_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str | None = None
    step_id: str | None = None
    tool_call_id: str | None = None

    def child(
        self,
        *,
        span_id: str | None = None,
        step_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> CorrelationContext:
        """Derive a child context (same run/trace, new span linkage)."""
        sid = span_id or new_span_id()
        return replace(
            self,
            parent_span_id=self.span_id or self.parent_span_id,
            span_id=sid,
            step_id=step_id,
            tool_call_id=tool_call_id,
        )


_CURRENT: ContextVar[CorrelationContext | None] = ContextVar(
    "research_engineer_correlation", default=None
)


def get_correlation() -> CorrelationContext | None:
    """Return the current correlation context, if any."""
    return _CURRENT.get()


def set_correlation(ctx: CorrelationContext) -> Token[CorrelationContext | None]:
    """Set the current correlation context and return a reset token."""
    return _CURRENT.set(ctx)


def reset_correlation(token: Token[CorrelationContext | None]) -> None:
    """Restore the previous correlation context."""
    _CURRENT.reset(token)


def stamp_correlation(event: dict[str, Any]) -> dict[str, Any]:
    """Stamp the current correlation identifiers onto ``event`` in place.

    Existing non-empty correlation fields on the event win, so explicit
    emitters (e.g. the gateway's ``call_id``) are preserved.
    """
    ctx = _CURRENT.get()
    if ctx is None:
        return event
    for key in _CORRELATION_KEY_ORDER:
        value = getattr(ctx, key, None)
        if value and not event.get(key):
            event[key] = value
    return event


__all__ = [
    "CorrelationContext",
    "get_correlation",
    "reset_correlation",
    "set_correlation",
    "stamp_correlation",
    "new_run_id",
    "new_span_id",
    "new_trace_id",
]
