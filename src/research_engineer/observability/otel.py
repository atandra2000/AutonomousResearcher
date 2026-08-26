"""E6 - Optional OpenTelemetry integration.

OpenTelemetry is *not* a mandatory runtime dependency: everything in this
module degrades gracefully to no-ops when the packages are missing, and a
failing/broken exporter can never break an agent run.

Provides:

* :func:`opentelemetry_available` — feature detection;
* :class:`OTelSpanSink` — an ``EventSink`` that mirrors each event into an
  OpenTelemetry span (used as an adapter on the existing EventBus);
* :func:`start_span` — a best-effort span context manager usable around any
  phase of code; when OTel is absent it yields a no-op;
* :class:`OTelMetricsBridge` — pushes a :class:`MetricsRegistry` snapshot
  into OpenTelemetry metrics (no-op without the SDK).
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from research_engineer.observability.privacy import TelemetryConfig, redact

logger = logging.getLogger(__name__)

#: Attribute key under which our application-level trace_id is exported.
TRACE_ID_ATTR = "research_engineer.trace_id"


def _sdk_available() -> bool:
    return (
        importlib.util.find_spec("opentelemetry") is not None
        and importlib.util.find_spec("opentelemetry.trace") is not None
    )


def opentelemetry_available() -> bool:
    """True when the OpenTelemetry API packages are importable."""
    try:
        return _sdk_available()
    except Exception:  # noqa: BLE001 - feature detection must not raise
        return False


def _tracer() -> Any:
    from opentelemetry import trace  # type: ignore[import-not-found]

    return trace.get_tracer("research_engineer")


def _flatten(value: Any) -> Any:
    """Convert event values into OTel-attribute-friendly scalars."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value][:64]
    if isinstance(value, dict):
        import json

        return json.dumps(value, default=str)[:2048]
    return str(value)


class OTelSpanSink:
    """EventSink that exports each observability event as an OTel span.

    Attach to an existing EventBus with::

        if opentelemetry_available():
            bus.add_sink(OTelSpanSink())
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self._config = config or TelemetryConfig()
        if not opentelemetry_available():
            raise ImportError(
                "OpenTelemetry is not installed. Install the 'telemetry' "
                "extra: uv add 'research-engineer[telemetry]'"
            )
        self._tracer = _tracer()

    def emit(self, event: dict[str, Any]) -> None:
        try:
            name = f"{event.get('kind', 'event')}.{event.get('event') or ''}".rstrip(".")
            attrs: dict[str, Any] = {}
            for key, value in redact(event, self._config).items():
                flattened = _flatten(value)
                if flattened is not None and len(str(key)) <= 128:
                    attrs[key] = flattened
            with self._tracer.start_as_current_span(name, attributes=attrs):
                pass  # instantaneous span representing the completed event
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            logger.debug("OTelSpanSink failed", exc_info=True)

    def close(self) -> None:
        return None


@contextmanager
def start_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Best-effort OTel span context manager.

    When OpenTelemetry is unavailable this yields ``None`` and does nothing.
    Exceptions inside the block are recorded on the span, never suppressed.
    """
    if not opentelemetry_available():
        yield None
        return
    try:
        tracer = _tracer()
        safe_attrs = {
            k: _flatten(v) for k, v in (attributes or {}).items() if v is not None
        }
        with tracer.start_as_current_span(name, attributes=safe_attrs) as span:
            yield span
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        yield None


class OTelMetricsBridge:
    """Push a :class:`MetricsRegistry` snapshot into OpenTelemetry metrics."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._instruments: dict[str, Any] = {}

    def export(self) -> bool:
        """Export one snapshot; returns False when OTel is unavailable."""
        if not opentelemetry_available():
            return False
        try:
            from opentelemetry.metrics import (
                get_meter,  # type: ignore[import-not-found]
            )

            meter = get_meter("research_engineer")
            for name, buckets in self._registry.snapshot().items():
                instrument = self._instruments.get(name)
                for label_key, value in buckets.items():
                    if isinstance(value, dict):  # histogram summary -> avg gauge
                        value = value.get("avg", 0.0)
                    if instrument is None:
                        instrument = meter.create_gauge(name)
                        self._instruments[name] = instrument
                    labels = _parse_labels(label_key)
                    instrument.set(value, labels)
            return True
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            logger.debug("OTelMetricsBridge export failed", exc_info=True)
            return False


def _parse_labels(label_key: str) -> dict[str, str]:
    label_key = label_key.strip("{}")
    out: dict[str, str] = {}
    for part in filter(None, label_key.split(",")):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


__all__ = [
    "TRACE_ID_ATTR",
    "OTelMetricsBridge",
    "OTelSpanSink",
    "opentelemetry_available",
    "start_span",
]
