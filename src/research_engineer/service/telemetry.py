"""E7 - Service/worker metrics bridged into the E6 telemetry stack.

The service layer reuses the E6 :class:`MetricsRegistry` and
:class:`OTelMetricsBridge` rather than adding a new observability stack.
Metric names are prefixed ``service_``; every lifecycle transition also emits
an event on the existing :class:`EventBus` preserving ``run_id``,
``execution_id``, ``trace_id``, ``step_id`` and ``tool_call_id`` attributes.
"""

from __future__ import annotations

import logging
from typing import Any

from research_engineer.observability import EventBus, get_event_bus
from research_engineer.observability.metrics import MetricsRegistry
from research_engineer.observability.otel import (
    OTelMetricsBridge,
    opentelemetry_available,
)

logger = logging.getLogger(__name__)


class ServiceTelemetry:
    """Counters/gauges/timings for API and worker behaviour."""

    def __init__(
        self,
        registry: MetricsRegistry | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.registry = registry or MetricsRegistry()
        self.bus = bus or get_event_bus()
        self._otel_bridge = OTelMetricsBridge(self.registry)

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def run_submitted(self, run_id: str) -> None:
        self.registry.increment("service_runs_submitted")
        self._emit("run_submitted", run_id=run_id)

    _active_hint: int = 0

    def run_started(self, run_id: str, execution_id: str) -> None:
        self.registry.increment("service_runs_started")
        self.registry.set_gauge(
            "service_active_runs",
            float(self._active_hint),
            {"source": "worker"},
        )
        self._emit(
            "run_started", run_id=run_id, execution_id=execution_id
        )

    def run_completed(self, run_id: str, duration_seconds: float) -> None:
        self.registry.increment("service_runs_completed")
        self.registry.observe("service_run_duration_seconds", duration_seconds)
        self._emit("run_completed", run_id=run_id)

    def run_failed(self, run_id: str, error: str) -> None:
        self.registry.increment("service_runs_failed")
        self.registry.increment("service_worker_failures")
        self._emit("run_failed", run_id=run_id, error=error[:256])

    def run_cancelled(self, run_id: str) -> None:
        self.registry.increment("service_runs_cancelled")
        self._emit("run_cancelled", run_id=run_id)

    def run_recovered(self, run_id: str) -> None:
        self.registry.increment("service_run_recoveries")
        self._emit("run_recovered", run_id=run_id)

    def queue_latency(self, run_id: str, seconds: float) -> None:
        self.registry.observe("service_queue_latency_seconds", seconds)
        self._emit("queue_latency", run_id=run_id, seconds=round(seconds, 4))

    def depth_gauges(self, queued: int, active: int) -> None:
        self.registry.set_gauge("service_queued_runs", float(queued))
        self.registry.set_gauge("service_active_runs", float(active))
        if opentelemetry_available():
            try:
                self._otel_bridge.export()
            except Exception:  # noqa: BLE001 - telemetry must not break runs
                logger.debug("otel bridge export failed", exc_info=True)

    def set_active_hint(self, count: int) -> None:
        self._active_hint = count

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event: str, **fields: Any) -> None:
        payload = {"kind": "service", "event": event}
        payload.update(fields)
        try:
            self.bus.emit(payload)
        except Exception:  # noqa: BLE001 - telemetry must not break runs
            logger.debug("event bus emit failed", exc_info=True)


__all__ = ["ServiceTelemetry"]
