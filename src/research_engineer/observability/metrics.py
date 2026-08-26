"""E6 - Telemetry metrics derived from existing observability events.

The :class:`MetricsSink` is an :class:`~research_engineer.observability.EventSink`
that consumes events already flowing through the :class:`EventBus` and turns
them into labeled counters/histograms held by a :class:`MetricsRegistry`.

Metrics are *derived*, never authoritative: cost accounting lives in the LLM
usage tracker, safety state in the SafetyController, evaluation results in
the eval harness — nothing here duplicates them.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

_MAX_SAMPLES = 4096


def _coerce(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return "{}"
    return "{" + ",".join(f"{k}={v}" for k, v in labels) + "}"


class MetricsRegistry:
    """Thread-safe registry of counters and value observations."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        self._values: dict[str, dict[tuple[tuple[str, str], ...], list[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )

    @staticmethod
    def _labels_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((k, str(v)) for k, v in (labels or {}).items()))

    def increment(
        self, name: str, *, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Add ``value`` to counter ``name``."""
        with self._lock:
            self._counters[name][self._labels_key(labels)] += value

    def observe(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record an observation of ``name`` (histogram/summary)."""
        with self._lock:
            series = self._values[name].setdefault(self._labels_key(labels), [])
            # Cap retained samples to bound memory in long-lived processes.
            if len(series) >= _MAX_SAMPLES:
                series.pop(0)
            series.append(float(value))

    def set_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set the latest value of a gauge-like metric."""
        with self._lock:
            self._values[name][self._labels_key(labels)] = [float(value)]

    def counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Read back a counter value (sums all buckets when unlabeled)."""
        with self._lock:
            buckets = self._counters.get(name)
            if buckets is None:
                return 0.0
            if labels is not None:
                return buckets.get(self._labels_key(labels), 0.0)
            return float(sum(buckets.values()))

    def values(self, name: str, labels: dict[str, str] | None = None) -> list[float]:
        """Return retained observations for ``name``."""
        with self._lock:
            return list(self._values.get(name, {}).get(self._labels_key(labels), []))

    def names(self) -> list[str]:
        """All metric names seen so far."""
        with self._lock:
            return sorted(set(self._counters) | set(self._values))

    def snapshot(self) -> dict[str, Any]:
        """A JSON-serializable summary of all metrics."""
        out: dict[str, Any] = {}
        with self._lock:
            for name, cbuckets in self._counters.items():
                counter_entry: dict[str, Any] = {}
                for clabels, cvalue in cbuckets.items():
                    counter_entry[_fmt_labels(clabels)] = round(cvalue, 6)
                out[name] = counter_entry
            for name, vbuckets in self._values.items():
                gauge_entry: dict[str, Any] = {}
                for vlabels, vseries in vbuckets.items():
                    if not vseries:
                        continue
                    gauge_entry[_fmt_labels(vlabels)] = {
                        "count": len(vseries),
                        "sum": round(sum(vseries), 6),
                        "avg": round(sum(vseries) / len(vseries), 6),
                        "max": max(vseries),
                    }
                if gauge_entry:
                    out[name] = gauge_entry
        return out


class MetricsSink:
    """Event-consumer that maintains a :class:`MetricsRegistry`.

    Implements the :class:`EventSink` protocol (an ``emit(event)`` method),
    so it can be attached directly to any :class:`EventBus`. All processing
    is best-effort: malformed events are skipped.
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()

    # -- EventSink protocol -------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._consume(dict(event))
        except Exception:  # noqa: BLE001 - metrics are best-effort
            pass

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<MetricsSink metrics={len(self.registry.names())}>"

    # -- Internals -----------------------------------------------------------
    def _consume(self, event: dict[str, Any]) -> None:  # noqa: C901 - dispatch table
        kind = event.get("kind")
        sub = event.get("event")
        r = self.registry

        if kind == "agent_runtime":
            execution_id = str(event.get("execution_id") or "")
            if sub == "start":
                r.increment("runs_started")
            elif sub == "terminate" and event.get("termination"):
                term = str(event["termination"]).rsplit(".", 1)[-1]
                outcome = "success" if term == "success" else "failure"
                r.increment(
                    "runs_total",
                    labels={"termination": term, "outcome": outcome},
                )
            elif sub == "step":
                r.increment("steps_total")
                step_no = event.get("step_number")
                if isinstance(step_no, int):
                    r.set_gauge(
                        "steps_per_run", int(step_no), {"execution_id": execution_id}
                    )
                score = _coerce(event.get("score"))
                if score is not None:
                    r.observe("evaluation_score", score)
                    r.increment("evaluations_total")
            elif sub == "error":
                r.increment("runtime_errors_total")
            elif sub == "checkpoint":
                r.increment("checkpoints_total")
            elif sub == "checkpoint_failed":
                r.increment("checkpoint_failures_total")
            elif sub == "resume":
                r.increment("resumes_total")
            elif sub == "safety_decision":
                action = str(event.get("action", ""))
                trigger = str(event.get("trigger", ""))
                r.increment(
                    "safety_decisions_total",
                    labels={"action": action, "trigger": trigger},
                )
                r.increment(
                    "safety_interventions_total", labels={"trigger": trigger}
                )
                if action in ("pause_for_approval",):
                    r.increment("human_interventions_total")

        elif kind == "llm_call":
            model = str(event.get("model", ""))
            provider = str(event.get("provider", ""))
            r.increment(
                "llm_calls_total", labels={"model": model, "provider": provider}
            )
            latency = _coerce(event.get("latency_seconds"))
            if latency is not None:
                r.observe("llm_latency_seconds", latency)
            tokens = event.get("tokens") or {}
            prompt = (
                _coerce(tokens.get("prompt")) if isinstance(tokens, dict) else None
            )
            completion = (
                _coerce(tokens.get("completion"))
                if isinstance(tokens, dict)
                else None
            )
            total = (
                _coerce(tokens.get("total")) if isinstance(tokens, dict) else None
            )
            if total is None and (prompt is not None or completion is not None):
                total = (prompt or 0) + (completion or 0)
            if total is not None:
                r.increment("llm_tokens", value=total, labels={"kind": "total"})
            if completion is not None:
                r.increment(
                    "llm_tokens", value=completion, labels={"kind": "completion"}
                )
            cost = _coerce(event.get("cost_usd"))
            if cost is not None:
                r.increment("llm_cost_usd", value=cost)

        elif kind == "tool_gateway":
            tool = str(event.get("tool_name", ""))
            if sub == "tool_call_end":
                status = str(event.get("status", ""))
                r.increment(
                    "tool_calls_total", labels={"tool": tool, "status": status}
                )
                latency = _coerce(event.get("duration_seconds"))
                if latency is not None:
                    r.observe("tool_latency_seconds", latency)
                    r.increment(
                        "tool_failures_total",
                        labels={"tool": tool, "status": status},
                    )


__all__ = ["MetricsRegistry", "MetricsSink"]
