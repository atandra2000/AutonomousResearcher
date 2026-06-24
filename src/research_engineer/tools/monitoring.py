"""Monitoring Tool for Phase 7.

Analyzes a completed (or in-progress) experiment's captured stdout/stderr,
scans the working directory for checkpoints, and detects lightweight
anomalies in real time.
"""

from __future__ import annotations

import re
from pathlib import Path

from research_engineer.models.experiment import (
    MetricReading,
    MetricType,
    MonitoringInput,
    MonitoringOutput,
)
from research_engineer.tools.base import Tool, ToolError

DEFAULT_METRIC_PATTERNS = [
    (r"loss[=: ]+([0-9.eE+-]+)", "loss", MetricType.LOSS),
    (r"eval[_-]?loss[=: ]+([0-9.eE+-]+)", "eval_loss", MetricType.LOSS),
    (r"acc(?:uracy)?[=: ]+([0-9.]+)", "accuracy", MetricType.ACCURACY),
    (r"lr[=: ]+([0-9.eE+-]+)", "learning_rate", MetricType.SCALAR),
    (r"epoch[=: ]+([0-9]+)", "epoch", MetricType.SCALAR),
    (r"step[=: ]+([0-9]+)", "step", MetricType.SCALAR),
]

CHECKPOINT_GLOBS = ["**/*.pt", "**/*.ckpt", "**/*.safetensors"]


class MonitoringTool(Tool[MonitoringInput, MonitoringOutput]):
    """Analyze experiment output for metrics, checkpoints, and anomalies."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: MonitoringInput) -> bool:
        return bool(input.experiment_id)

    async def execute(self, input: MonitoringInput) -> MonitoringOutput:
        try:
            stdout_lines = input.stdout.splitlines()
            stderr_lines = input.stderr.splitlines()

            metrics = self._scan_metrics(input.stdout, "stdout")
            metrics.extend(self._scan_metrics(input.stderr, "stderr"))

            anomalies: list[str] = []
            self._check_anomalies(metrics, anomalies)
            self._scan_text_anomalies(input.stdout, anomalies)
            self._scan_text_anomalies(input.stderr, anomalies)

            checkpoints = self._scan_checkpoints(input.working_dir)

            stdout_tail = "\n".join(stdout_lines[-input.log_tail_lines :])
            stderr_tail = "\n".join(stderr_lines[-input.log_tail_lines :])

            return MonitoringOutput(
                experiment_id=input.experiment_id,
                final_status=input.status,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
                total_stdout_lines=len(stdout_lines),
                total_stderr_lines=len(stderr_lines),
                elapsed_seconds=0.0,
                metrics_detected=metrics,
                checkpoints_found=checkpoints,
                anomalies=anomalies,
                poll_count=1,
            )
        except Exception as e:
            raise ToolError(f"Monitoring failed: {e}", input, e)

    def _scan_metrics(self, text: str, source: str) -> list[MetricReading]:
        """Scan text for metric readings."""
        readings: list[MetricReading] = []
        for pattern, name, mtype in DEFAULT_METRIC_PATTERNS:
            for match in re.finditer(pattern, text):
                try:
                    value = float(match.group(1))
                    readings.append(
                        MetricReading(
                            name=name,
                            value=value,
                            source=source,
                            metric_type=mtype,
                        )
                    )
                except (ValueError, IndexError):
                    pass
        return readings

    def _check_anomalies(
        self, metrics: list[MetricReading], anomalies: list[str]
    ) -> None:
        """Check metrics for lightweight anomalies."""
        for m in metrics:
            if "loss" in m.name.lower():
                if m.value != m.value:  # NaN check
                    anomalies.append(f"NaN loss detected at source {m.source}")
                elif m.value == float("inf"):
                    anomalies.append(
                        f"Infinite loss detected at source {m.source}"
                    )
                elif m.value > 1e6:
                    anomalies.append(
                        f"Loss explosion: {m.value} at {m.source}"
                    )

    def _scan_text_anomalies(self, text: str, anomalies: list[str]) -> None:
        """Scan raw text for anomaly patterns not captured by metrics."""
        lower = text.lower()
        if "loss=nan" in lower or "loss: nan" in lower or "loss nan" in lower:
            anomalies.append("NaN loss detected in output text")
        if "loss=inf" in lower or "loss: inf" in lower or "loss inf" in lower:
            anomalies.append("Infinite loss detected in output text")

    def _scan_checkpoints(self, working_dir: str | None) -> list[str]:
        """Scan working directory for checkpoint files."""
        if not working_dir:
            return []
        base = Path(working_dir)
        if not base.exists():
            return []
        found: list[str] = []
        for pattern in CHECKPOINT_GLOBS:
            for path in base.glob(pattern):
                found.append(str(path))
        return found
