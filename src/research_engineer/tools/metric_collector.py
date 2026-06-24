"""Metric Collector Tool for Phase 7.

Parses metrics from experiment stdout/stderr, JSON files, and CSV files.
Builds time-series and summary metrics using rule-based heuristics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from research_engineer.models.experiment import (
    MetricCollectorInput,
    MetricCollectorOutput,
    MetricPattern,
    MetricReading,
    MetricSeries,
    MetricType,
)
from research_engineer.tools.base import Tool, ToolError

DEFAULT_METRIC_PATTERNS: list[MetricPattern] = [
    MetricPattern(
        name="loss",
        regex=r"loss[=: ]+([0-9.eE+-]+)",
        metric_type=MetricType.LOSS,
        aggregate="last",
    ),
    MetricPattern(
        name="eval_loss",
        regex=r"eval[_-]?loss[=: ]+([0-9.eE+-]+)",
        metric_type=MetricType.LOSS,
        aggregate="min",
    ),
    MetricPattern(
        name="accuracy",
        regex=r"acc(?:uracy)?[=: ]+([0-9.]+)",
        metric_type=MetricType.ACCURACY,
        aggregate="max",
    ),
    MetricPattern(
        name="learning_rate",
        regex=r"lr[=: ]+([0-9.eE+-]+)",
        metric_type=MetricType.SCALAR,
        aggregate="last",
    ),
    MetricPattern(
        name="epoch",
        regex=r"epoch[=: ]+([0-9]+)",
        metric_type=MetricType.SCALAR,
        aggregate="max",
    ),
    MetricPattern(
        name="step",
        regex=r"step[=: ]+([0-9]+)",
        metric_type=MetricType.SCALAR,
        aggregate="max",
    ),
]


class MetricCollectorTool(Tool[MetricCollectorInput, MetricCollectorOutput]):
    """Collect and aggregate metrics from experiment output."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: MetricCollectorInput) -> bool:
        return bool(input.experiment_id)

    async def execute(self, input: MetricCollectorInput) -> MetricCollectorOutput:
        try:
            patterns = input.metric_patterns or DEFAULT_METRIC_PATTERNS
            readings: list[MetricReading] = []
            errors: list[str] = []
            sources: set[str] = set()

            # Parse stdout
            stdout_readings = self._parse_text(input.stdout, patterns, "stdout")
            readings.extend(stdout_readings)
            if input.stdout:
                sources.add("stdout")

            # Parse stderr
            stderr_readings = self._parse_text(input.stderr, patterns, "stderr")
            readings.extend(stderr_readings)
            if input.stderr:
                sources.add("stderr")

            # Parse files in output_dir
            if input.output_dir:
                file_readings, file_errors, file_sources = self._parse_files(
                    input.output_dir
                )
                readings.extend(file_readings)
                errors.extend(file_errors)
                sources.update(file_sources)

            # Build series and summaries
            series = self._build_series(readings)
            summary = self._build_summary(readings, patterns)

            return MetricCollectorOutput(
                metrics=readings,
                metric_series=series,
                summary_metrics=summary,
                parsing_errors=errors,
                sources=sorted(sources),
            )
        except Exception as e:
            raise ToolError(f"Metric collection failed: {e}", input, e)

    def _parse_text(
        self, text: str, patterns: list[MetricPattern], source: str
    ) -> list[MetricReading]:
        """Parse metric readings from text using patterns."""
        readings: list[MetricReading] = []
        if not text:
            return readings
        for pattern in patterns:
            try:
                regex = re.compile(pattern.regex)
            except re.error:
                continue
            for match in regex.finditer(text):
                try:
                    value = float(match.group(1))
                    readings.append(
                        MetricReading(
                            name=pattern.name,
                            value=value,
                            source=source,
                            metric_type=pattern.metric_type,
                        )
                    )
                except (ValueError, IndexError):
                    pass
        return readings

    def _parse_files(
        self, output_dir: str
    ) -> tuple[list[MetricReading], list[str], set[str]]:
        """Parse metric files (JSON, CSV) in a directory."""
        readings: list[MetricReading] = []
        errors: list[str] = []
        sources: set[str] = set()
        base = Path(output_dir)
        if not base.exists():
            return readings, errors, sources

        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    file_readings = self._parse_json_metrics(data, str(path))
                    readings.extend(file_readings)
                    sources.add(f"file:{path.name}")
                except (json.JSONDecodeError, OSError) as e:
                    errors.append(f"JSON parse error in {path}: {e}")
            elif path.suffix == ".csv":
                try:
                    file_readings = self._parse_csv_metrics(path)
                    readings.extend(file_readings)
                    sources.add(f"file:{path.name}")
                except (OSError, ValueError) as e:
                    errors.append(f"CSV parse error in {path}: {e}")
        return readings, errors, sources

    def _parse_json_metrics(
        self, data: object, source: str
    ) -> list[MetricReading]:
        """Parse metrics from a JSON structure."""
        readings: list[MetricReading] = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    readings.append(
                        MetricReading(
                            name=key,
                            value=float(value),
                            source=source,
                        )
                    )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            readings.append(
                                MetricReading(
                                    name=key,
                                    value=float(value),
                                    step=i,
                                    source=source,
                                )
                            )
        return readings

    def _parse_csv_metrics(self, path: Path) -> list[MetricReading]:
        """Parse metrics from a CSV file."""
        readings: list[MetricReading] = []
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return readings
        header = [h.strip() for h in lines[0].split(",")]
        for i, line in enumerate(lines[1:], start=1):
            cols = [c.strip() for c in line.split(",")]
            for j, col in enumerate(cols):
                if j >= len(header):
                    continue
                try:
                    value = float(col)
                    readings.append(
                        MetricReading(
                            name=header[j],
                            value=value,
                            step=i,
                            source=f"file:{path.name}",
                        )
                    )
                except ValueError:
                    pass
        return readings

    def _build_series(
        self, readings: list[MetricReading]
    ) -> list[MetricSeries]:
        """Build time-series from readings grouped by name."""
        by_name: dict[str, list[MetricReading]] = {}
        for r in readings:
            by_name.setdefault(r.name, []).append(r)

        series_list: list[MetricSeries] = []
        for name, items in by_name.items():
            values = [r.value for r in items]
            steps = [r.step if r.step is not None else i for i, r in enumerate(items)]
            if not values:
                continue
            best_val = self._aggregate(values, self._agg_for(name))
            final_val = values[-1]
            best_idx = values.index(best_val) if best_val in values else 0
            best_step = steps[best_idx] if best_idx < len(steps) else None
            series_list.append(
                MetricSeries(
                    name=name,
                    values=values,
                    steps=steps,
                    source=items[0].source,
                    best_value=best_val,
                    best_step=best_step,
                    final_value=final_val,
                )
            )
        return series_list

    def _build_summary(
        self,
        readings: list[MetricReading],
        patterns: list[MetricPattern],
    ) -> dict[str, float]:
        """Build summary metrics dict (name -> aggregated value)."""
        agg_map = {p.name: p.aggregate for p in patterns}
        by_name: dict[str, list[float]] = {}
        for r in readings:
            by_name.setdefault(r.name, []).append(r.value)

        summary: dict[str, float] = {}
        for name, values in by_name.items():
            if not values:
                continue
            agg = agg_map.get(name, "last")
            summary[name] = self._aggregate(values, agg)
        return summary

    @staticmethod
    def _aggregate(values: list[float], agg: str) -> float:
        """Aggregate a list of values."""
        if not values:
            return 0.0
        if agg == "max":
            return max(values)
        if agg == "min":
            return min(values)
        if agg == "mean":
            return sum(values) / len(values)
        return values[-1]  # last

    @staticmethod
    def _agg_for(name: str) -> str:
        """Default aggregate for a metric name."""
        lower = name.lower()
        if "accuracy" in lower:
            return "max"
        if "loss" in lower:
            return "min"
        return "last"
