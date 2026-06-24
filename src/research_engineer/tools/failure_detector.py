"""Failure Detector Tool for Phase 7.

Classifies experiment outcomes and detects failure modes automatically
using rule-based heuristics: exit code, log substring patterns, and
metric checks.
"""

from __future__ import annotations

import math

from research_engineer.models.experiment import (
    AnomalyIndicator,
    ExperimentRun,
    ExperimentStatus,
    FailureDetectorInput,
    FailureDetectorOutput,
    FailureSeverity,
    MetricReading,
)
from research_engineer.tools.base import Tool, ToolError

ERROR_PATTERNS: list[tuple[str, str, str]] = [
    # (substring, failure_mode, description)
    ("cuda out of memory", "memory_overflow", "GPU memory exceeded"),
    ("runtimeerror", "crash", "Runtime error encountered"),
    ("keyerror", "api_incompatibility", "Key error (API mismatch)"),
    ("importerror", "api_incompatibility", "Import error (missing module)"),
    ("attributeerror", "api_incompatibility", "Attribute error"),
    ("modulenotfounderror", "dependency_conflict", "Module not found"),
    ("filenotfounderror", "data_corruption", "File not found"),
    ("permissionerror", "data_corruption", "Permission denied"),
    ("valueerror", "numerical_instability", "Value error"),
    ("zerodivisionerror", "numerical_instability", "Division by zero"),
    ("oserror", "crash", "OS error"),
    ("syntaxerror", "crash", "Syntax error in code"),
    ("checkpoint", "checkpoint_failure", "Checkpoint error"),
]


class FailureDetectorTool(Tool[FailureDetectorInput, FailureDetectorOutput]):
    """Detect and classify experiment failures."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: FailureDetectorInput) -> bool:
        return input.run is not None

    async def execute(self, input: FailureDetectorInput) -> FailureDetectorOutput:
        try:
            run = input.run
            metrics = input.metrics
            anomalies: list[AnomalyIndicator] = []
            snippets: list[str] = []
            recommendations: list[str] = []
            lessons: list[str] = []

            detected = False
            failure_mode: str | None = None
            severity = FailureSeverity.NONE
            root_cause = ""

            # Success case
            if run.status == ExperimentStatus.COMPLETED and run.exit_code == 0:
                if not metrics and input.expected_metrics:
                    detected = True
                    failure_mode = "poor_performance"
                    severity = FailureSeverity.LOW
                    root_cause = (
                        "Experiment completed but no metrics were produced."
                    )
                    recommendations.append(
                        "Verify that the training script outputs metrics."
                    )
                    lessons.append(
                        "Completed runs without metrics are not useful; "
                        "ensure metric logging is enabled."
                    )
                else:
                    # Check for metric anomalies even on success
                    metric_anomalies = self._check_metric_anomalies(metrics)
                    if metric_anomalies:
                        detected = True
                        severity = FailureSeverity.HIGH
                        anomalies.extend(metric_anomalies)
                        first = metric_anomalies[0]
                        failure_mode = first.indicator
                        root_cause = first.description
                        lessons.extend(
                            [a.description for a in metric_anomalies]
                        )
                        recommendations.extend(
                            [
                                "Lower the learning rate.",
                                "Add gradient clipping.",
                                "Check for numerical instability in the model.",
                            ]
                        )
                        snippets.append(root_cause)
                    else:
                        return FailureDetectorOutput(
                            detected_failure=False,
                            failure_mode=None,
                            severity=FailureSeverity.NONE,
                            root_cause_hypothesis="Experiment completed successfully.",
                            recommendations=[],
                            lessons_learned=[],
                        )

            # Timeout
            if run.status == ExperimentStatus.TIMEOUT:
                detected = True
                failure_mode = None
                severity = FailureSeverity.MEDIUM
                root_cause = (
                    f"Experiment timed out after {run.timeout_seconds}s."
                )
                snippets.append(f"Timeout: {run.error_message or ''}")
                recommendations.append(
                    "Increase the timeout or reduce the workload (fewer "
                    "steps, smaller batch, smaller dataset)."
                )
                lessons.append(
                    f"Run exceeded {run.timeout_seconds}s timeout; consider "
                    "reducing problem size or increasing the limit."
                )

            # Cancelled
            if run.status == ExperimentStatus.CANCELLED:
                detected = True
                failure_mode = None
                severity = FailureSeverity.LOW
                root_cause = "Experiment was cancelled."
                recommendations.append("Investigate why the run was cancelled.")

            # Crashed (no exit code)
            if run.status == ExperimentStatus.CRASHED:
                detected = True
                failure_mode = "crash"
                severity = FailureSeverity.HIGH
                root_cause = run.error_message or "Process crashed."
                snippets.append(root_cause)
                recommendations.append(
                    "Check for environment issues, missing dependencies, or "
                    "invalid paths."
                )
                lessons.append("Process crashed before producing output.")

            # Non-zero exit or FAILED status: scan stderr for patterns
            combined = f"{run.stderr}\n{run.stdout}".lower()
            if run.status in (ExperimentStatus.FAILED, ExperimentStatus.CRASHED):
                for substring, mode, desc in ERROR_PATTERNS:
                    if substring in combined:
                        detected = True
                        failure_mode = mode
                        severity = self._severity_for_mode(mode)
                        root_cause = self._root_cause_for(mode, run)
                        snippet = self._extract_snippet(run.stderr, substring)
                        if snippet:
                            snippets.append(snippet)
                        recs = self._recommendations_for(mode)
                        recommendations.extend(recs)
                        lessons.append(self._lesson_for(mode))
                        anomalies.append(
                            AnomalyIndicator(
                                indicator=substring.replace(" ", "_"),
                                description=desc,
                                confidence=0.85,
                                evidence=[snippet] if snippet else [],
                            )
                        )
                        break  # Use first match

                # If no pattern matched but non-zero exit
                if not detected:
                    detected = True
                    failure_mode = "crash"
                    severity = FailureSeverity.HIGH
                    root_cause = (
                        f"Process exited with code {run.exit_code} but no "
                        f"known error pattern was detected."
                    )
                    snippets.append(run.stderr[-500:] if run.stderr else "")
                    recommendations.append(
                        "Inspect the full logs for the actual error."
                    )
                    lessons.append(
                        f"Exit code {run.exit_code} with unrecognized error; "
                        "manual log inspection required."
                    )

            # Metric-based checks (run regardless of exit code)
            # Skip if already checked during success path to avoid duplicates
            if not (
                run.status == ExperimentStatus.COMPLETED
                and run.exit_code == 0
                and metrics
            ):
                metric_anomalies = self._check_metric_anomalies(metrics)
                for anomaly in metric_anomalies:
                    anomalies.append(anomaly)
                    if anomaly.confidence > 0.7:
                        detected = True
                        if failure_mode is None:
                            failure_mode = anomaly.indicator
                            severity = FailureSeverity.HIGH
                            root_cause = anomaly.description
                        lessons.append(anomaly.description)

            # Expected metrics missing
            if input.expected_metrics and metrics:
                found_names = {m.name for m in metrics}
                missing = set(input.expected_metrics) - found_names
                if missing:
                    anomalies.append(
                        AnomalyIndicator(
                            indicator="missing_expected_metrics",
                            description=(
                                f"Expected metrics not found: {sorted(missing)}"
                            ),
                            confidence=0.6,
                            evidence=[],
                        )
                    )

            return FailureDetectorOutput(
                detected_failure=detected,
                failure_mode=failure_mode,
                severity=severity,
                root_cause_hypothesis=root_cause,
                error_snippets=snippets,
                anomaly_indicators=anomalies,
                recommendations=recommendations,
                lessons_learned=lessons,
            )
        except Exception as e:
            raise ToolError(f"Failure detection failed: {e}", input, e)

    def _check_metric_anomalies(
        self, metrics: list[MetricReading]
    ) -> list[AnomalyIndicator]:
        """Check for metric-based anomalies (NaN, inf, divergence)."""
        anomalies: list[AnomalyIndicator] = []
        loss_readings = [m for m in metrics if "loss" in m.name.lower()]
        for m in loss_readings:
            if math.isnan(m.value):
                anomalies.append(
                    AnomalyIndicator(
                        indicator="loss_nan",
                        description=(
                            f"Loss value is NaN (metric: {m.name}, "
                            f"source: {m.source})"
                        ),
                        confidence=0.95,
                        evidence=[f"{m.name}={m.value}"],
                    )
                )
            elif math.isinf(m.value):
                anomalies.append(
                    AnomalyIndicator(
                        indicator="loss_inf",
                        description=(
                            f"Loss value is infinite (metric: {m.name}, "
                            f"source: {m.source})"
                        ),
                        confidence=0.95,
                        evidence=[f"{m.name}={m.value}"],
                    )
                )

        # Divergence: loss increasing significantly
        loss_values = [m.value for m in loss_readings if not math.isnan(m.value)]
        if len(loss_values) >= 4:
            first_half = loss_values[: len(loss_values) // 2]
            second_half = loss_values[len(loss_values) // 2 :]
            if first_half and second_half:
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half)
                if avg_first > 0 and avg_second > avg_first * 1.5:
                    anomalies.append(
                        AnomalyIndicator(
                            indicator="loss_divergence",
                            description=(
                                f"Loss increased from avg {avg_first:.4f} to "
                                f"avg {avg_second:.4f} (divergence)"
                            ),
                            confidence=0.8,
                            evidence=[
                                f"first_half_avg={avg_first:.4f}",
                                f"second_half_avg={avg_second:.4f}",
                            ],
                        )
                    )

        return anomalies

    @staticmethod
    def _extract_snippet(text: str, pattern: str, context: int = 200) -> str:
        """Extract a snippet around a pattern match."""
        if not text:
            return ""
        idx = text.lower().find(pattern)
        if idx == -1:
            return ""
        start = max(0, idx - context)
        end = min(len(text), idx + len(pattern) + context)
        return text[start:end]

    @staticmethod
    def _severity_for_mode(mode: str) -> FailureSeverity:
        """Map a failure mode to a severity."""
        high = {"memory_overflow", "crash", "numerical_instability",
                "gradient_explosion", "gradient_vanishing"}
        medium = {"api_incompatibility", "dependency_conflict",
                  "data_corruption", "divergence", "checkpoint_failure"}
        if mode in high:
            return FailureSeverity.HIGH
        if mode in medium:
            return FailureSeverity.MEDIUM
        return FailureSeverity.LOW

    @staticmethod
    def _root_cause_for(mode: str, run: ExperimentRun) -> str:
        """Generate a root cause hypothesis for a failure mode."""
        causes = {
            "memory_overflow": (
                "GPU memory exceeded. Reduce batch size, use gradient "
                "accumulation, or enable mixed precision."
            ),
            "crash": (
                "Process crashed. Check for runtime errors, invalid "
                "configurations, or environment issues."
            ),
            "api_incompatibility": (
                "API mismatch detected. Check for version changes, renamed "
                "functions, or changed signatures."
            ),
            "dependency_conflict": (
                "Missing or conflicting dependency. Verify the environment "
                "and installed package versions."
            ),
            "data_corruption": (
                "Required file not found or inaccessible. Check dataset "
                "paths and permissions."
            ),
            "numerical_instability": (
                "Numerical instability detected. Check learning rate, "
                "gradient clipping, and mixed precision settings."
            ),
            "checkpoint_failure": (
                "Checkpoint error. Verify checkpoint paths and format "
                "compatibility."
            ),
            "poor_performance": (
                "Experiment produced no useful metrics. Verify the training "
                "script outputs results."
            ),
        }
        return causes.get(mode, f"Unknown failure mode: {mode}")

    @staticmethod
    def _recommendations_for(mode: str) -> list[str]:
        """Generate recommendations for a failure mode."""
        recs = {
            "memory_overflow": [
                "Reduce batch size by 50%.",
                "Enable gradient accumulation to maintain effective batch size.",
                "Use mixed precision (fp16/bf16) training.",
                "Reduce model size or sequence length.",
            ],
            "crash": [
                "Inspect the full traceback in the logs.",
                "Verify all file paths exist.",
                "Check for None values being passed to functions.",
            ],
            "api_incompatibility": [
                "Check the library version against the documented API.",
                "Update or pin the conflicting package.",
                "Review the changelog for renamed functions.",
            ],
            "dependency_conflict": [
                "Install the missing module.",
                "Create a clean virtual environment.",
                "Pin dependency versions in requirements.",
            ],
            "data_corruption": [
                "Verify dataset paths in the config.",
                "Check file permissions.",
                "Re-download or regenerate corrupted data.",
            ],
            "numerical_instability": [
                "Lower the learning rate.",
                "Add gradient clipping (e.g., max_norm=1.0).",
                "Disable mixed precision if using fp16; try bf16 instead.",
                "Add loss scaling for fp16 training.",
            ],
            "checkpoint_failure": [
                "Verify checkpoint file integrity.",
                "Check checkpoint format compatibility.",
                "Try loading with map_location='cpu'.",
            ],
        }
        return recs.get(mode, ["Inspect the logs for more details."])

    @staticmethod
    def _lesson_for(mode: str) -> str:
        """Generate a one-line lesson for a failure mode."""
        lessons = {
            "memory_overflow": (
                "OOM encountered; reduce batch size or enable mixed precision."
            ),
            "crash": "Process crashed; check logs for the root cause.",
            "api_incompatibility": (
                "API mismatch; verify library versions."
            ),
            "dependency_conflict": (
                "Missing dependency; pin versions in the environment."
            ),
            "data_corruption": (
                "File not found; verify data paths in config."
            ),
            "numerical_instability": (
                "Numerical instability; use gradient clipping and lower LR."
            ),
            "checkpoint_failure": (
                "Checkpoint error; verify checkpoint paths and formats."
            ),
            "poor_performance": (
                "No metrics produced; ensure the script logs results."
            ),
        }
        return lessons.get(mode, f"Failure mode: {mode}")
