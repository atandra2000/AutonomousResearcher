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
    ("out of memory", "memory_overflow", "Process memory exceeded"),
    ("memoryerror", "memory_overflow", "Python MemoryError encountered"),
    ("std::bad_alloc", "memory_overflow", "C++ memory allocation failed"),
    ("oom-killer", "memory_overflow", "Kernel OOM killer invoked"),
    ("loss: nan", "numerical_instability", "NaN loss detected in logs"),
    ("loss is nan", "numerical_instability", "NaN loss detected in logs"),
    ("nan loss", "numerical_instability", "NaN loss detected in logs"),
    ("loss: inf", "numerical_instability", "Infinite loss detected in logs"),
    ("loss is inf", "numerical_instability", "Infinite loss detected in logs"),
    ("overflowerror", "numerical_instability", "Numerical overflow encountered"),
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
                res = self._analyze_success_case(run, metrics, input)
                if isinstance(res, FailureDetectorOutput):
                    return res
                (
                    detected,
                    failure_mode,
                    severity,
                    root_cause,
                    snippets,
                    recommendations,
                    lessons,
                    anomalies,
                ) = res

            # Timeout, Cancelled, Crashed, or Failed status
            if run.status != ExperimentStatus.COMPLETED or run.exit_code != 0:
                (
                    s_detected,
                    s_mode,
                    s_sev,
                    s_cause,
                    s_snips,
                    s_recs,
                    s_less,
                    s_anom,
                ) = self._analyze_status_case(run)
                if s_detected:
                    detected = True
                    failure_mode = s_mode
                    severity = s_sev
                    root_cause = s_cause
                    snippets.extend(s_snips)
                    recommendations.extend(s_recs)
                    lessons.extend(s_less)
                    anomalies.extend(s_anom)

            # Metric-based checks (run regardless of exit code)
            if not (
                run.status == ExperimentStatus.COMPLETED
                and run.exit_code == 0
                and metrics
            ):
                m_detected, m_mode, m_sev, m_cause = self._process_metric_anomalies(
                    metrics, failure_mode, anomalies, lessons
                )
                if m_detected:
                    detected = True
                    failure_mode = m_mode
                    severity = m_sev
                    root_cause = m_cause

            # Expected metrics missing
            self._check_missing_expected_metrics(input, metrics, anomalies)

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

    def _process_metric_anomalies(
        self,
        metrics: list[MetricReading],
        current_failure_mode: str | None,
        anomalies: list[AnomalyIndicator],
        lessons: list[str],
    ) -> tuple[bool, str | None, FailureSeverity, str]:
        detected = False
        failure_mode = current_failure_mode
        severity = FailureSeverity.NONE
        root_cause = ""
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
        return detected, failure_mode, severity, root_cause

    @staticmethod
    def _check_missing_expected_metrics(
        input: FailureDetectorInput,
        metrics: list[MetricReading],
        anomalies: list[AnomalyIndicator],
    ) -> None:
        if input.expected_metrics and metrics:
            found_names = {m.name for m in metrics}
            missing = set(input.expected_metrics) - found_names
            if missing:
                anomalies.append(
                    AnomalyIndicator(
                        indicator="missing_expected_metrics",
                        description=f"Expected metrics not found: {sorted(missing)}",
                        confidence=0.6,
                        evidence=[],
                    )
                )

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

    def _analyze_success_case(
        self,
        run: ExperimentRun,
        metrics: list[MetricReading],
        input: FailureDetectorInput,
    ) -> FailureDetectorOutput | tuple[
        bool,
        str | None,
        FailureSeverity,
        str,
        list[str],
        list[str],
        list[str],
        list[AnomalyIndicator],
    ]:
        if not metrics and input.expected_metrics:
            return (
                True,
                "poor_performance",
                FailureSeverity.LOW,
                "Experiment completed but no metrics were produced.",
                [],
                ["Verify that the training script outputs metrics."],
                [
                    "Completed runs without metrics are not useful; "
                    "ensure metric logging is enabled."
                ],
                [],
            )
        metric_anomalies = self._check_metric_anomalies(metrics)
        if metric_anomalies:
            first = metric_anomalies[0]
            return (
                True,
                first.indicator,
                FailureSeverity.HIGH,
                first.description,
                [first.description],
                [
                    "Lower the learning rate.",
                    "Add gradient clipping.",
                    "Check for numerical instability in the model.",
                ],
                [a.description for a in metric_anomalies],
                metric_anomalies,
            )
        return FailureDetectorOutput(
            detected_failure=False,
            failure_mode=None,
            severity=FailureSeverity.NONE,
            root_cause_hypothesis="Experiment completed successfully.",
            recommendations=[],
            lessons_learned=[],
        )

    def _analyze_status_case(
        self, run: ExperimentRun
    ) -> tuple[
        bool,
        str | None,
        FailureSeverity,
        str,
        list[str],
        list[str],
        list[str],
        list[AnomalyIndicator],
    ]:
        if run.status == ExperimentStatus.TIMEOUT:
            return (
                True,
                "timeout",
                FailureSeverity.HIGH,
                f"Experiment timed out after {run.timeout_seconds}s.",
                [f"Timeout: {run.error_message or ''}"],
                [
                    "Increase the timeout or reduce the workload (fewer steps, smaller batch, smaller dataset)."
                ],
                [
                    f"Run exceeded {run.timeout_seconds}s timeout; consider reducing problem size or increasing the limit."
                ],
                [],
            )
        if run.status == ExperimentStatus.CANCELLED:
            return (
                True,
                None,
                FailureSeverity.LOW,
                "Experiment was cancelled.",
                [],
                ["Investigate why the run was cancelled."],
                [],
                [],
            )
        if run.status in (ExperimentStatus.FAILED, ExperimentStatus.CRASHED):
            return self._detect_stderr_patterns(run)

        return False, None, FailureSeverity.NONE, "", [], [], [], []

    def _detect_stderr_patterns(
        self, run: ExperimentRun
    ) -> tuple[
        bool,
        str | None,
        FailureSeverity,
        str,
        list[str],
        list[str],
        list[str],
        list[AnomalyIndicator],
    ]:
        """Scan process logs/exit code for error patterns."""
        anomalies: list[AnomalyIndicator] = []
        snippets: list[str] = []
        recommendations: list[str] = []
        lessons: list[str] = []
        combined = f"{run.stderr}\n{run.stdout}".lower()
        for substring, mode, desc in ERROR_PATTERNS:
            if substring in combined:
                severity = self._severity_for_mode(mode)
                root_cause = self._root_cause_for(mode, run)
                snippet = self._extract_snippet(run.stderr, substring)
                if snippet:
                    snippets.append(snippet)
                recommendations.extend(self._recommendations_for(mode))
                lessons.append(self._lesson_for(mode))
                anomalies.append(
                    AnomalyIndicator(
                        indicator=substring.replace(" ", "_"),
                        description=desc,
                        confidence=0.85,
                        evidence=[snippet] if snippet else [],
                    )
                )
                return (
                    True,
                    mode,
                    severity,
                    root_cause,
                    snippets,
                    recommendations,
                    lessons,
                    anomalies,
                )

        if run.exit_code == 137:
            return (
                True,
                "memory_overflow",
                FailureSeverity.HIGH,
                "Process was killed (exit code 137, SIGKILL), likely due to Out-Of-Memory (OOM).",
                [],
                [
                    "Reduce batch size or model memory overhead.",
                    "Increase memory limit or use gradient accumulation.",
                ],
                ["Exit code 137 indicates OOM killer termination."],
                [],
            )

        root_cause = (
            f"Process exited with code {run.exit_code} but no known error pattern was detected."
        )
        snippet = run.stderr[-500:] if run.stderr else ""
        return (
            True,
            "crash",
            FailureSeverity.HIGH,
            root_cause,
            [snippet] if snippet else [],
            ["Inspect the full logs for the actual error."],
            [f"Exit code {run.exit_code} with unrecognized error; manual log inspection required."],
            [],
        )

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
            "timeout": (
                "Experiment timed out before completion. Increase timeout "
                "or reduce workload."
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
            "timeout": [
                "Increase timeout_seconds parameter.",
                "Reduce dataset size or batch count for faster testing.",
                "Optimize model training speed or enable mixed precision.",
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
            "timeout": (
                "Experiment timed out; increase limit or optimize runtime."
            ),
        }
        return lessons.get(mode, f"Failure mode: {mode}")
