"""Experiment Comparison Tool for Phase 8.

Compares two or more experiments across metrics, status, duration, and
failure modes. Produces a winner, delta table, and qualitative findings.
"""

from __future__ import annotations

from research_engineer.models.evaluation import (
    ExperimentComparisonInput,
    ExperimentComparisonOutput,
    MetricDelta,
)
from research_engineer.models.experiment import ExperimentRecord
from research_engineer.tools.base import Tool, ToolError


class ExperimentComparisonTool(
    Tool[ExperimentComparisonInput, ExperimentComparisonOutput]
):
    """Compare two or more experiments."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ExperimentComparisonInput) -> bool:
        return len(input.experiments) >= 2

    async def execute(
        self, input: ExperimentComparisonInput
    ) -> ExperimentComparisonOutput:
        try:
            if len(input.experiments) < 2:
                raise ToolError(
                    "Need at least 2 experiments to compare", input
                )

            experiments = input.experiments
            shared, unique = self._metric_sets(experiments)

            metric_deltas: list[MetricDelta] = []
            for metric in sorted(shared):
                metric_deltas.append(
                    self._compute_delta(
                        metric, experiments, input.higher_is_better,
                        input.baseline_experiment_id,
                    )
                )

            winner = self._determine_winner(
                experiments, input.primary_metric, input.higher_is_better
            )

            status_summary = {
                e.experiment_id: e.status.value for e in experiments
            }
            duration_summary = {
                e.experiment_id: e.duration_seconds for e in experiments
            }
            failure_summary = {
                e.experiment_id: e.failure_mode for e in experiments
            }

            findings = self._generate_findings(
                experiments, winner, input.primary_metric,
                input.higher_is_better, shared, failure_summary,
            )
            recommendation = self._recommendation(
                winner, input.primary_metric, findings,
            )

            return ExperimentComparisonOutput(
                experiments_compared=len(experiments),
                primary_metric=input.primary_metric,
                winner_experiment_id=winner,
                metric_deltas=metric_deltas,
                shared_metrics=sorted(shared),
                unique_metrics=unique,
                status_summary=status_summary,
                duration_summary=duration_summary,
                failure_summary=failure_summary,
                findings=findings,
                recommendation=recommendation,
            )
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Comparison failed: {e}", input, e)

    def _metric_sets(
        self, experiments: list[ExperimentRecord]
    ) -> tuple[set[str], dict[str, list[str]]]:
        """Return (shared metrics, unique metrics per experiment)."""
        metric_sets = [
            set(e.metrics.keys()) for e in experiments
        ]
        if not metric_sets:
            return set(), {}
        shared = set.intersection(*metric_sets) if metric_sets else set()
        unique: dict[str, list[str]] = {}
        for e, mset in zip(experiments, metric_sets):
            only = mset - shared
            unique[e.experiment_id] = sorted(only)
        return shared, unique

    def _compute_delta(
        self,
        metric: str,
        experiments: list[ExperimentRecord],
        higher_is_better: bool,
        baseline_id: str | None,
    ) -> MetricDelta:
        values = {
            e.experiment_id: e.metrics.get(metric, 0.0)
            for e in experiments
        }
        present = {
            eid: v for eid, v in values.items() if v is not None
        }
        if not present:
            return MetricDelta(metric=metric)
        if higher_is_better:
            best_id = max(present, key=present.get)  # type: ignore[arg-type]
        else:
            best_id = min(present, key=present.get)  # type: ignore[arg-type]
        best_value = present[best_id]
        deltas = {
            eid: v - best_value for eid, v in present.items()
        }
        improvement: dict[str, float] = {}
        if baseline_id and baseline_id in present:
            base_val = present[baseline_id]
            for eid, v in present.items():
                if base_val == 0:
                    improvement[eid] = 0.0
                else:
                    improvement[eid] = (
                        (base_val - v) / abs(base_val) * 100.0
                        if not higher_is_better
                        else (v - base_val) / abs(base_val) * 100.0
                    )
        return MetricDelta(
            metric=metric,
            values=values,
            best_experiment_id=best_id,
            best_value=best_value,
            deltas=deltas,
            improvement_pct=improvement,
        )

    def _determine_winner(
        self,
        experiments: list[ExperimentRecord],
        primary_metric: str | None,
        higher_is_better: bool,
    ) -> str | None:
        if not primary_metric:
            return None
        candidates = [
            e for e in experiments
            if primary_metric in e.metrics
        ]
        if not candidates:
            return None
        if higher_is_better:
            best = max(
                candidates, key=lambda e: e.metrics[primary_metric]
            )
        else:
            best = min(
                candidates, key=lambda e: e.metrics[primary_metric]
            )
        return best.experiment_id

    def _generate_findings(
        self,
        experiments: list[ExperimentRecord],
        winner: str | None,
        primary_metric: str | None,
        higher_is_better: bool,
        shared: set[str],
        failure_summary: dict[str, str | None],
    ) -> list[str]:
        findings: list[str] = []
        if winner and primary_metric:
            w = next(
                (e for e in experiments if e.experiment_id == winner),
                None,
            )
            if w:
                val = w.metrics.get(primary_metric)
                findings.append(
                    f"Winner on '{primary_metric}': {winner} "
                    f"({val}) "
                    f"({'higher is better' if higher_is_better else 'lower is better'})"
                )
        if shared:
            findings.append(
                f"Shared metrics across runs: {', '.join(sorted(shared))}"
            )
        else:
            findings.append(
                "No shared metrics across runs; comparison limited."
            )
        failed = [
            eid for eid, fm in failure_summary.items() if fm
        ]
        if failed:
            findings.append(
                f"Failed runs: {', '.join(failed)} "
                f"({len(failed)}/{len(experiments)})"
            )
        else:
            findings.append("All compared runs completed without failure.")
        durations = {
            e.experiment_id: e.duration_seconds for e in experiments
        }
        if durations:
            fastest = min(durations, key=durations.get)  # type: ignore[arg-type]
            findings.append(
                f"Fastest run: {fastest} "
                f"({durations[fastest]:.1f}s)"
            )
        return findings

    @staticmethod
    def _recommendation(
        winner: str | None,
        primary_metric: str | None,
        findings: list[str],
    ) -> str:
        if winner and primary_metric:
            return (
                f"Adopt configuration of {winner} "
                f"(best on '{primary_metric}'). "
                f"Ablate the winning change to confirm causality."
            )
        if findings:
            return (
                "No clear winner on primary metric. "
                "Consider running more seeds or exploring orthogonal "
                "axes (data, architecture)."
            )
        return "Insufficient data to recommend a run."
