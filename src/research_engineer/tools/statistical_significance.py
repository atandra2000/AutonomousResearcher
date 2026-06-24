"""Statistical Significance Tool for Phase 8.

Evaluates statistical significance of metric differences across
experiments using Welch's t-test, Cohen's d effect size, and 95%
confidence intervals. Pure-Python (no SciPy) via the _stats helper.
"""

from __future__ import annotations

from itertools import combinations

from research_engineer.models.evaluation import (
    SignificanceResult,
    StatisticalSignificanceInput,
    StatisticalSignificanceOutput,
)
from research_engineer.models.experiment import (
    ExperimentRecord,
)
from research_engineer.tools import _stats
from research_engineer.tools.base import Tool, ToolError


class StatisticalSignificanceTool(
    Tool[StatisticalSignificanceInput, StatisticalSignificanceOutput]
):
    """Test statistical significance of metric differences."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: StatisticalSignificanceInput) -> bool:
        return len(input.experiments) >= 2 and bool(input.metric)

    async def execute(
        self, input: StatisticalSignificanceInput
    ) -> StatisticalSignificanceOutput:
        try:
            if len(input.experiments) < 2:
                raise ToolError(
                    "Need at least 2 experiments for significance test",
                    input,
                )
            experiments = input.experiments
            series_map = {
                e.experiment_id: self._extract_series(e, input.metric)
                for e in experiments
            }
            results: list[SignificanceResult] = []
            insufficient: list[str] = []

            for a, b in combinations(experiments, 2):
                sa = series_map[a.experiment_id]
                sb = series_map[b.experiment_id]
                if (
                    len(sa) < input.min_samples
                    or len(sb) < input.min_samples
                ):
                    insufficient.append(
                        f"{a.experiment_id} vs {b.experiment_id}"
                    )
                    continue
                results.append(
                    self._test_pair(
                        a.experiment_id,
                        b.experiment_id,
                        sa,
                        sb,
                        input.metric,
                        input.alpha,
                        input.higher_is_better,
                    )
                )

            best = self._best_experiment(
                experiments, series_map, input.higher_is_better,
                input.min_samples,
            )
            verdict = self._verdict(results, best, input.metric)

            return StatisticalSignificanceOutput(
                results=results,
                overall_verdict=verdict,
                best_experiment_id=best,
                pairwise_count=len(results),
                insufficient_data_pairs=insufficient,
            )
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Significance test failed: {e}", input, e)

    def _extract_series(
        self, exp: ExperimentRecord, metric: str
    ) -> list[float]:
        """Extract a metric series from an experiment record."""
        for s in exp.metric_series:
            if s.name == metric or s.name.lower() == metric.lower():
                return list(s.values)
        lower = metric.lower()
        for s in exp.metric_series:
            if s.name.lower() == lower:
                return list(s.values)
        if metric in exp.metrics:
            return [float(exp.metrics[metric])]
        return []

    def _test_pair(
        self,
        id_a: str,
        id_b: str,
        sa: list[float],
        sb: list[float],
        metric: str,
        alpha: float,
        higher_is_better: bool,
    ) -> SignificanceResult:
        t, p = _stats.welch_p_value(sa, sb)
        df = _stats.welch_degrees_of_freedom(sa, sb)
        ma, mb = _stats.mean(sa), _stats.mean(sb)
        std_a, std_b = _stats.std(sa), _stats.std(sb)
        d = _stats.cohens_d(sa, sb)
        ci = _stats.mean_diff_ci_95(sa, sb, df)
        significant = p < alpha
        if ma > mb:
            direction = "a>b"
        elif ma < mb:
            direction = "a<b"
        else:
            direction = "no difference"
        return SignificanceResult(
            comparison=f"{id_a} vs {id_b}",
            metric=metric,
            mean_a=ma,
            mean_b=mb,
            std_a=std_a,
            std_b=std_b,
            n_a=len(sa),
            n_b=len(sb),
            t_statistic=t,
            p_value=p,
            degrees_of_freedom=df,
            significant=significant,
            effect_size=d,
            effect_size_label=_stats.effect_size_label(d),
            confidence_interval_95=ci,
            direction=direction,
        )

    def _best_experiment(
        self,
        experiments: list[ExperimentRecord],
        series_map: dict[str, list[float]],
        higher_is_better: bool,
        min_samples: int,
    ) -> str | None:
        candidates = {
            eid: vals
            for eid, vals in series_map.items()
            if len(vals) >= min_samples and vals
        }
        if not candidates:
            return None
        means = {eid: _stats.mean(vals) for eid, vals in candidates.items()}
        if higher_is_better:
            return max(means, key=means.get)  # type: ignore[arg-type]
        return min(means, key=means.get)  # type: ignore[arg-type]

    @staticmethod
    def _verdict(
        results: list[SignificanceResult],
        best: str | None,
        metric: str,
    ) -> str:
        if not results:
            return (
                f"Insufficient data to test significance on '{metric}'."
            )
        sig = [r for r in results if r.significant]
        if not sig:
            return (
                f"No statistically significant differences found on "
                f"'{metric}' across {len(results)} pair(s). "
                f"Consider running more seeds."
            )
        sig_summary = "; ".join(
            f"{r.comparison}: p={r.p_value:.4f} ({r.effect_size_label})"
            for r in sig
        )
        best_str = f" Best: {best}." if best else ""
        return (
            f"Found {len(sig)} significant difference(s) on '{metric}'. "
            f"{sig_summary}.{best_str}"
        )
