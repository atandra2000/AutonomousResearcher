"""Tests for Phase 8 evaluation tools."""

import pytest

from research_engineer.models.evaluation import (
    DynamicsPatternType,
    ExperimentComparisonInput,
    NextExperimentInput,
    RecommendationPriority,
    StatisticalSignificanceInput,
    TrainingDynamicsInput,
)
from research_engineer.models.experiment import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentType,
    MetricSeries,
)
from research_engineer.tools import _stats
from research_engineer.tools.evaluation_storage import EvaluationStorageTool
from research_engineer.tools.experiment_comparison import (
    ExperimentComparisonTool,
)
from research_engineer.tools.next_experiment import NextExperimentTool
from research_engineer.tools.statistical_significance import (
    StatisticalSignificanceTool,
)
from research_engineer.tools.training_dynamics import TrainingDynamicsTool


def _record(
    exp_id: str = "exp1",
    metrics: dict[str, float] | None = None,
    series: list[MetricSeries] | None = None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    failure_mode: str | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=exp_id,
        repo_path="./repo",
        command=["python", "train.py"],
        experiment_type=ExperimentType.TRAINING,
        status=status,
        start_time="2026-01-01T00:00:00",
        metrics=metrics or {},
        metric_series=series or [],
        failure_mode=failure_mode,
    )


def _series(name: str, values: list[float]) -> MetricSeries:
    return MetricSeries(
        name=name,
        values=values,
        steps=list(range(len(values))),
        best_value=min(values) if "loss" in name.lower() else max(values),
        best_step=values.index(min(values))
        if "loss" in name.lower()
        else values.index(max(values)),
        final_value=values[-1],
    )


# ---------------------------------------------------------------------------
# ExperimentComparisonTool
# ---------------------------------------------------------------------------


class TestExperimentComparison:
    @pytest.mark.asyncio
    async def test_two_run_comparison_loss(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 1.5}),
                    _record("b", {"loss": 1.0}),
                ],
                primary_metric="loss",
                higher_is_better=False,
            )
        )
        assert out.experiments_compared == 2
        assert out.winner_experiment_id == "b"
        assert out.shared_metrics == ["loss"]
        assert "loss" in [d.metric for d in out.metric_deltas]
        delta = next(d for d in out.metric_deltas if d.metric == "loss")
        assert delta.best_experiment_id == "b"

    @pytest.mark.asyncio
    async def test_higher_is_better(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"accuracy": 0.8}),
                    _record("b", {"accuracy": 0.9}),
                ],
                primary_metric="accuracy",
                higher_is_better=True,
            )
        )
        assert out.winner_experiment_id == "b"

    @pytest.mark.asyncio
    async def test_three_run_comparison(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 1.5, "acc": 0.7}),
                    _record("b", {"loss": 1.0, "acc": 0.8}),
                    _record("c", {"loss": 1.2, "acc": 0.9}),
                ],
                primary_metric="loss",
                higher_is_better=False,
            )
        )
        assert out.experiments_compared == 3
        assert out.winner_experiment_id == "b"

    @pytest.mark.asyncio
    async def test_baseline_improvement(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 2.0}),
                    _record("b", {"loss": 1.0}),
                ],
                primary_metric="loss",
                higher_is_better=False,
                baseline_experiment_id="a",
            )
        )
        delta = next(d for d in out.metric_deltas if d.metric == "loss")
        assert "a" in delta.improvement_pct
        assert delta.improvement_pct["b"] == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_no_shared_metrics(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 1.0}),
                    _record("b", {"acc": 0.9}),
                ],
            )
        )
        assert out.shared_metrics == []
        assert "No shared metrics" in " ".join(out.findings)

    @pytest.mark.asyncio
    async def test_findings_and_recommendation(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 1.5}, status=ExperimentStatus.FAILED, failure_mode="crash"),
                    _record("b", {"loss": 1.0}),
                ],
                primary_metric="loss",
                higher_is_better=False,
            )
        )
        assert any("Winner" in f for f in out.findings)
        assert any("Failed" in f for f in out.findings)
        assert "Adopt configuration of b" in out.recommendation

    @pytest.mark.asyncio
    async def test_too_few_experiments_raises(self):
        tool = ExperimentComparisonTool()
        with pytest.raises(Exception):
            await tool.execute(
                ExperimentComparisonInput(experiments=[_record("a")])
            )

    @pytest.mark.asyncio
    async def test_no_primary_metric_no_winner(self):
        tool = ExperimentComparisonTool()
        out = await tool.execute(
            ExperimentComparisonInput(
                experiments=[
                    _record("a", {"loss": 1.5}),
                    _record("b", {"loss": 1.0}),
                ],
            )
        )
        assert out.winner_experiment_id is None


# ---------------------------------------------------------------------------
# TrainingDynamicsTool
# ---------------------------------------------------------------------------


class TestTrainingDynamics:
    @pytest.mark.asyncio
    async def test_overfitting(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series("loss", [2.0, 1.5, 1.0, 0.8, 0.6, 0.5])
        eval_s = _series("eval_loss", [2.0, 1.8, 1.6, 1.5, 1.4, 1.35])
        out = await tool.execute(
            TrainingDynamicsInput(
                experiment=rec,
                metric_series=[train, eval_s],
                overfit_threshold=0.1,
            )
        )
        patterns = {p.pattern_type: p for p in out.patterns}
        assert patterns[DynamicsPatternType.OVERFITTING].detected
        assert out.train_eval_gap is not None

    @pytest.mark.asyncio
    async def test_underfitting(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series("loss", [5.0, 4.9, 4.8, 4.8, 4.8, 4.8])
        out = await tool.execute(
            TrainingDynamicsInput(experiment=rec, metric_series=[train])
        )
        patterns = {p.pattern_type: p for p in out.patterns}
        assert patterns[DynamicsPatternType.UNDERFITTING].detected

    @pytest.mark.asyncio
    async def test_convergence(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series(
            "loss",
            [3.0, 2.0, 1.0, 0.5, 0.45, 0.44, 0.439, 0.438, 0.437, 0.437, 0.437, 0.437],
        )
        out = await tool.execute(
            TrainingDynamicsInput(
                experiment=rec,
                metric_series=[train],
                convergence_window=5,
            )
        )
        patterns = {p.pattern_type: p for p in out.patterns}
        assert patterns[DynamicsPatternType.CONVERGENCE].detected

    @pytest.mark.asyncio
    async def test_instability(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series("loss", [1.0, 5.0, 0.5, 4.5, 0.6, 4.0, 0.4, 3.5])
        out = await tool.execute(
            TrainingDynamicsInput(
                experiment=rec,
                metric_series=[train],
                instability_threshold=0.3,
            )
        )
        patterns = {p.pattern_type: p for p in out.patterns}
        assert patterns[DynamicsPatternType.INSTABILITY].detected

    @pytest.mark.asyncio
    async def test_divergence(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series("loss", [0.5, 0.8, 1.2, 2.0, 3.5, 5.0])
        out = await tool.execute(
            TrainingDynamicsInput(experiment=rec, metric_series=[train])
        )
        patterns = {p.pattern_type: p for p in out.patterns}
        assert patterns[DynamicsPatternType.DIVERGENCE].detected

    @pytest.mark.asyncio
    async def test_healthy(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series(
            "loss",
            [2.0, 1.5, 1.2, 1.0, 0.85, 0.75, 0.7, 0.68, 0.67, 0.67, 0.67, 0.67],
        )
        out = await tool.execute(
            TrainingDynamicsInput(
                experiment=rec,
                metric_series=[train],
                convergence_window=5,
            )
        )
        assert out.primary_pattern() == DynamicsPatternType.HEALTHY

    @pytest.mark.asyncio
    async def test_missing_series_fallback(self):
        tool = TrainingDynamicsTool()
        rec = _record("a", metrics={"loss": 0.5, "eval_loss": 0.6})
        out = await tool.execute(
            TrainingDynamicsInput(experiment=rec, metric_series=[])
        )
        assert out.final_train_metric == 0.5
        assert out.final_eval_metric == 0.6
        assert out.train_eval_gap == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_empty_series(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        out = await tool.execute(
            TrainingDynamicsInput(experiment=rec, metric_series=[])
        )
        assert out.final_train_metric is None
        assert out.summary

    @pytest.mark.asyncio
    async def test_stability_score_bounds(self):
        tool = TrainingDynamicsTool()
        rec = _record("a")
        train = _series("loss", [1.0, 1.01, 0.99, 1.0, 1.005])
        out = await tool.execute(
            TrainingDynamicsInput(experiment=rec, metric_series=[train])
        )
        assert 0.0 <= out.stability_score <= 1.0


# ---------------------------------------------------------------------------
# _stats helpers
# ---------------------------------------------------------------------------


class TestStatsHelpers:
    def test_mean(self):
        assert _stats.mean([1, 2, 3]) == pytest.approx(2.0)
        assert _stats.mean([]) == 0.0

    def test_variance(self):
        v = _stats.variance([1, 2, 3])
        assert v == pytest.approx(1.0)

    def test_std(self):
        assert _stats.std([1, 2, 3]) == pytest.approx(1.0)

    def test_welch_t_statistic(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert _stats.welch_t_statistic(a, b) == pytest.approx(0.0)

    def test_welch_t_statistic_different(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 5.0, 6.0, 7.0]
        t = _stats.welch_t_statistic(a, b)
        assert t < 0  # a mean < b mean

    def test_p_value_bounds(self):
        p = _stats.student_t_p_value_two_tailed(2.0, 10.0)
        assert 0.0 <= p <= 1.0
        p0 = _stats.student_t_p_value_two_tailed(0.0, 10.0)
        assert p0 == pytest.approx(1.0)

    def test_cohens_d(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert _stats.cohens_d(a, b) == pytest.approx(0.0)

    def test_effect_size_labels(self):
        assert _stats.effect_size_label(0.1) == "negligible"
        assert _stats.effect_size_label(0.3) == "small"
        assert _stats.effect_size_label(0.6) == "medium"
        assert _stats.effect_size_label(1.0) == "large"

    def test_ci_contains_mean_diff(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        df = _stats.welch_degrees_of_freedom(a, b)
        ci = _stats.mean_diff_ci_95(a, b, df)
        diff = _stats.mean(a) - _stats.mean(b)
        assert ci[0] <= diff <= ci[1]

    def test_regularized_incomplete_beta_bounds(self):
        assert _stats.regularized_incomplete_beta(1, 1, 0.0) == 0.0
        assert _stats.regularized_incomplete_beta(1, 1, 1.0) == 1.0
        assert _stats.regularized_incomplete_beta(1, 1, 0.5) == pytest.approx(0.5)

    def test_inverse_t_cdf_roundtrip(self):
        df = 20.0
        t = 2.086
        cdf = _stats.student_t_cdf(t, df)
        t_back = _stats.inverse_student_t_cdf(cdf, df)
        assert abs(t_back - t) < 0.05


# ---------------------------------------------------------------------------
# StatisticalSignificanceTool
# ---------------------------------------------------------------------------


class TestStatisticalSignificance:
    @pytest.mark.asyncio
    async def test_significant_difference(self):
        tool = StatisticalSignificanceTool()
        a = _record(
            "a",
            series=[_series("loss", [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02])],
        )
        b = _record(
            "b",
            series=[_series("loss", [3.0, 3.1, 2.9, 3.0, 3.05, 2.95, 3.0, 3.02])],
        )
        out = await tool.execute(
            StatisticalSignificanceInput(
                experiments=[a, b], metric="loss", min_samples=3
            )
        )
        assert out.pairwise_count == 1
        assert out.results[0].significant is True
        assert out.best_experiment_id == "a"

    @pytest.mark.asyncio
    async def test_no_difference(self):
        tool = StatisticalSignificanceTool()
        vals = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.02]
        a = _record("a", series=[_series("loss", vals)])
        b = _record("b", series=[_series("loss", vals)])
        out = await tool.execute(
            StatisticalSignificanceInput(
                experiments=[a, b], metric="loss", min_samples=3
            )
        )
        assert out.results[0].significant is False

    @pytest.mark.asyncio
    async def test_insufficient_samples(self):
        tool = StatisticalSignificanceTool()
        a = _record("a", series=[_series("loss", [1.0, 1.1])])
        b = _record("b", series=[_series("loss", [3.0, 3.1])])
        out = await tool.execute(
            StatisticalSignificanceInput(
                experiments=[a, b], metric="loss", min_samples=3
            )
        )
        assert out.pairwise_count == 0
        assert len(out.insufficient_data_pairs) == 1

    @pytest.mark.asyncio
    async def test_higher_is_better_best(self):
        tool = StatisticalSignificanceTool()
        a = _record(
            "a",
            series=[_series("accuracy", [0.7, 0.71, 0.69, 0.7, 0.705])],
        )
        b = _record(
            "b",
            series=[_series("accuracy", [0.9, 0.91, 0.89, 0.9, 0.905])],
        )
        out = await tool.execute(
            StatisticalSignificanceInput(
                experiments=[a, b],
                metric="accuracy",
                min_samples=3,
                higher_is_better=True,
            )
        )
        assert out.best_experiment_id == "b"

    @pytest.mark.asyncio
    async def test_verdict_with_significance(self):
        tool = StatisticalSignificanceTool()
        a = _record(
            "a", series=[_series("loss", [1.0, 1.1, 0.9, 1.0, 1.05])]
        )
        b = _record(
            "b", series=[_series("loss", [5.0, 5.1, 4.9, 5.0, 5.05])]
        )
        out = await tool.execute(
            StatisticalSignificanceInput(
                experiments=[a, b], metric="loss", min_samples=3
            )
        )
        assert "significant" in out.overall_verdict.lower()


# ---------------------------------------------------------------------------
# NextExperimentTool
# ---------------------------------------------------------------------------


class TestNextExperiment:
    @pytest.mark.asyncio
    async def test_overfitting_recommendation(self):
        from research_engineer.models.evaluation import (
            DynamicsPattern,
            TrainingDynamicsOutput,
        )

        tool = NextExperimentTool()
        dyn = TrainingDynamicsOutput(
            experiment_id="a",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.OVERFITTING,
                    detected=True,
                    confidence=0.9,
                )
            ],
        )
        out = await tool.execute(
            NextExperimentInput(experiments=[_record("a")], dynamics=[dyn])
        )
        titles = [r.title for r in out.experiment_recommendations]
        assert any("overfitting" in t.lower() for t in titles)
        assert out.experiment_recommendations[0].priority == RecommendationPriority.HIGH

    @pytest.mark.asyncio
    async def test_divergence_critical(self):
        from research_engineer.models.evaluation import (
            DynamicsPattern,
            TrainingDynamicsOutput,
        )

        tool = NextExperimentTool()
        dyn = TrainingDynamicsOutput(
            experiment_id="a",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.DIVERGENCE,
                    detected=True,
                    confidence=0.9,
                )
            ],
        )
        out = await tool.execute(
            NextExperimentInput(experiments=[_record("a")], dynamics=[dyn])
        )
        assert out.experiment_recommendations[0].priority == RecommendationPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_all_failed(self):
        tool = NextExperimentTool()
        out = await tool.execute(
            NextExperimentInput(
                experiments=[
                    _record("a", status=ExperimentStatus.FAILED),
                    _record("b", status=ExperimentStatus.CRASHED),
                ]
            )
        )
        assert out.experiment_recommendations[0].priority == RecommendationPriority.CRITICAL
        assert "minimal" in out.experiment_recommendations[0].title.lower()

    @pytest.mark.asyncio
    async def test_healthy_low_priority(self):
        from research_engineer.models.evaluation import (
            DynamicsPattern,
            TrainingDynamicsOutput,
        )

        tool = NextExperimentTool()
        dyn = TrainingDynamicsOutput(
            experiment_id="a",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.HEALTHY,
                    detected=True,
                    confidence=0.8,
                )
            ],
        )
        out = await tool.execute(
            NextExperimentInput(experiments=[_record("a")], dynamics=[dyn])
        )
        assert out.experiment_recommendations[0].priority == RecommendationPriority.LOW

    @pytest.mark.asyncio
    async def test_empty_experiments_baseline(self):
        tool = NextExperimentTool()
        out = await tool.execute(NextExperimentInput())
        assert len(out.experiment_recommendations) >= 1
        assert out.paper_query

    @pytest.mark.asyncio
    async def test_max_recommendations(self):
        from research_engineer.models.evaluation import (
            DynamicsPattern,
            TrainingDynamicsOutput,
        )

        tool = NextExperimentTool()
        dyns = [
            TrainingDynamicsOutput(
                experiment_id=f"e{i}",
                patterns=[
                    DynamicsPattern(
                        pattern_type=DynamicsPatternType.OVERFITTING,
                        detected=True,
                        confidence=0.8,
                    )
                ],
            )
            for i in range(5)
        ]
        out = await tool.execute(
            NextExperimentInput(
                experiments=[_record(f"e{i}") for i in range(5)],
                dynamics=dyns,
                max_recommendations=2,
            )
        )
        assert len(out.experiment_recommendations) <= 2

    @pytest.mark.asyncio
    async def test_paper_query_built(self):
        from research_engineer.models.evaluation import (
            DynamicsPattern,
            TrainingDynamicsOutput,
        )

        tool = NextExperimentTool()
        dyn = TrainingDynamicsOutput(
            experiment_id="a",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.OVERFITTING,
                    detected=True,
                    confidence=0.9,
                )
            ],
        )
        out = await tool.execute(
            NextExperimentInput(
                experiments=[_record("a")],
                dynamics=[dyn],
                recommend_papers=True,
                paper_id="2503.12345",
            )
        )
        assert out.paper_query
        assert "2503.12345" in out.paper_query

    @pytest.mark.asyncio
    async def test_paper_suggestions_default_empty(self):
        tool = NextExperimentTool()
        out = await tool.execute(NextExperimentInput())
        assert out.paper_suggestions == []


# ---------------------------------------------------------------------------
# EvaluationStorageTool
# ---------------------------------------------------------------------------


class TestEvaluationStorage:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationRecord,
            EvaluationStorageInput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        rec = EvaluationRecord(
            experiment_ids=["a", "b"],
            paper_id="2503.12345",
            repo_path="./repo",
            summary="test",
            conclusions=["c1"],
            tags=["eval"],
        )
        store_out = await tool.execute(
            EvaluationStorageInput(evaluation=rec)
        )
        assert store_out.success
        got = await tool.get_by_id(rec.evaluation_id)
        assert got is not None
        assert got.experiment_ids == ["a", "b"]
        assert got.paper_id == "2503.12345"

    @pytest.mark.asyncio
    async def test_query_by_paper(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationQueryInput,
            EvaluationRecord,
            EvaluationStorageInput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        rec = EvaluationRecord(
            experiment_ids=["a"], paper_id="p1", repo_path="./r"
        )
        await tool.execute(EvaluationStorageInput(evaluation=rec))
        out = await tool.execute(EvaluationQueryInput(paper_id="p1"))
        assert out.total == 1
        assert out.evaluations[0].paper_id == "p1"

    @pytest.mark.asyncio
    async def test_query_by_experiment_id(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationQueryInput,
            EvaluationRecord,
            EvaluationStorageInput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        rec = EvaluationRecord(experiment_ids=["exp_a", "exp_b"])
        await tool.execute(EvaluationStorageInput(evaluation=rec))
        out = await tool.execute(
            EvaluationQueryInput(experiment_id="exp_a")
        )
        assert out.total == 1

    @pytest.mark.asyncio
    async def test_text_search(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationQueryInput,
            EvaluationRecord,
            EvaluationStorageInput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        rec = EvaluationRecord(
            experiment_ids=["a"], summary="overfitting detected"
        )
        await tool.execute(EvaluationStorageInput(evaluation=rec))
        out = await tool.execute(
            EvaluationQueryInput(search_text="overfitting")
        )
        assert out.total == 1

    @pytest.mark.asyncio
    async def test_roundtrip_with_nested_models(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationRecord,
            EvaluationStorageInput,
            ExperimentComparisonOutput,
            NextExperimentOutput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        rec = EvaluationRecord(
            experiment_ids=["a", "b"],
            comparison=ExperimentComparisonOutput(
                experiments_compared=2,
                winner_experiment_id="a",
            ),
            next_experiments=NextExperimentOutput(
                overall_strategy="test strategy"
            ),
        )
        await tool.execute(EvaluationStorageInput(evaluation=rec))
        got = await tool.get_by_id(rec.evaluation_id)
        assert got is not None
        assert got.comparison is not None
        assert got.comparison.winner_experiment_id == "a"
        assert got.next_experiments is not None
        assert got.next_experiments.overall_strategy == "test strategy"

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path):
        from research_engineer.models.evaluation import (
            EvaluationQueryInput,
            EvaluationRecord,
            EvaluationStorageInput,
        )

        tool = EvaluationStorageTool(db_path=str(tmp_path / "test.db"))
        for i in range(5):
            await tool.execute(
                EvaluationStorageInput(
                    evaluation=EvaluationRecord(
                        experiment_ids=[f"e{i}"], paper_id="p1"
                    )
                )
            )
        out = await tool.execute(
            EvaluationQueryInput(paper_id="p1", limit=2, offset=0)
        )
        assert len(out.evaluations) == 2
        assert out.total == 5
