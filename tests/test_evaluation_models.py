"""Tests for Phase 8 evaluation models."""


from research_engineer.models.evaluation import (
    DynamicsPattern,
    DynamicsPatternType,
    EvaluationConfig,
    EvaluationQueryInput,
    EvaluationQueryOutput,
    EvaluationRecord,
    EvaluationResult,
    EvaluationStorageInput,
    EvaluationStorageOutput,
    ExperimentComparisonInput,
    ExperimentComparisonOutput,
    ExperimentRecommendation,
    MetricDelta,
    NextExperimentInput,
    NextExperimentOutput,
    PaperSuggestion,
    RecommendationPriority,
    SignificanceResult,
    StatisticalSignificanceInput,
    StatisticalSignificanceOutput,
    TrainingDynamicsInput,
    TrainingDynamicsOutput,
)
from research_engineer.models.experiment import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentType,
)


def _make_record(exp_id: str = "exp1") -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=exp_id,
        repo_path="./repo",
        command=["python", "train.py"],
        experiment_type=ExperimentType.TRAINING,
        status=ExperimentStatus.COMPLETED,
        start_time="2026-01-01T00:00:00",
    )


class TestEnums:
    def test_dynamics_pattern_type_values(self):
        assert DynamicsPatternType.OVERFITTING.value == "overfitting"
        assert DynamicsPatternType.UNDERFITTING.value == "underfitting"
        assert DynamicsPatternType.CONVERGENCE.value == "convergence"
        assert DynamicsPatternType.INSTABILITY.value == "instability"
        assert DynamicsPatternType.DIVERGENCE.value == "divergence"
        assert DynamicsPatternType.PLATEAU.value == "plateau"
        assert DynamicsPatternType.HEALTHY.value == "healthy"

    def test_recommendation_priority_values(self):
        assert RecommendationPriority.LOW.value == "low"
        assert RecommendationPriority.MEDIUM.value == "medium"
        assert RecommendationPriority.HIGH.value == "high"
        assert RecommendationPriority.CRITICAL.value == "critical"


class TestExperimentComparisonModels:
    def test_metric_delta_defaults(self):
        d = MetricDelta(metric="loss")
        assert d.values == {}
        assert d.best_experiment_id == ""
        assert d.deltas == {}
        assert d.improvement_pct == {}

    def test_comparison_input_defaults(self):
        inp = ExperimentComparisonInput(experiments=[_make_record()])
        assert inp.primary_metric is None
        assert inp.higher_is_better is False
        assert inp.baseline_experiment_id is None

    def test_comparison_output_defaults(self):
        out = ExperimentComparisonOutput()
        assert out.experiments_compared == 0
        assert out.winner_experiment_id is None
        assert out.metric_deltas == []
        assert out.findings == []


class TestTrainingDynamicsModels:
    def test_dynamics_pattern_defaults(self):
        p = DynamicsPattern(pattern_type=DynamicsPatternType.HEALTHY)
        assert p.detected is False
        assert p.confidence == 0.0
        assert p.evidence == []
        assert p.recommendation == ""

    def test_dynamics_input_defaults(self):
        inp = TrainingDynamicsInput(experiment=_make_record())
        assert inp.train_metric == "loss"
        assert inp.eval_metric == "eval_loss"
        assert inp.overfit_threshold == 0.15
        assert inp.convergence_window == 10
        assert inp.convergence_tolerance == 1e-3
        assert inp.instability_threshold == 0.5

    def test_dynamics_output_defaults(self):
        out = TrainingDynamicsOutput(experiment_id="exp1")
        assert out.patterns == []
        assert out.convergence_step is None
        assert out.stability_score == 0.0
        assert out.summary == ""

    def test_primary_pattern_none(self):
        out = TrainingDynamicsOutput(experiment_id="exp1")
        assert out.primary_pattern() is None

    def test_primary_pattern_detected(self):
        out = TrainingDynamicsOutput(
            experiment_id="exp1",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.HEALTHY,
                    detected=True,
                    confidence=0.4,
                ),
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.OVERFITTING,
                    detected=True,
                    confidence=0.9,
                ),
            ],
        )
        assert out.primary_pattern() == DynamicsPatternType.OVERFITTING

    def test_primary_pattern_ignores_not_detected(self):
        out = TrainingDynamicsOutput(
            experiment_id="exp1",
            patterns=[
                DynamicsPattern(
                    pattern_type=DynamicsPatternType.OVERFITTING,
                    detected=False,
                    confidence=0.99,
                ),
            ],
        )
        assert out.primary_pattern() is None


class TestStatisticalSignificanceModels:
    def test_significance_result_defaults(self):
        r = SignificanceResult(comparison="a vs b", metric="loss")
        assert r.mean_a == 0.0
        assert r.p_value == 1.0
        assert r.significant is False
        assert r.effect_size_label == "negligible"
        assert r.direction == "no difference"
        assert r.confidence_interval_95 == [0.0, 0.0]

    def test_significance_input_validation(self):
        inp = StatisticalSignificanceInput(
            experiments=[_make_record(), _make_record("exp2")],
            metric="loss",
        )
        assert inp.alpha == 0.05
        assert inp.min_samples == 3
        assert inp.higher_is_better is False

    def test_significance_output_defaults(self):
        out = StatisticalSignificanceOutput()
        assert out.results == []
        assert out.best_experiment_id is None
        assert out.pairwise_count == 0
        assert out.insufficient_data_pairs == []


class TestNextExperimentModels:
    def test_recommendation_defaults(self):
        r = ExperimentRecommendation(rank=1, title="t", rationale="r")
        assert r.suggested_type == ExperimentType.TRAINING
        assert r.priority == RecommendationPriority.MEDIUM
        assert r.estimated_effort == "medium"
        assert r.suggested_changes == []

    def test_paper_suggestion_defaults(self):
        p = PaperSuggestion(paper_id="p1", title="t")
        assert p.reason == ""
        assert p.relevance == 0.0

    def test_next_input_defaults(self):
        inp = NextExperimentInput()
        assert inp.experiments == []
        assert inp.dynamics == []
        assert inp.significance is None
        assert inp.comparison is None
        assert inp.max_recommendations == 5
        assert inp.recommend_papers is True

    def test_next_output_defaults(self):
        out = NextExperimentOutput()
        assert out.experiment_recommendations == []
        assert out.paper_suggestions == []
        assert out.overall_strategy == ""
        assert out.open_questions == []
        assert out.paper_query == ""


class TestEvaluationStorageModels:
    def test_record_defaults(self):
        rec = EvaluationRecord()
        assert rec.evaluation_id  # auto-generated
        assert rec.experiment_ids == []
        assert rec.comparison is None
        assert rec.dynamics == []
        assert rec.conclusions == []
        assert rec.memory_ids == []
        assert rec.tags == []

    def test_storage_input_defaults(self):
        rec = EvaluationRecord()
        inp = EvaluationStorageInput(evaluation=rec)
        assert inp.operation == "store"

    def test_storage_output(self):
        out = EvaluationStorageOutput(
            evaluation_id="e1", success=True, message="ok"
        )
        assert out.success is True

    def test_query_input_defaults(self):
        q = EvaluationQueryInput()
        assert q.limit == 100
        assert q.offset == 0

    def test_query_output_defaults(self):
        out = EvaluationQueryOutput()
        assert out.evaluations == []
        assert out.total == 0


class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult(evaluation_id="e1")
        assert r.result_id  # auto
        assert r.experiment_ids == []
        assert r.processing_time_seconds == 0.0
        assert r.has_findings() is False
        assert r.is_positive() is False

    def test_has_findings_with_comparison(self):
        r = EvaluationResult(
            evaluation_id="e1",
            comparison=ExperimentComparisonOutput(experiments_compared=2),
        )
        assert r.has_findings() is True

    def test_has_findings_with_dynamics(self):
        r = EvaluationResult(
            evaluation_id="e1",
            dynamics=[TrainingDynamicsOutput(experiment_id="exp1")],
        )
        assert r.has_findings() is True

    def test_is_positive_with_low_priority_rec(self):
        r = EvaluationResult(
            evaluation_id="e1",
            next_experiments=NextExperimentOutput(
                experiment_recommendations=[
                    ExperimentRecommendation(
                        rank=1,
                        title="scale up",
                        rationale="healthy",
                        priority=RecommendationPriority.LOW,
                    )
                ]
            ),
        )
        assert r.is_positive() is True

    def test_is_positive_with_winner(self):
        r = EvaluationResult(
            evaluation_id="e1",
            comparison=ExperimentComparisonOutput(
                winner_experiment_id="exp1"
            ),
        )
        assert r.is_positive() is True

    def test_serialization_roundtrip(self):
        r = EvaluationResult(
            evaluation_id="e1",
            experiment_ids=["exp1", "exp2"],
            comparison=ExperimentComparisonOutput(
                experiments_compared=2,
                winner_experiment_id="exp1",
                findings=["exp1 better"],
            ),
        )
        data = r.model_dump(mode="json")
        r2 = EvaluationResult.model_validate(data)
        assert r2.evaluation_id == r.evaluation_id
        assert r2.comparison is not None
        assert r2.comparison.winner_experiment_id == "exp1"


class TestEvaluationConfig:
    def test_defaults(self):
        c = EvaluationConfig()
        assert c.store_conclusions is True
        assert c.update_graph is True
        assert c.recommend_papers is True
        assert c.output_dir == "output/evaluations"
        assert c.overfit_threshold == 0.15
        assert c.significance_alpha == 0.05
        assert c.max_next_recommendations == 5
