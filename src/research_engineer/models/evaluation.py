"""Phase 8 - Evaluation models.

Typed Pydantic models for experiment comparison, training dynamics
analysis, statistical significance testing, next-experiment
recommendations, and evaluation storage.
"""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from research_engineer.models.experiment import (
    ExperimentRecord,
    ExperimentType,
    MetricSeries,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DynamicsPatternType(StrEnum):
    """Types of training dynamics patterns."""

    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    CONVERGENCE = "convergence"
    INSTABILITY = "instability"
    DIVERGENCE = "divergence"
    PLATEAU = "plateau"
    HEALTHY = "healthy"


class RecommendationPriority(StrEnum):
    """Priority of a next-experiment recommendation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Experiment Comparison
# ---------------------------------------------------------------------------


class MetricDelta(BaseModel):
    """Delta of a single metric across experiments."""

    metric: str = Field(..., description="Metric name")
    values: dict[str, float] = Field(
        default_factory=dict, description="experiment_id -> value"
    )
    best_experiment_id: str = Field("", description="Experiment with best value")
    best_value: float = Field(0.0, description="Best value")
    deltas: dict[str, float] = Field(
        default_factory=dict, description="experiment_id -> delta vs best"
    )
    improvement_pct: dict[str, float] = Field(
        default_factory=dict,
        description="experiment_id -> improvement pct vs baseline",
    )


class ExperimentComparisonInput(BaseModel):
    """Input for experiment comparison."""

    experiments: list[ExperimentRecord] = Field(
        ..., description="Experiments to compare (>=2)"
    )
    primary_metric: str | None = Field(
        None, description="Metric to determine the winner"
    )
    higher_is_better: bool = Field(
        False, description="True for accuracy, False for loss"
    )
    baseline_experiment_id: str | None = Field(
        None, description="Baseline experiment for improvement calc"
    )


class ExperimentComparisonOutput(BaseModel):
    """Output from experiment comparison."""

    experiments_compared: int = Field(0, description="Number compared")
    primary_metric: str | None = Field(
        None, description="Primary metric used"
    )
    winner_experiment_id: str | None = Field(
        None, description="Winning experiment ID"
    )
    metric_deltas: list[MetricDelta] = Field(
        default_factory=list, description="Per-metric deltas"
    )
    shared_metrics: list[str] = Field(
        default_factory=list, description="Metrics present in all runs"
    )
    unique_metrics: dict[str, list[str]] = Field(
        default_factory=dict,
        description="experiment_id -> metrics only it has",
    )
    status_summary: dict[str, str] = Field(
        default_factory=dict, description="experiment_id -> status"
    )
    duration_summary: dict[str, float] = Field(
        default_factory=dict, description="experiment_id -> seconds"
    )
    failure_summary: dict[str, str | None] = Field(
        default_factory=dict, description="experiment_id -> failure_mode"
    )
    findings: list[str] = Field(
        default_factory=list, description="Qualitative findings"
    )
    recommendation: str = Field(
        "", description="Which run to keep and why"
    )


# ---------------------------------------------------------------------------
# Training Dynamics
# ---------------------------------------------------------------------------


class DynamicsPattern(BaseModel):
    """A detected training dynamics pattern."""

    pattern_type: DynamicsPatternType = Field(
        ..., description="Type of pattern"
    )
    detected: bool = Field(False, description="Whether detected")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence 0-1"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    recommendation: str = Field("", description="Suggested action")


class TrainingDynamicsInput(BaseModel):
    """Input for training dynamics analysis."""

    experiment: ExperimentRecord = Field(
        ..., description="Experiment to analyze"
    )
    metric_series: list[MetricSeries] = Field(
        default_factory=list,
        description="Series (inferred from record if empty)",
    )
    train_metric: str = Field("loss", description="Training metric name")
    eval_metric: str = Field("eval_loss", description="Eval metric name")
    overfit_threshold: float = Field(
        0.15, description="Eval-train gap fraction for overfitting"
    )
    convergence_window: int = Field(
        10, ge=2, description="Last N steps for plateau"
    )
    convergence_tolerance: float = Field(
        1e-3, description="Slope tolerance for plateau"
    )
    instability_threshold: float = Field(
        0.5, description="Coeff of variation for instability"
    )


class TrainingDynamicsOutput(BaseModel):
    """Output from training dynamics analysis."""

    experiment_id: str = Field(..., description="Experiment ID")
    patterns: list[DynamicsPattern] = Field(
        default_factory=list, description="Detected patterns"
    )
    convergence_step: int | None = Field(
        None, description="Step where plateaued"
    )
    final_train_metric: float | None = Field(
        None, description="Final train metric"
    )
    final_eval_metric: float | None = Field(
        None, description="Final eval metric"
    )
    best_eval_metric: float | None = Field(
        None, description="Best eval metric"
    )
    best_eval_step: int | None = Field(
        None, description="Step of best eval metric"
    )
    train_eval_gap: float | None = Field(
        None, description="Final eval - final train"
    )
    stability_score: float = Field(
        0.0, ge=0.0, le=1.0, description="0 unstable - 1 stable"
    )
    summary: str = Field("", description="Narrative summary")

    def primary_pattern(self) -> DynamicsPatternType | None:
        """Return the highest-confidence detected pattern type."""
        detected = [p for p in self.patterns if p.detected]
        if not detected:
            return None
        detected.sort(key=lambda p: p.confidence, reverse=True)
        return detected[0].pattern_type


# ---------------------------------------------------------------------------
# Statistical Significance
# ---------------------------------------------------------------------------


class SignificanceResult(BaseModel):
    """Result of a pairwise significance test."""

    comparison: str = Field(..., description="e.g. 'exp_a vs exp_b'")
    metric: str = Field(..., description="Metric compared")
    mean_a: float = Field(0.0, description="Mean of series A")
    mean_b: float = Field(0.0, description="Mean of series B")
    std_a: float = Field(0.0, description="Std of series A")
    std_b: float = Field(0.0, description="Std of series B")
    n_a: int = Field(0, description="Sample count A")
    n_b: int = Field(0, description="Sample count B")
    t_statistic: float = Field(0.0, description="Welch t-statistic")
    p_value: float = Field(
        1.0, ge=0.0, le=1.0, description="Two-tailed p-value"
    )
    degrees_of_freedom: float = Field(
        0.0, description="Welch-Satterthwaite df"
    )
    significant: bool = Field(False, description="p < alpha")
    effect_size: float = Field(0.0, description="Cohen's d")
    effect_size_label: str = Field(
        "negligible", description="negligible/small/medium/large"
    )
    confidence_interval_95: list[float] = Field(
        default_factory=lambda: [0.0, 0.0],
        description="[low, high] for mean difference",
    )
    direction: str = Field(
        "no difference", description="a>b / a<b / no difference"
    )


class StatisticalSignificanceInput(BaseModel):
    """Input for statistical significance testing."""

    experiments: list[ExperimentRecord] = Field(
        ..., description="Experiments to compare (>=2)"
    )
    metric: str = Field(..., description="Metric to compare")
    alpha: float = Field(
        0.05, gt=0.0, lt=1.0, description="Significance level"
    )
    min_samples: int = Field(
        3, ge=2, description="Min points per series"
    )
    higher_is_better: bool = Field(
        False, description="True for accuracy, False for loss"
    )


class StatisticalSignificanceOutput(BaseModel):
    """Output from statistical significance testing."""

    results: list[SignificanceResult] = Field(
        default_factory=list, description="Pairwise results"
    )
    overall_verdict: str = Field("", description="Summary statement")
    best_experiment_id: str | None = Field(
        None, description="Best experiment by mean"
    )
    pairwise_count: int = Field(0, description="Pairs compared")
    insufficient_data_pairs: list[str] = Field(
        default_factory=list, description="Pairs skipped (low samples)"
    )


# ---------------------------------------------------------------------------
# Next-Experiment Recommendations
# ---------------------------------------------------------------------------


class ExperimentRecommendation(BaseModel):
    """A single next-experiment recommendation."""

    rank: int = Field(..., description="Rank (1 = top)")
    title: str = Field(..., description="Short title")
    rationale: str = Field(..., description="Why this experiment")
    suggested_type: ExperimentType = Field(
        default_factory=lambda: ExperimentType.TRAINING,
        description="Suggested experiment type",
    )
    suggested_changes: list[str] = Field(
        default_factory=list, description="Concrete changes"
    )
    priority: RecommendationPriority = Field(
        RecommendationPriority.MEDIUM, description="Priority"
    )
    expected_impact: str = Field("", description="Expected impact")
    estimated_effort: str = Field(
        "medium", description="low/medium/high"
    )


class PaperSuggestion(BaseModel):
    """A paper suggestion for next steps."""

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    reason: str = Field("", description="Why relevant")
    relevance: float = Field(
        0.0, ge=0.0, le=1.0, description="Relevance 0-1"
    )


class NextExperimentInput(BaseModel):
    """Input for next-experiment recommendations."""

    experiments: list[ExperimentRecord] = Field(
        default_factory=list, description="Past experiments"
    )
    dynamics: list[TrainingDynamicsOutput] = Field(
        default_factory=list, description="Dynamics analyses"
    )
    significance: StatisticalSignificanceOutput | None = Field(
        None, description="Significance results"
    )
    comparison: ExperimentComparisonOutput | None = Field(
        None, description="Comparison results"
    )
    paper_id: str | None = Field(None, description="Associated paper")
    repo_path: str | None = Field(None, description="Repository path")
    max_recommendations: int = Field(
        5, ge=1, le=50, description="Max recommendations"
    )
    recommend_papers: bool = Field(
        True, description="Whether to suggest papers"
    )


class NextExperimentOutput(BaseModel):
    """Output from next-experiment recommendations."""

    experiment_recommendations: list[ExperimentRecommendation] = Field(
        default_factory=list, description="Recommendations"
    )
    paper_suggestions: list[PaperSuggestion] = Field(
        default_factory=list, description="Paper suggestions"
    )
    overall_strategy: str = Field(
        "", description="Narrative next-steps summary"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Open questions"
    )
    paper_query: str = Field(
        "", description="Query used for paper search"
    )


# ---------------------------------------------------------------------------
# Evaluation Storage
# ---------------------------------------------------------------------------


class EvaluationRecord(BaseModel):
    """A persisted evaluation record."""

    evaluation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique evaluation ID",
    )
    experiment_ids: list[str] = Field(
        default_factory=list, description="Experiments analyzed"
    )
    paper_id: str | None = Field(None, description="Associated paper")
    repo_path: str | None = Field(None, description="Repository path")
    comparison: ExperimentComparisonOutput | None = Field(
        None, description="Comparison results"
    )
    dynamics: list[TrainingDynamicsOutput] = Field(
        default_factory=list, description="Dynamics per experiment"
    )
    significance: StatisticalSignificanceOutput | None = Field(
        None, description="Significance results"
    )
    next_experiments: NextExperimentOutput | None = Field(
        None, description="Next-experiment recommendations"
    )
    summary: str = Field("", description="Evaluation summary")
    conclusions: list[str] = Field(
        default_factory=list, description="Key conclusions"
    )
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    tags: list[str] = Field(default_factory=list, description="Tags")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime | None = Field(
        None, description="Last update timestamp"
    )


class EvaluationStorageInput(BaseModel):
    """Input for evaluation storage operations."""

    evaluation: EvaluationRecord = Field(
        ..., description="Evaluation to store"
    )
    operation: str = Field("store", description="store, update")


class EvaluationStorageOutput(BaseModel):
    """Output from evaluation storage operations."""

    evaluation_id: str = Field(..., description="Evaluation ID")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field("", description="Status message")


class EvaluationQueryInput(BaseModel):
    """Input for querying evaluations."""

    evaluation_id: str | None = Field(
        None, description="Specific evaluation ID"
    )
    experiment_id: str | None = Field(
        None, description="Find evals containing this experiment"
    )
    paper_id: str | None = Field(None, description="Filter by paper ID")
    repo_path: str | None = Field(
        None, description="Filter by repository path"
    )
    search_text: str | None = Field(None, description="Text search")
    limit: int = Field(100, ge=1, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class EvaluationQueryOutput(BaseModel):
    """Output from querying evaluations."""

    evaluations: list[EvaluationRecord] = Field(
        default_factory=list, description="Matching evaluations"
    )
    total: int = Field(0, description="Total matching count")


# ---------------------------------------------------------------------------
# Top-Level Result
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Top-level result from the full evaluation workflow."""

    result_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Result ID"
    )
    evaluation_id: str = Field(..., description="Evaluation ID")
    experiment_ids: list[str] = Field(
        default_factory=list, description="Experiments analyzed"
    )
    paper_id: str | None = Field(None, description="Paper ID")
    repo_path: str | None = Field(None, description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Result timestamp"
    )
    comparison: ExperimentComparisonOutput | None = Field(
        None, description="Comparison results"
    )
    dynamics: list[TrainingDynamicsOutput] = Field(
        default_factory=list, description="Dynamics results"
    )
    significance: StatisticalSignificanceOutput | None = Field(
        None, description="Significance results"
    )
    next_experiments: NextExperimentOutput | None = Field(
        None, description="Next-experiment recommendations"
    )
    record: EvaluationRecord | None = Field(
        None, description="Stored record"
    )
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    output_dir: str | None = Field(None, description="Output directory")
    generated_files: list[str] = Field(
        default_factory=list, description="Generated file paths"
    )
    processing_time_seconds: float = Field(0.0, description="Total time")

    def has_findings(self) -> bool:
        """Return True if any analysis produced findings."""
        return bool(
            self.comparison
            or self.dynamics
            or self.significance
            or self.next_experiments
        )

    def is_positive(self) -> bool:
        """Return True if the evaluation validated a successful approach."""
        if self.next_experiments:
            for rec in self.next_experiments.experiment_recommendations:
                if rec.priority == RecommendationPriority.LOW:
                    return True
        if self.comparison and self.comparison.winner_experiment_id:
            return True
        return False


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Configuration for Evaluation Agent."""

    store_conclusions: bool = Field(
        True, description="Store conclusions in memory"
    )
    update_graph: bool = Field(
        True, description="Auto-update knowledge graph"
    )
    recommend_papers: bool = Field(
        True, description="Use LiteratureAgent for paper recs"
    )
    output_dir: str = Field(
        "output/evaluations", description="Default output directory"
    )
    overfit_threshold: float = Field(
        0.15, description="Overfitting gap threshold"
    )
    convergence_window: int = Field(
        10, ge=2, description="Convergence window size"
    )
    convergence_tolerance: float = Field(
        1e-3, description="Convergence slope tolerance"
    )
    instability_threshold: float = Field(
        0.5, description="Instability coefficient of variation"
    )
    significance_alpha: float = Field(
        0.05, gt=0.0, lt=1.0, description="Significance alpha"
    )
    min_samples_for_stats: int = Field(
        3, ge=2, description="Min samples for stats"
    )
    max_next_recommendations: int = Field(
        5, ge=1, le=50, description="Max next-experiment recs"
    )
