"""Phase 3 - Experiment Planner domain models.

All Pydantic models for the Experiment Planner Agent, including
compatibility analysis, implementation planning, experiment design,
validation, risk assessment, compute estimation, impact analysis,
and result prediction.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CompatibilityLevel(StrEnum):
    """Compatibility level between paper technique and repository."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskLevel(StrEnum):
    """Risk level for implementation or operational risks."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ConfidenceLevel(StrEnum):
    """Confidence level for predictions and estimates."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class DifficultyLevel(StrEnum):
    """Implementation difficulty level."""

    TRIVIAL = "Trivial"
    EASY = "Easy"
    MODERATE = "Moderate"
    HARD = "Hard"
    VERY_HARD = "Very Hard"


class ExperimentType(StrEnum):
    """Type of experiment."""

    BASELINE = "baseline"
    MINIMUM_VIABLE = "minimum_viable"
    ABLATION = "ablation"
    STRESS = "stress"
    SCALING = "scaling"


class ValidationTestType(StrEnum):
    """Type of validation test."""

    UNIT = "unit"
    INTEGRATION = "integration"
    NUMERICAL_EQUIVALENCE = "numerical_equivalence"
    REGRESSION = "regression"
    PERFORMANCE = "performance"
    CHECKPOINT_COMPAT = "checkpoint_compatibility"


# --- Compatibility Analysis ---


class CompatibilityDimension(BaseModel):
    """Compatibility analysis for a single dimension."""

    dimension: str = Field(..., description="Dimension name")
    level: CompatibilityLevel = Field(
        ..., description="Compatibility level"
    )
    reasoning: str = Field(..., description="Explanation of compatibility")
    evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    blockers: list[str] = Field(
        default_factory=list, description="Blocking issues"
    )


class CompatibilityReport(BaseModel):
    """Full compatibility analysis between paper technique and repository."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )
    architecture_compatibility: CompatibilityDimension = Field(
        ..., description="Architecture compatibility"
    )
    training_compatibility: CompatibilityDimension = Field(
        ..., description="Training compatibility"
    )
    inference_compatibility: CompatibilityDimension = Field(
        ..., description="Inference compatibility"
    )
    config_compatibility: CompatibilityDimension = Field(
        ..., description="Configuration compatibility"
    )
    checkpoint_compatibility: CompatibilityDimension = Field(
        ..., description="Checkpoint compatibility"
    )
    distributed_compatibility: CompatibilityDimension = Field(
        ..., description="Distributed training compatibility"
    )
    evaluation_compatibility: CompatibilityDimension = Field(
        ..., description="Evaluation compatibility"
    )
    overall_compatibility: CompatibilityLevel = Field(
        ..., description="Overall compatibility score"
    )
    overall_reasoning: str = Field(
        ..., description="Summary reasoning for overall score"
    )
    recommended_actions: list[str] = Field(
        default_factory=list, description="Recommended actions"
    )


# --- Implementation Planning ---


class ImplementationTarget(BaseModel):
    """A target file/class/interface requiring modification."""

    file_path: str = Field(..., description="Path to file")
    class_name: str | None = Field(
        None, description="Class name if applicable"
    )
    method_name: str | None = Field(
        None, description="Method name if applicable"
    )
    target_type: str = Field(
        ..., description="Type: attention, optimizer, dataset, loss, metric, config, etc."
    )
    modification_type: str = Field(
        ..., description="Modification: add, modify, refactor, replace, extend"
    )
    description: str = Field(
        ..., description="Description of required changes"
    )
    estimated_lines: int = Field(
        default=50, description="Estimated lines of code"
    )
    complexity: DifficultyLevel = Field(
        default=DifficultyLevel.MODERATE, description="Modification complexity"
    )


class ImplementationStep(BaseModel):
    """A single step in the implementation plan."""

    step_number: int = Field(..., description="Step number (1-indexed)")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed description")
    targets: list[ImplementationTarget] = Field(
        default_factory=list, description="Files/classes to modify"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MODERATE, description="Step difficulty"
    )
    dependencies: list[int] = Field(
        default_factory=list, description="Step numbers this depends on"
    )
    estimated_effort: str = Field(
        default="1-2 days", description="Estimated effort"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="Risk level"
    )
    validation_criteria: list[str] = Field(
        default_factory=list, description="How to validate this step"
    )


class ImplementationPlan(BaseModel):
    """Complete implementation plan for integrating a paper into a repo."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Plan creation timestamp"
    )
    steps: list[ImplementationStep] = Field(
        ..., min_length=1, description="Ordered implementation steps"
    )
    targets: list[ImplementationTarget] = Field(
        ..., description="All modification targets"
    )
    total_estimated_effort: str = Field(
        default="1-2 weeks", description="Total estimated effort"
    )
    prerequisite_knowledge: list[str] = Field(
        default_factory=list, description="Required knowledge areas"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="Plan success criteria"
    )


# --- Impact Analysis ---


class ImpactDimension(BaseModel):
    """Impact on a single engineering dimension."""

    dimension: str = Field(..., description="Dimension name")
    current_estimate: str = Field(
        ..., description="Current baseline estimate"
    )
    projected_estimate: str = Field(
        ..., description="Projected estimate after integration"
    )
    change_direction: str = Field(
        ..., description="increase, decrease, or neutral"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Confidence in estimate"
    )
    reasoning: str = Field(..., description="Explanation")


class ImpactReport(BaseModel):
    """Engineering impact analysis."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Analysis timestamp"
    )
    memory_impact: ImpactDimension = Field(
        ..., description="Memory usage impact"
    )
    training_speed_impact: ImpactDimension = Field(
        ..., description="Training speed impact"
    )
    inference_speed_impact: ImpactDimension = Field(
        ..., description="Inference speed impact"
    )
    model_quality_impact: ImpactDimension = Field(
        ..., description="Model quality impact"
    )
    checkpoint_size_impact: ImpactDimension = Field(
        ..., description="Checkpoint size impact"
    )
    communication_impact: ImpactDimension = Field(
        ..., description="Communication overhead impact"
    )
    distributed_complexity_impact: ImpactDimension = Field(
        ..., description="Distributed training complexity impact"
    )
    overall_assessment: str = Field(
        ..., description="Overall impact summary"
    )


# --- Experiment Design ---


class MetricDefinition(BaseModel):
    """Definition of a metric to track."""

    name: str = Field(..., description="Metric name")
    description: str = Field(..., description="What this metric measures")
    unit: str = Field(default="", description="Unit of measurement")
    why_important: str = Field(
        ..., description="Why this metric matters"
    )
    target_value: str | None = Field(
        None, description="Target value if known"
    )


class Experiment(BaseModel):
    """A single experiment definition."""

    experiment_id: str = Field(..., description="Unique experiment ID")
    experiment_type: ExperimentType = Field(
        ..., description="Type of experiment"
    )
    title: str = Field(..., description="Experiment title")
    description: str = Field(..., description="Experiment description")
    hypothesis: str = Field(..., description="Hypothesis being tested")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Configuration overrides"
    )
    baseline_id: str | None = Field(
        None, description="ID of baseline experiment for comparison"
    )
    expected_duration: str = Field(
        default="1-2 hours", description="Expected duration"
    )
    required_resources: list[str] = Field(
        default_factory=list, description="Required resources (GPUs, etc.)"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="Success criteria"
    )


class ExperimentGroup(BaseModel):
    """A group of related experiments."""

    group_name: str = Field(..., description="Group name")
    group_type: ExperimentType = Field(
        ..., description="Type of experiments in this group"
    )
    description: str = Field(..., description="Group description")
    experiments: list[Experiment] = Field(
        ..., min_length=1, description="Experiments in this group"
    )


class ExperimentMatrix(BaseModel):
    """Complete experiment matrix for a paper-repo integration."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metrics: list[MetricDefinition] = Field(
        ..., description="Metrics to track"
    )
    groups: list[ExperimentGroup] = Field(
        ..., min_length=1, description="Experiment groups"
    )
    total_experiments: int = Field(
        ..., description="Total number of experiments"
    )
    estimated_total_gpu_hours: float = Field(
        default=0.0, description="Estimated total GPU hours"
    )


# --- Validation Planning ---


class TestCase(BaseModel):
    """A single test case definition."""

    test_name: str = Field(..., description="Test name")
    test_type: ValidationTestType = Field(..., description="Type of test")
    description: str = Field(..., description="What this test validates")
    test_file: str = Field(
        default="", description="File path for the test"
    )
    assertion: str = Field(
        ..., description="What assertion to make"
    )
    priority: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="Test priority"
    )


class TestSuite(BaseModel):
    """A suite of related tests."""

    suite_name: str = Field(..., description="Suite name")
    test_type: ValidationTestType = Field(..., description="Type of tests")
    description: str = Field(..., description="Suite description")
    test_cases: list[TestCase] = Field(
        ..., min_length=1, description="Test cases"
    )


class ValidationPlan(BaseModel):
    """Complete validation strategy."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Plan creation timestamp"
    )
    test_suites: list[TestSuite] = Field(
        ..., min_length=1, description="Test suites"
    )
    total_test_cases: int = Field(
        ..., description="Total number of test cases"
    )
    validation_approach: str = Field(
        ..., description="Overall validation approach"
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Acceptance criteria"
    )


# --- Risk Assessment ---


class RiskItem(BaseModel):
    """A single risk item."""

    risk_id: str = Field(..., description="Unique risk identifier")
    category: str = Field(
        ..., description="Category: implementation, training, memory, performance, distributed, inference, checkpoint"
    )
    description: str = Field(..., description="Risk description")
    level: RiskLevel = Field(..., description="Risk level")
    probability: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Probability of occurrence"
    )
    impact: str = Field(..., description="Impact if risk materializes")
    mitigation: str = Field(
        default="", description="Mitigation strategy"
    )
    contingency: str = Field(
        default="", description="Contingency plan if risk occurs"
    )


class RiskAssessment(BaseModel):
    """Complete risk assessment."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Assessment timestamp"
    )
    implementation_risks: list[RiskItem] = Field(
        default_factory=list, description="Implementation risks"
    )
    training_risks: list[RiskItem] = Field(
        default_factory=list, description="Training risks"
    )
    memory_risks: list[RiskItem] = Field(
        default_factory=list, description="Memory risks"
    )
    performance_risks: list[RiskItem] = Field(
        default_factory=list, description="Performance risks"
    )
    distributed_risks: list[RiskItem] = Field(
        default_factory=list, description="Distributed training risks"
    )
    inference_risks: list[RiskItem] = Field(
        default_factory=list, description="Inference risks"
    )
    checkpoint_risks: list[RiskItem] = Field(
        default_factory=list, description="Checkpoint migration risks"
    )
    overall_risk_level: RiskLevel = Field(
        ..., description="Overall risk level"
    )
    overall_risk_reasoning: str = Field(
        ..., description="Overall risk reasoning"
    )


# --- Compute Estimation ---


class ComputeEstimate(BaseModel):
    """Compute cost estimation."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Estimation timestamp"
    )
    gpu_type: str = Field(
        default="A100", description="Assumed GPU type"
    )
    gpu_count_per_experiment: int = Field(
        default=1, description="GPUs per experiment"
    )
    estimated_gpu_hours_per_experiment: float = Field(
        default=0.0, description="GPU hours per experiment"
    )
    total_experiments: int = Field(
        default=0, description="Total number of experiments"
    )
    total_gpu_hours: float = Field(
        default=0.0, description="Total estimated GPU hours"
    )
    estimated_training_duration: str = Field(
        default="Unknown", description="Estimated wall time"
    )
    peak_memory_per_gpu_gb: float = Field(
        default=0.0, description="Peak memory per GPU in GB"
    )
    total_storage_gb: float = Field(
        default=0.0, description="Total storage needed in GB"
    )
    approximate_cloud_cost_usd: float = Field(
        default=0.0, description="Approximate cloud cost in USD"
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.LOW, description="Confidence in estimate"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made"
    )


# --- Result Prediction ---


class ScenarioOutcome(BaseModel):
    """A predicted outcome scenario."""

    scenario: str = Field(..., description="Scenario type")
    description: str = Field(..., description="Outcome description")
    metrics: dict[str, str] = Field(
        default_factory=dict, description="Predicted metrics"
    )
    probability: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Estimated probability"
    )


class FailureMode(BaseModel):
    """A potential failure mode."""

    failure_id: str = Field(..., description="Failure ID")
    description: str = Field(..., description="What fails")
    trigger: str = Field(..., description="What triggers the failure")
    detection: str = Field(..., description="How to detect the failure")
    recovery: str = Field(..., description="How to recover")


class ResultPrediction(BaseModel):
    """Predicted results for the integration."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )
    best_case: ScenarioOutcome = Field(
        ..., description="Best case outcome"
    )
    likely_case: ScenarioOutcome = Field(
        ..., description="Most likely outcome"
    )
    worst_case: ScenarioOutcome = Field(
        ..., description="Worst case outcome"
    )
    failure_modes: list[FailureMode] = Field(
        default_factory=list, description="Potential failure modes"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="Success criteria"
    )
    overall_confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM, description="Overall confidence"
    )


# --- Plan Result (Aggregate) ---


class PlanResult(BaseModel):
    """Aggregate result of the entire planning process."""

    plan_id: str = Field(..., description="Unique plan identifier")
    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Plan creation timestamp"
    )
    compatibility: CompatibilityReport = Field(
        ..., description="Compatibility analysis"
    )
    implementation: ImplementationPlan = Field(
        ..., description="Implementation plan"
    )
    impact: ImpactReport = Field(
        ..., description="Engineering impact analysis"
    )
    experiments: ExperimentMatrix = Field(
        ..., description="Experiment matrix"
    )
    validation: ValidationPlan = Field(
        ..., description="Validation strategy"
    )
    risks: RiskAssessment = Field(
        ..., description="Risk assessment"
    )
    compute: ComputeEstimate = Field(
        ..., description="Compute cost estimation"
    )
    predictions: ResultPrediction = Field(
        ..., description="Result prediction"
    )

    def to_engineering_report(self) -> str:
        """Generate a comprehensive engineering report in markdown."""
        lines = [
            f"# Engineering Report: {self.paper_id}",
            "",
            f"**Repository**: {self.repo_path}",
            f"**Date**: {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"**Overall Compatibility**: {self.compatibility.overall_compatibility.value}",
            f"**Overall Risk**: {self.risks.overall_risk_level.value}",
            f"**Prediction Confidence**: {self.predictions.overall_confidence.value}",
            "",
            "## Compatibility Analysis",
            "",
            "| Dimension | Level | Reasoning |",
            "|-----------|-------|-----------|",
        ]
        compat_dims: list[CompatibilityDimension] = [
            self.compatibility.architecture_compatibility,
            self.compatibility.training_compatibility,
            self.compatibility.inference_compatibility,
            self.compatibility.config_compatibility,
            self.compatibility.checkpoint_compatibility,
            self.compatibility.distributed_compatibility,
            self.compatibility.evaluation_compatibility,
        ]
        for cdim in compat_dims:
            lines.append(
                f"| {cdim.dimension} | {cdim.level.value} | {cdim.reasoning[:80]} |"
            )
        lines.extend([
            "",
            f"**Overall**: {self.compatibility.overall_reasoning}",
            "",
            "## Implementation Plan",
            "",
        ])
        for step in self.implementation.steps:
            lines.append(
                f"### Step {step.step_number}: {step.title}"
            )
            lines.append(f"{step.description}")
            lines.append(
                f"- Difficulty: {step.difficulty.value}"
            )
            lines.append(f"- Effort: {step.estimated_effort}")
            lines.append(f"- Risk: {step.risk_level.value}")
            lines.append("")
        lines.extend([
            "",
            "## Engineering Impact",
            "",
        ])
        impact_dims: list[ImpactDimension] = [
            self.impact.memory_impact,
            self.impact.training_speed_impact,
            self.impact.inference_speed_impact,
            self.impact.model_quality_impact,
            self.impact.checkpoint_size_impact,
            self.impact.communication_impact,
            self.impact.distributed_complexity_impact,
        ]
        for idim in impact_dims:
            lines.append(
                f"- **{idim.dimension}**: {idim.change_direction} "
                f"(confidence: {idim.confidence.value})"
            )
        lines.extend([
            "",
            "## Experiments",
            "",
            f"Total: {self.experiments.total_experiments}",
            f"Estimated GPU hours: {self.compute.total_gpu_hours}",
            "",
        ])
        for group in self.experiments.groups:
            lines.append(f"### {group.group_name}")
            for exp in group.experiments:
                lines.append(
                    f"- {exp.experiment_id}: {exp.title}"
                )
        lines.extend([
            "",
            "## Risk Assessment",
            "",
        ])
        risk_categories = [
            ("Implementation", self.risks.implementation_risks),
            ("Training", self.risks.training_risks),
            ("Memory", self.risks.memory_risks),
            ("Performance", self.risks.performance_risks),
            ("Distributed", self.risks.distributed_risks),
            ("Inference", self.risks.inference_risks),
            ("Checkpoint", self.risks.checkpoint_risks),
        ]
        for cat_name, risk_items in risk_categories:
            if risk_items:
                lines.append(f"### {cat_name} Risks")
                for risk in risk_items:
                    lines.append(
                        f"- [{risk.level.value}] {risk.description} "
                        f"- Mitigation: {risk.mitigation}"
                    )
                lines.append("")
        lines.extend([
            "",
            "## Compute Cost",
            "",
            f"- GPU type: {self.compute.gpu_type}",
            f"- GPU hours: {self.compute.total_gpu_hours}",
            f"- Duration: {self.compute.estimated_training_duration}",
            f"- Memory/GPU: {self.compute.peak_memory_per_gpu_gb} GB",
            f"- Storage: {self.compute.total_storage_gb} GB",
            f"- Cloud cost: ${self.compute.approximate_cloud_cost_usd:.2f}",
            "",
            "## Predicted Results",
            "",
            f"- **Best case**: {self.predictions.best_case.description}",
            f"- **Likely case**: {self.predictions.likely_case.description}",
            f"- **Worst case**: {self.predictions.worst_case.description}",
            f"- **Confidence**: {self.predictions.overall_confidence.value}",
            "",
            "## Success Criteria",
            "",
        ])
        for criterion in self.predictions.success_criteria:
            lines.append(f"- {criterion}")
        return "\n".join(lines)
