"""Compute Estimator Tool for Phase 3 - Experiment Planner.

Estimates GPU hours, training duration, memory requirements,
storage requirements, experiment count, and approximate cloud cost.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    ComputeEstimate,
    ConfidenceLevel,
    ExperimentMatrix,
)
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ComputeEstimatorInput(BaseModel):
    """Input for compute estimation."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    experiment_matrix: ExperimentMatrix = Field(
        ..., description="Experiment matrix"
    )


class ComputeEstimatorOutput(BaseModel):
    """Output from compute estimation."""

    estimate: ComputeEstimate = Field(..., description="Compute estimate")


_GPU_COST_PER_HOUR: dict[str, float] = {
    "A100": 2.50,
    "A100_80GB": 3.50,
    "H100": 4.00,
    "V100": 1.50,
    "T4": 0.50,
    "T4_16GB": 0.75,
}

_MODEL_SIZE_MEMORY_GB: dict[str, float] = {
    "small": 2.0,
    "medium": 8.0,
    "large": 24.0,
    "xl": 40.0,
    "xxl": 80.0,
}


class ComputeEstimatorTool(Tool[ComputeEstimatorInput, ComputeEstimatorOutput]):
    """Estimate compute costs for experiment execution."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ComputeEstimatorInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary)

    async def execute(self, input: ComputeEstimatorInput) -> ComputeEstimatorOutput:
        try:
            summary = input.summary
            matrix = input.experiment_matrix

            gpu_type = self._determine_gpu_type(summary)
            gpu_count = self._estimate_gpu_count(summary)
            gpu_hours_per_exp = self._estimate_gpu_hours_per_experiment(summary)
            total_experiments = matrix.total_experiments
            total_gpu_hours = gpu_hours_per_exp * total_experiments
            estimated_duration = self._estimate_duration(summary, total_gpu_hours, gpu_count)
            peak_memory = self._estimate_peak_memory(summary)
            storage = self._estimate_storage(summary, total_experiments)
            cloud_cost = self._estimate_cloud_cost(total_gpu_hours, gpu_type, gpu_count)
            confidence = self._estimate_confidence(summary)

            estimate = ComputeEstimate(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                gpu_type=gpu_type,
                gpu_count_per_experiment=gpu_count,
                estimated_gpu_hours_per_experiment=round(gpu_hours_per_exp, 1),
                total_experiments=total_experiments,
                total_gpu_hours=round(total_gpu_hours, 1),
                estimated_training_duration=estimated_duration,
                peak_memory_per_gpu_gb=round(peak_memory, 1),
                total_storage_gb=round(storage, 1),
                approximate_cloud_cost_usd=round(cloud_cost, 2),
                confidence=confidence,
                assumptions=self._list_assumptions(summary, gpu_type, gpu_count),
            )

            return ComputeEstimatorOutput(estimate=estimate)

        except Exception as e:
            raise ToolError(f"Compute estimation failed: {e}", input, e)

    def _determine_gpu_type(self, summary: ResearchSummary) -> str:
        arch = summary.model_architecture.lower()
        train = summary.training_methodology.lower()

        if any(kw in arch for kw in ["moe", "mixture", "70b", "175b", "405b"]):
            return "A100_80GB"
        elif any(kw in arch for kw in ["flash attention", "ring attention", "triton"]):
            return "A100"
        elif any(kw in train for kw in ["distributed", "multi-node"]):
            return "A100"
        else:
            return "A100"

    def _estimate_gpu_count(self, summary: ResearchSummary) -> int:
        arch = summary.model_architecture.lower()
        train = summary.training_methodology.lower()

        if any(kw in arch for kw in ["70b", "175b", "405b", "moe", "mixture"]):
            return 8
        elif any(kw in train for kw in ["distributed", "multi-gpu", "8 gpu"]):
            return 4
        elif any(kw in arch for kw in ["flash attention", "ring attention", "large"]):
            return 2
        else:
            return 1

    def _estimate_gpu_hours_per_experiment(self, summary: ResearchSummary) -> float:
        comp = getattr(summary, "compute_requirements", "").lower()

        train = summary.training_methodology.lower()
        reps = summary.reproduction_challenges
        repro_text = " ".join(reps).lower() if reps else ""

        base_hours = 4.0

        if any(kw in comp for kw in ["high", "multi-gpu", "large-scale"]):
            base_hours = 24.0
        elif any(kw in comp for kw in ["medium"]):
            base_hours = 8.0

        if any(kw in train for kw in ["days", "1-2 days", "long training"]):
            base_hours = max(base_hours, 24.0)
        elif any(kw in train for kw in ["hours", "several hours"]):
            base_hours = max(base_hours, 6.0)

        if any(kw in repro_text for kw in ["significant compute", "large-scale", "expensive"]):
            base_hours *= 1.5

        return base_hours

    def _estimate_duration(self, summary: ResearchSummary, total_gpu_hours: float, gpu_count: int) -> str:
        if gpu_count <= 0:
            gpu_count = 1

        wall_hours = total_gpu_hours / gpu_count

        if wall_hours < 4:
            return "1-4 hours"
        elif wall_hours < 12:
            return "4-12 hours"
        elif wall_hours < 24:
            return "12 hours - 1 day"
        elif wall_hours < 72:
            return "1-3 days"
        elif wall_hours < 168:
            return "3-7 days"
        else:
            return "1-2+ weeks"

    def _estimate_peak_memory(self, summary: ResearchSummary) -> float:
        arch = summary.model_architecture.lower()

        if any(kw in arch for kw in ["70b", "moe", "mixture"]):
            return 70.0
        elif any(kw in arch for kw in ["13b", "7b", "large"]):
            return 40.0
        elif any(kw in arch for kw in ["small", "mini"]):
            return 8.0
        else:
            return 24.0

    def _estimate_storage(self, summary: ResearchSummary, total_experiments: int) -> float:
        base_per_exp_gb = 5.0
        dataset_gb = 10.0
        checkpoints_per_exp = 3.0

        base_total = (base_per_exp_gb + checkpoints_per_exp) * total_experiments + dataset_gb

        arch = summary.model_architecture.lower()
        if any(kw in arch for kw in ["moe", "mixture"]):
            base_total *= 2.0

        return base_total

    def _estimate_cloud_cost(self, total_gpu_hours: float, gpu_type: str, gpu_count: int) -> float:
        cost_per_hour = _GPU_COST_PER_HOUR.get(gpu_type, 2.50)
        return total_gpu_hours * cost_per_hour

    def _estimate_confidence(self, summary: ResearchSummary) -> ConfidenceLevel:
        repro = summary.reproduction_challenges
        repro_text = " ".join(repro).lower() if repro else ""

        if any(kw in repro_text for kw in ["not specify", "unclear", "proprietary"]):
            return ConfidenceLevel.LOW
        elif any(kw in repro_text for kw in ["resource", "compute", "significant"]):
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.MEDIUM

    def _list_assumptions(self, summary: ResearchSummary, gpu_type: str, gpu_count: int) -> list[str]:
        return [
            f"GPU type: {gpu_type}",
            f"GPUs per experiment: {gpu_count}",
            "Experiments run sequentially unless otherwise noted",
            "Cloud pricing based on on-demand rates",
            "No pre-emptible or spot instance pricing assumed",
            "Storage estimates include checkpoints and logs",
            "Dataset access costs not included",
            "Costs are approximate and may vary by provider and region",
        ]
