"""Impact Analysis Tool for Phase 3 - Experiment Planner.

Predicts the engineering impact of integrating a paper's technique
into a target repository across multiple dimensions.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    ConfidenceLevel,
    ImpactDimension,
    ImpactReport,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ImpactAnalysisInput(BaseModel):
    """Input for impact analysis."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(..., description="Repository summary")


class ImpactAnalysisOutput(BaseModel):
    """Output from impact analysis."""

    impact: ImpactReport = Field(..., description="Impact report")


class ImpactAnalysisTool(Tool[ImpactAnalysisInput, ImpactAnalysisOutput]):
    """Predict engineering impact of integrating technique into repo."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ImpactAnalysisInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary and input.repo_summary)

    async def execute(self, input: ImpactAnalysisInput) -> ImpactAnalysisOutput:
        try:
            impact = ImpactReport(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                memory_impact=self._assess_memory_impact(input),
                training_speed_impact=self._assess_training_speed(input),
                inference_speed_impact=self._assess_inference_speed(input),
                model_quality_impact=self._assess_model_quality(input),
                checkpoint_size_impact=self._assess_checkpoint_size(input),
                communication_impact=self._assess_communication(input),
                distributed_complexity_impact=self._assess_distributed_complexity(input),
                overall_assessment=self._overall_assessment(input),
            )
            return ImpactAnalysisOutput(impact=impact)
        except Exception as e:
            raise ToolError(f"Impact analysis failed: {e}", input, e)

    def _assess_memory_impact(self, input: ImpactAnalysisInput) -> ImpactDimension:
        arch = input.summary.model_architecture.lower()

        if any(kw in arch for kw in ["flash attention", "ring attention", "sparse"]):
            return ImpactDimension(
                dimension="Memory Usage",
                current_estimate="Standard transformer memory footprint",
                projected_estimate="Reduced memory via efficient attention",
                change_direction="decrease",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Efficient attention variants reduce peak memory by recomputing or distributing attention",
            )
        elif any(kw in arch for kw in ["moe", "mixture of experts"]):
            return ImpactDimension(
                dimension="Memory Usage",
                current_estimate="Dense model memory footprint",
                projected_estimate="Increased memory for expert parameters; reduced activation memory",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="MoE adds expert parameters but reduces per-token compute, overall memory increases",
            )
        elif any(kw in arch for kw in ["quantiz", "4-bit", "8-bit"]):
            return ImpactDimension(
                dimension="Memory Usage",
                current_estimate="FP32/FP16/BF16 weights",
                projected_estimate="Significantly reduced memory from quantization",
                change_direction="decrease",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Quantization reduces memory proportionally to bit reduction",
            )
        else:
            return ImpactDimension(
                dimension="Memory Usage",
                current_estimate="Standard memory footprint",
                projected_estimate="Moderate increase from new components",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="New components may add parameters and activations",
            )

    def _assess_training_speed(self, input: ImpactAnalysisInput) -> ImpactDimension:
        arch = input.summary.model_architecture.lower()
        train = input.summary.training_methodology.lower()

        if any(kw in arch for kw in ["flash attention", "ring attention"]):
            return ImpactDimension(
                dimension="Training Speed",
                current_estimate="Standard attention training speed",
                projected_estimate="Improved training throughput from efficient attention",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Efficient attention reduces compute from O(n^2) and improves GPU utilization",
            )
        elif any(kw in train for kw in ["distributed", "multi-gpu", "multi-node"]):
            return ImpactDimension(
                dimension="Training Speed",
                current_estimate="Single-node training",
                projected_estimate="Distributed training with communication overhead",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Distributed training adds throughput but also communication overhead",
            )
        elif "moe" in arch or "mixture" in arch:
            return ImpactDimension(
                dimension="Training Speed",
                current_estimate="Dense model training speed",
                projected_estimate="Slower training iterations but better compute utilization",
                change_direction="decrease",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="MoE has auxiliary losses and routing overhead per iteration",
            )
        else:
            return ImpactDimension(
                dimension="Training Speed",
                current_estimate="Baseline training speed",
                projected_estimate="Similar or slightly slower training speed",
                change_direction="neutral",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Minor overhead from new components",
            )

    def _assess_inference_speed(self, input: ImpactAnalysisInput) -> ImpactDimension:
        arch = input.summary.model_architecture.lower()

        if any(kw in arch for kw in ["flash attention", "ring attention"]):
            return ImpactDimension(
                dimension="Inference Speed",
                current_estimate="Standard inference latency",
                projected_estimate="Reduced inference latency from efficient attention",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Efficient attention reduces memory and improves cache utilization",
            )
        elif any(kw in arch for kw in ["moe", "mixture"]):
            return ImpactDimension(
                dimension="Inference Speed",
                current_estimate="Dense model inference latency",
                projected_estimate="Potentially faster per-token but expert loading overhead",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="MoE reduces per-token compute but has expert routing overhead",
            )
        elif any(kw in arch for kw in ["kv-cache", "kv cache", "speculative"]):
            return ImpactDimension(
                dimension="Inference Speed",
                current_estimate="Standard inference throughput",
                projected_estimate="Improved throughput via caching or speculation",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="KV cache optimization or speculative decoding improves throughput",
            )
        else:
            return ImpactDimension(
                dimension="Inference Speed",
                current_estimate="Baseline inference speed",
                projected_estimate="Similar inference speed, minor overhead from new components",
                change_direction="neutral",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="No major inference optimization proposed",
            )

    def _assess_model_quality(self, input: ImpactAnalysisInput) -> ImpactDimension:
        results = input.summary.key_results
        arch = input.summary.model_architecture.lower()

        has_quality_claims = any(
            any(kw in r.lower() for kw in ["improve", "better", "state-of-the-art", "sota", "outperform", "increase"])
            for r in results
        ) if results else False

        if has_quality_claims:
            return ImpactDimension(
                dimension="Model Quality",
                current_estimate="Baseline model quality",
                projected_estimate="Improved quality per paper claims",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Paper claims quality improvements, but reproduction may not match exactly",
            )
        elif "efficien" in arch or "speed" in arch or "fast" in arch:
            return ImpactDimension(
                dimension="Model Quality",
                current_estimate="Baseline model quality",
                projected_estimate="Similar quality with efficiency gains",
                change_direction="neutral",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Efficiency-focused technique with quality preservation",
            )
        else:
            return ImpactDimension(
                dimension="Model Quality",
                current_estimate="Baseline model quality",
                projected_estimate="Uncertain impact on quality",
                change_direction="neutral",
                confidence=ConfidenceLevel.LOW,
                reasoning="Quality impact depends on correct implementation and hyperparameter tuning",
            )

    def _assess_checkpoint_size(self, input: ImpactAnalysisInput) -> ImpactDimension:
        arch = input.summary.model_architecture.lower()

        if "moe" in arch or "mixture" in arch:
            return ImpactDimension(
                dimension="Checkpoint Size",
                current_estimate="Standard model checkpoint size",
                projected_estimate="Significantly larger (multiple expert parameters)",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="MoE checkpoints include all expert weights, increasing size proportionally",
            )
        elif any(kw in arch for kw in ["quantiz", "4-bit", "8-bit"]):
            return ImpactDimension(
                dimension="Checkpoint Size",
                current_estimate="FP32/FP16 checkpoint size",
                projected_estimate="Reduced checkpoint size from quantization",
                change_direction="decrease",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Quantized checkpoints store weights in lower precision",
            )
        else:
            return ImpactDimension(
                dimension="Checkpoint Size",
                current_estimate="Standard checkpoint size",
                projected_estimate="Slightly larger from new parameters",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="New components add parameters, modest size increase",
            )

    def _assess_communication(self, input: ImpactAnalysisInput) -> ImpactDimension:
        train = input.summary.training_methodology.lower()
        arch = input.summary.model_architecture.lower()

        if any(kw in train for kw in ["distributed", "multi-gpu", "multi-node", "pipeline parallel"]):
            return ImpactDimension(
                dimension="Communication Overhead",
                current_estimate="Standard single-node communication",
                projected_estimate="Significant communication overhead from distributed setup",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Distributed techniques require inter-node communication",
            )
        elif any(kw in arch for kw in ["ring attention", "sequence parallel"]):
            return ImpactDimension(
                dimension="Communication Overhead",
                current_estimate="Standard attention communication",
                projected_estimate="Increased communication for distributed attention",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Ring/sequence parallelism requires communication between devices",
            )
        else:
            return ImpactDimension(
                dimension="Communication Overhead",
                current_estimate="Minimal communication overhead",
                projected_estimate="No significant change in communication overhead",
                change_direction="neutral",
                confidence=ConfidenceLevel.HIGH,
                reasoning="No distributed communication requirements detected",
            )

    def _assess_distributed_complexity(self, input: ImpactAnalysisInput) -> ImpactDimension:
        train = input.summary.training_methodology.lower()
        arch = input.summary.model_architecture.lower()

        if any(kw in train for kw in ["distributed", "multi-node", "pipeline parallel", "tensor parallel"]):
            return ImpactDimension(
                dimension="Distributed Complexity",
                current_estimate="Single-node training",
                projected_estimate="Multi-node distributed training required",
                change_direction="increase",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Distributed training adds significant complexity in setup, debugging, and monitoring",
            )
        elif any(kw in arch for kw in ["ring attention", "sequence parallel", "moe"]):
            return ImpactDimension(
                dimension="Distributed Complexity",
                current_estimate="Standard data parallel training",
                projected_estimate="Enhanced parallelism strategy needed",
                change_direction="increase",
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Technique requires specialized parallelism beyond data parallel",
            )
        else:
            return ImpactDimension(
                dimension="Distributed Complexity",
                current_estimate="Standard training setup",
                projected_estimate="No change in distributed complexity",
                change_direction="neutral",
                confidence=ConfidenceLevel.HIGH,
                reasoning="Technique does not require distributed training changes",
            )

    def _overall_assessment(self, input: ImpactAnalysisInput) -> str:
        dims = [
            self._assess_memory_impact(input),
            self._assess_training_speed(input),
            self._assess_model_quality(input),
        ]
        increases = sum(1 for d in dims if d.change_direction == "increase")
        decreases = sum(1 for d in dims if d.change_direction == "decrease")

        if increases >= 2:
            return "Integration is expected to improve key metrics. Monitor for tradeoffs."
        elif decreases >= 2:
            return "Integration may reduce some metrics. Verify tradeoffs are acceptable."
        else:
            return "Integration has mixed impact. Verify each dimension independently."
