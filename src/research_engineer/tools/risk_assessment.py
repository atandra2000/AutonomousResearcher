"""Risk Assessment Tool for Phase 3 - Experiment Planner.

Identifies and assesses risks across implementation, training,
memory, performance, distributed training, inference, and
checkpoint migration dimensions.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    ConfidenceLevel,
    RiskAssessment,
    RiskItem,
    RiskLevel,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class RiskAssessmentInput(BaseModel):
    """Input for risk assessment."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(..., description="Repository summary")


class RiskAssessmentOutput(BaseModel):
    """Output from risk assessment."""

    assessment: RiskAssessment = Field(..., description="Risk assessment")


class RiskAssessmentTool(Tool[RiskAssessmentInput, RiskAssessmentOutput]):
    """Assess risks for paper-repo integration."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: RiskAssessmentInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary)

    async def execute(self, input: RiskAssessmentInput) -> RiskAssessmentOutput:
        try:
            impl_risks = self._assess_implementation_risks(input)
            train_risks = self._assess_training_risks(input)
            memory_risks = self._assess_memory_risks(input)
            perf_risks = self._assess_performance_risks(input)
            dist_risks = self._assess_distributed_risks(input)
            infer_risks = self._assess_inference_risks(input)
            ckpt_risks = self._assess_checkpoint_risks(input)

            all_risks = (
                impl_risks + train_risks + memory_risks
                + perf_risks + dist_risks + infer_risks + ckpt_risks
            )
            overall = self._compute_overall(all_risks)

            assessment = RiskAssessment(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                implementation_risks=impl_risks,
                training_risks=train_risks,
                memory_risks=memory_risks,
                performance_risks=perf_risks,
                distributed_risks=dist_risks,
                inference_risks=infer_risks,
                checkpoint_risks=ckpt_risks,
                overall_risk_level=overall,
                overall_risk_reasoning=self._overall_reasoning(all_risks, overall),
            )

            return RiskAssessmentOutput(assessment=assessment)

        except Exception as e:
            raise ToolError(f"Risk assessment failed: {e}", input, e)

    def _assess_implementation_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        arch = input.summary.model_architecture.lower()
        repro = input.summary.reproduction_challenges
        risks: list[RiskItem] = []

        risks.append(RiskItem(
            risk_id="IMPL-001",
            category="implementation",
            description="Core technique implementation may differ from paper's intent",
            level=RiskLevel.MEDIUM,
            probability=ConfidenceLevel.MEDIUM,
            impact="Implementation may not reproduce paper results exactly",
            mitigation="Implement test cases first; verify numerical equivalence against reference",
            contingency="Iterate on implementation, adjust hyperparameters",
        ))

        if any(kw in arch for kw in ["flash attention", "ring attention", "triton", "cuda kernel"]):
            risks.append(RiskItem(
                risk_id="IMPL-002",
                category="implementation",
                description="Custom CUDA kernel or Triton implementation required",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.HIGH,
                impact="Kernel bugs can cause silent correctness errors",
                mitigation="Extensive unit testing; use torch.compile as fallback; numerical gradient checking",
                contingency="Fall back to pure PyTorch implementation with reduced performance",
            ))

        if "moe" in arch or "mixture" in arch:
            risks.append(RiskItem(
                risk_id="IMPL-003",
                category="implementation",
                description="MoE routing requires careful implementation to avoid load imbalance",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.MEDIUM,
                impact="Load imbalance wastes compute and degrades quality",
                mitigation="Implement load balancing loss; monitor expert utilization; add capacity factor",
                contingency="Simplify to fewer experts or top-1 routing",
            ))

        repro_text = " ".join(repro).lower() if repro else ""
        if any(kw in repro_text for kw in ["resource", "compute", "gpu", "memory"]):
            risks.append(RiskItem(
                risk_id="IMPL-004",
                category="implementation",
                description="Paper reports significant compute requirements for reproduction",
                level=RiskLevel.MEDIUM,
                probability=ConfidenceLevel.MEDIUM,
                impact="May not be able to run full-scale experiments",
                mitigation="Design small-scale validation experiments first; use gradient accumulation",
                contingency="Reduce model size or sequence length for initial testing",
            ))

        return risks

    def _assess_training_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        train = input.summary.training_methodology.lower()
        risks: list[RiskItem] = []

        risks.append(RiskItem(
            risk_id="TRAIN-001",
            category="training",
            description="Training may be unstable with new technique",
            level=RiskLevel.MEDIUM,
            probability=ConfidenceLevel.MEDIUM,
            impact="Training diverges or produces NaN losses",
            mitigation="Use gradient clipping; monitor loss; start with lower learning rate",
            contingency="Revert to baseline training configuration",
        ))

        if any(kw in train for kw in ["custom loss", "auxiliary", "modified loss", "novel loss"]):
            risks.append(RiskItem(
                risk_id="TRAIN-002",
                category="training",
                description="Custom loss function may cause training instability",
                level=RiskLevel.MEDIUM,
                probability=ConfidenceLevel.MEDIUM,
                impact="Loss spikes, NaN, or poor convergence",
                mitigation="Loss scaling; gradient monitoring; loss weight scheduling",
                contingency="Reduce loss weight or remove auxiliary loss",
            ))

        if any(kw in train for kw in ["distributed", "multi-gpu", "multi-node"]):
            risks.append(RiskItem(
                risk_id="TRAIN-003",
                category="training",
                description="Distributed training introduces synchronization and debugging challenges",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.MEDIUM,
                impact="Difficult to debug distributed issues; may waste GPU hours",
                mitigation="Test single-GPU first; use torchrun for fault tolerance; log extensively",
                contingency="Reduce to single-node training with gradient accumulation",
            ))

        return risks

    def _assess_memory_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        arch = input.summary.model_architecture.lower()
        risks: list[RiskItem] = []

        risks.append(RiskItem(
            risk_id="MEM-001",
            category="memory",
            description="New technique may exceed available GPU memory",
            level=RiskLevel.MEDIUM,
            probability=ConfidenceLevel.MEDIUM,
            impact="Out-of-memory errors during training or inference",
            mitigation="Profile memory usage; implement gradient checkpointing; reduce batch size",
            contingency="Use CPU offloading or reduce model dimension",
        ))

        if "moe" in arch or "mixture" in arch:
            risks.append(RiskItem(
                risk_id="MEM-002",
                category="memory",
                description="MoE expert parameters significantly increase model memory",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.HIGH,
                impact="Memory pressure limits batch size and sequence length",
                mitigation="Expert parallelism; CPU offloading for inactive experts; gradient checkpointing",
                contingency="Reduce number of experts",
            ))

        if any(kw in arch for kw in ["long context", "long sequence", "32k", "128k", "1m"]):
            risks.append(RiskItem(
                risk_id="MEM-003",
                category="memory",
                description="Long context sequences consume significant memory",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.HIGH,
                impact="O(n^2) memory for standard attention limits sequence length",
                mitigation="Use efficient attention; implement chunked processing; gradient checkpointing",
                contingency="Reduce maximum sequence length",
            ))

        return risks

    def _assess_performance_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        risks: list[RiskItem] = [
            RiskItem(
                risk_id="PERF-001",
                category="performance",
                description="Implementation may not achieve claimed speedup",
                level=RiskLevel.MEDIUM,
                probability=ConfidenceLevel.MEDIUM,
                impact="Slower than expected training or inference",
                mitigation="Profile and optimize hot paths; use torch.compile; benchmark early",
                contingency="Accept reduced performance or simplify implementation",
            ),
        ]
        return risks

    def _assess_distributed_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        train = input.summary.training_methodology.lower()
        arch = input.summary.model_architecture.lower()
        risks: list[RiskItem] = []

        if any(kw in train for kw in ["distributed", "multi-node", "pipeline"]):
            risks.append(RiskItem(
                risk_id="DIST-001",
                category="distributed",
                description="Multi-node training coordination may fail",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.MEDIUM,
                impact="Training crashes or hangs in distributed setting",
                mitigation="Test single-node first; use robust distributed launchers; health checks",
                contingency="Fall back to single-node training",
            ))

        if any(kw in arch for kw in ["ring attention", "sequence parallel"]):
            risks.append(RiskItem(
                risk_id="DIST-002",
                category="distributed",
                description="Sequence parallelism requires specific communication patterns",
                level=RiskLevel.HIGH,
                probability=ConfidenceLevel.MEDIUM,
                impact="Communication overhead may negate benefits",
                mitigation="Profile communication vs computation ratio; optimize overlap",
                contingency="Use tensor parallelism instead",
            ))

        return risks

    def _assess_inference_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        arch = input.summary.model_architecture.lower()
        risks: list[RiskItem] = []

        if "kv cache" in arch or "kv-cache" in arch:
            risks.append(RiskItem(
                risk_id="INFER-001",
                category="inference",
                description="KV cache modifications may cause inference issues",
                level=RiskLevel.MEDIUM,
                probability=ConfidenceLevel.MEDIUM,
                impact="Increased inference latency or memory for serving",
                mitigation="Test inference throughput explicitly; measure KV cache size",
                contingency="Use standard KV cache with reduced context",
            ))

        risks.append(RiskItem(
            risk_id="INFER-002",
            category="inference",
            description="New technique may change inference serving characteristics",
            level=RiskLevel.LOW,
            probability=ConfidenceLevel.LOW,
            impact="Serving infrastructure may need updates",
            mitigation="Benchmark inference before deployment; test with realistic traffic patterns",
            contingency="Revert to baseline for serving",
        ))

        return risks

    def _assess_checkpoint_risks(self, input: RiskAssessmentInput) -> list[RiskItem]:
        risks: list[RiskItem] = [
            RiskItem(
                risk_id="CKPT-001",
                category="checkpoint",
                description="New model parameters may break checkpoint loading",
                level=RiskLevel.MEDIUM,
                probability=ConfidenceLevel.MEDIUM,
                impact="Cannot resume from existing checkpoints",
                mitigation="Implement checkpoint migration utility; add version field; test roundtrip",
                contingency="Train from scratch with new architecture",
            ),
        ]
        return risks

    def _compute_overall(self, all_risks: list[RiskItem]) -> RiskLevel:
        high_count = sum(1 for r in all_risks if r.level == RiskLevel.HIGH)

        if high_count >= 3:
            return RiskLevel.HIGH
        elif high_count >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _overall_reasoning(self, all_risks: list[RiskItem], overall: RiskLevel) -> str:
        high = [r for r in all_risks if r.level == RiskLevel.HIGH]
        med = [r for r in all_risks if r.level == RiskLevel.MEDIUM]

        reasoning = f"Overall risk level: {overall.value}. "
        reasoning += f"Found {len(high)} high-risk and {len(med)} medium-risk items. "

        if high:
            reasoning += "Key high-risk areas: " + ", ".join(
                f"{r.risk_id}: {r.description[:60]}" for r in high[:3]
            )
        elif med:
            reasoning += "Key medium-risk areas: " + ", ".join(
                f"{r.risk_id}: {r.description[:60]}" for r in med[:3]
            )

        return reasoning
