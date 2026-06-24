"""Result Prediction Tool for Phase 3 - Experiment Planner.

Predicts best-case, likely-case, and worst-case outcomes,
failure modes, success criteria, and confidence levels.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    ConfidenceLevel,
    FailureMode,
    ResultPrediction,
    ScenarioOutcome,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ResultPredictionInput(BaseModel):
    """Input for result prediction."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(..., description="Repository summary")


class ResultPredictionOutput(BaseModel):
    """Output from result prediction."""

    prediction: ResultPrediction = Field(..., description="Result prediction")


class ResultPredictionTool(Tool[ResultPredictionInput, ResultPredictionOutput]):
    """Predict outcomes for paper-repo integration."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ResultPredictionInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary)

    async def execute(self, input: ResultPredictionInput) -> ResultPredictionOutput:
        try:
            best = self._predict_best_case(input)
            likely = self._predict_likely_case(input)
            worst = self._predict_worst_case(input)
            failures = self._predict_failure_modes(input)
            criteria = self._define_success_criteria(input)
            overall_conf = self._estimate_confidence(input)

            prediction = ResultPrediction(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                best_case=best,
                likely_case=likely,
                worst_case=worst,
                failure_modes=failures,
                success_criteria=criteria,
                overall_confidence=overall_conf,
            )

            return ResultPredictionOutput(prediction=prediction)

        except Exception as e:
            raise ToolError(f"Result prediction failed: {e}", input, e)

    def _predict_best_case(self, input: ResultPredictionInput) -> ScenarioOutcome:
        arch = input.summary.model_architecture.lower()
        results = input.summary.key_results

        metrics: dict[str, str] = {
            "implementation_success": "Full reproduction of paper results",
            "quality": "Matches or exceeds paper benchmarks",
            "efficiency": "Achieves claimed efficiency gains",
        }

        if "attention" in arch:
            metrics["speedup"] = "1.5-3x training speedup"
            metrics["memory_reduction"] = "2-4x memory reduction"

        if "moe" in arch or "mixture" in arch:
            metrics["quality"] = "Better quality per compute unit"
            metrics["throughput"] = "2-3x throughput for same quality"

        result_str = " ".join(results).lower() if results else ""
        if "sota" in result_str or "state-of-the-art" in result_str:
            metrics["quality"] = "Matches SOTA results"

        return ScenarioOutcome(
            scenario="best_case",
            description="Implementation works correctly on first attempt. "
            "Results match or exceed paper claims. No major issues encountered.",
            metrics=metrics,
            probability=ConfidenceLevel.LOW,
        )

    def _predict_likely_case(self, input: ResultPredictionInput) -> ScenarioOutcome:
        arch = input.summary.model_architecture.lower()

        metrics: dict[str, str] = {
            "implementation_success": "Working implementation with minor issues",
            "quality": "Within 5-10% of paper results",
            "efficiency": "70-90% of claimed efficiency improvements",
        }

        if "attention" in arch:
            metrics["speedup"] = "1.2-2x speedup (less than paper claims)"
            metrics["memory_reduction"] = "1.5-3x memory reduction"

        if "moe" in arch or "mixture" in arch:
            metrics["quality"] = "MoE quality improvements observed but routing takes tuning"
            metrics["throughput"] = "1.5-2x throughput improvement"

        return ScenarioOutcome(
            scenario="likely_case",
            description="Implementation works with some debugging required. "
            "Results are within reasonable range of paper claims. "
            "Some hyperparameter tuning needed.",
            metrics=metrics,
            probability=ConfidenceLevel.MEDIUM,
        )

    def _predict_worst_case(self, input: ResultPredictionInput) -> ScenarioOutcome:
        return ScenarioOutcome(
            scenario="worst_case",
            description="Implementation encounters significant issues. "
            "Paper results may not be reproducible. Requires substantial debugging "
            "and may need design changes.",
            metrics={
                "implementation_success": "Working but suboptimal implementation",
                "quality": "Significant gap (>15%) from paper results",
                "efficiency": "No measurable efficiency improvement",
                "timeline": "2-3x longer than estimated",
            },
            probability=ConfidenceLevel.LOW,
        )

    def _predict_failure_modes(self, input: ResultPredictionInput) -> list[FailureMode]:
        arch = input.summary.model_architecture.lower()
        failures: list[FailureMode] = []

        failures.append(FailureMode(
            failure_id="FAIL-001",
            description="Training diverges (NaN/Inf losses)",
            trigger="Incorrect initialization or loss function implementation",
            detection="Monitor loss for NaN/Inf; set up early stopping on loss spikes",
            recovery="Reduce learning rate; check loss function implementation; verify gradient norms",
        ))

        failures.append(FailureMode(
            failure_id="FAIL-002",
            description="OOM during training or inference",
            trigger="Memory exceeds GPU capacity",
            detection="CUDA OOM error; memory profiling shows peak near capacity",
            recovery="Reduce batch size; enable gradient checkpointing; use CPU offloading",
        ))

        if "attention" in arch:
            failures.append(FailureMode(
                failure_id="FAIL-003",
                description="Attention produces incorrect results",
                trigger="Bug in attention implementation (masking, softmax, reshaping)",
                detection="Numerical equivalence test fails; loss diverges from baseline",
                recovery="Compare against reference implementation; simplify attention to debug",
            ))

        if "moe" in arch or "mixture" in arch:
            failures.append(FailureMode(
                failure_id="FAIL-004",
                description="Expert routing collapses to single expert",
                trigger="Load balancing loss not working; router not learning",
                detection="Monitor expert utilization distribution; should be roughly uniform",
                recovery="Increase load balancing loss weight; add noise to routing; adjust capacity factor",
            ))

        failures.append(FailureMode(
            failure_id="FAIL-005",
            description="Results significantly worse than paper claims",
            trigger="Implementation bug or missing detail from paper",
            detection="Metrics fall outside expected range after converged training",
            recovery="Re-read paper for missing details; check hyperparameters; simplify to match paper setup",
        ))

        failures.append(FailureMode(
            failure_id="FAIL-006",
            description="Checkpoint incompatibility",
            trigger="New model parameters don't match old checkpoint format",
            detection="Checkpoint load fails or produces wrong results",
            recovery="Write migration script; add backward-compatible parameter names",
        ))

        return failures

    def _define_success_criteria(self, input: ResultPredictionInput) -> list[str]:
        criteria = [
            "Implementation produces valid output shapes for all input sizes",
            "Training converges without NaN/Inf errors",
            "Baseline metrics are reproduced within 5% of reference",
            "New technique shows measurable improvement over baseline",
            "All unit and integration tests pass",
            "No memory regressions beyond acceptable threshold",
            "Checkpoint save/load works correctly",
        ]

        arch = input.summary.model_architecture.lower()
        if "attention" in arch:
            criteria.append("Attention mechanism produces numerically correct results")
            criteria.append("Memory usage is within claimed bounds")
        if "moe" in arch or "mixture" in arch:
            criteria.append("Expert utilization is balanced (>80% load balance)")
            criteria.append("Routing accuracy improves during training")

        return criteria

    def _estimate_confidence(self, input: ResultPredictionInput) -> ConfidenceLevel:
        repro = input.summary.reproduction_challenges
        repro_text = " ".join(repro).lower() if repro else ""

        if any(kw in repro_text for kw in ["proprietary", "not possible", "impossible"]):
            return ConfidenceLevel.LOW

        arch = input.summary.model_architecture.lower()
        if any(kw in arch for kw in ["moe", "mixture", "ring attention"]):
            return ConfidenceLevel.MEDIUM

        return ConfidenceLevel.MEDIUM
