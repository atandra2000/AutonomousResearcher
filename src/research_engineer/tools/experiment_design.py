"""Experiment Design Tool for Phase 3 - Experiment Planner.

Creates experiment matrices with baseline, minimum viable,
ablation, stress, and scaling experiments.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    Experiment,
    ExperimentGroup,
    ExperimentMatrix,
    ExperimentType,
    MetricDefinition,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ExperimentDesignInput(BaseModel):
    """Input for experiment design."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(..., description="Repository summary")


class ExperimentDesignOutput(BaseModel):
    """Output from experiment design."""

    matrix: ExperimentMatrix = Field(..., description="Experiment matrix")


class ExperimentDesignTool(Tool[ExperimentDesignInput, ExperimentDesignOutput]):
    """Design experiment matrix for paper-repo integration."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ExperimentDesignInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary)

    async def execute(self, input: ExperimentDesignInput) -> ExperimentDesignOutput:
        try:
            metrics = self._define_metrics(input)
            groups = self._create_experiment_groups(input)
            total = sum(len(g.experiments) for g in groups)
            gpu_hours = self._estimate_gpu_hours(groups)

            matrix = ExperimentMatrix(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                metrics=metrics,
                groups=groups,
                total_experiments=total,
                estimated_total_gpu_hours=gpu_hours,
            )

            return ExperimentDesignOutput(matrix=matrix)

        except Exception as e:
            raise ToolError(f"Experiment design failed: {e}", input, e)

    def _define_metrics(self, input: ExperimentDesignInput) -> list[MetricDefinition]:
        metrics = [
            MetricDefinition(
                name="loss",
                description="Training and validation loss",
                unit="nats",
                target_value=None, why_important="Primary indicator of model convergence",
            ),
            MetricDefinition(
                name="perplexity",
                description="Language model perplexity on validation set",
                unit="PPL",
                target_value=None, why_important="Standard LM quality metric",
            ),
            MetricDefinition(
                name="tokens_per_second",
                description="Training throughput in tokens/second",
                unit="tokens/s",
                target_value=None, why_important="Measures training efficiency",
            ),
            MetricDefinition(
                name="mfu",
                description="Model FLOPs Utilization",
                unit="%",
                target_value=None, why_important="Measures hardware utilization efficiency",
            ),
            MetricDefinition(
                name="gpu_memory_peak",
                description="Peak GPU memory usage during training",
                unit="GB",
                target_value=None, why_important="Determines maximum model size and batch size",
            ),
            MetricDefinition(
                name="validation_loss",
                description="Loss on held-out validation set",
                unit="nats",
                target_value=None, why_important="Tracks overfitting and generalization",
            ),
        ]

        arch = input.summary.model_architecture.lower()
        if "attention" in arch:
            metrics.append(MetricDefinition(
                name="attention_latency",
                description="Attention mechanism forward pass latency",
                unit="ms",
                target_value=None, why_important="Directly measures the technique's efficiency",
            ))
        if "moe" in arch or "mixture" in arch:
            metrics.append(MetricDefinition(
                name="expert_utilization",
                description="Distribution of tokens across MoE experts",
                unit="ratio",
                target_value=None, why_important="Ensures balanced expert usage",
            ))
            metrics.append(MetricDefinition(
                name="routing_accuracy",
                description="Expert routing accuracy",
                unit="%",
                target_value=None, why_important="Measures how well the router selects experts",
            ))

        return metrics

    def _create_experiment_groups(self, input: ExperimentDesignInput) -> list[ExperimentGroup]:
        groups: list[ExperimentGroup] = []
        paper_id_short = input.paper_id.replace(".", "_")

        groups.append(self._create_baseline_group(input, paper_id_short))
        groups.append(self._create_mve_group(input, paper_id_short))
        groups.append(self._create_ablation_group(input, paper_id_short))
        groups.append(self._create_stress_group(input, paper_id_short))
        groups.append(self._create_scaling_group(input, paper_id_short))

        return groups

    def _create_baseline_group(self, input: ExperimentDesignInput, pid: str) -> ExperimentGroup:
        return ExperimentGroup(
            group_name="Baseline Experiments",
            group_type=ExperimentType.BASELINE,
            description="Run existing implementation to establish baseline metrics before any changes",
            experiments=[
                Experiment(
                    experiment_id=f"{pid}_baseline_vanilla",
                    experiment_type=ExperimentType.BASELINE,
                    title="Vanilla Baseline",
                    description="Run the existing model without any modifications to establish baseline metrics",
                    hypothesis="Baseline produces expected metrics from existing implementation",
                    configuration={"model": "baseline", "batch_size": "default", "learning_rate": "default"},
                    baseline_id=None,
                    expected_duration="2-4 hours",
                    required_resources=["1-2 GPUs"],
                    success_criteria=[
                        "Baseline loss converges",
                        "Baseline perplexity matches expectations",
                        "No training errors",
                    ],
                ),
                Experiment(
                    experiment_id=f"{pid}_baseline_extended",
                    experiment_type=ExperimentType.BASELINE,
                    title="Extended Baseline",
                    description="Run baseline with extended training to establish long-run performance",
                    hypothesis="Extended training shows stable convergence pattern",
                    configuration={"model": "baseline", "training_steps": "2x default"},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="4-8 hours",
                    required_resources=["1-2 GPUs"],
                    success_criteria=[
                        "No training instability",
                        "Loss continues to decrease",
                        "No NaN/Inf errors",
                    ],
                ),
            ],
        )

    def _create_mve_group(self, input: ExperimentDesignInput, pid: str) -> ExperimentGroup:
        return ExperimentGroup(
            group_name="Minimum Viable Experiments",
            group_type=ExperimentType.MINIMUM_VIABLE,
            description="Smallest experiments to validate the new technique works at all",
            experiments=[
                Experiment(
                    experiment_id=f"{pid}_mve_small",
                    experiment_type=ExperimentType.MINIMUM_VIABLE,
                    title="Small-Scale MVE",
                    description="Test the new technique on a small model/data subset to verify it runs correctly",
                    hypothesis="New technique produces valid output shapes and training converges",
                    configuration={"model": "new_technique", "scale": "small", "batch_size": "default"},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="1-2 hours",
                    required_resources=["1 GPU"],
                    success_criteria=[
                        "Forward pass produces correct shapes",
                        "Loss decreases during training",
                        "No NaN/Inf errors",
                        "Memory usage within GPU limits",
                    ],
                ),
            ],
        )

    def _create_ablation_group(self, input: ExperimentDesignInput, pid: str) -> ExperimentGroup:
        arch = input.summary.model_architecture.lower()
        experiments: list[Experiment] = [
            Experiment(
                experiment_id=f"{pid}_ablation_no_technique",
                experiment_type=ExperimentType.ABLATION,
                title="Ablation: Without New Technique",
                description="Run model without the new technique to isolate its contribution",
                hypothesis="Removing the technique degrades performance compared to MVE",
                configuration={"model": "baseline", "note": "Same as baseline, for comparison reference"},
                baseline_id=f"{pid}_baseline_vanilla",
                expected_duration="2-4 hours",
                required_resources=["1-2 GPUs"],
                success_criteria=["Results match baseline", "Technique contribution is measurable"],
            ),
        ]

        if "attention" in arch:
            experiments.append(Experiment(
                experiment_id=f"{pid}_ablation_attention_variant",
                experiment_type=ExperimentType.ABLATION,
                title="Ablation: Attention Variant Comparison",
                description="Compare different attention configurations to isolate the contribution",
                hypothesis="Full technique outperforms simplified variants",
                configuration={"model": "attention_variant", "variant": "simplified"},
                baseline_id=f"{pid}_baseline_vanilla",
                expected_duration="4-8 hours",
                required_resources=["1-2 GPUs"],
                success_criteria=["Each variant trains successfully", "Performance gaps are measurable"],
            ))

        if "moe" in arch or "mixture" in arch:
            experiments.append(Experiment(
                experiment_id=f"{pid}_ablation_expert_count",
                experiment_type=ExperimentType.ABLATION,
                title="Ablation: Expert Count",
                description="Vary number of experts to understand impact on quality and efficiency",
                hypothesis="Quality improves with more experts up to diminishing returns",
                configuration={"model": "moe", "expert_counts": [2, 4, 8, 16]},
                baseline_id=f"{pid}_baseline_vanilla",
                expected_duration="8-16 hours",
                required_resources=["2-4 GPUs"],
                success_criteria=["Each expert count trains", "Quality trend is measurable"],
            ))

        return ExperimentGroup(
            group_name="Ablation Experiments",
            group_type=ExperimentType.ABLATION,
            description="Systematically remove or vary components to understand their contribution",
            experiments=experiments,
        )

    def _create_stress_group(self, input: ExperimentDesignInput, pid: str) -> ExperimentGroup:
        return ExperimentGroup(
            group_name="Stress Tests",
            group_type=ExperimentType.STRESS,
            description="Test the implementation under extreme conditions",
            experiments=[
                Experiment(
                    experiment_id=f"{pid}_stress_long_seq",
                    experiment_type=ExperimentType.STRESS,
                    title="Long Sequence Stress Test",
                    description="Test with very long sequences to verify memory and correctness under pressure",
                    hypothesis="Implementation handles max sequence length without OOM or errors",
                    configuration={"model": "new_technique", "seq_length": "max_supported", "batch_size": 1},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="1-2 hours",
                    required_resources=["1-2 GPUs"],
                    success_criteria=[
                        "No out-of-memory errors",
                        "Correct attention pattern for long sequences",
                        "Numerical stability maintained",
                    ],
                ),
                Experiment(
                    experiment_id=f"{pid}_stress_large_batch",
                    experiment_type=ExperimentType.STRESS,
                    title="Large Batch Stress Test",
                    description="Test with large batch sizes to verify memory scaling",
                    hypothesis="Implementation scales gracefully with batch size",
                    configuration={"model": "new_technique", "batch_size": "large"},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="1-2 hours",
                    required_resources=["1-2 GPUs"],
                    success_criteria=[
                        "Batch scaling is near-linear",
                        "No OOM with reasonable batch sizes",
                    ],
                ),
                Experiment(
                    experiment_id=f"{pid}_stress_gradient",
                    experiment_type=ExperimentType.STRESS,
                    title="Gradient Stability Test",
                    description="Monitor gradient norms to verify numerical stability",
                    hypothesis="Gradients remain stable without explosion or vanishing",
                    configuration={"model": "new_technique", "gradient_monitoring": True},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="1-2 hours",
                    required_resources=["1 GPU"],
                    success_criteria=[
                        "No gradient explosion (norm < 1000)",
                        "No gradient vanishing (norm > 1e-7)",
                        "Training loss decreases smoothly",
                    ],
                ),
            ],
        )

    def _create_scaling_group(self, input: ExperimentDesignInput, pid: str) -> ExperimentGroup:
        return ExperimentGroup(
            group_name="Scaling Experiments",
            group_type=ExperimentType.SCALING,
            description="Test how the technique scales with model size and data",
            experiments=[
                Experiment(
                    experiment_id=f"{pid}_scale_model_small",
                    experiment_type=ExperimentType.SCALING,
                    title="Small Model Scale",
                    description="Test technique on smallest model configuration",
                    hypothesis="Technique works correctly at small scale",
                    configuration={"model_size": "small", "model": "new_technique"},
                    baseline_id=f"{pid}_baseline_vanilla",
                    expected_duration="2-4 hours",
                    required_resources=["1 GPU"],
                    success_criteria=["Small model trains successfully", "Metrics are reasonable"],
                ),
                Experiment(
                    experiment_id=f"{pid}_scale_model_medium",
                    experiment_type=ExperimentType.SCALING,
                    title="Medium Model Scale",
                    description="Test technique on medium model configuration",
                    hypothesis="Technique scales reasonably to medium size",
                    configuration={"model_size": "medium", "model": "new_technique"},
                    baseline_id=f"{pid}_scale_model_small",
                    expected_duration="4-8 hours",
                    required_resources=["2-4 GPUs"],
                    success_criteria=["Medium model trains successfully", "Quality scales as expected"],
                ),
                Experiment(
                    experiment_id=f"{pid}_scale_model_large",
                    experiment_type=ExperimentType.SCALING,
                    title="Large Model Scale",
                    description="Test technique on target model size (if resources allow)",
                    hypothesis="Technique maintains benefits at scale",
                    configuration={"model_size": "large", "model": "new_technique"},
                    baseline_id=f"{pid}_scale_model_medium",
                    expected_duration="12-24 hours",
                    required_resources=["4-8 GPUs"],
                    success_criteria=["Large model trains successfully", "Quality improvement is sustained"],
                ),
            ],
        )

    def _estimate_gpu_hours(self, groups: list[ExperimentGroup]) -> float:  # noqa: C901
        total_hours = 0.0
        hour_map = {
            "1 gpu": 2.0,
            "1-2 gpus": 4.0,
            "2-4 gpus": 12.0,
            "4-8 gpus": 48.0,
        }

        for group in groups:
            for exp in group.experiments:
                dur = exp.expected_duration.lower()
                hours = 2.0
                if "1-2 hour" in dur:
                    hours = 1.5
                elif "2-4 hour" in dur:
                    hours = 3.0
                elif "4-8 hour" in dur:
                    hours = 6.0
                elif "8-16 hour" in dur:
                    hours = 12.0
                elif "12-24 hour" in dur:
                    hours = 18.0

                gpu_count: float = 1.0
                for res in exp.required_resources:
                    for key, val in hour_map.items():
                        if key in res.lower():
                            gpu_count = val / 2.0

                total_hours += hours * max(gpu_count, 1)

        return total_hours
