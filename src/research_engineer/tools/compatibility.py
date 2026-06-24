"""Compatibility Analysis Tool for Phase 3 - Experiment Planner.

Analyzes compatibility between a research paper's technique
and a target repository across 7 dimensions:
architecture, training, inference, config, checkpoint,
distributed, and evaluation.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    CompatibilityDimension,
    CompatibilityLevel,
    CompatibilityReport,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class CompatibilityInput(BaseModel):
    """Input for compatibility analysis."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(
        ..., description="Repository summary"
    )


class CompatibilityOutput(BaseModel):
    """Output from compatibility analysis."""

    report: CompatibilityReport = Field(
        ..., description="Compatibility report"
    )


class CompatibilityAnalysisTool(Tool[CompatibilityInput, CompatibilityOutput]):
    """Analyze compatibility between paper technique and repository."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: CompatibilityInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary and input.repo_summary)

    async def execute(self, input: CompatibilityInput) -> CompatibilityOutput:
        try:
            arch_compat = self._analyze_architecture(input)
            train_compat = self._analyze_training(input)
            infer_compat = self._analyze_inference(input)
            config_compat = self._analyze_config(input)
            ckpt_compat = self._analyze_checkpoint(input)
            dist_compat = self._analyze_distributed(input)
            eval_compat = self._analyze_evaluation(input)

            dimensions = [
                arch_compat, train_compat, infer_compat,
                config_compat, ckpt_compat, dist_compat, eval_compat,
            ]
            overall = self._compute_overall(dimensions)

            report = CompatibilityReport(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                architecture_compatibility=arch_compat,
                training_compatibility=train_compat,
                inference_compatibility=infer_compat,
                config_compatibility=config_compat,
                checkpoint_compatibility=ckpt_compat,
                distributed_compatibility=dist_compat,
                evaluation_compatibility=eval_compat,
                overall_compatibility=overall,
                overall_reasoning=self._compute_overall_reasoning(dimensions, overall),
                recommended_actions=self._generate_recommendations(dimensions),
            )

            return CompatibilityOutput(report=report)

        except Exception as e:
            raise ToolError(f"Compatibility analysis failed: {e}", input, e)

    def _analyze_architecture(self, input: CompatibilityInput) -> CompatibilityDimension:
        summary = input.summary
        repo = input.repo_summary
        arch_text = summary.model_architecture.lower()
        repo_type = repo.project_type.lower() if repo.project_type else ""
        evidence = []
        blockers = []

        has_transformer_keywords = any(
            kw in arch_text
            for kw in ["transformer", "attention", "encoder", "decoder", "mlp"]
        )
        has_specialized = any(
            kw in arch_text
            for kw in ["moe", "mixture of experts", "ring attention",
                        "flash attention", "mla", "multi-head latent"]
        )

        repo_supports_arch = any(
            kw in repo_type.lower()
            for kw in ["training", "llm", "inference", "fine"]
        )

        if has_transformer_keywords and repo_supports_arch:
            level = CompatibilityLevel.HIGH
            evidence.append("Repository type supports transformer architecture")
        elif has_transformer_keywords:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Transformer architecture detected but repo type unclear")
        elif has_specialized:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Specialized architecture detected - may require significant changes")
            blockers.append("Specialized architecture requires careful integration")
        else:
            level = CompatibilityLevel.LOW
            blockers.append("Architecture type unclear from paper summary")

        if "moe" in arch_text or "mixture of experts" in arch_text:
            blockers.append("MoE architectures require routing infrastructure")

        return CompatibilityDimension(
            dimension="Architecture",
            level=level,
            reasoning=f"Architecture compatibility: {level.value}. "
            f"Paper describes: {summary.model_architecture[:200]}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_training(self, input: CompatibilityInput) -> CompatibilityDimension:
        summary = input.summary
        repo = input.repo_summary
        train_text = summary.training_methodology.lower()
        evidence = []
        blockers = []

        has_standard_training = any(
            kw in train_text
            for kw in ["adam", "sgd", "learning rate", "gradient", "loss"]
        )
        has_custom_loss = any(
            kw in train_text
            for kw in ["custom loss", "novel loss", "auxiliary loss", "modified loss"]
        )

        repo_has_training = any(
            kw in str(repo.architecture_summary).lower()
            for kw in ["train", "optimizer", "loss", "gradient"]
        )

        if has_standard_training and repo_has_training:
            level = CompatibilityLevel.HIGH
            evidence.append("Standard training methodology compatible with repo")
        elif has_standard_training:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Standard training but repo lacks training infrastructure")
        elif has_custom_loss:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Custom loss function requires integration")
            blockers.append("Custom loss implementation needed")
        else:
            level = CompatibilityLevel.LOW
            blockers.append("Training methodology unclear from paper")

        return CompatibilityDimension(
            dimension="Training",
            level=level,
            reasoning=f"Training compatibility: {level.value}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_inference(self, input: CompatibilityInput) -> CompatibilityDimension:
        summary = input.summary
        infer_text = (summary.model_architecture + " " + summary.training_methodology).lower()
        evidence = []
        blockers = []

        has_kv_cache = "kv cache" in infer_text or "key-value" in infer_text or "kv-cache" in infer_text
        has_quantization = any(kw in infer_text for kw in ["quantiz", "int8", "int4", "4-bit", "8-bit"])
        has_new_inference = any(
            kw in infer_text
            for kw in ["flash attention", "ring attention", "speculative decoding"]
        )

        if has_new_inference:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Novel inference technique detected")
            if has_kv_cache:
                blockers.append("KV cache modifications required")
            if has_quantization:
                blockers.append("Quantization support may need custom implementation")
        elif has_kv_cache or has_quantization:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Inference optimizations detected")
        else:
            level = CompatibilityLevel.HIGH
            evidence.append("No unusual inference requirements")

        return CompatibilityDimension(
            dimension="Inference",
            level=level,
            reasoning=f"Inference compatibility: {level.value}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_config(self, input: CompatibilityInput) -> CompatibilityDimension:
        repo = input.repo_summary
        evidence = []
        blockers = []

        config_files = repo.configuration_analysis.config_files
        config_framework = repo.configuration_analysis.config_framework

        if config_files:
            level = CompatibilityLevel.HIGH
            evidence.append(f"Repository has {len(config_files)} config files ({config_framework})")
        else:
            level = CompatibilityLevel.MEDIUM
            evidence.append("No config files detected - may need manual configuration")
            blockers.append("Configuration system unknown")

        return CompatibilityDimension(
            dimension="Configuration",
            level=level,
            reasoning=f"Config compatibility: {level.value}. Framework: {config_framework}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_checkpoint(self, input: CompatibilityInput) -> CompatibilityDimension:
        repo = input.repo_summary
        summary = input.summary
        evidence = []
        blockers = []

        checkpoint_settings = repo.configuration_analysis.checkpoint_settings
        has_model_change = any(
            kw in summary.model_architecture.lower()
            for kw in ["new layer", "modified", "additional", "extra", "novel"]
        )

        if checkpoint_settings:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Repository has checkpoint infrastructure")
            if has_model_change:
                blockers.append(
                    "Model architecture changes require checkpoint migration"
                )
        else:
            level = CompatibilityLevel.LOW
            evidence.append("No checkpoint settings found")
            blockers.append("Checkpoint compatibility unknown")

        if has_model_change:
            level = CompatibilityLevel.LOW if level == CompatibilityLevel.LOW else CompatibilityLevel.MEDIUM

        return CompatibilityDimension(
            dimension="Checkpoint",
            level=level,
            reasoning=f"Checkpoint compatibility: {level.value}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_distributed(self, input: CompatibilityInput) -> CompatibilityDimension:
        summary = input.summary
        repo = input.repo_summary
        train_text = summary.training_methodology.lower()
        evidence = []
        blockers = []

        distributed_settings = repo.configuration_analysis.distributed_settings
        has_distributed_paper = any(
            kw in train_text
            for kw in ["distributed", "multi-gpu", "multi-node", "ddp", "fsdp", "pipeline"]
        )

        if has_distributed_paper and distributed_settings:
            level = CompatibilityLevel.HIGH
            evidence.append("Both paper and repo support distributed training")
        elif has_distributed_paper:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Paper requires distributed training, repo may need setup")
            blockers.append("Distributed training infrastructure may need modification")
        elif distributed_settings:
            level = CompatibilityLevel.HIGH
            evidence.append("Repo supports distributed training")
        else:
            level = CompatibilityLevel.HIGH
            evidence.append("No distributed training requirements detected")

        return CompatibilityDimension(
            dimension="Distributed Training",
            level=level,
            reasoning=f"Distributed compatibility: {level.value}",
            evidence=evidence,
            blockers=blockers,
        )

    def _analyze_evaluation(self, input: CompatibilityInput) -> CompatibilityDimension:
        summary = input.summary
        eval_text = summary.evaluation_methodology.lower()
        evidence = []
        blockers = []

        has_standard_metrics = any(
            kw in eval_text
            for kw in ["perplexity", "accuracy", "bleu", "rouge", "f1", "auc"]
        )
        has_custom_metrics = any(
            kw in eval_text
            for kw in ["custom metric", "novel metric", "proposed metric", "new evaluation"]
        )

        if has_standard_metrics:
            level = CompatibilityLevel.HIGH
            evidence.append("Standard evaluation metrics used")
        elif has_custom_metrics:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Custom metrics require implementation")
            blockers.append("Custom evaluation metrics need implementation")
        else:
            level = CompatibilityLevel.MEDIUM
            evidence.append("Evaluation methodology needs clarification")

        return CompatibilityDimension(
            dimension="Evaluation",
            level=level,
            reasoning=f"Evaluation compatibility: {level.value}",
            evidence=evidence,
            blockers=blockers,
        )

    def _compute_overall(self, dimensions: list[CompatibilityDimension]) -> CompatibilityLevel:
        scores = {CompatibilityLevel.HIGH: 3, CompatibilityLevel.MEDIUM: 2, CompatibilityLevel.LOW: 1}
        values = [scores[d.level] for d in dimensions]
        avg = sum(values) / len(values) if values else 2

        if avg >= 2.5:
            return CompatibilityLevel.HIGH
        elif avg >= 1.5:
            return CompatibilityLevel.MEDIUM
        else:
            return CompatibilityLevel.LOW

    def _compute_overall_reasoning(
        self, dimensions: list[CompatibilityDimension], overall: CompatibilityLevel
    ) -> str:
        high_count = sum(1 for d in dimensions if d.level == CompatibilityLevel.HIGH)
        med_count = sum(1 for d in dimensions if d.level == CompatibilityLevel.MEDIUM)
        low_count = sum(1 for d in dimensions if d.level == CompatibilityLevel.LOW)
        return (
            f"Overall compatibility is {overall.value}: "
            f"{high_count} High, {med_count} Medium, {low_count} Low dimensions. "
            f"Key blockers: "
            + "; ".join(
                f"{d.dimension}: {d.blockers[0]}"
                for d in dimensions
                if d.blockers
            )
        )

    def _generate_recommendations(self, dimensions: list[CompatibilityDimension]) -> list[str]:
        recommendations = []
        for d in dimensions:
            if d.level == CompatibilityLevel.LOW:
                recommendations.append(
                    f"Address {d.dimension.lower()} blockers before proceeding: "
                    + "; ".join(d.blockers)
                )
            elif d.level == CompatibilityLevel.MEDIUM and d.blockers:
                recommendations.append(
                    f"Plan for {d.dimension.lower()} integration challenges: "
                    + "; ".join(d.blockers)
                )
        if not recommendations:
            recommendations.append("No major compatibility issues detected")
        return recommendations
