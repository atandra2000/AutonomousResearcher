"""Next Experiment Tool for Phase 8.

Recommends the next experiment(s) to run based on evaluation findings,
failure modes, and dynamics patterns. Rule-based, no LLM.
"""

from __future__ import annotations

from research_engineer.models.evaluation import (
    DynamicsPatternType,
    ExperimentRecommendation,
    NextExperimentInput,
    NextExperimentOutput,
    RecommendationPriority,
    TrainingDynamicsOutput,
)
from research_engineer.models.experiment import (
    ExperimentStatus,
    ExperimentType,
)
from research_engineer.tools.base import Tool, ToolError

_PRIORITY_ORDER = {
    RecommendationPriority.CRITICAL: 0,
    RecommendationPriority.HIGH: 1,
    RecommendationPriority.MEDIUM: 2,
    RecommendationPriority.LOW: 3,
}


class NextExperimentTool(Tool[NextExperimentInput, NextExperimentOutput]):
    """Recommend next experiments from evaluation findings."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: NextExperimentInput) -> bool:
        return True

    async def execute(
        self, input: NextExperimentInput
    ) -> NextExperimentOutput:
        try:
            recs: list[ExperimentRecommendation] = []
            open_questions: list[str] = []

            all_failed = bool(input.experiments) and all(
                e.status
                in (ExperimentStatus.FAILED, ExperimentStatus.CRASHED)
                for e in input.experiments
            )

            if all_failed:
                recs.append(
                    ExperimentRecommendation(
                        rank=1,
                        title="Reproduce with minimal config",
                        rationale=(
                            "All experiments failed; isolate the failure "
                            "with a smaller, faster run first."
                        ),
                        suggested_type=ExperimentType.TRAINING,
                        suggested_changes=[
                            "Reduce batch size and model size",
                            "Use a tiny dataset subset",
                            "Fix data pipeline before scaling",
                        ],
                        priority=RecommendationPriority.CRITICAL,
                        expected_impact=(
                            "Confirm the pipeline works end-to-end"
                        ),
                        estimated_effort="low",
                    )
                )
                open_questions.append(
                    "What is the minimal config that reproduces the failure?"
                )
            else:
                recs.extend(self._from_dynamics(input.dynamics))
                recs.extend(self._from_significance(input.significance))
                recs.extend(self._from_comparison(input.comparison))

            if not recs:
                recs.append(
                    ExperimentRecommendation(
                        rank=1,
                        title="Run baseline ablation",
                        rationale=(
                            "No strong signal from prior runs; establish a "
                            "clean baseline and ablate one variable at a time."
                        ),
                        suggested_type=ExperimentType.TRAINING,
                        suggested_changes=[
                            "Fix seeds and hyperparameters",
                            "Run baseline + one ablation",
                        ],
                        priority=RecommendationPriority.MEDIUM,
                        expected_impact=(
                            "Establish a reliable comparison point"
                        ),
                        estimated_effort="medium",
                    )
                )

            recs = self._rank_and_cap(recs, input.max_recommendations)
            strategy = self._strategy(recs, all_failed, input)
            query = self._paper_query(recs, input) if input.recommend_papers else ""

            return NextExperimentOutput(
                experiment_recommendations=recs,
                paper_suggestions=[],  # filled in by the agent
                overall_strategy=strategy,
                open_questions=open_questions,
                paper_query=query,
            )
        except Exception as e:
            raise ToolError(f"Next-experiment recommendation failed: {e}", input, e)

    def _from_dynamics(
        self, dynamics: list[TrainingDynamicsOutput]
    ) -> list[ExperimentRecommendation]:
        recs: list[ExperimentRecommendation] = []
        for dyn in dynamics:
            pattern = dyn.primary_pattern()
            if pattern is None:
                continue
            rec = self._dynamics_rec(pattern)
            if rec is not None:
                recs.append(rec)
        return recs

    @staticmethod
    def _dynamics_rec(
        pattern: DynamicsPatternType,
    ) -> ExperimentRecommendation | None:
        table = {
            DynamicsPatternType.OVERFITTING: ExperimentRecommendation(
                rank=0,
                title="Add regularization against overfitting",
                rationale="Eval loss diverges from train loss (overfitting).",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Increase dropout",
                    "Add weight decay",
                    "Use data augmentation",
                    "Enable early stopping",
                ],
                priority=RecommendationPriority.HIGH,
                expected_impact="Close the train-eval gap",
                estimated_effort="low",
            ),
            DynamicsPatternType.UNDERFITTING: ExperimentRecommendation(
                rank=0,
                title="Increase capacity to fix underfitting",
                rationale="Both train and eval loss remain high.",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Increase model width/depth",
                    "Train for more epochs",
                    "Raise the learning rate",
                ],
                priority=RecommendationPriority.HIGH,
                expected_impact="Lower both train and eval loss",
                estimated_effort="medium",
            ),
            DynamicsPatternType.DIVERGENCE: ExperimentRecommendation(
                rank=0,
                title="Stabilize training to stop divergence",
                rationale="Loss is increasing over time (divergence).",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Lower the learning rate",
                    "Add gradient clipping (max_norm=1.0)",
                    "Check the LR schedule",
                ],
                priority=RecommendationPriority.CRITICAL,
                expected_impact="Restore stable loss descent",
                estimated_effort="low",
            ),
            DynamicsPatternType.INSTABILITY: ExperimentRecommendation(
                rank=0,
                title="Reduce training instability",
                rationale="Metric variance is high across steps.",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Reduce batch size or use gradient accumulation",
                    "Lower the learning rate",
                    "Use a warmup schedule",
                ],
                priority=RecommendationPriority.HIGH,
                expected_impact="Smooth out the loss curve",
                estimated_effort="low",
            ),
            DynamicsPatternType.CONVERGENCE: ExperimentRecommendation(
                rank=0,
                title="Apply LR decay after plateau",
                rationale="Training has plateaued (convergence reached).",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Switch to cosine LR decay",
                    "If still underfitting, increase capacity",
                ],
                priority=RecommendationPriority.MEDIUM,
                expected_impact="Squeeze additional gains",
                estimated_effort="low",
            ),
            DynamicsPatternType.HEALTHY: ExperimentRecommendation(
                rank=0,
                title="Scale up the healthy run",
                rationale="Run is healthy and converged.",
                suggested_type=ExperimentType.TRAINING,
                suggested_changes=[
                    "Scale to more data / GPUs",
                    "Enable mixed precision",
                ],
                priority=RecommendationPriority.LOW,
                expected_impact="Higher final metric / faster wall-clock",
                estimated_effort="high",
            ),
            DynamicsPatternType.PLATEAU: None,
        }
        return table.get(pattern)

    def _from_significance(
        self, significance: object | None
    ) -> list[ExperimentRecommendation]:
        if significance is None:
            return []
        recs: list[ExperimentRecommendation] = []
        results = getattr(significance, "results", []) or []
        sig = [r for r in results if getattr(r, "significant", False)]
        if not results:
            return recs
        if not sig:
            recs.append(
                ExperimentRecommendation(
                    rank=0,
                    title="Run more seeds for statistical power",
                    rationale=(
                        "No significant difference detected; increase sample "
                        "size (seeds) before drawing conclusions."
                    ),
                    suggested_type=ExperimentType.TRAINING,
                    suggested_changes=[
                        "Run 3+ random seeds per config",
                        "Report mean +/- std",
                    ],
                    priority=RecommendationPriority.MEDIUM,
                    expected_impact="Reach statistical significance",
                    estimated_effort="medium",
                )
            )
        else:
            best = getattr(significance, "best_experiment_id", None)
            recs.append(
                ExperimentRecommendation(
                    rank=0,
                    title="Adopt winning configuration and ablate",
                    rationale=(
                        "A statistically significant winner was found"
                        + (f" ({best})" if best else "")
                        + "."
                    ),
                    suggested_type=ExperimentType.VALIDATION,
                    suggested_changes=[
                        f"Adopt config of {best}" if best else "Adopt winner config",
                        "Ablate the winning change to confirm causality",
                    ],
                    priority=RecommendationPriority.HIGH,
                    expected_impact="Confirm and lock in the improvement",
                    estimated_effort="medium",
                )
            )
        return recs

    def _from_comparison(
        self, comparison: object | None
    ) -> list[ExperimentRecommendation]:
        if comparison is None:
            return []
        winner = getattr(comparison, "winner_experiment_id", None)
        if winner:
            return [
                ExperimentRecommendation(
                    rank=0,
                    title="Ablate the winning change",
                    rationale=(
                        f"Comparison found {winner} as winner; "
                        "confirm the cause with an ablation."
                    ),
                    suggested_type=ExperimentType.VALIDATION,
                    suggested_changes=[
                        f"Revert the winning change in {winner}",
                        "Re-run to confirm the metric drops",
                    ],
                    priority=RecommendationPriority.HIGH,
                    expected_impact="Causal confirmation of the win",
                    estimated_effort="low",
                )
            ]
        return []

    def _rank_and_cap(
        self,
        recs: list[ExperimentRecommendation],
        max_recs: int,
    ) -> list[ExperimentRecommendation]:
        recs.sort(key=lambda r: _PRIORITY_ORDER.get(r.priority, 99))
        capped = recs[:max_recs]
        for i, r in enumerate(capped, 1):
            r.rank = i
        return capped

    @staticmethod
    def _strategy(
        recs: list[ExperimentRecommendation],
        all_failed: bool,
        input: NextExperimentInput,
    ) -> str:
        if all_failed:
            return (
                "All experiments failed. First reproduce with a minimal "
                "config to isolate the failure, then scale back up."
            )
        if not recs:
            return "No actionable recommendations; run a clean baseline."
        top = recs[0]
        lines = [
            f"Top priority: {top.title} ({top.priority.value}).",
            f"Rationale: {top.rationale}",
        ]
        if input.paper_id:
            lines.append(
                f"Context: paper {input.paper_id}."
            )
        return " ".join(lines)

    @staticmethod
    def _paper_query(
        recs: list[ExperimentRecommendation],
        input: NextExperimentInput,
    ) -> str:
        keywords: list[str] = []
        if recs:
            keywords.append(recs[0].title.split()[0].lower())
        for r in recs[:2]:
            for change in r.suggested_changes[:1]:
                keywords.extend(change.lower().split()[:2])
        base = " ".join(keywords[:6])
        if input.paper_id:
            base = f"{input.paper_id} {base}".strip()
        return base or "machine learning training"
