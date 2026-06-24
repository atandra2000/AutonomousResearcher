"""Relevance Scoring Tool for Phase 6 - Literature Intelligence.

Scores how relevant a paper is to a target repository for implementation,
across 6 dimensions: architecture, training, inference, evaluation, data,
and feasibility.
"""

from research_engineer.models.literature import (
    RelevanceDimension,
    RelevanceLevel,
    RelevanceScore,
    RelevanceScoringInput,
    RelevanceScoringOutput,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.tools.base import Tool, ToolError

DIMENSIONS = [
    ("architecture", 0.25),
    ("training", 0.20),
    ("inference", 0.15),
    ("evaluation", 0.15),
    ("data", 0.15),
    ("feasibility", 0.10),
]

DIMENSION_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "architecture": {
        "paper": [
            "transformer", "attention", "encoder", "decoder", "mlp",
            "cnn", "rnn", "lstm", "moe", "diffusion", "vae", "gan",
            "resnet", "architecture", "network", "model",
        ],
        "repo": [
            "transformer", "attention", "encoder", "decoder", "mlp",
            "model", "network", "architecture", "layer", "module",
        ],
    },
    "training": {
        "paper": [
            "training", "optimizer", "adam", "sgd", "learning rate",
            "batch size", "loss", "gradient", "fine-tune", "pretrain",
        ],
        "repo": [
            "train", "optimizer", "loss", "gradient", "training",
            "fine-tune", "pretrain", "scheduler",
        ],
    },
    "inference": {
        "paper": [
            "inference", "kv cache", "quantiz", "int8", "int4",
            "flash attention", "speculative", "decoding",
        ],
        "repo": [
            "inference", "serve", "deploy", "quantiz", "cache",
            "latency", "throughput",
        ],
    },
    "evaluation": {
        "paper": [
            "evaluate", "metric", "benchmark", "accuracy", "perplexity",
            "bleu", "rouge", "f1", "baseline",
        ],
        "repo": [
            "eval", "metric", "benchmark", "accuracy", "validate",
            "test", "score",
        ],
    },
    "data": {
        "paper": [
            "dataset", "corpus", "c4", "wikipedia", "imagenet",
            "cifar", "glue", "data",
        ],
        "repo": [
            "dataset", "data", "dataloader", "corpus", "tokenize",
            "preprocess",
        ],
    },
    "feasibility": {
        "paper": [
            "simple", "straightforward", "standard", "modular",
            "custom", "complex", "distributed", "novel",
        ],
        "repo": [
            "config", "yaml", "json", "api", "interface", "plugin",
            "module", "extension",
        ],
    },
}


class RelevanceScoringTool(Tool[RelevanceScoringInput, RelevanceScoringOutput]):
    """Score implementation relevance of a paper to a repository."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: RelevanceScoringInput) -> bool:
        return bool(input.paper and input.repo_summary)

    async def execute(self, input: RelevanceScoringInput) -> RelevanceScoringOutput:
        try:
            repo = self._coerce_repo_summary(input.repo_summary)
            paper_text = f"{input.paper.title} {input.paper.abstract}".lower()
            repo_text = self._build_repo_text(repo).lower()

            dimensions: list[RelevanceDimension] = []
            weighted_sum = 0.0
            total_weight = 0.0

            for dim_name, weight in DIMENSIONS:
                score = self._score_dimension(dim_name, paper_text, repo_text)
                reasoning = self._dimension_reasoning(dim_name, score, paper_text, repo_text)
                evidence = self._dimension_evidence(dim_name, paper_text, repo_text)

                dimensions.append(
                    RelevanceDimension(
                        dimension=dim_name,
                        score=round(score, 3),
                        reasoning=reasoning,
                        evidence=evidence,
                    )
                )
                weighted_sum += score * weight
                total_weight += weight

            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
            level = self._classify_level(overall)

            relevance_score = RelevanceScore(
                paper_id=input.paper.paper_id,
                repo_path=getattr(repo, "repository_name", "unknown"),
                overall_score=round(overall, 3),
                relevance_level=level,
            )

            recommendations = self._generate_recommendations(dimensions, level)

            return RelevanceScoringOutput(
                score=relevance_score,
                dimension_scores=dimensions,
                recommendations=recommendations,
            )
        except Exception as e:
            raise ToolError(f"Relevance scoring failed: {e}", input, e)

    def _coerce_repo_summary(self, repo: object) -> RepositorySummary:
        if isinstance(repo, RepositorySummary):
            return repo
        if isinstance(repo, dict):
            return RepositorySummary(**repo)
        raise ToolError(f"Expected RepositorySummary, got {type(repo).__name__}", None)

    def _build_repo_text(self, repo: RepositorySummary) -> str:
        parts = [
            repo.repository_name,
            repo.project_type,
            repo.architecture_summary,
            repo.training_pipeline,
        ]
        parts.extend(f.file_path for f in repo.important_files)
        parts.extend(
            f"{t.file_path}:{t.target_type}:{t.insertion_point}"
            for t in repo.implementation_targets
        )
        return " ".join(parts)

    def _score_dimension(
        self, dim_name: str, paper_text: str, repo_text: str
    ) -> float:
        keywords = DIMENSION_KEYWORDS.get(dim_name, {})
        paper_kws = keywords.get("paper", [])
        repo_kws = keywords.get("repo", [])

        paper_matches = sum(1 for kw in paper_kws if kw in paper_text)
        repo_matches = sum(1 for kw in repo_kws if kw in repo_text)

        paper_score = min(1.0, paper_matches / max(1, len(paper_kws) * 0.3))
        repo_score = min(1.0, repo_matches / max(1, len(repo_kws) * 0.3))

        shared = sum(
            1 for kw in paper_kws
            if kw in paper_text and kw in repo_text
        )
        overlap_score = min(1.0, shared / max(1, len(paper_kws) * 0.2))

        score = 0.3 * paper_score + 0.3 * repo_score + 0.4 * overlap_score
        return min(1.0, score)

    def _dimension_reasoning(
        self, dim_name: str, score: float, paper_text: str, repo_text: str
    ) -> str:
        if score >= 0.7:
            return f"{dim_name}: High relevance ({score:.2f}). Paper and repo share key concepts."
        if score >= 0.4:
            return f"{dim_name}: Medium relevance ({score:.2f}). Some overlap detected."
        return f"{dim_name}: Low relevance ({score:.2f}). Limited conceptual overlap."

    def _dimension_evidence(
        self, dim_name: str, paper_text: str, repo_text: str
    ) -> list[str]:
        keywords = DIMENSION_KEYWORDS.get(dim_name, {})
        evidence: list[str] = []
        for kw in keywords.get("paper", []):
            if kw in paper_text and kw in repo_text:
                evidence.append(f"Shared keyword: '{kw}'")
        if not evidence:
            for kw in keywords.get("paper", []):
                if kw in paper_text:
                    evidence.append(f"Paper mentions: '{kw}'")
                    break
        return evidence[:5]

    def _classify_level(self, score: float) -> RelevanceLevel:
        if score >= 0.7:
            return RelevanceLevel.HIGH
        if score >= 0.4:
            return RelevanceLevel.MEDIUM
        return RelevanceLevel.LOW

    def _generate_recommendations(
        self, dimensions: list[RelevanceDimension], level: RelevanceLevel
    ) -> list[str]:
        recs: list[str] = []
        if level == RelevanceLevel.HIGH:
            recs.append("Strong candidate for implementation. Proceed with planning.")
        elif level == RelevanceLevel.MEDIUM:
            recs.append("Moderate relevance. Review low-scoring dimensions before proceeding.")
        else:
            recs.append("Low relevance. May require significant adaptation.")

        for dim in dimensions:
            if dim.score < 0.4:
                recs.append(
                    f"Address {dim.dimension} gap: {dim.reasoning}"
                )
        return recs
