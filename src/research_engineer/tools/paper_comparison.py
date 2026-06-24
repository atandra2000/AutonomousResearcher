"""Paper Comparison Tool for Phase 6 - Literature Intelligence.

Compares papers on the same topic across multiple dimensions, detecting
similarities, differences, consensus findings, and conflicts.
"""

from difflib import SequenceMatcher

from research_engineer.models.literature import (
    ComparisonDimension,
    ComparisonMatrix,
    ConflictItem,
    DifferencePair,
    PaperComparisonInput,
    PaperComparisonOutput,
    PaperRanking,
    PaperSummary,
    SimilarityPair,
)
from research_engineer.tools.base import Tool, ToolError

DEFAULT_DIMENSIONS = [
    "Architecture",
    "Training",
    "Evaluation",
    "Dataset",
    "Results",
    "Novelty",
    "Scalability",
]

DIMENSION_KEYWORDS: dict[str, list[str]] = {
    "Architecture": [
        "transformer", "attention", "encoder", "decoder", "cnn", "rnn",
        "lstm", "mlp", "moe", "diffusion", "vae", "gan", "resnet",
        "architecture", "network", "model", "layer", "module",
    ],
    "Training": [
        "training", "optimizer", "adam", "sgd", "learning rate",
        "batch size", "loss", "gradient", "fine-tune", "pretrain",
        "schedule", "warmup", "curriculum",
    ],
    "Evaluation": [
        "evaluate", "metric", "benchmark", "accuracy", "perplexity",
        "bleu", "rouge", "f1", "auc", "baseline", "comparison",
    ],
    "Dataset": [
        "dataset", "data", "corpus", "benchmark", "c4", "wikipedia",
        "imagenet", "cifar", "glue", "superglue",
    ],
    "Results": [
        "result", "achieve", "outperform", "state-of-the-art", "sota",
        "improve", "better", "speedup", "reduction", "gain",
    ],
    "Novelty": [
        "novel", "new", "propose", "introduce", "first", "unique",
        "innovative", "original", "unlike", "different",
    ],
    "Scalability": [
        "scale", "scalable", "distributed", "multi-gpu", "multi-node",
        "parallel", "efficient", "large-scale", "billion", "trillion",
    ],
}

NEGATION_PATTERNS = [
    "unlike", "in contrast", "however", "unlike previous",
    "contrary to", "differs from", "as opposed to",
]


class PaperComparisonTool(Tool[PaperComparisonInput, PaperComparisonOutput]):
    """Compare papers across multiple dimensions."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: PaperComparisonInput) -> bool:
        return len(input.papers) >= 2

    async def execute(self, input: PaperComparisonInput) -> PaperComparisonOutput:
        try:
            dimensions = self._get_dimensions(input.dimensions)
            matrix = self._build_matrix(input.papers, dimensions)
            similarities = self._compute_similarities(input.papers)
            differences = self._compute_differences(input.papers, dimensions, matrix)
            consensus = self._find_consensus(input.papers)
            conflicts = self._find_conflicts(input.papers)
            ranking = self._rank_papers(input.papers)

            return PaperComparisonOutput(
                comparison=matrix,
                similarities=similarities,
                differences=differences,
                consensus_findings=consensus,
                conflicting_findings=conflicts,
                ranking=ranking,
            )
        except Exception as e:
            raise ToolError(f"Paper comparison failed: {e}", input, e)

    def _get_dimensions(self, override: list[str] | None) -> list[ComparisonDimension]:
        if override:
            return [ComparisonDimension(name=d, description=d) for d in override]
        return [
            ComparisonDimension(name=d, description=d) for d in DEFAULT_DIMENSIONS
        ]

    def _build_matrix(
        self, papers: list[PaperSummary], dimensions: list[ComparisonDimension]
    ) -> ComparisonMatrix:
        paper_ids = [p.paper_id for p in papers]
        matrix: dict[str, dict[str, str]] = {}
        for paper in papers:
            matrix[paper.paper_id] = {}
            text = f"{paper.title} {paper.abstract} {' '.join(paper.key_contributions)}".lower()
            for dim in dimensions:
                keywords = DIMENSION_KEYWORDS.get(dim.name, [])
                matched = [kw for kw in keywords if kw in text]
                matrix[paper.paper_id][dim.name] = (
                    ", ".join(matched[:3]) if matched else "N/A"
                )
        return ComparisonMatrix(
            papers=paper_ids,
            dimensions=dimensions,
            matrix=matrix,
        )

    def _compute_similarities(self, papers: list[PaperSummary]) -> list[SimilarityPair]:
        results: list[SimilarityPair] = []
        for i, a in enumerate(papers):
            for b in papers[i + 1:]:
                score = self._similarity_score(a, b)
                shared = self._shared_aspects(a, b)
                if score > 0.3:
                    results.append(
                        SimilarityPair(
                            paper_a=a.paper_id,
                            paper_b=b.paper_id,
                            similarity_score=round(score, 3),
                            shared_aspects=shared,
                        )
                    )
        return results

    def _similarity_score(self, a: PaperSummary, b: PaperSummary) -> float:
        text_a = f"{a.title} {a.abstract}".lower()
        text_b = f"{b.title} {b.abstract}".lower()
        ratio = SequenceMatcher(None, text_a, text_b).ratio()
        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        return 0.5 * ratio + 0.5 * overlap

    def _shared_aspects(self, a: PaperSummary, b: PaperSummary) -> list[str]:
        text_a = f"{a.title} {a.abstract}".lower()
        text_b = f"{b.title} {b.abstract}".lower()
        shared: list[str] = []
        for dim, keywords in DIMENSION_KEYWORDS.items():
            a_match = any(kw in text_a for kw in keywords)
            b_match = any(kw in text_b for kw in keywords)
            if a_match and b_match:
                shared.append(dim)
        return shared

    def _compute_differences(
        self,
        papers: list[PaperSummary],
        dimensions: list[ComparisonDimension],
        matrix: ComparisonMatrix,
    ) -> list[DifferencePair]:
        results: list[DifferencePair] = []
        for i, a in enumerate(papers):
            for b in papers[i + 1:]:
                for dim in dimensions:
                    val_a = matrix.matrix.get(a.paper_id, {}).get(dim.name, "")
                    val_b = matrix.matrix.get(b.paper_id, {}).get(dim.name, "")
                    if val_a != val_b and val_a != "N/A" and val_b != "N/A":
                        results.append(
                            DifferencePair(
                                paper_a=a.paper_id,
                                paper_b=b.paper_id,
                                dimension=dim.name,
                                value_a=val_a,
                                value_b=val_b,
                            )
                        )
        return results

    def _find_consensus(self, papers: list[PaperSummary]) -> list[str]:
        """Find findings shared across multiple papers."""
        all_results: dict[str, list[str]] = {}
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            for kw in DIMENSION_KEYWORDS["Results"]:
                if kw in text:
                    all_results.setdefault(kw, []).append(paper.paper_id)

        consensus: list[str] = []
        for finding, pids in all_results.items():
            if len(pids) >= 2:
                consensus.append(
                    f"Multiple papers report: {finding} (papers: {', '.join(pids[:3])})"
                )
        return consensus

    def _find_conflicts(self, papers: list[PaperSummary]) -> list[ConflictItem]:
        """Detect conflicting claims via negation patterns."""
        conflicts: list[ConflictItem] = []
        for i, a in enumerate(papers):
            text_a = f"{a.title} {a.abstract}".lower()
            has_negation_a = any(pat in text_a for pat in NEGATION_PATTERNS)
            if not has_negation_a:
                continue
            for b in papers[i + 1:]:
                shared_topic = self._shared_aspects(a, b)
                if shared_topic:
                    conflicts.append(
                        ConflictItem(
                            papers=[a.paper_id, b.paper_id],
                            topic=shared_topic[0],
                            positions={
                                a.paper_id: "contrasting approach",
                                b.paper_id: "standard approach",
                            },
                        )
                    )
        return conflicts

    def _rank_papers(self, papers: list[PaperSummary]) -> list[PaperRanking]:
        """Rank papers by impact (citation count * novelty)."""
        ranked: list[tuple[PaperSummary, float]] = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            novelty_keywords = DIMENSION_KEYWORDS["Novelty"]
            novelty = sum(1 for kw in novelty_keywords if kw in text)
            novelty_score = min(1.0, novelty / 3.0)
            impact = paper.citation_count * (0.5 + novelty_score)
            ranked.append((paper, impact))

        ranked.sort(key=lambda x: x[1], reverse=True)
        results: list[PaperRanking] = []
        for rank, (paper, score) in enumerate(ranked, 1):
            results.append(
                PaperRanking(
                    paper_id=paper.paper_id,
                    rank=rank,
                    score=round(score, 3),
                    rationale=f"Citations={paper.citation_count}, novelty keywords detected",
                )
            )
        return results
