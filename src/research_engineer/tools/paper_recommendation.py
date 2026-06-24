"""Paper Recommendation Tool for Phase 6 - Literature Intelligence.

Recommends papers worth implementing based on impact, novelty,
implementability, and relevance to specified criteria.
"""

from datetime import datetime

from research_engineer.models.literature import (
    PaperRecommendation,
    PaperRecommendationInput,
    PaperRecommendationOutput,
    PaperSummary,
    RecommendationCriteria,
)
from research_engineer.tools.base import Tool, ToolError

COMPLEXITY_INDICATORS = [
    "novel architecture",
    "custom",
    "specialized",
    "complex",
    "intricate",
    "distributed",
    "multi-stage",
]

SIMPLICITY_INDICATORS = [
    "simple", "straightforward", "standard", "modular",
    "drop-in", "plug-and-play", "minimal",
]


class PaperRecommendationTool(Tool[PaperRecommendationInput, PaperRecommendationOutput]):
    """Recommend papers worth implementing."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: PaperRecommendationInput) -> bool:
        return len(input.papers) >= 1

    async def execute(self, input: PaperRecommendationInput) -> PaperRecommendationOutput:
        try:
            criteria = input.criteria or RecommendationCriteria()
            current_year = datetime.now().year

            candidates = self._filter_papers(input.papers, criteria, current_year)

            scored: list[tuple[PaperSummary, PaperRecommendation]] = []
            for paper in candidates:
                impact = self._impact_score(paper, candidates)
                novelty = self._novelty_score(paper, candidates)
                implementability = self._implementability_score(paper)
                relevance = self._relevance_score(paper, criteria)

                overall = (
                    0.3 * impact
                    + 0.25 * novelty
                    + 0.25 * implementability
                    + 0.2 * relevance
                )

                strengths = self._identify_strengths(
                    paper, impact, novelty, implementability
                )
                challenges = self._identify_challenges(paper, implementability)

                rec = PaperRecommendation(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    rank=0,
                    overall_score=round(overall, 3),
                    impact_score=round(impact, 3),
                    novelty_score=round(novelty, 3),
                    implementability_score=round(implementability, 3),
                    relevance_score=round(relevance, 3),
                    rationale=self._build_rationale(
                        paper, impact, novelty, implementability, relevance
                    ),
                    key_strengths=strengths,
                    potential_challenges=challenges,
                )
                scored.append((paper, rec))

            scored.sort(key=lambda x: x[1].overall_score, reverse=True)

            for rank, (_, rec) in enumerate(scored[: input.max_recommendations], 1):
                rec.rank = rank

            top = [rec for _, rec in scored[: input.max_recommendations]]
            rationale = self._build_overall_rationale(top)

            return PaperRecommendationOutput(
                recommendations=top,
                ranking_rationale=rationale,
            )
        except Exception as e:
            raise ToolError(f"Paper recommendation failed: {e}", input, e)

    def _filter_papers(
        self,
        papers: list[PaperSummary],
        criteria: RecommendationCriteria,
        current_year: int,
    ) -> list[PaperSummary]:
        filtered = [
            p for p in papers if p.citation_count >= criteria.min_citation_count
        ]
        if criteria.max_age_years is not None:
            max_age = criteria.max_age_years
            filtered = [
                p for p in filtered
                if p.year is None or (current_year - p.year) <= max_age
            ]
        if criteria.topics:
            topic_set = {t.lower() for t in criteria.topics}
            filtered = [
                p for p in filtered
                if any(
                    topic in f"{p.title} {p.abstract}".lower()
                    for topic in topic_set
                )
            ]
        return filtered

    def _impact_score(
        self, paper: PaperSummary, all_papers: list[PaperSummary]
    ) -> float:
        max_citations = max(
            (p.citation_count for p in all_papers), default=1
        )
        if max_citations == 0:
            return 0.5
        return min(1.0, paper.citation_count / max_citations)

    def _novelty_score(
        self, paper: PaperSummary, all_papers: list[PaperSummary]
    ) -> float:
        text = f"{paper.title} {paper.abstract}".lower()
        novelty_keywords = [
            "novel", "new", "first", "unique", "innovative", "original",
            "unlike", "different", "breakthrough",
        ]
        novelty_count = sum(1 for kw in novelty_keywords if kw in text)

        all_text = " ".join(
            f"{p.title} {p.abstract}" for p in all_papers
        ).lower()
        rarity: list[float] = []
        for kw in novelty_keywords:
            freq = all_text.count(kw)
            rarity.append(1.0 / (1.0 + freq))
        avg_rarity = sum(rarity) / len(rarity) if rarity else 0.5

        score = min(1.0, (novelty_count / 3.0) * 0.5 + avg_rarity * 0.5)
        return max(0.1, score)

    def _implementability_score(self, paper: PaperSummary) -> float:
        text = f"{paper.title} {paper.abstract}".lower()
        simplicity = sum(1 for kw in SIMPLICITY_INDICATORS if kw in text)
        complexity = sum(1 for kw in COMPLEXITY_INDICATORS if kw in text)

        score = 0.5 + 0.1 * simplicity - 0.1 * complexity
        return max(0.1, min(1.0, score))

    def _relevance_score(
        self, paper: PaperSummary, criteria: RecommendationCriteria
    ) -> float:
        if not criteria.topics:
            return 0.7
        text = f"{paper.title} {paper.abstract}".lower()
        topic_set = {t.lower() for t in criteria.topics}
        matched = sum(1 for t in topic_set if t in text)
        return matched / len(topic_set) if topic_set else 0.7

    def _identify_strengths(
        self,
        paper: PaperSummary,
        impact: float,
        novelty: float,
        implementability: float,
    ) -> list[str]:
        strengths: list[str] = []
        if impact > 0.7:
            strengths.append(f"High impact ({paper.citation_count} citations)")
        if novelty > 0.7:
            strengths.append("Novel contribution")
        if implementability > 0.7:
            strengths.append("Straightforward to implement")
        if paper.citation_count > 100:
            strengths.append("Well-validated by community")
        if not strengths:
            strengths.append("Solid contribution to the field")
        return strengths

    def _identify_challenges(
        self, paper: PaperSummary, implementability: float
    ) -> list[str]:
        challenges: list[str] = []
        text = f"{paper.title} {paper.abstract}".lower()
        if implementability < 0.4:
            challenges.append("Complex implementation")
        if "distributed" in text or "multi-gpu" in text:
            challenges.append("Requires distributed infrastructure")
        if "custom" in text:
            challenges.append("Custom components needed")
        if not challenges:
            challenges.append("No major challenges identified")
        return challenges

    def _build_rationale(
        self,
        paper: PaperSummary,
        impact: float,
        novelty: float,
        implementability: float,
        relevance: float,
    ) -> str:
        return (
            f"Paper '{paper.title[:60]}' scored: impact={impact:.2f}, "
            f"novelty={novelty:.2f}, implementability={implementability:.2f}, "
            f"relevance={relevance:.2f}. Recommended for implementation "
            f"based on weighted combination."
        )

    def _build_overall_rationale(
        self, recommendations: list[PaperRecommendation]
    ) -> str:
        if not recommendations:
            return "No recommendations generated."
        top = recommendations[0]
        return (
            f"Top recommendation: '{top.title[:60]}' with overall score "
            f"{top.overall_score:.2f}. {len(recommendations)} papers recommended "
            f"based on impact, novelty, implementability, and relevance."
        )
