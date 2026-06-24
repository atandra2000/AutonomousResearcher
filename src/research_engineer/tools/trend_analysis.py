"""Trend Analysis Tool for Phase 6 - Literature Intelligence.

Identifies research trends over time from a collection of papers. Detects
rising, stable, and declining topics, plus emerging and hot areas.
"""

from datetime import datetime

from research_engineer.models.literature import (
    PaperSummary,
    ResearchTrend,
    TopicEntry,
    TrendAnalysisInput,
    TrendAnalysisOutput,
    TrendDirection,
)
from research_engineer.tools.base import Tool, ToolError

STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to",
    "with", "from", "by", "at", "is", "are", "was", "were", "be",
    "this", "that", "we", "our", "their", "it", "as", "which", "but",
    "not", "can", "will", "has", "have", "had", "been", "more", "than",
    "also", "such", "these", "those", "its", "about", "into", "over",
    "after", "before", "between", "through", "during", "based", "using",
    "use", "used", "uses", "method", "methods", "paper", "model",
    "approach", "proposed", "propose", "show", "shown", "results",
    "performance", "task", "tasks", "problem", "work", "study",
    "present", "introduce", "demonstrate", "achieve", "provide",
    "given", "via", "each", "both", "new", "novel", "recent",
    "existing", "previous", "prior", "well", "same",
})


class TrendAnalysisTool(Tool[TrendAnalysisInput, TrendAnalysisOutput]):
    """Analyze research trends from a collection of papers."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: TrendAnalysisInput) -> bool:
        return len(input.papers) >= 1

    async def execute(self, input: TrendAnalysisInput) -> TrendAnalysisOutput:
        try:
            current_year = datetime.now().year
            start_year = current_year - input.time_window_years + 1

            papers_with_year = [p for p in input.papers if p.year is not None]
            papers_in_window = [
                p for p in papers_with_year if (p.year or 0) >= start_year
            ]

            if not papers_in_window:
                return TrendAnalysisOutput(
                    trend_summary="No papers with year metadata in the time window."
                )

            topics = self._extract_topics(papers_in_window)
            trends: list[ResearchTrend] = []
            emerging: list[TopicEntry] = []
            declining: list[TopicEntry] = []
            hot: list[TopicEntry] = []

            for topic in topics:
                topic_papers = [
                    p for p in papers_in_window if self._paper_has_topic(p, topic)
                ]
                if len(topic_papers) < input.min_papers_per_trend:
                    continue

                counts_by_year = self._count_by_year(topic_papers, start_year, current_year)
                growth_rate = self._compute_growth_rate(counts_by_year)
                direction = self._classify_direction(growth_rate)
                years = [p.year for p in topic_papers if p.year]
                start = min(years) if years else None
                peak = max(years, key=lambda y: counts_by_year.get(str(y), 0)) if years else None
                key_papers = sorted(
                    topic_papers, key=lambda p: p.citation_count, reverse=True
                )[:5]
                key_ids = [p.paper_id for p in key_papers]

                trend = ResearchTrend(
                    topic=topic,
                    direction=direction,
                    paper_count_by_year=counts_by_year,
                    growth_rate=round(growth_rate, 1),
                    key_papers=key_ids,
                    description=self._trend_description(topic, direction, len(topic_papers), growth_rate),
                    start_year=start,
                    peak_year=peak,
                )
                trends.append(trend)

                entry = TopicEntry(
                    topic=topic,
                    paper_count=len(topic_papers),
                    growth_rate=round(growth_rate, 1),
                    recent_papers=[
                        p.paper_id for p in topic_papers if p.year == current_year
                    ],
                    first_seen_year=start,
                )

                if start and start >= current_year - 2 and growth_rate > 20:
                    emerging.append(entry)
                if growth_rate < -10:
                    declining.append(entry)
                if counts_by_year.get(str(current_year), 0) >= max(
                    counts_by_year.values()
                ) and counts_by_year.get(str(current_year), 0) >= 2:
                    hot.append(entry)

            trends.sort(key=lambda t: abs(t.growth_rate), reverse=True)
            emerging.sort(key=lambda e: e.growth_rate, reverse=True)
            hot.sort(key=lambda h: h.paper_count, reverse=True)
            declining.sort(key=lambda d: d.growth_rate)

            summary = self._build_summary(trends, emerging, hot, declining)

            return TrendAnalysisOutput(
                trends=trends,
                emerging_topics=emerging,
                declining_topics=declining,
                hot_topics=hot,
                trend_summary=summary,
            )
        except Exception as e:
            raise ToolError(f"Trend analysis failed: {e}", input, e)

    def _extract_topics(self, papers: list[PaperSummary]) -> list[str]:
        """Extract topics via keyword frequency from titles and abstracts."""
        word_freq: dict[str, int] = {}
        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            words = text.replace(",", " ").replace(".", " ").split()
            for word in words:
                word = word.strip()
                if len(word) < 4 or word in STOPWORDS or word.isdigit():
                    continue
                word_freq[word] = word_freq.get(word, 0) + 1

        min_count = max(2, len(papers) // 10)
        topics = [w for w, c in word_freq.items() if c >= min_count]
        topics.sort(key=lambda w: word_freq[w], reverse=True)
        return topics[:20]

    def _paper_has_topic(self, paper: PaperSummary, topic: str) -> bool:
        text = f"{paper.title} {paper.abstract}".lower()
        return topic in text

    def _count_by_year(
        self, papers: list[PaperSummary], start_year: int, end_year: int
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for y in range(start_year, end_year + 1):
            counts[str(y)] = 0
        for p in papers:
            if p.year:
                key = str(p.year)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _compute_growth_rate(self, counts_by_year: dict[str, int]) -> float:
        """Compute annual growth rate via linear regression slope."""
        years = sorted(counts_by_year.keys(), key=int)
        if len(years) < 2:
            return 0.0
        values = [counts_by_year[y] for y in years]
        first = values[0]
        last = values[-1]
        if first == 0:
            return 100.0 if last > 0 else 0.0
        n = len(years)
        growth = ((last - first) / first) * 100.0
        return growth / max(1, n - 1)

    def _classify_direction(self, growth_rate: float) -> TrendDirection:
        if growth_rate > 10:
            return TrendDirection.RISING
        if growth_rate < -10:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def _trend_description(
        self, topic: str, direction: TrendDirection, count: int, growth: float
    ) -> str:
        return (
            f"Topic '{topic}' is {direction.value} with {count} papers "
            f"and {growth:.1f}% annual growth rate."
        )

    def _build_summary(
        self,
        trends: list[ResearchTrend],
        emerging: list[TopicEntry],
        hot: list[TopicEntry],
        declining: list[TopicEntry],
    ) -> str:
        lines: list[str] = []
        lines.append(f"Analyzed {len(trends)} trends.")
        rising = sum(1 for t in trends if t.direction == TrendDirection.RISING)
        stable = sum(1 for t in trends if t.direction == TrendDirection.STABLE)
        declining_count = sum(
            1 for t in trends if t.direction == TrendDirection.DECLINING
        )
        lines.append(f"Rising: {rising}, Stable: {stable}, Declining: {declining_count}")
        if emerging:
            lines.append(f"Emerging topics: {', '.join(e.topic for e in emerging[:5])}")
        if hot:
            lines.append(f"Hot topics: {', '.join(h.topic for h in hot[:5])}")
        if declining:
            lines.append(
                f"Declining topics: {', '.join(d.topic for d in declining[:5])}"
            )
        return "\n".join(lines)
