"""Literature Review Tool for Phase 6 - Literature Intelligence.

Generates structured literature reviews from a set of papers, producing
executive summaries, topic sections, timelines, and markdown output.
"""

from datetime import datetime

from research_engineer.models.literature import (
    LiteratureReview,
    LiteratureReviewInput,
    LiteratureReviewOutput,
    PaperSummary,
    ReviewDepth,
    ReviewSection,
    TimelineEntry,
)
from research_engineer.tools.base import Tool, ToolError

SECTION_TOPICS = [
    ("Architecture & Methods", ["architecture", "model", "network", "attention", "transformer"]),
    ("Training & Optimization", ["training", "optimizer", "loss", "learning rate", "gradient"]),
    ("Evaluation & Benchmarks", ["evaluate", "benchmark", "metric", "accuracy", "perplexity"]),
    ("Datasets & Data", ["dataset", "data", "corpus", "benchmark"]),
    ("Results & Performance", ["result", "achieve", "outperform", "improve", "speedup"]),
]


class LiteratureReviewTool(Tool[LiteratureReviewInput, LiteratureReviewOutput]):
    """Generate structured literature reviews from papers."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: LiteratureReviewInput) -> bool:
        return bool(input.topic) and len(input.papers) >= 1

    async def execute(self, input: LiteratureReviewInput) -> LiteratureReviewOutput:
        try:
            exec_summary = self._generate_executive_summary(input.topic, input.papers)
            sections = self._build_sections(input.papers, input.review_depth)
            methodology = self._build_methodology_landscape(input.papers)
            key_findings = self._extract_key_findings(input.papers)
            gaps = self._identify_gaps(input.papers, input.topic)
            recommendations = self._generate_recommendations(input.papers, input.topic)
            timeline = self._build_timeline(input.papers)
            citation_net = self._build_citation_network(input.papers)

            review = LiteratureReview(
                topic=input.topic,
                timestamp=datetime.now(),
                executive_summary=exec_summary,
                papers_analyzed=len(input.papers),
                sections=sections,
                methodology_landscape=methodology,
                key_findings=key_findings,
                research_gaps=gaps,
                recommendations=recommendations,
                timeline=timeline,
                citation_network=citation_net,
            )

            markdown = self._generate_markdown(review)

            return LiteratureReviewOutput(
                review=review,
                markdown=markdown,
                generated_files=[],
            )
        except Exception as e:
            raise ToolError(f"Literature review generation failed: {e}", input, e)

    def _generate_executive_summary(self, topic: str, papers: list[PaperSummary]) -> str:
        if not papers:
            return f"No papers found for topic '{topic}'."

        years = [p.year for p in papers if p.year]
        year_range = ""
        if years:
            year_range = f" from {min(years)} to {max(years)}"

        top_paper = max(papers, key=lambda p: p.citation_count) if papers else None
        top_str = ""
        if top_paper:
            top_str = f" The most cited paper is '{top_paper.title}' with {top_paper.citation_count} citations."

        return (
            f"This literature review analyzes {len(papers)} papers on '{topic}'"
            f"{year_range}.{top_str} The review covers methodology landscape, "
            f"key findings, research gaps, and recommendations for future work."
        )

    def _build_sections(
        self, papers: list[PaperSummary], depth: ReviewDepth
    ) -> list[ReviewSection]:
        max_sections = {
            ReviewDepth.BRIEF: 3,
            ReviewDepth.STANDARD: 5,
            ReviewDepth.COMPREHENSIVE: 10,
        }[depth]

        sections: list[ReviewSection] = []
        for title, keywords in SECTION_TOPICS[:max_sections]:
            relevant = [
                p for p in papers
                if any(kw in f"{p.title} {p.abstract}".lower() for kw in keywords)
            ]
            if not relevant:
                continue

            content = self._build_section_content(title, relevant)
            key_points = self._extract_section_points(relevant, keywords)
            sections.append(
                ReviewSection(
                    title=title,
                    content=content,
                    papers_referenced=[p.paper_id for p in relevant],
                    key_points=key_points,
                )
            )
        return sections

    def _build_section_content(
        self, title: str, papers: list[PaperSummary]
    ) -> str:
        lines = [f"## {title}", ""]
        for p in papers[:10]:
            author_str = ", ".join(p.authors[:3]) if p.authors else "Unknown"
            year_str = f" ({p.year})" if p.year else ""
            lines.append(f"- **{p.title}** [{author_str}{year_str}] - {p.abstract[:200]}...")
        return "\n".join(lines)

    def _extract_section_points(
        self, papers: list[PaperSummary], keywords: list[str]
    ) -> list[str]:
        points: list[str] = []
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            matched = [kw for kw in keywords if kw in text]
            if matched:
                points.append(
                    f"{p.paper_id}: Uses {', '.join(matched[:3])}"
                )
        return points[:10]

    def _build_methodology_landscape(self, papers: list[PaperSummary]) -> str:
        methods: dict[str, int] = {}
        method_keywords = [
            "supervised", "unsupervised", "self-supervised", "reinforcement",
            "transfer", "fine-tuning", "pretraining", "distillation",
            "contrastive", "zero-shot", "few-shot",
        ]
        for p in papers:
            text = f"{p.title} {p.abstract}".lower()
            for kw in method_keywords:
                if kw in text:
                    methods[kw] = methods.get(kw, 0) + 1

        if not methods:
            return "No specific methodologies detected from the analyzed papers."

        lines = ["Methodology distribution:"]
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {method}: {count} papers")
        return "\n".join(lines)

    def _extract_key_findings(self, papers: list[PaperSummary]) -> list[str]:
        findings: list[str] = []
        result_keywords = [
            "achieve", "outperform", "improve", "state-of-the-art",
            "demonstrate", "show", "result",
        ]
        for p in papers:
            sentences = p.abstract.split(". ") if p.abstract else []
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in result_keywords):
                    findings.append(f"{p.paper_id}: {sentence.strip()[:200]}")
                    break
        return findings[:15]

    def _identify_gaps(self, papers: list[PaperSummary], topic: str) -> list[str]:
        gaps: list[str] = []
        all_text = " ".join(f"{p.title} {p.abstract}" for p in papers).lower()

        gap_indicators = {
            "limited evaluation": "limited" in all_text and "evaluat" in all_text,
            "small-scale experiments": "small" in all_text and "scale" in all_text,
            "lack of theoretical analysis": "theoretical" not in all_text,
            "limited reproducibility": "reproduc" not in all_text,
            "no real-world deployment": "deploy" not in all_text and "real-world" not in all_text,
        }
        for gap, condition in gap_indicators.items():
            if condition:
                gaps.append(f"Research gap: {gap} for topic '{topic}'")
        if not gaps:
            gaps.append(f"No obvious gaps detected; topic '{topic}' is well-covered.")
        return gaps

    def _generate_recommendations(
        self, papers: list[PaperSummary], topic: str
    ) -> list[str]:
        recs: list[str] = []
        recs.append(
            f"Focus on papers with highest citation count for '{topic}' "
            f"as they represent established contributions."
        )
        recent = [p for p in papers if p.year and p.year >= datetime.now().year - 2]
        if recent:
            recs.append(
                f"Recent papers ({len(recent)} in last 2 years) may contain "
                f"the latest advances worth implementing."
            )
        if len(papers) > 10:
            recs.append(
                "With a large body of work, consider a systematic review "
                "to identify consensus and conflicts."
            )
        gaps_found = self._identify_gaps(papers, topic)
        if any("limited" in g for g in gaps_found):
            recs.append("Address identified research gaps in future implementations.")
        return recs

    def _build_timeline(self, papers: list[PaperSummary]) -> list[TimelineEntry]:
        by_year: dict[int, list[PaperSummary]] = {}
        for p in papers:
            if p.year:
                by_year.setdefault(p.year, []).append(p)

        entries: list[TimelineEntry] = []
        for year in sorted(by_year.keys()):
            year_papers = by_year[year]
            milestone = (
                f"{len(year_papers)} paper(s) published. "
                f"Key: {year_papers[0].title[:80]}"
            )
            entries.append(
                TimelineEntry(
                    year=year,
                    papers=[p.paper_id for p in year_papers],
                    milestone=milestone,
                )
            )
        return entries

    def _build_citation_network(
        self, papers: list[PaperSummary]
    ) -> dict[str, list[str]]:
        """Build a simple citation network from shared references."""
        import re

        network: dict[str, list[str]] = {}
        arxiv_pattern = re.compile(r"\b(\d{4}\.\d{4,5})\b")

        for paper in papers:
            cited_by: list[str] = []
            text = f"{paper.title} {paper.abstract}"
            refs = arxiv_pattern.findall(text)
            ref_set = set(refs) - {paper.paper_id}
            for other in papers:
                if other.paper_id in ref_set:
                    cited_by.append(other.paper_id)
            if cited_by:
                network[paper.paper_id] = cited_by

        for paper in papers:
            if paper.paper_id not in network:
                network[paper.paper_id] = []
        return network

    def _generate_markdown(self, review: LiteratureReview) -> str:
        lines: list[str] = [
            f"# Literature Review: {review.topic}",
            "",
            f"**Generated**: {review.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"**Papers Analyzed**: {review.papers_analyzed}",
            "",
            "## Executive Summary",
            "",
            review.executive_summary,
            "",
        ]
        lines.extend(self._md_sections(review))
        lines.extend(self._md_methodology(review))
        lines.extend(self._md_findings(review))
        lines.extend(self._md_gaps(review))
        lines.extend(self._md_timeline(review))
        lines.extend(self._md_recommendations(review))
        return "\n".join(lines)

    def _md_sections(self, review: LiteratureReview) -> list[str]:
        lines: list[str] = []
        if review.sections:
            lines.append("## Detailed Analysis")
            lines.append("")
            for section in review.sections:
                lines.append(f"### {section.title}")
                lines.append("")
                lines.append(section.content)
                lines.append("")
                if section.key_points:
                    lines.append("**Key Points:**")
                    for kp in section.key_points:
                        lines.append(f"- {kp}")
                    lines.append("")
        return lines

    def _md_methodology(self, review: LiteratureReview) -> list[str]:
        return [
            "## Methodology Landscape",
            "",
            review.methodology_landscape,
            "",
        ]

    def _md_findings(self, review: LiteratureReview) -> list[str]:
        lines: list[str] = []
        if review.key_findings:
            lines.append("## Key Findings")
            lines.append("")
            for finding in review.key_findings:
                lines.append(f"- {finding}")
            lines.append("")
        return lines

    def _md_gaps(self, review: LiteratureReview) -> list[str]:
        lines: list[str] = []
        if review.research_gaps:
            lines.append("## Research Gaps")
            lines.append("")
            for gap in review.research_gaps:
                lines.append(f"- {gap}")
            lines.append("")
        return lines

    def _md_timeline(self, review: LiteratureReview) -> list[str]:
        lines: list[str] = []
        if review.timeline:
            lines.append("## Timeline")
            lines.append("")
            lines.append("| Year | Papers | Milestone |")
            lines.append("|------|--------|-----------|")
            for entry in review.timeline:
                lines.append(
                    f"| {entry.year} | {len(entry.papers)} | {entry.milestone} |"
                )
            lines.append("")
        return lines

    def _md_recommendations(self, review: LiteratureReview) -> list[str]:
        lines: list[str] = []
        if review.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in review.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        return lines
