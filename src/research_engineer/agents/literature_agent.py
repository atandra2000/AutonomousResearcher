"""Literature Agent for Phase 6 - Literature Intelligence.

Orchestrates paper search, comparison, review generation, relationship
detection, trend analysis, recommendations, and relevance scoring. Stores
all findings in the Memory Agent and updates the knowledge graph.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.llm import LLMProvider
from research_engineer.models.literature import (
    LiteratureResult,
    LiteratureReviewInput,
    LiteratureReviewOutput,
    PaperComparisonInput,
    PaperComparisonOutput,
    PaperRecommendationInput,
    PaperRecommendationOutput,
    PaperRelationshipInput,
    PaperRelationshipOutput,
    PaperSearchInput,
    PaperSearchOutput,
    PaperSummary,
    RelevanceScoringInput,
    RelevanceScoringOutput,
    ReviewDepth,
    SearchSource,
    TrendAnalysisInput,
    TrendAnalysisOutput,
)
from research_engineer.models.memory import (
    MemoryRelationship,
    RelationshipType,
)
from research_engineer.tools.base import ToolError
from research_engineer.tools.literature_review import LiteratureReviewTool
from research_engineer.tools.paper_comparison import PaperComparisonTool
from research_engineer.tools.paper_recommendation import PaperRecommendationTool
from research_engineer.tools.paper_relationship import (
    PaperRelationship,
    PaperRelationshipTool,
    PaperRelationType,
)
from research_engineer.tools.paper_search import PaperSearchTool
from research_engineer.tools.relevance_scoring import RelevanceScoringTool
from research_engineer.tools.trend_analysis import TrendAnalysisTool


class LiteratureConfig(BaseModel):
    """Configuration for Literature Agent."""

    max_papers: int = Field(20, description="Max papers to search")
    review_depth: ReviewDepth = Field(
        ReviewDepth.STANDARD, description="Review depth"
    )
    store_findings: bool = Field(True, description="Store findings in memory")
    update_graph: bool = Field(True, description="Auto-update knowledge graph")
    output_dir: str = Field("output/literature", description="Output directory")


class LiteratureAgent:
    """Agent for literature intelligence and research discovery.

    Orchestrates:
    1. Paper search (multi-source)
    2. Paper comparison
    3. Relationship detection
    4. Trend analysis
    5. Literature review generation
    6. Paper recommendations
    7. Repository relevance scoring
    8. Memory storage + graph update
    """

    def __init__(
        self,
        memory_agent: Any | None = None,
        search_tool: PaperSearchTool | None = None,
        comparison_tool: PaperComparisonTool | None = None,
        review_tool: LiteratureReviewTool | None = None,
        relationship_tool: PaperRelationshipTool | None = None,
        trend_tool: TrendAnalysisTool | None = None,
        recommendation_tool: PaperRecommendationTool | None = None,
        relevance_tool: RelevanceScoringTool | None = None,
        config: LiteratureConfig | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "LiteratureAgent"
        self.config = config or LiteratureConfig()
        self.memory = memory_agent
        self.search = search_tool or PaperSearchTool()
        self.comparison = comparison_tool or PaperComparisonTool()
        self.review = review_tool or LiteratureReviewTool()
        self.relationship = relationship_tool or PaperRelationshipTool()
        self.trend = trend_tool or TrendAnalysisTool()
        self.recommendation = recommendation_tool or PaperRecommendationTool()
        self.relevance = relevance_tool or RelevanceScoringTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def discover(
        self,
        topic: str,
        repo_path: str | None = None,
        output_dir: str | None = None,
        max_papers: int | None = None,
    ) -> LiteratureResult:
        """Full literature discovery workflow.

        Args:
            topic: Research topic to explore
            repo_path: Optional repository path for relevance scoring
            output_dir: Output directory for generated files
            max_papers: Override max papers to search

        Returns:
            LiteratureResult with all findings
        """
        start = time.time()
        out_dir = output_dir or self.config.output_dir
        max_p = max_papers or self.config.max_papers

        result = LiteratureResult(topic=topic, timestamp=datetime.now())

        # Step 1: Search papers
        search_output = await self.search_papers(topic, max_results=max_p)
        result.search_results = search_output

        papers = self._to_summaries(search_output)

        if not papers:
            result.processing_time_seconds = round(time.time() - start, 2)
            return result

        # Step 2: Detect relationships
        rel_output = await self.detect_relationships(papers)
        result.relationships = rel_output

        # Step 3: Analyze trends
        trend_output = await self.analyze_trends(papers)
        result.trends = trend_output

        # Step 4: Compare papers
        if len(papers) >= 2:
            comp_output = await self.compare_papers(papers)
            result.comparison = comp_output

        # Step 5: Generate literature review
        review_output = await self.generate_review(topic, papers)
        result.review = review_output

        # Step 6: Recommend papers
        rec_output = await self.recommend_papers(papers)
        result.recommendations = rec_output

        # Step 7: Score relevance (if repo provided)
        if repo_path:
            repo_summary = await self._load_repo_summary(repo_path)
            if repo_summary:
                rel_output = await self.score_relevance(
                    papers[0] if papers else None,
                    repo_summary,
                )
                if rel_output:
                    result.relevance = rel_output

        # Step 8: Store findings in memory + update graph
        if self.config.store_findings and self.memory:
            memory_ids = await self._store_findings(result)
            result.memory_ids = memory_ids

        # Step 9: Generate output files
        generated = await self._write_output_files(result, out_dir, topic)
        result.generated_files = generated
        result.output_dir = out_dir

        result.processing_time_seconds = round(time.time() - start, 2)
        return result

    async def search_papers(
        self,
        query: str,
        max_results: int = 20,
        sources: list[SearchSource] | None = None,
    ) -> PaperSearchOutput:
        """Search for papers across multiple sources."""
        search_input = PaperSearchInput(
            query=query,
            sources=sources or [SearchSource.LOCAL, SearchSource.ARXIV],
            max_results=max_results,
        )
        return await self.search.execute(search_input)

    async def compare_papers(
        self, papers: list[PaperSummary]
    ) -> PaperComparisonOutput:
        """Compare papers across multiple dimensions."""
        return await self.comparison.execute(
            PaperComparisonInput(papers=papers)
        )

    async def detect_relationships(
        self, papers: list[PaperSummary]
    ) -> PaperRelationshipOutput:
        """Detect relationships between papers."""
        return await self.relationship.execute(
            PaperRelationshipInput(papers=papers)
        )

    async def analyze_trends(
        self, papers: list[PaperSummary]
    ) -> TrendAnalysisOutput:
        """Analyze research trends."""
        return await self.trend.execute(
            TrendAnalysisInput(papers=papers)
        )

    async def generate_review(
        self,
        topic: str,
        papers: list[PaperSummary],
        depth: ReviewDepth | None = None,
    ) -> LiteratureReviewOutput:
        """Generate a structured literature review."""
        return await self.review.execute(
            LiteratureReviewInput(
                topic=topic,
                papers=papers,
                review_depth=depth or self.config.review_depth,
            )
        )

    async def recommend_papers(
        self,
        papers: list[PaperSummary],
        max_recommendations: int = 10,
    ) -> PaperRecommendationOutput:
        """Recommend papers worth implementing."""
        return await self.recommendation.execute(
            PaperRecommendationInput(
                papers=papers,
                max_recommendations=max_recommendations,
            )
        )

    async def score_relevance(
        self,
        paper: PaperSummary | None,
        repo_summary: object,
    ) -> RelevanceScoringOutput | None:
        """Score paper relevance to a repository."""
        if not paper:
            return None
        try:
            return await self.relevance.execute(
                RelevanceScoringInput(paper=paper, repo_summary=repo_summary)
            )
        except ToolError:
            return None

    # --- Memory & Graph Integration ---

    async def _store_findings(self, result: LiteratureResult) -> list[str]:
        """Store all findings in Memory Agent and update graph."""
        memory_ids: list[str] = []

        if result.search_results:
            mem_id = await self._store_search_result(result.topic, result.search_results)
            if mem_id:
                memory_ids.append(mem_id)

        if result.relationships:
            graph_ids = await self._store_relationships(result.relationships)
            memory_ids.extend(graph_ids)

        if result.trends:
            mem_id = await self._store_trends(result.topic, result.trends)
            if mem_id:
                memory_ids.append(mem_id)

        memory_ids.extend(await self._store_secondary_findings(result))

        return memory_ids

    async def _store_secondary_findings(self, result: LiteratureResult) -> list[str]:
        """Store review, recommendations, and relevance findings."""
        memory_ids: list[str] = []

        if result.review:
            mem_id = await self._store_review(result.topic, result.review)
            if mem_id:
                memory_ids.append(mem_id)

        if result.recommendations:
            mem_id = await self._store_recommendations(
                result.topic, result.recommendations
            )
            if mem_id:
                memory_ids.append(mem_id)

        if result.relevance:
            mem_id = await self._store_relevance(result.relevance)
            if mem_id:
                memory_ids.append(mem_id)

        return memory_ids

    async def _store_search_result(
        self, topic: str, output: PaperSearchOutput
    ) -> str | None:
        if not self.memory:
            return None
        try:
            description = (
                f"Found {output.total_found} papers for '{topic}' from "
                f"sources: {', '.join(output.sources_searched)}"
            )
            return await self.memory.store_insight(
                insight_type="pattern",
                domain=topic,
                description=description,
                evidence=[p.paper_id for p in output.papers[:20]],
                applicability=["literature_search"],
            )
        except Exception:
            return None

    async def _store_relationships(
        self, output: PaperRelationshipOutput
    ) -> list[str]:
        """Store paper relationships in memory storage + graph."""
        ids: list[str] = []
        if not self.memory:
            return ids

        for rel in output.relationships:
            try:
                mem_rel = self._to_memory_relationship(rel)
                await self.memory.storage.store_relationship(mem_rel)
                if self.config.update_graph and self.memory.graph is not None:
                    self.memory.graph.add_node(rel.source_paper_id, "paper")
                    self.memory.graph.add_node(rel.target_paper_id, "paper")
                    self.memory.graph.add_relationship(mem_rel)
                ids.append(mem_rel.relationship_id)
            except Exception:
                pass
        return ids

    def _to_memory_relationship(
        self, rel: PaperRelationship
    ) -> MemoryRelationship:
        """Convert PaperRelationship to MemoryRelationship."""
        type_map = {
            PaperRelationType.CITES: RelationshipType.CITES,
            PaperRelationType.EXTENDS: RelationshipType.EXTENDS,
            PaperRelationType.SIMILAR_TO: RelationshipType.SIMILAR_TO,
            PaperRelationType.CONTRADICTS: RelationshipType.CONTRADICTS,
            PaperRelationType.BUILDS_ON: RelationshipType.DERIVED_FROM,
            PaperRelationType.REPRODUCES: RelationshipType.VALIDATES,
            PaperRelationType.IMPROVES_UPON: RelationshipType.EXTENDS,
        }
        return MemoryRelationship(
            source_memory_id=rel.source_paper_id,
            target_memory_id=rel.target_paper_id,
            relationship_type=type_map.get(
                rel.relationship_type, RelationshipType.SIMILAR_TO
            ),
            confidence=rel.confidence,
        )

    async def _store_trends(
        self, topic: str, output: TrendAnalysisOutput
    ) -> str | None:
        if not self.memory:
            return None
        try:
            description = (
                f"Trend analysis for '{topic}': {output.trend_summary}"
            )
            return await self.memory.store_insight(
                insight_type="empirical_finding",
                domain=topic,
                description=description,
                evidence=[t.topic for t in output.trends],
                applicability=["trend_analysis"],
            )
        except Exception:
            return None

    async def _store_review(
        self, topic: str, output: LiteratureReviewOutput
    ) -> str | None:
        if not self.memory:
            return None
        try:
            return await self.memory.store_insight(
                insight_type="best_practice",
                domain=topic,
                description=f"Literature review for '{topic}': {output.review.executive_summary[:200]}",
                evidence=[s.title for s in output.review.sections],
                applicability=["literature_review"],
            )
        except Exception:
            return None

    async def _store_recommendations(
        self, topic: str, output: PaperRecommendationOutput
    ) -> str | None:
        if not self.memory:
            return None
        try:
            return await self.memory.store_insight(
                insight_type="optimization",
                domain=topic,
                description=f"Paper recommendations: {output.ranking_rationale}",
                evidence=[r.paper_id for r in output.recommendations],
                applicability=["paper_recommendation"],
            )
        except Exception:
            return None

    async def _store_relevance(
        self, output: RelevanceScoringOutput
    ) -> str | None:
        if not self.memory:
            return None
        try:
            return await self.memory.store_insight(
                insight_type="implementation_trick",
                domain=f"relevance:{output.score.paper_id}",
                description=(
                    f"Relevance score {output.score.overall_score:.2f} "
                    f"({output.score.relevance_level.value}) for "
                    f"{output.score.paper_id} to {output.score.repo_path}"
                ),
                evidence=[d.dimension for d in output.dimension_scores],
                applicability=["relevance_scoring"],
            )
        except Exception:
            return None

    # --- Helpers ---

    def _to_summaries(self, output: PaperSearchOutput) -> list[PaperSummary]:
        """Convert SearchResult list to PaperSummary list."""
        return [
            PaperSummary(
                paper_id=r.paper_id,
                title=r.title,
                abstract=r.abstract,
                authors=r.authors,
                year=r.year,
                citation_count=r.citation_count,
                url=r.url,
                doi=r.doi,
                source=r.source,
            )
            for r in output.papers
        ]

    async def _load_repo_summary(self, repo_path: str) -> object | None:
        """Load repository summary from Phase 2 output or analyze on the fly."""
        try:
            from research_engineer.agents.repository_agent import RepositoryAgent

            agent = RepositoryAgent(
                enable_caching=False,
                rate_limit_enabled=False,
                llm_enabled=False,
            )
            result = await agent.analyze(repo_path, output_dir="output")
            from research_engineer.models.repo import (
                ConfigurationAnalysis,
                FileImportance,
                KnowledgeGraph,
                RepositorySummary,
            )

            important_files = [
                FileImportance(**f) if isinstance(f, dict) else f
                for f in result.get("important_files", [])
            ]
            kg_data = result.get("knowledge_graph", {})
            kg = KnowledgeGraph(**kg_data) if isinstance(kg_data, dict) else kg_data
            config_data = result.get("configuration_analysis", {})
            config = (
                ConfigurationAnalysis(**config_data)
                if isinstance(config_data, dict)
                else config_data
            )
            return RepositorySummary(
                repository_name=result.get("repository_name", Path(repo_path).name),
                project_type=result.get("project_type", "Unknown"),
                architecture_summary=result.get("architecture_summary", ""),
                important_files=important_files,
                training_pipeline=str(result.get("training_pipeline", "")),
                knowledge_graph=kg,
                implementation_targets=[],
                configuration_analysis=config,
                analysis_timestamp=datetime.now(),
            )
        except Exception:
            return None

    async def _write_output_files(
        self, result: LiteratureResult, output_dir: str, topic: str
    ) -> list[str]:
        """Write output files for the literature result."""
        try:
            slug = topic.lower().replace(" ", "_")[:50]
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            out_path = Path(output_dir) / f"{slug}_{timestamp}"
            out_path.mkdir(parents=True, exist_ok=True)

            generated: list[str] = []

            if result.search_results:
                f = out_path / "search_results.json"
                f.write_text(
                    result.search_results.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                generated.append(str(f))

            if result.comparison:
                f = out_path / "paper_comparison.md"
                f.write_text(
                    self._format_comparison_md(result.comparison),
                    encoding="utf-8",
                )
                generated.append(str(f))

            if result.review:
                f = out_path / "literature_review.md"
                f.write_text(result.review.markdown, encoding="utf-8")
                generated.append(str(f))

            if result.relationships:
                f = out_path / "relationships.json"
                f.write_text(
                    result.relationships.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                generated.append(str(f))

            if result.trends:
                f = out_path / "trend_analysis.md"
                f.write_text(
                    self._format_trends_md(result.trends),
                    encoding="utf-8",
                )
                generated.append(str(f))

            if result.recommendations:
                f = out_path / "recommendations.md"
                f.write_text(
                    self._format_recommendations_md(result.recommendations),
                    encoding="utf-8",
                )
                generated.append(str(f))

            if result.relevance:
                f = out_path / "relevance_scores.json"
                f.write_text(
                    result.relevance.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                generated.append(str(f))

            f = out_path / "literature_result.json"
            f.write_text(
                result.model_dump_json(indent=2, default=str),
                encoding="utf-8",
            )
            generated.append(str(f))

            return generated
        except Exception:
            return []

    def _format_comparison_md(self, output: PaperComparisonOutput) -> str:
        lines = ["# Paper Comparison", ""]
        lines.append("## Comparison Matrix")
        lines.append("")
        matrix = output.comparison
        header = "| Paper | " + " | ".join(d.name for d in matrix.dimensions) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(matrix.dimensions) + 1)) + "|")
        for pid in matrix.papers:
            row = f"| {pid} |"
            for dim in matrix.dimensions:
                row += f" {matrix.matrix.get(pid, {}).get(dim.name, 'N/A')} |"
            lines.append(row)
        lines.append("")
        if output.similarities:
            lines.append("## Similarities")
            lines.append("")
            for sim in output.similarities:
                lines.append(
                    f"- {sim.paper_a} <-> {sim.paper_b}: {sim.similarity_score:.2f}"
                )
            lines.append("")
        if output.conflicting_findings:
            lines.append("## Conflicts")
            lines.append("")
            for conf in output.conflicting_findings:
                lines.append(f"- {conf.topic}: {', '.join(conf.papers)}")
            lines.append("")
        if output.ranking:
            lines.append("## Ranking")
            lines.append("")
            for rank in output.ranking:
                lines.append(
                    f"{rank.rank}. {rank.paper_id} (score: {rank.score:.2f})"
                )
        return "\n".join(lines)

    def _format_trends_md(self, output: TrendAnalysisOutput) -> str:
        lines = ["# Trend Analysis", "", output.trend_summary, ""]
        if output.trends:
            lines.append("## Trends")
            lines.append("")
            for t in output.trends:
                lines.append(f"### {t.topic} ({t.direction.value})")
                lines.append(f"- Growth rate: {t.growth_rate:.1f}%")
                lines.append(f"- Papers: {sum(t.paper_count_by_year.values())}")
                lines.append(f"- {t.description}")
                lines.append("")
        if output.emerging_topics:
            lines.append("## Emerging Topics")
            lines.append("")
            for e in output.emerging_topics:
                lines.append(f"- {e.topic} ({e.growth_rate:.1f}% growth)")
            lines.append("")
        if output.hot_topics:
            lines.append("## Hot Topics")
            lines.append("")
            for h in output.hot_topics:
                lines.append(f"- {h.topic} ({h.paper_count} papers)")
            lines.append("")
        return "\n".join(lines)

    def _format_recommendations_md(
        self, output: PaperRecommendationOutput
    ) -> str:
        lines = ["# Paper Recommendations", "", output.ranking_rationale, ""]
        for rec in output.recommendations:
            lines.append(f"## {rec.rank}. {rec.title}")
            lines.append(f"- **Overall Score**: {rec.overall_score:.2f}")
            lines.append(f"- **Impact**: {rec.impact_score:.2f}")
            lines.append(f"- **Novelty**: {rec.novelty_score:.2f}")
            lines.append(f"- **Implementability**: {rec.implementability_score:.2f}")
            lines.append(f"- **Rationale**: {rec.rationale}")
            if rec.key_strengths:
                lines.append("- **Strengths**:")
                for s in rec.key_strengths:
                    lines.append(f"  - {s}")
            if rec.potential_challenges:
                lines.append("- **Challenges**:")
                for c in rec.potential_challenges:
                    lines.append(f"  - {c}")
            lines.append("")
        return "\n".join(lines)
