"""Tests for Phase 6 literature tools."""

import pytest

from research_engineer.models.literature import (
    PaperComparisonInput,
    PaperRecommendationInput,
    PaperRelationshipInput,
    PaperSummary,
    RelevanceScoringInput,
    ReviewDepth,
    SearchSource,
    TrendAnalysisInput,
)
from research_engineer.models.repo import (
    ConfigurationAnalysis,
    FileImportance,
    KnowledgeGraph,
    RepositorySummary,
)
from research_engineer.tools.paper_comparison import PaperComparisonTool
from research_engineer.tools.paper_recommendation import PaperRecommendationTool
from research_engineer.tools.paper_relationship import PaperRelationshipTool
from research_engineer.tools.paper_search import PaperSearchTool
from research_engineer.tools.relevance_scoring import RelevanceScoringTool
from research_engineer.tools.trend_analysis import TrendAnalysisTool
from research_engineer.tools.literature_review import LiteratureReviewTool


def _make_paper(
    paper_id: str = "2503.12345",
    title: str = "FlashAttention: Fast and Memory-Efficient Exact Attention",
    abstract: str = "We propose a novel attention mechanism that is IO-aware and achieves 2-4x memory reduction with 15-20% speedup.",
    year: int = 2024,
    citation_count: int = 100,
) -> PaperSummary:
    return PaperSummary(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["Author A"],
        year=year,
        citation_count=citation_count,
    )


def _make_repo_summary() -> RepositorySummary:
    return RepositorySummary(
        repository_name="test-repo",
        project_type="LLMTrainingFramework",
        architecture_summary="Transformer training framework with attention modules",
        important_files=[
            FileImportance(
                file_path="models/attention.py",
                importance="Critical",
                reason="Attention implementation",
                complexity="High",
            )
        ],
        training_pipeline="Standard training loop with AdamW optimizer",
        knowledge_graph=KnowledgeGraph(
            nodes=[], edges=[], communities=[], central_nodes=[], relationships_by_type={}
        ),
        implementation_targets=[],
        configuration_analysis=ConfigurationAnalysis(
            config_files=["configs/model.yaml"],
            config_framework="yaml",
        ),
    )


# --- PaperSearchTool ---


class TestPaperSearchTool:
    @pytest.mark.asyncio
    async def test_validate_valid(self):
        from research_engineer.models.literature import PaperSearchInput
        tool = PaperSearchTool()
        inp = PaperSearchInput(query="attention")
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_validate_empty_query(self):
        from research_engineer.models.literature import PaperSearchInput
        tool = PaperSearchTool()
        inp = PaperSearchInput(query="")
        assert await tool.validate(inp) is False

    @pytest.mark.asyncio
    async def test_execute_no_storage(self):
        from research_engineer.models.literature import PaperSearchInput
        tool = PaperSearchTool(storage_tool=None)
        inp = PaperSearchInput(
            query="attention",
            sources=[SearchSource.LOCAL],
        )
        out = await tool.execute(inp)
        assert out.total_found == 0

    @pytest.mark.asyncio
    async def test_deduplicate(self):
        from research_engineer.models.literature import SearchResult
        tool = PaperSearchTool()
        results = [
            SearchResult(paper_id="1", title="A", source=SearchSource.ARXIV, citation_count=5),
            SearchResult(paper_id="1", title="A", source=SearchSource.SEMANTIC_SCHOLAR, citation_count=10),
            SearchResult(paper_id="2", title="B", source=SearchSource.ARXIV),
        ]
        deduped = tool._deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0].citation_count == 10

    @pytest.mark.asyncio
    async def test_rank_relevance(self):
        from research_engineer.models.literature import SearchResult
        tool = PaperSearchTool()
        results = [
            SearchResult(paper_id="1", title="attention mechanism", abstract="", source=SearchSource.ARXIV),
            SearchResult(paper_id="2", title="something else", abstract="", source=SearchSource.ARXIV),
        ]
        ranked = tool._rank_results(results, "attention", "relevance")
        assert ranked[0].paper_id == "1"

    @pytest.mark.asyncio
    async def test_rank_by_citations(self):
        from research_engineer.models.literature import SearchResult
        tool = PaperSearchTool()
        results = [
            SearchResult(paper_id="1", title="A", source=SearchSource.ARXIV, citation_count=5),
            SearchResult(paper_id="2", title="B", source=SearchSource.ARXIV, citation_count=100),
        ]
        ranked = tool._rank_results(results, "test", "citationCount")
        assert ranked[0].paper_id == "2"


# --- PaperComparisonTool ---


class TestPaperComparisonTool:
    @pytest.mark.asyncio
    async def test_validate_two_papers(self):
        tool = PaperComparisonTool()
        papers = [
            PaperSummary(paper_id="1", title="A"),
            PaperSummary(paper_id="2", title="B"),
        ]
        assert await tool.validate(PaperComparisonInput(papers=papers)) is True

    @pytest.mark.asyncio
    async def test_validate_single_paper(self):
        tool = PaperComparisonTool()
        papers = [PaperSummary(paper_id="1", title="A")]
        with pytest.raises(Exception):
            PaperComparisonInput(papers=papers)

    @pytest.mark.asyncio
    async def test_execute_comparison(self):
        tool = PaperComparisonTool()
        papers = [
            _make_paper("1", "FlashAttention", "IO-aware attention mechanism"),
            _make_paper("2", "RingAttention", "Distributed ring attention for long sequences"),
        ]
        out = await tool.execute(PaperComparisonInput(papers=papers))
        assert len(out.comparison.papers) == 2
        assert len(out.comparison.dimensions) > 0
        assert len(out.ranking) == 2

    @pytest.mark.asyncio
    async def test_similarity_detection(self):
        tool = PaperComparisonTool()
        papers = [
            _make_paper("1", "Attention is all you need", "We propose transformer attention"),
            _make_paper("2", "Attention mechanism for NLP", "We propose transformer attention mechanism"),
        ]
        out = await tool.execute(PaperComparisonInput(papers=papers))
        assert len(out.similarities) > 0

    @pytest.mark.asyncio
    async def test_consensus_findings(self):
        tool = PaperComparisonTool()
        papers = [
            _make_paper("1", "Paper A", "We achieve state-of-the-art results"),
            _make_paper("2", "Paper B", "We achieve improved results"),
        ]
        out = await tool.execute(PaperComparisonInput(papers=papers))
        assert isinstance(out.consensus_findings, list)


# --- PaperRelationshipTool ---


class TestPaperRelationshipTool:
    @pytest.mark.asyncio
    async def test_validate(self):
        tool = PaperRelationshipTool()
        inp = PaperRelationshipInput(papers=[PaperSummary(paper_id="1", title="A")])
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_similarity_detection(self):
        tool = PaperRelationshipTool()
        papers = [
            PaperSummary(paper_id="1", title="attention transformer", abstract="attention transformer model"),
            PaperSummary(paper_id="2", title="attention transformer v2", abstract="attention transformer model v2"),
        ]
        out = await tool.execute(PaperRelationshipInput(papers=papers))
        types = [r.relationship_type.value for r in out.relationships]
        assert "similar_to" in types

    @pytest.mark.asyncio
    async def test_citation_detection(self):
        tool = PaperRelationshipTool()
        papers = [
            PaperSummary(
                paper_id="2401.00001",
                title="Paper A",
                abstract="We build on 2503.12345 for our approach",
            ),
            PaperSummary(
                paper_id="2503.12345",
                title="Paper B",
                abstract="Original work",
            ),
        ]
        out = await tool.execute(PaperRelationshipInput(papers=papers))
        cites = [r for r in out.relationships if r.relationship_type.value == "cites"]
        assert len(cites) > 0

    @pytest.mark.asyncio
    async def test_single_paper_no_relationships(self):
        tool = PaperRelationshipTool()
        papers = [PaperSummary(paper_id="1", title="A", abstract="test")]
        out = await tool.execute(PaperRelationshipInput(papers=papers))
        assert len(out.relationships) == 0

    @pytest.mark.asyncio
    async def test_min_confidence_filtering(self):
        tool = PaperRelationshipTool()
        papers = [
            PaperSummary(paper_id="1", title="alpha beta", abstract="alpha beta gamma"),
            PaperSummary(paper_id="2", title="delta epsilon", abstract="delta epsilon zeta"),
        ]
        out = await tool.execute(
            PaperRelationshipInput(papers=papers, min_confidence=0.99)
        )
        assert len(out.relationships) == 0


# --- TrendAnalysisTool ---


class TestTrendAnalysisTool:
    @pytest.mark.asyncio
    async def test_validate(self):
        tool = TrendAnalysisTool()
        inp = TrendAnalysisInput(papers=[PaperSummary(paper_id="1", title="A")])
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_rising_trend(self):
        tool = TrendAnalysisTool()
        from research_engineer.tools.trend_analysis import TrendAnalysisTool as TA
        # Patch current_year to 2024 so the window is 2020-2024
        import research_engineer.tools.trend_analysis as ta_mod
        from unittest.mock import patch
        from datetime import datetime

        papers = []
        # 2020: 2 papers, 2021: 4, 2022: 8, 2023: 12, 2024: 16
        for year, count in [(2020, 2), (2021, 4), (2022, 8), (2023, 12), (2024, 16)]:
            for i in range(count):
                papers.append(PaperSummary(
                    paper_id=f"p{year}_{i}",
                    title=f"attention paper {i}",
                    abstract="attention mechanism",
                    year=year,
                    citation_count=10,
                ))

        with patch.object(ta_mod, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 6, 15)
            mock_dt.side_effect = datetime
            out = await tool.execute(TrendAnalysisInput(papers=papers, min_papers_per_trend=2))

        assert len(out.trends) > 0
        assert any(t.direction.value == "rising" for t in out.trends)

    @pytest.mark.asyncio
    async def test_no_papers_in_window(self):
        tool = TrendAnalysisTool()
        papers = [PaperSummary(paper_id="1", title="test", abstract="test", year=2000)]
        out = await tool.execute(TrendAnalysisInput(papers=papers))
        assert len(out.trends) == 0

    @pytest.mark.asyncio
    async def test_empty_papers(self):
        tool = TrendAnalysisTool()
        out = await tool.execute(TrendAnalysisInput(papers=[]))
        assert len(out.trends) == 0

    @pytest.mark.asyncio
    async def test_growth_rate_computation(self):
        tool = TrendAnalysisTool()
        counts = {"2020": 2, "2021": 4, "2022": 8}
        rate = tool._compute_growth_rate(counts)
        assert rate > 0


# --- LiteratureReviewTool ---


class TestLiteratureReviewTool:
    @pytest.mark.asyncio
    async def test_validate(self):
        from research_engineer.models.literature import LiteratureReviewInput
        tool = LiteratureReviewTool()
        inp = LiteratureReviewInput(
            topic="attention",
            papers=[PaperSummary(paper_id="1", title="A")],
        )
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_brief_review(self):
        from research_engineer.models.literature import LiteratureReviewInput
        tool = LiteratureReviewTool()
        papers = [
            _make_paper("1", "Paper A", "We propose attention mechanism", 2024),
            _make_paper("2", "Paper B", "We propose transformer architecture", 2023),
        ]
        out = await tool.execute(
            LiteratureReviewInput(topic="attention", papers=papers, review_depth=ReviewDepth.BRIEF)
        )
        assert out.review.papers_analyzed == 2
        assert len(out.markdown) > 0
        assert "attention" in out.markdown.lower()

    @pytest.mark.asyncio
    async def test_standard_review(self):
        from research_engineer.models.literature import LiteratureReviewInput
        tool = LiteratureReviewTool()
        papers = [_make_paper(str(i), f"Paper {i}", "attention mechanism", 2024) for i in range(5)]
        out = await tool.execute(
            LiteratureReviewInput(topic="attention", papers=papers, review_depth=ReviewDepth.STANDARD)
        )
        assert out.review.papers_analyzed == 5

    @pytest.mark.asyncio
    async def test_timeline(self):
        from research_engineer.models.literature import LiteratureReviewInput
        tool = LiteratureReviewTool()
        papers = [
            _make_paper("1", "A", "test", 2022),
            _make_paper("2", "B", "test", 2023),
            _make_paper("3", "C", "test", 2024),
        ]
        out = await tool.execute(
            LiteratureReviewInput(topic="test", papers=papers, review_depth=ReviewDepth.BRIEF)
        )
        assert len(out.review.timeline) == 3

    @pytest.mark.asyncio
    async def test_markdown_structure(self):
        from research_engineer.models.literature import LiteratureReviewInput
        tool = LiteratureReviewTool()
        papers = [_make_paper("1", "A", "attention")]
        out = await tool.execute(
            LiteratureReviewInput(topic="attention", papers=papers, review_depth=ReviewDepth.BRIEF)
        )
        assert "# Literature Review" in out.markdown
        assert "## Executive Summary" in out.markdown


# --- PaperRecommendationTool ---


class TestPaperRecommendationTool:
    @pytest.mark.asyncio
    async def test_validate(self):
        tool = PaperRecommendationTool()
        inp = PaperRecommendationInput(papers=[PaperSummary(paper_id="1", title="A")])
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_basic_ranking(self):
        tool = PaperRecommendationTool()
        papers = [
            _make_paper("1", "Paper A", "novel attention", citation_count=100),
            _make_paper("2", "Paper B", "standard approach", citation_count=10),
        ]
        out = await tool.execute(PaperRecommendationInput(papers=papers))
        assert len(out.recommendations) == 2
        assert out.recommendations[0].rank == 1
        assert out.recommendations[0].overall_score >= out.recommendations[1].overall_score

    @pytest.mark.asyncio
    async def test_max_recommendations(self):
        tool = PaperRecommendationTool()
        papers = [_make_paper(str(i), f"Paper {i}", "attention") for i in range(10)]
        out = await tool.execute(PaperRecommendationInput(papers=papers, max_recommendations=3))
        assert len(out.recommendations) == 3

    @pytest.mark.asyncio
    async def test_criteria_filtering(self):
        from research_engineer.models.literature import RecommendationCriteria
        tool = PaperRecommendationTool()
        papers = [
            _make_paper("1", "Paper A", "attention", citation_count=5),
            _make_paper("2", "Paper B", "attention", citation_count=50),
        ]
        out = await tool.execute(
            PaperRecommendationInput(
                papers=papers,
                criteria=RecommendationCriteria(min_citation_count=20),
            )
        )
        assert len(out.recommendations) == 1
        assert out.recommendations[0].paper_id == "2"

    @pytest.mark.asyncio
    async def test_empty_papers(self):
        tool = PaperRecommendationTool()
        out = await tool.execute(PaperRecommendationInput(papers=[]))
        assert len(out.recommendations) == 0


# --- RelevanceScoringTool ---


class TestRelevanceScoringTool:
    @pytest.mark.asyncio
    async def test_validate(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(paper_id="1", title="A")
        inp = RelevanceScoringInput(paper=paper, repo_summary=_make_repo_summary())
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_high_relevance(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(
            paper_id="1",
            title="FlashAttention: transformer attention mechanism",
            abstract="We propose a novel transformer attention architecture for training with AdamW optimizer",
        )
        out = await tool.execute(
            RelevanceScoringInput(paper=paper, repo_summary=_make_repo_summary())
        )
        assert out.score.overall_score > 0.0
        assert len(out.dimension_scores) == 6

    @pytest.mark.asyncio
    async def test_low_relevance(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(
            paper_id="1",
            title="Cooking recipes for Italian food",
            abstract="We propose a new pasta recipe",
        )
        out = await tool.execute(
            RelevanceScoringInput(paper=paper, repo_summary=_make_repo_summary())
        )
        assert out.score.relevance_level.value == "low"

    @pytest.mark.asyncio
    async def test_all_dimensions_scored(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(paper_id="1", title="test", abstract="test")
        out = await tool.execute(
            RelevanceScoringInput(paper=paper, repo_summary=_make_repo_summary())
        )
        dim_names = [d.dimension for d in out.dimension_scores]
        assert "architecture" in dim_names
        assert "training" in dim_names
        assert "feasibility" in dim_names

    @pytest.mark.asyncio
    async def test_recommendations_generated(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(paper_id="1", title="test", abstract="test")
        out = await tool.execute(
            RelevanceScoringInput(paper=paper, repo_summary=_make_repo_summary())
        )
        assert len(out.recommendations) > 0

    @pytest.mark.asyncio
    async def test_dict_repo_summary(self):
        tool = RelevanceScoringTool()
        paper = PaperSummary(paper_id="1", title="test", abstract="test")
        repo_dict = _make_repo_summary().model_dump()
        out = await tool.execute(
            RelevanceScoringInput(paper=paper, repo_summary=repo_dict)
        )
        assert out.score is not None