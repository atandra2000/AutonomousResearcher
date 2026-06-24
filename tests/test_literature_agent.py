"""Tests for LiteratureAgent (Phase 6)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_engineer.agents.literature_agent import LiteratureAgent, LiteratureConfig
from research_engineer.models.literature import (
    LiteratureReviewOutput,
    PaperComparisonOutput,
    PaperRecommendationOutput,
    PaperRelationshipOutput,
    PaperSearchOutput,
    PaperSummary,
    RelevanceScoringOutput,
    SearchResult,
    SearchSource,
    TrendAnalysisOutput,
)


def _make_papers(n: int = 3) -> list[PaperSummary]:
    return [
        PaperSummary(
            paper_id=f"2503.{i:05d}",
            title=f"Paper {i} about attention",
            abstract=f"We propose a novel attention mechanism {i}",
            authors=["Author"],
            year=2024,
            citation_count=10 * (i + 1),
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_memory_agent():
    agent = MagicMock()
    agent.storage = AsyncMock()
    agent.storage.store_relationship = AsyncMock(return_value=MagicMock(relationship_id="rel_1"))
    agent.graph = MagicMock()
    agent.graph.add_node = MagicMock()
    agent.graph.add_relationship = MagicMock()
    agent.store_insight = AsyncMock(return_value="mem_123")
    return agent


@pytest.fixture
def literature_agent(mock_memory_agent):
    return LiteratureAgent(
        memory_agent=mock_memory_agent,
        config=LiteratureConfig(store_findings=True, update_graph=True),
    )


class TestLiteratureAgentInit:
    def test_defaults(self):
        agent = LiteratureAgent()
        assert agent.memory is None
        assert agent.search is not None
        assert agent.comparison is not None
        assert agent.review is not None
        assert agent.relationship is not None
        assert agent.trend is not None
        assert agent.recommendation is not None
        assert agent.relevance is not None

    def test_config(self):
        agent = LiteratureAgent()
        assert agent.config.max_papers == 20


class TestSearchPapers:
    @pytest.mark.asyncio
    async def test_search_returns_output(self, literature_agent):
        result = await literature_agent.search_papers("attention", max_results=5)
        assert isinstance(result, PaperSearchOutput)


class TestComparePapers:
    @pytest.mark.asyncio
    async def test_compare(self, literature_agent):
        papers = _make_papers(3)
        result = await literature_agent.compare_papers(papers)
        assert isinstance(result, PaperComparisonOutput)
        assert len(result.ranking) == 3


class TestDetectRelationships:
    @pytest.mark.asyncio
    async def test_detect(self, literature_agent):
        papers = _make_papers(3)
        result = await literature_agent.detect_relationships(papers)
        assert isinstance(result, PaperRelationshipOutput)


class TestAnalyzeTrends:
    @pytest.mark.asyncio
    async def test_trends(self, literature_agent):
        papers = _make_papers(5)
        result = await literature_agent.analyze_trends(papers)
        assert isinstance(result, TrendAnalysisOutput)


class TestGenerateReview:
    @pytest.mark.asyncio
    async def test_review(self, literature_agent):
        papers = _make_papers(3)
        result = await literature_agent.generate_review("attention", papers)
        assert isinstance(result, LiteratureReviewOutput)
        assert result.review.papers_analyzed == 3
        assert len(result.markdown) > 0


class TestRecommendPapers:
    @pytest.mark.asyncio
    async def test_recommend(self, literature_agent):
        papers = _make_papers(3)
        result = await literature_agent.recommend_papers(papers)
        assert isinstance(result, PaperRecommendationOutput)
        assert len(result.recommendations) > 0


class TestScoreRelevance:
    @pytest.mark.asyncio
    async def test_score_with_none(self, literature_agent):
        result = await literature_agent.score_relevance(None, MagicMock())
        assert result is None


class TestDiscoverWorkflow:
    @pytest.mark.asyncio
    async def test_discover_basic(self, literature_agent, tmp_path):
        result = await literature_agent.discover(
            "attention",
            output_dir=str(tmp_path / "lit"),
            max_papers=5,
        )
        assert result.topic == "attention"
        assert result.search_results is not None
        assert result.processing_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_discover_stores_in_memory(self, literature_agent, mock_memory_agent, tmp_path):
        await literature_agent.discover(
            "attention",
            output_dir=str(tmp_path / "lit"),
            max_papers=3,
        )
        assert mock_memory_agent.store_insight.called


class TestToSummaries:
    def test_convert(self, literature_agent):
        from research_engineer.models.literature import PaperSearchOutput, SearchResult
        output = PaperSearchOutput(
            papers=[
                SearchResult(paper_id="1", title="A", abstract="test", source=SearchSource.ARXIV),
                SearchResult(paper_id="2", title="B", abstract="test", source=SearchSource.LOCAL),
            ],
            total_found=2,
        )
        summaries = literature_agent._to_summaries(output)
        assert len(summaries) == 2
        assert summaries[0].paper_id == "1"


class TestStoreRelationships:
    @pytest.mark.asyncio
    async def test_store_relationships(self, literature_agent, mock_memory_agent):
        from research_engineer.models.literature import (
            PaperRelationType,
            PaperRelationship,
            PaperRelationshipOutput,
        )
        output = PaperRelationshipOutput(
            relationships=[
                PaperRelationship(
                    source_paper_id="1",
                    target_paper_id="2",
                    relationship_type=PaperRelationType.CITES,
                    confidence=0.9,
                )
            ],
        )
        ids = await literature_agent._store_relationships(output)
        assert len(ids) == 1
        assert mock_memory_agent.storage.store_relationship.called
        assert mock_memory_agent.graph.add_relationship.called

    @pytest.mark.asyncio
    async def test_store_no_memory(self):
        agent = LiteratureAgent(memory_agent=None)
        from research_engineer.models.literature import PaperRelationshipOutput
        ids = await agent._store_relationships(PaperRelationshipOutput())
        assert ids == []