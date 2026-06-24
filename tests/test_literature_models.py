"""Tests for Phase 6 literature models."""

import pytest

from research_engineer.models.literature import (
    ComparisonDimension,
    ComparisonMatrix,
    ConflictItem,
    DifferencePair,
    LiteratureResult,
    LiteratureReview,
    PaperComparisonInput,
    PaperComparisonOutput,
    PaperRecommendation,
    PaperRecommendationInput,
    PaperRecommendationOutput,
    PaperRelationship,
    PaperRelationshipInput,
    PaperRelationshipOutput,
    PaperRelationType,
    PaperSearchInput,
    PaperSearchOutput,
    PaperSummary,
    RecommendationCriteria,
    RelevanceDimension,
    RelevanceLevel,
    RelevanceScore,
    RelevanceScoringInput,
    RelevanceScoringOutput,
    ResearchTrend,
    ReviewDepth,
    ReviewSection,
    SearchResult,
    SearchSource,
    SimilarityPair,
    TimelineEntry,
    TopicEntry,
    TrendAnalysisInput,
    TrendAnalysisOutput,
    TrendDirection,
)


class TestEnums:
    def test_search_source_values(self):
        assert SearchSource.LOCAL.value == "local"
        assert SearchSource.ARXIV.value == "arxiv"
        assert SearchSource.SEMANTIC_SCHOLAR.value == "semantic_scholar"

    def test_review_depth_values(self):
        assert ReviewDepth.BRIEF.value == "brief"
        assert ReviewDepth.STANDARD.value == "standard"
        assert ReviewDepth.COMPREHENSIVE.value == "comprehensive"

    def test_paper_relation_type_values(self):
        assert PaperRelationType.CITES.value == "cites"
        assert PaperRelationType.SIMILAR_TO.value == "similar_to"
        assert PaperRelationType.CONTRADICTS.value == "contradicts"

    def test_trend_direction_values(self):
        assert TrendDirection.RISING.value == "rising"
        assert TrendDirection.DECLINING.value == "declining"
        assert TrendDirection.STABLE.value == "stable"

    def test_relevance_level_values(self):
        assert RelevanceLevel.HIGH.value == "high"
        assert RelevanceLevel.MEDIUM.value == "medium"
        assert RelevanceLevel.LOW.value == "low"


class TestPaperSummary:
    def test_minimal(self):
        p = PaperSummary(paper_id="2503.12345", title="Test Paper")
        assert p.paper_id == "2503.12345"
        assert p.abstract == ""
        assert p.authors == []
        assert p.citation_count == 0

    def test_full(self):
        p = PaperSummary(
            paper_id="2503.12345",
            title="Test Paper",
            abstract="An abstract",
            authors=["Author A"],
            year=2024,
            citation_count=50,
            key_contributions=["contribution 1"],
            methodology="transformer",
            url="https://arxiv.org/abs/2503.12345",
            doi="10.1234/test",
            source=SearchSource.ARXIV,
        )
        assert p.year == 2024
        assert p.citation_count == 50
        assert p.source == SearchSource.ARXIV


class TestPaperSearchModels:
    def test_search_input_defaults(self):
        inp = PaperSearchInput(query="attention")
        assert inp.max_results == 20
        assert inp.sort == "relevance"
        assert SearchSource.LOCAL in inp.sources

    def test_search_result(self):
        r = SearchResult(
            paper_id="2503.12345",
            title="Test",
            source=SearchSource.ARXIV,
        )
        assert r.citation_count == 0
        assert r.relevance_score == 0.0

    def test_search_output(self):
        out = PaperSearchOutput(
            papers=[],
            total_found=0,
            sources_searched=["arxiv"],
            search_time_seconds=0.5,
        )
        assert out.total_found == 0
        assert "arxiv" in out.sources_searched


class TestPaperComparisonModels:
    def test_comparison_input_min_length(self):
        with pytest.raises(Exception):
            PaperComparisonInput(papers=[PaperSummary(paper_id="1", title="A")])

    def test_comparison_input_valid(self):
        papers = [
            PaperSummary(paper_id="1", title="A"),
            PaperSummary(paper_id="2", title="B"),
        ]
        inp = PaperComparisonInput(papers=papers)
        assert len(inp.papers) == 2

    def test_comparison_matrix(self):
        m = ComparisonMatrix(
            papers=["1", "2"],
            dimensions=[ComparisonDimension(name="Architecture", description="")],
            matrix={"1": {"Architecture": "transformer"}, "2": {"Architecture": "cnn"}},
        )
        assert m.matrix["1"]["Architecture"] == "transformer"

    def test_similarity_pair(self):
        s = SimilarityPair(paper_a="1", paper_b="2", similarity_score=0.8)
        assert s.similarity_score == 0.8

    def test_difference_pair(self):
        d = DifferencePair(paper_a="1", paper_b="2", dimension="Training", value_a="adam", value_b="sgd")
        assert d.dimension == "Training"

    def test_conflict_item(self):
        c = ConflictItem(papers=["1", "2"], topic="evaluation", positions={"1": "A", "2": "B"})
        assert c.positions["1"] == "A"


class TestLiteratureReviewModels:
    def test_review_section(self):
        s = ReviewSection(title="Architecture", content="content", papers_referenced=["1"])
        assert s.title == "Architecture"
        assert s.key_points == []

    def test_timeline_entry(self):
        e = TimelineEntry(year=2024, papers=["1"], milestone="test")
        assert e.year == 2024

    def test_literature_review_defaults(self):
        r = LiteratureReview(topic="attention")
        assert r.papers_analyzed == 0
        assert r.sections == []
        assert r.executive_summary == ""

    def test_review_output(self):
        from research_engineer.models.literature import LiteratureReviewOutput
        review = LiteratureReview(topic="attention")
        out = LiteratureReviewOutput(review=review, markdown="# Review")
        assert out.markdown == "# Review"
        assert out.review.topic == "attention"


class TestPaperRelationshipModels:
    def test_relationship(self):
        r = PaperRelationship(
            source_paper_id="1",
            target_paper_id="2",
            relationship_type=PaperRelationType.CITES,
            confidence=0.9,
        )
        assert r.relationship_type == PaperRelationType.CITES
        assert r.confidence == 0.9

    def test_relationship_input_defaults(self):
        inp = PaperRelationshipInput(papers=[PaperSummary(paper_id="1", title="A")])
        assert inp.detect_citations is True
        assert inp.min_confidence == 0.6

    def test_relationship_output(self):
        out = PaperRelationshipOutput(
            relationships=[],
            relationship_graph={"1": ["2"]},
            summary={"cites": 1},
        )
        assert out.relationship_graph["1"] == ["2"]


class TestTrendModels:
    def test_research_trend(self):
        t = ResearchTrend(
            topic="attention",
            direction=TrendDirection.RISING,
            paper_count_by_year={"2024": 10},
            growth_rate=25.0,
        )
        assert t.direction == TrendDirection.RISING
        assert t.growth_rate == 25.0

    def test_topic_entry(self):
        e = TopicEntry(topic="moe", paper_count=5, growth_rate=30.0)
        assert e.topic == "moe"

    def test_trend_output(self):
        out = TrendAnalysisOutput(trend_summary="Summary")
        assert out.trends == []


class TestRecommendationModels:
    def test_criteria_defaults(self):
        c = RecommendationCriteria()
        assert c.min_citation_count == 0
        assert c.min_novelty_score == 0.5
        assert c.prefer_recent is True

    def test_paper_recommendation(self):
        r = PaperRecommendation(
            paper_id="1",
            title="Test",
            rank=1,
            overall_score=0.85,
        )
        assert r.rank == 1
        assert r.key_strengths == []

    def test_recommendation_output(self):
        out = PaperRecommendationOutput(recommendations=[], ranking_rationale="test")
        assert out.ranking_rationale == "test"


class TestRelevanceModels:
    def test_relevance_score(self):
        s = RelevanceScore(
            paper_id="1",
            repo_path="./repo",
            overall_score=0.75,
            relevance_level=RelevanceLevel.HIGH,
        )
        assert s.relevance_level == RelevanceLevel.HIGH

    def test_relevance_dimension(self):
        d = RelevanceDimension(dimension="architecture", score=0.8, reasoning="match")
        assert d.score == 0.8
        assert d.evidence == []


class TestLiteratureResult:
    def test_defaults(self):
        r = LiteratureResult(topic="attention")
        assert r.search_results is None
        assert r.comparison is None
        assert r.memory_ids == []
        assert r.processing_time_seconds == 0.0

    def test_serialization(self):
        r = LiteratureResult(topic="attention")
        data = r.model_dump()
        assert data["topic"] == "attention"
        r2 = LiteratureResult(**data)
        assert r2.topic == "attention"