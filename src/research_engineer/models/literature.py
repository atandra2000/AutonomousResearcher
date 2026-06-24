"""Literature intelligence models for Phase 6.

Provides typed Pydantic models for paper search, comparison, review
generation, relationship detection, trend analysis, recommendations, and
repository relevance scoring.
"""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchSource(StrEnum):
    """Sources for paper search."""

    LOCAL = "local"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class ReviewDepth(StrEnum):
    """Depth levels for literature reviews."""

    BRIEF = "brief"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class PaperRelationType(StrEnum):
    """Types of relationships between papers."""

    CITES = "cites"
    EXTENDS = "extends"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    BUILDS_ON = "builds_on"
    REPRODUCES = "reproduces"
    IMPROVES_UPON = "improves_upon"


class TrendDirection(StrEnum):
    """Direction of a research trend."""

    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"


class RelevanceLevel(StrEnum):
    """Relevance level for repository implementation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Paper Summary (shared across tools)
# ---------------------------------------------------------------------------


class PaperSummary(BaseModel):
    """Summary of a paper used across Phase 6 tools."""

    paper_id: str = Field(..., description="Paper ID (arXiv or DOI)")
    title: str = Field(..., description="Paper title")
    abstract: str = Field("", description="Paper abstract")
    authors: list[str] = Field(default_factory=list, description="Author names")
    year: int | None = Field(None, description="Publication year")
    citation_count: int = Field(0, description="Citation count")
    key_contributions: list[str] = Field(
        default_factory=list, description="Key contributions"
    )
    methodology: str = Field("", description="Methodology description")
    url: str | None = Field(None, description="Paper URL")
    doi: str | None = Field(None, description="DOI")
    source: SearchSource | None = Field(None, description="Where found")


# ---------------------------------------------------------------------------
# Paper Search
# ---------------------------------------------------------------------------


class PaperSearchInput(BaseModel):
    """Input for paper search."""

    query: str = Field(..., description="Search query")
    sources: list[SearchSource] = Field(
        default_factory=lambda: [SearchSource.LOCAL, SearchSource.ARXIV],
        description="Sources to search",
    )
    max_results: int = Field(20, ge=1, le=100, description="Max results per source")
    year_range: str | None = Field(None, description="Year range e.g. 2020-2025")
    fields_of_study: list[str] | None = Field(
        None, description="Filter by fields of study"
    )
    min_citation_count: int | None = Field(
        None, description="Minimum citation count"
    )
    sort: str = Field("relevance", description="Sort: relevance, citationCount, year")


class SearchResult(BaseModel):
    """A single search result."""

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Author names")
    abstract: str = Field("", description="Paper abstract")
    year: int | None = Field(None, description="Publication year")
    citation_count: int = Field(0, description="Citation count")
    source: SearchSource = Field(..., description="Source where found")
    url: str | None = Field(None, description="Paper URL")
    doi: str | None = Field(None, description="DOI")
    relevance_score: float = Field(0.0, description="Relevance to query (0-1)")


class PaperSearchOutput(BaseModel):
    """Output from paper search."""

    papers: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    total_found: int = Field(0, description="Total papers found")
    sources_searched: list[str] = Field(
        default_factory=list, description="Sources searched"
    )
    search_time_seconds: float = Field(0.0, description="Search duration")


# ---------------------------------------------------------------------------
# Paper Comparison
# ---------------------------------------------------------------------------


class ComparisonDimension(BaseModel):
    """A dimension for comparing papers."""

    name: str = Field(..., description="Dimension name")
    description: str = Field("", description="Dimension description")


class ComparisonMatrix(BaseModel):
    """Matrix of paper x dimension values."""

    papers: list[str] = Field(..., description="Paper IDs")
    dimensions: list[ComparisonDimension] = Field(
        default_factory=list, description="Comparison dimensions"
    )
    matrix: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="paper_id -> dimension -> value"
    )


class SimilarityPair(BaseModel):
    """Similarity between two papers."""

    paper_a: str = Field(..., description="First paper ID")
    paper_b: str = Field(..., description="Second paper ID")
    similarity_score: float = Field(0.0, description="Similarity 0-1")
    shared_aspects: list[str] = Field(
        default_factory=list, description="Shared aspects"
    )


class DifferencePair(BaseModel):
    """Difference between two papers on a dimension."""

    paper_a: str = Field(..., description="First paper ID")
    paper_b: str = Field(..., description="Second paper ID")
    dimension: str = Field(..., description="Dimension name")
    value_a: str = Field("", description="Value for paper A")
    value_b: str = Field("", description="Value for paper B")


class ConflictItem(BaseModel):
    """A conflicting finding across papers."""

    papers: list[str] = Field(..., description="Conflicting paper IDs")
    topic: str = Field(..., description="Topic of conflict")
    positions: dict[str, str] = Field(
        default_factory=dict, description="paper_id -> position"
    )


class PaperRanking(BaseModel):
    """A paper's rank in a comparison."""

    paper_id: str = Field(..., description="Paper ID")
    rank: int = Field(..., description="Rank position (1 = best)")
    score: float = Field(0.0, description="Ranking score")
    rationale: str = Field("", description="Why this rank")


class PaperComparisonInput(BaseModel):
    """Input for paper comparison."""

    papers: list[PaperSummary] = Field(
        ..., min_length=2, description="Papers to compare"
    )
    dimensions: list[str] | None = Field(
        None, description="Override default dimensions"
    )


class PaperComparisonOutput(BaseModel):
    """Output from paper comparison."""

    comparison: ComparisonMatrix = Field(..., description="Comparison matrix")
    similarities: list[SimilarityPair] = Field(
        default_factory=list, description="Similarities"
    )
    differences: list[DifferencePair] = Field(
        default_factory=list, description="Differences"
    )
    consensus_findings: list[str] = Field(
        default_factory=list, description="Agreed findings"
    )
    conflicting_findings: list[ConflictItem] = Field(
        default_factory=list, description="Conflicts"
    )
    ranking: list[PaperRanking] = Field(
        default_factory=list, description="Paper rankings"
    )


# ---------------------------------------------------------------------------
# Literature Review
# ---------------------------------------------------------------------------


class ReviewSection(BaseModel):
    """A section of a literature review."""

    title: str = Field(..., description="Section title")
    content: str = Field("", description="Section content")
    papers_referenced: list[str] = Field(
        default_factory=list, description="Referenced paper IDs"
    )
    key_points: list[str] = Field(
        default_factory=list, description="Key points"
    )


class TimelineEntry(BaseModel):
    """An entry in the research timeline."""

    year: int = Field(..., description="Year")
    papers: list[str] = Field(
        default_factory=list, description="Paper IDs in that year"
    )
    milestone: str = Field("", description="What happened this year")


class LiteratureReview(BaseModel):
    """A structured literature review."""

    review_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Review ID"
    )
    topic: str = Field(..., description="Review topic")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    executive_summary: str = Field("", description="Executive summary")
    papers_analyzed: int = Field(0, description="Number of papers analyzed")
    sections: list[ReviewSection] = Field(
        default_factory=list, description="Review sections"
    )
    methodology_landscape: str = Field("", description="Methodology landscape")
    key_findings: list[str] = Field(
        default_factory=list, description="Key findings"
    )
    research_gaps: list[str] = Field(
        default_factory=list, description="Research gaps"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )
    timeline: list[TimelineEntry] = Field(
        default_factory=list, description="Chronological timeline"
    )
    citation_network: dict[str, list[str]] = Field(
        default_factory=dict, description="paper_id -> cited_by"
    )


class LiteratureReviewInput(BaseModel):
    """Input for literature review generation."""

    topic: str = Field(..., description="Review topic")
    papers: list[PaperSummary] = Field(..., description="Papers to include")
    review_depth: ReviewDepth = Field(
        ReviewDepth.STANDARD, description="Review depth"
    )
    output_format: str = Field("markdown", description="Output format")


class LiteratureReviewOutput(BaseModel):
    """Output from literature review generation."""

    review: LiteratureReview = Field(..., description="The review")
    markdown: str = Field("", description="Full markdown document")
    generated_files: list[str] = Field(
        default_factory=list, description="Generated file paths"
    )


# ---------------------------------------------------------------------------
# Paper Relationships
# ---------------------------------------------------------------------------


class PaperRelationship(BaseModel):
    """A relationship between two papers."""

    source_paper_id: str = Field(..., description="Source paper ID")
    target_paper_id: str = Field(..., description="Target paper ID")
    relationship_type: PaperRelationType = Field(
        ..., description="Type of relationship"
    )
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    evidence: str = Field("", description="Evidence for the relationship")


class PaperRelationshipInput(BaseModel):
    """Input for paper relationship detection."""

    papers: list[PaperSummary] = Field(..., description="Papers to analyze")
    detect_citations: bool = Field(True, description="Detect citations")
    detect_extensions: bool = Field(True, description="Detect extensions")
    detect_similarities: bool = Field(True, description="Detect similarities")
    detect_contradictions: bool = Field(True, description="Detect contradictions")
    min_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Min confidence")


class PaperRelationshipOutput(BaseModel):
    """Output from paper relationship detection."""

    relationships: list[PaperRelationship] = Field(
        default_factory=list, description="Detected relationships"
    )
    relationship_graph: dict[str, list[str]] = Field(
        default_factory=dict, description="paper_id -> related_ids"
    )
    summary: dict[str, int] = Field(
        default_factory=dict, description="Counts by relationship type"
    )


# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------


class ResearchTrend(BaseModel):
    """A research trend over time."""

    topic: str = Field(..., description="Trend topic")
    direction: TrendDirection = Field(..., description="Trend direction")
    paper_count_by_year: dict[str, int] = Field(
        default_factory=dict, description="year -> count"
    )
    growth_rate: float = Field(0.0, description="Annual growth rate %")
    key_papers: list[str] = Field(
        default_factory=list, description="Key paper IDs"
    )
    description: str = Field("", description="Trend description")
    start_year: int | None = Field(None, description="First year seen")
    peak_year: int | None = Field(None, description="Peak year")


class TopicEntry(BaseModel):
    """A topic entry for emerging/hot/declining lists."""

    topic: str = Field(..., description="Topic name")
    paper_count: int = Field(0, description="Number of papers")
    growth_rate: float = Field(0.0, description="Growth rate %")
    recent_papers: list[str] = Field(
        default_factory=list, description="Recent paper IDs"
    )
    first_seen_year: int | None = Field(None, description="First year seen")


class TrendAnalysisInput(BaseModel):
    """Input for trend analysis."""

    papers: list[PaperSummary] = Field(..., description="Papers to analyze")
    time_window_years: int = Field(5, ge=1, description="How far back to look")
    granularity: str = Field("year", description="Granularity: year, biannual")
    min_papers_per_trend: int = Field(
        3, ge=1, description="Min papers to form a trend"
    )


class TrendAnalysisOutput(BaseModel):
    """Output from trend analysis."""

    trends: list[ResearchTrend] = Field(
        default_factory=list, description="Detected trends"
    )
    emerging_topics: list[TopicEntry] = Field(
        default_factory=list, description="Emerging topics"
    )
    declining_topics: list[TopicEntry] = Field(
        default_factory=list, description="Declining topics"
    )
    hot_topics: list[TopicEntry] = Field(
        default_factory=list, description="Hot topics"
    )
    trend_summary: str = Field("", description="Summary of trends")


# ---------------------------------------------------------------------------
# Paper Recommendations
# ---------------------------------------------------------------------------


class RecommendationCriteria(BaseModel):
    """Criteria for paper recommendations."""

    min_citation_count: int = Field(0, ge=0, description="Min citation count")
    min_novelty_score: float = Field(0.5, ge=0.0, le=1.0, description="Min novelty")
    min_implementability: float = Field(
        0.4, ge=0.0, le=1.0, description="Min implementability"
    )
    prefer_recent: bool = Field(True, description="Prefer recent papers")
    max_age_years: int | None = Field(None, description="Max paper age in years")
    topics: list[str] | None = Field(None, description="Filter by topics")


class PaperRecommendation(BaseModel):
    """A single paper recommendation."""

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    rank: int = Field(..., description="Rank (1 = best)")
    overall_score: float = Field(0.0, description="Overall score 0-1")
    impact_score: float = Field(0.0, description="Impact score")
    novelty_score: float = Field(0.0, description="Novelty score")
    implementability_score: float = Field(0.0, description="Implementability")
    relevance_score: float = Field(0.0, description="Relevance to criteria")
    rationale: str = Field("", description="Recommendation rationale")
    key_strengths: list[str] = Field(
        default_factory=list, description="Key strengths"
    )
    potential_challenges: list[str] = Field(
        default_factory=list, description="Potential challenges"
    )


class PaperRecommendationInput(BaseModel):
    """Input for paper recommendations."""

    papers: list[PaperSummary] = Field(..., description="Papers to rank")
    criteria: RecommendationCriteria | None = Field(
        None, description="Recommendation criteria"
    )
    max_recommendations: int = Field(10, ge=1, le=100, description="Max results")


class PaperRecommendationOutput(BaseModel):
    """Output from paper recommendations."""

    recommendations: list[PaperRecommendation] = Field(
        default_factory=list, description="Recommendations"
    )
    ranking_rationale: str = Field("", description="Overall rationale")


# ---------------------------------------------------------------------------
# Relevance Scoring
# ---------------------------------------------------------------------------


class RelevanceScore(BaseModel):
    """Overall relevance score for a paper to a repository."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall score")
    relevance_level: RelevanceLevel = Field(
        RelevanceLevel.LOW, description="Relevance level"
    )


class RelevanceDimension(BaseModel):
    """A dimension of relevance scoring."""

    dimension: str = Field(..., description="Dimension name")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Score 0-1")
    reasoning: str = Field("", description="Reasoning for score")
    evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence"
    )


class RelevanceScoringInput(BaseModel):
    """Input for relevance scoring."""

    paper: PaperSummary = Field(..., description="Paper to score")
    repo_summary: object = Field(
        ..., description="RepositorySummary from Phase 2"
    )
    additional_papers: list[PaperSummary] = Field(
        default_factory=list, description="Context papers"
    )


class RelevanceScoringOutput(BaseModel):
    """Output from relevance scoring."""

    score: RelevanceScore = Field(..., description="Overall relevance score")
    dimension_scores: list[RelevanceDimension] = Field(
        default_factory=list, description="Per-dimension scores"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )


# ---------------------------------------------------------------------------
# Top-Level Result (discover workflow)
# ---------------------------------------------------------------------------


class LiteratureResult(BaseModel):
    """Top-level result from the full literature discovery workflow."""

    result_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Result ID"
    )
    topic: str = Field(..., description="Search topic")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    search_results: PaperSearchOutput | None = Field(
        None, description="Search results"
    )
    comparison: PaperComparisonOutput | None = Field(
        None, description="Comparison results"
    )
    relationships: PaperRelationshipOutput | None = Field(
        None, description="Relationship results"
    )
    trends: TrendAnalysisOutput | None = Field(
        None, description="Trend results"
    )
    review: LiteratureReviewOutput | None = Field(
        None, description="Literature review"
    )
    recommendations: PaperRecommendationOutput | None = Field(
        None, description="Recommendations"
    )
    relevance: RelevanceScoringOutput | None = Field(
        None, description="Relevance score (if repo provided)"
    )
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    output_dir: str | None = Field(None, description="Output directory")
    generated_files: list[str] = Field(
        default_factory=list, description="Generated file paths"
    )
    processing_time_seconds: float = Field(0.0, description="Total time")
