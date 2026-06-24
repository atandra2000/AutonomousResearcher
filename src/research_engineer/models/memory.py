"""Memory models for persistent knowledge storage."""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MemoryType(StrEnum):
    """Types of memories in the system."""

    PAPER = "paper"
    REPOSITORY = "repository"
    EXPERIMENT_PLAN = "experiment_plan"
    PATCH = "patch"
    ARCHITECTURE_DECISION = "architecture_decision"
    RESEARCH_INSIGHT = "research_insight"
    FAILED_APPROACH = "failed_approach"
    SUCCESSFUL_APPROACH = "successful_approach"


class InsightType(StrEnum):
    """Types of research insights."""

    PATTERN = "pattern"
    ANTI_PATTERN = "anti_pattern"
    OPTIMIZATION = "optimization"
    BEST_PRACTICE = "best_practice"
    EMPIRICAL_FINDING = "empirical_finding"
    THEORETICAL_RESULT = "theoretical_result"
    IMPLEMENTATION_TRICK = "implementation_trick"
    HYPERPARAMETER_GUIDELINE = "hyperparameter_guideline"


class FailureMode(StrEnum):
    """Types of failure modes."""

    CRASH = "crash"
    DIVERGENCE = "divergence"
    POOR_PERFORMANCE = "poor_performance"
    MEMORY_OVERFLOW = "memory_overflow"
    NUMERICAL_INSTABILITY = "numerical_instability"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    DATA_CORRUPTION = "data_corruption"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    DISTRIBUTION_SHIFT = "distribution_shift"
    API_INCOMPATIBILITY = "api_incompatibility"
    DEPENDENCY_CONFLICT = "dependency_conflict"


class ExecutionOutcome(StrEnum):
    """Outcome of experiment execution."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    NOT_EXECUTED = "not_executed"


class MemoryBase(BaseModel):
    """Base class for all memory types."""

    memory_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique memory ID")
    memory_type: MemoryType = Field(..., description="Type of memory")
    tags: list[str] = Field(default_factory=list, description="Auto-generated tags")
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Quality/confidence score")
    accessed_count: int = Field(0, description="Number of times accessed")
    embedding_key: str | None = Field(None, description="Reference to vector store")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")
    last_accessed_at: datetime | None = Field(None, description="Last access timestamp")
    is_archived: bool = Field(False, description="Whether memory is archived")

    def mark_accessed(self):
        """Mark memory as accessed."""
        self.accessed_count += 1
        self.last_accessed_at = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump(mode="json")


class PaperMemory(MemoryBase):
    """Memory for analyzed papers."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "paper_id": "2503.12345",
            "title": "Example Paper",
            "abstract": "This paper presents...",
        }
    })

    memory_type: MemoryType = MemoryType.PAPER
    paper_id: str = Field(..., description="Paper ID (arXiv or DOI)")
    title: str = Field(..., description="Paper title")
    authors: list[dict] = Field(default_factory=list, description="Author information")
    abstract: str = Field(..., description="Paper abstract")
    research_summary: dict = Field(default_factory=dict, description="Serialized ResearchSummary")
    engineering_report: dict = Field(default_factory=dict, description="Serialized EngineeringReport")
    full_text_embedding_key: str | None = Field(None, description="Full text embedding key")
    pdf_path: str | None = Field(None, description="Path to stored PDF")


class RepositoryMemory(MemoryBase):
    """Memory for analyzed repositories."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "repo_path": "./my_repo",
            "repo_name": "my_repo",
            "architecture_summary": "PyTorch-based training pipeline...",
        }
    })

    memory_type: MemoryType = MemoryType.REPOSITORY
    repo_path: str = Field(..., description="Repository path")
    repo_name: str = Field(..., description="Repository name")
    repo_type: str = Field(..., description="Type of repository")
    architecture_summary: str = Field(..., description="Architecture description")
    key_components: list[str] = Field(default_factory=list, description="Key components")
    training_pipeline: dict = Field(default_factory=dict, description="Training pipeline info")
    config_structure: dict = Field(default_factory=dict, description="Config file structure")
    dependencies: list[str] = Field(default_factory=list, description="Key dependencies")
    code_embedding_key: str | None = Field(None, description="Code embedding key")


class ExperimentPlanMemory(MemoryBase):
    """Memory for experiment plans."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "plan_id": "plan_123",
            "paper_id": "2503.12345",
            "repo_path": "./my_repo",
        }
    })

    memory_type: MemoryType = MemoryType.EXPERIMENT_PLAN
    plan_id: str = Field(..., description="Plan ID")
    paper_id: str = Field(..., description="Associated paper ID")
    repo_path: str = Field(..., description="Target repository")
    compatibility_report: dict = Field(default_factory=dict, description="Compatibility analysis")
    implementation_plan: dict = Field(default_factory=dict, description="Implementation plan")
    impact_report: dict = Field(default_factory=dict, description="Impact analysis")
    experiment_matrix: dict = Field(default_factory=dict, description="Experiment matrix")
    validation_plan: dict = Field(default_factory=dict, description="Validation plan")
    risk_assessment: dict = Field(default_factory=dict, description="Risk assessment")
    compute_estimate: dict = Field(default_factory=dict, description="Compute estimates")
    result_prediction: dict = Field(default_factory=dict, description="Result predictions")
    execution_outcome: ExecutionOutcome | None = Field(None, description="Actual outcome")
    output_dir: str | None = Field(None, description="Output directory path")


class PatchMemory(MemoryBase):
    """Memory for generated patches."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "patch_id": "patch_123",
            "files_modified": ["src/model.py"],
            "change_type": "modification",
        }
    })

    memory_type: MemoryType = MemoryType.PATCH
    patch_id: str = Field(..., description="Patch ID")
    implementation_id: str = Field(..., description="Implementation ID")
    paper_id: str | None = Field(None, description="Associated paper ID")
    repo_path: str = Field(..., description="Target repository")
    patch_content: str = Field(..., description="Unified diff content")
    files_modified: list[str] = Field(default_factory=list, description="Modified files")
    change_type: str = Field(..., description="Type of change")
    test_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Test coverage")
    review_result: dict = Field(default_factory=dict, description="Review results")
    applied_successfully: bool = Field(False, description="Whether patch was applied")
    rollback_available: bool = Field(True, description="Whether rollback is available")
    performance_impact: dict | None = Field(None, description="Performance metrics")
    patch_file_path: str | None = Field(None, description="Path to patch file")


class ArchitectureDecisionMemory(MemoryBase):
    """Memory for architecture decisions."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "context": "Need to choose optimizer for training",
            "decision": "Use AdamW with lr=1e-4",
            "rationale": "Better convergence than Adam...",
        }
    })

    memory_type: MemoryType = MemoryType.ARCHITECTURE_DECISION
    decision_id: str = Field(default_factory=lambda: str(uuid4()), description="Decision ID")
    context: str = Field(..., description="What prompted this decision")
    decision: str = Field(..., description="What was decided")
    rationale: str = Field(..., description="Why this choice was made")
    alternatives_considered: list[str] = Field(default_factory=list, description="Alternatives")
    consequences: list[str] = Field(default_factory=list, description="Consequences")
    related_papers: list[str] = Field(default_factory=list, description="Related paper IDs")
    related_repos: list[str] = Field(default_factory=list, description="Related repos")
    confidence: str = Field("high", description="Confidence level")
    validated: bool = Field(False, description="Whether tested in practice")


class ResearchInsightMemory(MemoryBase):
    """Memory for research insights."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "insight_type": "pattern",
            "domain": "attention",
            "description": "Multi-head attention works best with...",
        }
    })

    memory_type: MemoryType = MemoryType.RESEARCH_INSIGHT
    insight_type: InsightType = Field(..., description="Type of insight")
    domain: str = Field(..., description="Research domain (e.g., attention, normalization)")
    description: str = Field(..., description="Insight description")
    evidence: list[str] = Field(default_factory=list, description="Supporting paper/repo IDs")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in insight")
    applicability: list[str] = Field(default_factory=list, description="Where this applies")
    derived_from: list[str] = Field(default_factory=list, description="Source memory IDs")


class FailedApproachMemory(MemoryBase):
    """Memory for failed approaches."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "context": "Training with batch size 1024",
            "approach_description": "Attempted gradient accumulation...",
            "failure_mode": "gradient_explosion",
        }
    })

    memory_type: MemoryType = MemoryType.FAILED_APPROACH
    failure_id: str = Field(default_factory=lambda: str(uuid4()), description="Failure ID")
    context: str = Field(..., description="What was attempted")
    approach_description: str = Field(..., description="Description of approach")
    failure_mode: FailureMode = Field(..., description="Type of failure")
    error_details: str | None = Field(None, description="Error message/details")
    root_cause_analysis: str | None = Field(None, description="Root cause if known")
    lessons_learned: list[str] = Field(default_factory=list, description="Lessons learned")
    related_papers: list[str] = Field(default_factory=list, description="Related paper IDs")
    related_repos: list[str] = Field(default_factory=list, description="Related repos")
    recovery_action: str | None = Field(None, description="Action taken to recover")


class SuccessfulApproachMemory(MemoryBase):
    """Memory for successful approaches."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "context": "Achieved 95% accuracy on ImageNet",
            "approach_description": "Used mixup augmentation with...",
            "success_metrics": {"accuracy": 0.95, "f1": 0.94},
        }
    })

    memory_type: MemoryType = MemoryType.SUCCESSFUL_APPROACH
    success_id: str = Field(default_factory=lambda: str(uuid4()), description="Success ID")
    context: str = Field(..., description="What was achieved")
    approach_description: str = Field(..., description="Description of approach")
    success_metrics: dict[str, float] = Field(default_factory=dict, description="Success metrics")
    key_factors: list[str] = Field(default_factory=list, description="What made it work")
    reproducibility_notes: str = Field("", description="How to reproduce")
    related_papers: list[str] = Field(default_factory=list, description="Related paper IDs")
    related_repos: list[str] = Field(default_factory=list, description="Related repos")
    transferable_to: list[str] = Field(default_factory=list, description="Where else this applies")


class RelationshipType(StrEnum):
    """Types of relationships between memories."""

    CITES = "cites"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    CONFLICTS_WITH = "conflicts_with"
    VALIDATES = "validates"
    FAILED_WITH = "failed_with"
    SUCCEEDED_WITH = "succeeded_with"
    INSPIRED_BY = "inspired_by"
    CRITIQUES = "critiques"
    COMBINES = "combines"
    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"
    CONFIRMS = "confirms"


class MemoryRelationship(BaseModel):
    """Relationship between two memories."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "source_memory_id": "mem_123",
            "target_memory_id": "mem_456",
            "relationship_type": "cites",
        }
    })

    relationship_id: str = Field(default_factory=lambda: str(uuid4()), description="Relationship ID")
    source_memory_id: str = Field(..., description="Source memory ID")
    target_memory_id: str = Field(..., description="Target memory ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in relationship")
    metadata: dict = Field(default_factory=dict, description="Relationship-specific metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    validated: bool = Field(False, description="Whether relationship is validated")


class MemoryAccessLog(BaseModel):
    """Log entry for memory access."""

    id: int | None = Field(None, description="Database record ID")
    memory_id: str = Field(..., description="Memory ID accessed")
    access_type: str = Field(..., description="Type of access (read, write, update, delete)")
    accessed_by: str | None = Field(None, description="Agent/user identifier")
    context: str | None = Field(None, description="Query or operation context")
    accessed_at: datetime = Field(default_factory=datetime.now, description="Access timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "mem_123",
                "access_type": "read",
                "context": "semantic search for attention",
            }
        }


class MemoryVersion(BaseModel):
    """Versioned memory content for audit trail."""

    version_id: str = Field(default_factory=lambda: str(uuid4()), description="Version ID")
    memory_id: str = Field(..., description="Memory ID")
    version_number: int = Field(..., description="Version number")
    content_json: str = Field(..., description="Serialized content")
    change_summary: str = Field("", description="Summary of changes")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "mem_123",
                "version_number": 2,
                "change_summary": "Updated confidence score",
            }
        }


class MemoryStats(BaseModel):
    """Statistics about memory storage."""

    total_memories: int = Field(..., description="Total number of memories")
    memories_by_type: dict[str, int] = Field(default_factory=dict, description="Count by type")
    total_relationships: int = Field(..., description="Total relationships")
    archived_count: int = Field(..., description="Number of archived memories")
    avg_confidence: float = Field(..., description="Average confidence score")
    most_accessed: list[str] = Field(default_factory=list, description="Most accessed memory IDs")
    recent_memories: list[str] = Field(default_factory=list, description="Recent memory IDs")
    storage_size_mb: float = Field(0.0, description="Storage size in MB")

    class Config:
        json_schema_extra = {
            "example": {
                "total_memories": 1000,
                "memories_by_type": {"paper": 500, "repository": 200},
                "total_relationships": 1500,
            }
        }


class MemoryFilters(BaseModel):
    """Filters for memory queries."""

    memory_types: list[MemoryType] | None = Field(None, description="Filter by types")
    tags: list[str] | None = Field(None, description="Filter by tags")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence")
    date_start: datetime | None = Field(None, description="Start date filter")
    date_end: datetime | None = Field(None, description="End date filter")
    exclude_archived: bool = Field(True, description="Exclude archived memories")
    ids: list[str] | None = Field(None, description="Filter by specific IDs")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_types": ["paper", "insight"],
                "min_confidence": 0.8,
                "exclude_archived": True,
            }
        }


class MemoryResult(BaseModel):
    """Result from memory retrieval."""

    memory: MemoryBase | dict = Field(..., description="Retrieved memory")
    score: float = Field(1.0, description="Relevance score")
    match_type: str = Field("exact", description="Type of match (exact, semantic, graph)")
    relationship_path: list[str] | None = Field(None, description="Path in graph if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "memory": {"memory_id": "mem_123", "memory_type": "paper"},
                "score": 0.95,
                "match_type": "semantic",
            }
        }


class MemoryRecommendation(BaseModel):
    """Memory recommendation."""

    memory_id: str = Field(..., description="Recommended memory ID")
    reason: str = Field(..., description="Why this is recommended")
    relevance_score: float = Field(..., description="Relevance score")
    memory_type: MemoryType = Field(..., description="Type of memory")
    preview: str = Field(..., description="Short preview text")

    class Config:
        json_schema_extra = {
            "example": {
                "memory_id": "mem_123",
                "reason": "Similar architecture to your current project",
                "relevance_score": 0.92,
            }
        }
