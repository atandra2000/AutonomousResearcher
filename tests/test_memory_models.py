"""Tests for memory models."""

import pytest
from datetime import datetime

from research_engineer.models.memory import (
    MemoryType,
    MemoryBase,
    PaperMemory,
    RepositoryMemory,
    ExperimentPlanMemory,
    PatchMemory,
    ArchitectureDecisionMemory,
    ResearchInsightMemory,
    FailedApproachMemory,
    SuccessfulApproachMemory,
    InsightType,
    FailureMode,
    RelationshipType,
    MemoryRelationship,
    MemoryFilters,
    MemoryResult,
)


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_type_values(self):
        """Test memory type enum values."""
        assert MemoryType.PAPER == "paper"
        assert MemoryType.REPOSITORY == "repository"
        assert MemoryType.EXPERIMENT_PLAN == "experiment_plan"
        assert MemoryType.PATCH == "patch"
        assert MemoryType.ARCHITECTURE_DECISION == "architecture_decision"
        assert MemoryType.RESEARCH_INSIGHT == "research_insight"
        assert MemoryType.FAILED_APPROACH == "failed_approach"
        assert MemoryType.SUCCESSFUL_APPROACH == "successful_approach"


class TestPaperMemory:
    """Test PaperMemory model."""

    def test_create_paper_memory(self):
        """Test creating a paper memory."""
        memory = PaperMemory(
            paper_id="2503.12345",
            title="Test Paper",
            abstract="This is a test abstract",
            authors=[{"name": "Test Author"}],
        )

        assert memory.memory_type == MemoryType.PAPER
        assert memory.paper_id == "2503.12345"
        assert memory.title == "Test Paper"
        assert memory.confidence_score == 1.0
        assert memory.is_archived is False
        assert memory.memory_id is not None

    def test_paper_memory_serialization(self):
        """Test paper memory serialization."""
        memory = PaperMemory(
            paper_id="2503.12345",
            title="Test Paper",
            abstract="Test abstract",
        )

        data = memory.to_dict()
        assert data["memory_type"] == "paper"
        assert data["paper_id"] == "2503.12345"
        assert "memory_id" in data


class TestRepositoryMemory:
    """Test RepositoryMemory model."""

    def test_create_repository_memory(self):
        """Test creating a repository memory."""
        memory = RepositoryMemory(
            repo_path="./test_repo",
            repo_name="test_repo",
            repo_type="pytorch",
            architecture_summary="Test architecture",
        )

        assert memory.memory_type == MemoryType.REPOSITORY
        assert memory.repo_path == "./test_repo"
        assert memory.confidence_score == 1.0


class TestExperimentPlanMemory:
    """Test ExperimentPlanMemory model."""

    def test_create_plan_memory(self):
        """Test creating a plan memory."""
        memory = ExperimentPlanMemory(
            plan_id="plan_123",
            paper_id="2503.12345",
            repo_path="./test_repo",
        )

        assert memory.memory_type == MemoryType.EXPERIMENT_PLAN
        assert memory.plan_id == "plan_123"


class TestPatchMemory:
    """Test PatchMemory model."""

    def test_create_patch_memory(self):
        """Test creating a patch memory."""
        memory = PatchMemory(
            patch_id="patch_123",
            implementation_id="impl_123",
            repo_path="./test_repo",
            patch_content="diff --git a/test.py b/test.py",
            files_modified=["test.py"],
            change_type="modification",
        )

        assert memory.memory_type == MemoryType.PATCH
        assert memory.patch_id == "patch_123"
        assert memory.files_modified == ["test.py"]


class TestArchitectureDecisionMemory:
    """Test ArchitectureDecisionMemory model."""

    def test_create_decision_memory(self):
        """Test creating a decision memory."""
        memory = ArchitectureDecisionMemory(
            context="Need to choose optimizer",
            decision="Use AdamW",
            rationale="Better convergence",
        )

        assert memory.memory_type == MemoryType.ARCHITECTURE_DECISION
        assert memory.decision == "Use AdamW"
        assert memory.confidence == "high"


class TestResearchInsightMemory:
    """Test ResearchInsightMemory model."""

    def test_create_insight_memory(self):
        """Test creating an insight memory."""
        memory = ResearchInsightMemory(
            insight_type=InsightType.PATTERN,
            domain="attention",
            description="Multi-head attention works best with...",
        )

        assert memory.memory_type == MemoryType.RESEARCH_INSIGHT
        assert memory.insight_type == InsightType.PATTERN
        assert memory.domain == "attention"


class TestFailedApproachMemory:
    """Test FailedApproachMemory model."""

    def test_create_failure_memory(self):
        """Test creating a failure memory."""
        memory = FailedApproachMemory(
            context="Training with batch size 1024",
            approach_description="Attempted gradient accumulation",
            failure_mode=FailureMode.GRADIENT_EXPLOSION,
        )

        assert memory.memory_type == MemoryType.FAILED_APPROACH
        assert memory.failure_mode == FailureMode.GRADIENT_EXPLOSION


class TestSuccessfulApproachMemory:
    """Test SuccessfulApproachMemory model."""

    def test_create_success_memory(self):
        """Test creating a success memory."""
        memory = SuccessfulApproachMemory(
            context="Achieved 95% accuracy",
            approach_description="Used mixup augmentation",
            success_metrics={"accuracy": 0.95},
        )

        assert memory.memory_type == MemoryType.SUCCESSFUL_APPROACH
        assert memory.success_metrics == {"accuracy": 0.95}


class TestMemoryRelationship:
    """Test MemoryRelationship model."""

    def test_create_relationship(self):
        """Test creating a memory relationship."""
        relationship = MemoryRelationship(
            source_memory_id="mem_123",
            target_memory_id="mem_456",
            relationship_type=RelationshipType.CITES,
        )

        assert relationship.relationship_type == RelationshipType.CITES
        assert relationship.confidence == 1.0
        assert relationship.relationship_id is not None

    def test_relationship_types(self):
        """Test relationship type enum values."""
        assert RelationshipType.CITES == "cites"
        assert RelationshipType.IMPLEMENTS == "implements"
        assert RelationshipType.SIMILAR_TO == "similar_to"
        assert RelationshipType.DERIVED_FROM == "derived_from"


class TestMemoryFilters:
    """Test MemoryFilters model."""

    def test_create_filters(self):
        """Test creating memory filters."""
        filters = MemoryFilters(
            memory_types=[MemoryType.PAPER, MemoryType.REPOSITORY],
            min_confidence=0.8,
            exclude_archived=True,
        )

        assert len(filters.memory_types) == 2
        assert filters.min_confidence == 0.8
        assert filters.exclude_archived is True


class TestMemoryResult:
    """Test MemoryResult model."""

    def test_create_result(self):
        """Test creating a memory result."""
        memory = PaperMemory(
            paper_id="2503.12345",
            title="Test Paper",
            abstract="Test",
        )

        result = MemoryResult(
            memory=memory.to_dict(),
            score=0.95,
            match_type="semantic",
        )

        assert result.score == 0.95
        assert result.match_type == "semantic"


class TestMemoryBase:
    """Test MemoryBase functionality."""

    def test_mark_accessed(self):
        """Test marking memory as accessed."""
        memory = PaperMemory(
            paper_id="2503.12345",
            title="Test",
            abstract="Test",
        )

        initial_count = memory.accessed_count
        memory.mark_accessed()

        assert memory.accessed_count == initial_count + 1
        assert memory.last_accessed_at is not None

    def test_memory_id_generation(self):
        """Test that memory IDs are unique."""
        memory1 = PaperMemory(paper_id="1", title="T1", abstract="A1")
        memory2 = PaperMemory(paper_id="2", title="T2", abstract="A2")

        assert memory1.memory_id != memory2.memory_id


class TestInsightType:
    """Test InsightType enum."""

    def test_insight_types(self):
        """Test insight type values."""
        assert InsightType.PATTERN == "pattern"
        assert InsightType.ANTI_PATTERN == "anti_pattern"
        assert InsightType.OPTIMIZATION == "optimization"
        assert InsightType.BEST_PRACTICE == "best_practice"


class TestFailureMode:
    """Test FailureMode enum."""

    def test_failure_modes(self):
        """Test failure mode values."""
        assert FailureMode.CRASH == "crash"
        assert FailureMode.DIVERGENCE == "divergence"
        assert FailureMode.GRADIENT_EXPLOSION == "gradient_explosion"
        assert FailureMode.OVERFITTING == "overfitting"
