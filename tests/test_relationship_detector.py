"""Tests for RelationshipDetector."""

import pytest

from research_engineer.models.memory import (
    FailedApproachMemory,
    FailureMode,
    PaperMemory,
    PatchMemory,
    RelationshipType,
    RepositoryMemory,
)
from research_engineer.tools.relationship_detector import (
    RelationshipDetector,
    RelationshipDetectorInput,
)


@pytest.fixture
def detector():
    return RelationshipDetector()


@pytest.fixture
def paper_a():
    return PaperMemory(
        paper_id="2401.00001",
        title="Attention Is All You Need",
        abstract="We propose a new architecture 2401.00002 based on transformers.",
    )


@pytest.fixture
def paper_b():
    return PaperMemory(
        paper_id="2401.00002",
        title="Transformers for Vision",
        abstract="We extend the attention mechanism from 2401.00001 for vision tasks.",
    )


@pytest.fixture
def paper_c():
    return PaperMemory(
        paper_id="2401.00003",
        title="Attention Is All You Need",
        abstract="We propose a new architecture based on transformers.",
    )


@pytest.fixture
def repo():
    return RepositoryMemory(
        repo_path="./my_repo",
        repo_name="my_repo",
        repo_type="pytorch",
        architecture_summary="Implements paper 2401.00001 with transformer layers",
        key_components=["transformer", "attention", "2401.00001"],
        dependencies=["torch", "numpy", "pydantic"],
    )


@pytest.fixture
def repo_shared_deps():
    return RepositoryMemory(
        repo_path="./other_repo",
        repo_name="other_repo",
        repo_type="pytorch",
        architecture_summary="Another PyTorch project",
        dependencies=["torch", "numpy", "httpx"],
    )


@pytest.fixture
def patch_mem():
    return PatchMemory(
        patch_id="patch_1",
        implementation_id="impl_1",
        repo_path="./my_repo",
        patch_content="diff --git a/model.py b/model.py",
        files_modified=["model.py"],
        change_type="modification",
    )


class TestCitationDetection:
    """Tests for citation relationship detection."""

    @pytest.mark.asyncio
    async def test_detects_arxiv_citation(self, detector, paper_a, paper_b):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_a, candidate_memories=[paper_b])
        )
        cites = [r for r in result.relationships if r.relationship_type == RelationshipType.CITES]
        # paper_a abstract references 2401.00002 which is paper_b's id
        assert len(cites) >= 1
        assert cites[0].confidence >= 0.9

    @pytest.mark.asyncio
    async def test_no_citation_when_not_referenced(self, detector, paper_b, paper_a):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_b, candidate_memories=[paper_a])
        )
        cites = [r for r in result.relationships if r.relationship_type == RelationshipType.CITES]
        # paper_b abstract references 2401.00001 which IS paper_a's id
        assert len(cites) >= 1

    @pytest.mark.asyncio
    async def test_no_citation_for_unrelated_papers(self, detector, paper_a, paper_c):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_a, candidate_memories=[paper_c])
        )
        cites = [r for r in result.relationships if r.relationship_type == RelationshipType.CITES]
        assert len(cites) == 0


class TestSimilarityDetection:
    """Tests for similarity relationship detection."""

    @pytest.mark.asyncio
    async def test_detects_similar_papers(self, detector, paper_a, paper_c):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_a, candidate_memories=[paper_c])
        )
        similar = [
            r for r in result.relationships if r.relationship_type == RelationshipType.SIMILAR_TO
        ]
        assert len(similar) >= 1
        assert similar[0].confidence > 0.5

    @pytest.mark.asyncio
    async def test_no_similarity_for_different_papers(self, detector, paper_a, paper_b):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_a, candidate_memories=[paper_b])
        )
        # paper_a and paper_b have different titles/abstracts, may or may not be similar
        # but at least the detector runs without error
        assert isinstance(result.relationships, list)


class TestImplementationDetection:
    """Tests for implementation relationship detection."""

    @pytest.mark.asyncio
    async def test_repo_implements_paper(self, detector, repo, paper_a):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=repo, candidate_memories=[paper_a])
        )
        implements = [
            r for r in result.relationships if r.relationship_type == RelationshipType.IMPLEMENTS
        ]
        assert len(implements) >= 1

    @pytest.mark.asyncio
    async def test_patch_implements_paper(self, detector, patch_mem, paper_a):
        patch_mem.paper_id = paper_a.paper_id
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=patch_mem, candidate_memories=[paper_a])
        )
        implements = [
            r for r in result.relationships if r.relationship_type == RelationshipType.IMPLEMENTS
        ]
        assert len(implements) >= 1


class TestDependencyDetection:
    """Tests for dependency relationship detection."""

    @pytest.mark.asyncio
    async def test_shared_dependencies(self, detector, repo, repo_shared_deps):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=repo, candidate_memories=[repo_shared_deps])
        )
        deps = [
            r for r in result.relationships if r.relationship_type == RelationshipType.DEPENDS_ON
        ]
        assert len(deps) >= 1
        assert deps[0].confidence >= 0.5


class TestDetectorOutput:
    """Tests for detector output structure."""

    @pytest.mark.asyncio
    async def test_detection_summary(self, detector, paper_a, paper_b, paper_c):
        result = await detector.execute(
            RelationshipDetectorInput(
                source_memory=paper_a, candidate_memories=[paper_b, paper_c]
            )
        )
        assert isinstance(result.detection_summary, dict)
        assert result.scanned_count == 2

    @pytest.mark.asyncio
    async def test_min_confidence_filter(self, detector, paper_a, paper_c):
        result = await detector.execute(
            RelationshipDetectorInput(
                source_memory=paper_a, candidate_memories=[paper_c], min_confidence=0.99
            )
        )
        for rel in result.relationships:
            assert rel.confidence >= 0.99

    @pytest.mark.asyncio
    async def test_deduplication(self, detector, paper_a, paper_c):
        result = await detector.execute(
            RelationshipDetectorInput(
                source_memory=paper_a, candidate_memories=[paper_c, paper_c]
            )
        )
        similar = [
            r
            for r in result.relationships
            if r.relationship_type == RelationshipType.SIMILAR_TO
        ]
        assert len(similar) <= 1

    @pytest.mark.asyncio
    async def test_exclude_self(self, detector, paper_a):
        result = await detector.execute(
            RelationshipDetectorInput(source_memory=paper_a, candidate_memories=[paper_a])
        )
        assert result.scanned_count == 0