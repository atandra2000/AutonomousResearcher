"""Automatic relationship detection between memories.

Discovers citations, semantic similarity, implementation matches, and
dependency relationships between stored memories. Rule-based (no LLM) in
keeping with Phases 1-3 constraints.
"""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from research_engineer.models.memory import (
    MemoryBase,
    MemoryRelationship,
    MemoryType,
    PaperMemory,
    PatchMemory,
    RelationshipType,
    RepositoryMemory,
)
from research_engineer.tools.base import Tool, ToolError

if TYPE_CHECKING:
    from research_engineer.tools.memory_storage import MemoryStorageTool


ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;\"')\]]+", re.IGNORECASE)
REPO_PATH_PATTERN = re.compile(r"(\.{0,2}/[\w./-]+|[A-Za-z_][\w-]*/[\w-]+)")


class RelationshipDetectorInput(BaseModel):
    """Input for relationship detection."""

    source_memory: MemoryBase = Field(..., description="Memory to find relationships for")
    candidate_memories: list[MemoryBase] = Field(
        default_factory=list,
        description="Other memories to compare against (empty = query storage)",
    )
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum relationship confidence")
    detect_citations: bool = Field(True, description="Detect citation relationships")
    detect_similarity: bool = Field(True, description="Detect semantic similarity relationships")
    detect_implementation: bool = Field(True, description="Detect implementation relationships")
    detect_dependency: bool = Field(True, description="Detect dependency relationships")


class RelationshipDetectorOutput(BaseModel):
    """Output from relationship detection."""

    relationships: list[MemoryRelationship] = Field(
        default_factory=list, description="Discovered relationships"
    )
    detection_summary: dict[str, int] = Field(
        default_factory=dict, description="Counts by relationship type"
    )
    scanned_count: int = Field(0, description="Number of candidate memories scanned")


class RelationshipDetector(Tool[RelationshipDetectorInput, RelationshipDetectorOutput]):
    """Automatically discover relationships between memories.

    Detection strategies:
    - Citations: arXiv IDs / DOIs appearing in paper text referencing other papers
    - Similarity: token-overlap + SequenceMatcher on abstracts/descriptions
    - Implementation: repo paths / paper IDs shared between patches/plans and papers
    - Dependency: shared dependency lists between repos
    """

    SIMILARITY_THRESHOLD = 0.6
    TOKEN_OVERLAP_THRESHOLD = 0.3

    def __init__(self, storage: MemoryStorageTool | None = None) -> None:
        self.storage = storage

    async def validate(self, input: RelationshipDetectorInput) -> bool:
        return bool(input.source_memory and input.source_memory.memory_id)

    async def execute(self, input: RelationshipDetectorInput) -> RelationshipDetectorOutput:
        try:
            candidates = input.candidate_memories
            if not candidates and self.storage is not None:
                candidates = await self._load_candidates(input.source_memory)

            candidates = [c for c in candidates if c.memory_id != input.source_memory.memory_id]

            relationships: list[MemoryRelationship] = []

            for candidate in candidates:
                if input.detect_citations:
                    relationships.extend(
                        self._detect_citations(input.source_memory, candidate)
                    )
                if input.detect_similarity:
                    relationships.extend(
                        self._detect_similarity(input.source_memory, candidate)
                    )
                if input.detect_implementation:
                    relationships.extend(
                        self._detect_implementation(input.source_memory, candidate)
                    )
                if input.detect_dependency:
                    relationships.extend(
                        self._detect_dependency(input.source_memory, candidate)
                    )

            filtered = [
                r for r in relationships if r.confidence >= input.min_confidence
            ]
            filtered = self._deduplicate(filtered)

            summary: dict[str, int] = {}
            for rel in filtered:
                key = rel.relationship_type.value
                summary[key] = summary.get(key, 0) + 1

            return RelationshipDetectorOutput(
                relationships=filtered,
                detection_summary=summary,
                scanned_count=len(candidates),
            )
        except Exception as e:
            raise ToolError(f"Relationship detection failed: {e}", input, e)

    async def _load_candidates(self, source: MemoryBase) -> list[MemoryBase]:
        """Load candidate memories from storage when none provided."""
        from research_engineer.tools.memory_storage import MemoryQueryInput

        try:
            output = await self.storage.execute(MemoryQueryInput(limit=200))  # type: ignore[union-attr]
            return [self._dict_to_memory(m) for m in output.memories]
        except Exception:
            return []

    def _dict_to_memory(self, data: dict) -> MemoryBase:
        """Best-effort reconstruction of a MemoryBase from stored dict."""
        content = data.get("content_json", {})
        mtype = data.get("memory_type", MemoryType.PAPER.value)
        try:
            type_enum = MemoryType(mtype)
        except ValueError:
            type_enum = MemoryType.PAPER

        common = {
            "memory_id": data.get("memory_id", ""),
            "tags": data.get("tags", []),
            "confidence_score": data.get("confidence_score", 1.0),
        }
        if type_enum == MemoryType.PAPER:
            return PaperMemory(**{**common, "paper_id": content.get("paper_id", ""), "title": content.get("title", ""), "abstract": content.get("abstract", "")})
        if type_enum == MemoryType.REPOSITORY:
            return RepositoryMemory(**{**common, "repo_path": content.get("repo_path", ""), "repo_name": content.get("repo_name", ""), "repo_type": content.get("repo_type", ""), "architecture_summary": content.get("architecture_summary", "")})
        if type_enum == MemoryType.PATCH:
            return PatchMemory(**{**common, "patch_id": content.get("patch_id", ""), "implementation_id": content.get("implementation_id", ""), "repo_path": content.get("repo_path", ""), "patch_content": content.get("patch_content", ""), "change_type": content.get("change_type", "modification")})
        return MemoryBase(**common, memory_type=type_enum)

    def _detect_citations(self, source: MemoryBase, target: MemoryBase) -> list[MemoryRelationship]:
        """Detect citation relationships by scanning paper text for arXiv IDs / DOIs."""
        rels: list[MemoryRelationship] = []
        if not isinstance(source, PaperMemory) or not isinstance(target, PaperMemory):
            return rels

        text = " ".join([source.abstract, source.title, str(source.research_summary)])
        target_ids = {target.paper_id}
        target_ids.update(DOI_PATTERN.findall(target.abstract or ""))

        for ref in ARXIV_ID_PATTERN.findall(text):
            if ref in target_ids:
                rels.append(self._make(source, target, RelationshipType.CITES, 0.95))
        for ref in DOI_PATTERN.findall(text):
            if ref in target_ids:
                rels.append(self._make(source, target, RelationshipType.CITES, 0.9))
        return rels

    def _detect_similarity(self, source: MemoryBase, target: MemoryBase) -> list[MemoryRelationship]:
        """Detect semantic similarity via token overlap and string ratio."""
        s_text = self._text_for_similarity(source)
        t_text = self._text_for_similarity(target)
        if not s_text or not t_text:
            return []

        ratio = SequenceMatcher(None, s_text.lower(), t_text.lower()).ratio()
        overlap = self._token_overlap(s_text, t_text)
        score = 0.5 * ratio + 0.5 * overlap

        if score >= self.SIMILARITY_THRESHOLD:
            return [self._make(source, target, RelationshipType.SIMILAR_TO, round(score, 3))]
        return []

    def _detect_implementation(self, source: MemoryBase, target: MemoryBase) -> list[MemoryRelationship]:
        """Detect implementation relationships (repo/paper shared references)."""
        rels: list[MemoryRelationship] = []

        if isinstance(source, RepositoryMemory) and isinstance(target, PaperMemory):
            if target.paper_id in (source.architecture_summary + " " + " ".join(source.key_components)):
                rels.append(self._make(source, target, RelationshipType.IMPLEMENTS, 0.85))

        if isinstance(source, PatchMemory) and isinstance(target, PaperMemory):
            if source.paper_id and source.paper_id == target.paper_id:
                rels.append(self._make(source, target, RelationshipType.IMPLEMENTS, 0.95))

        if isinstance(source, PatchMemory) and isinstance(target, RepositoryMemory):
            if source.repo_path and source.repo_path == target.repo_path:
                rels.append(self._make(source, target, RelationshipType.DEPENDS_ON, 0.8))

        return rels

    def _detect_dependency(self, source: MemoryBase, target: MemoryBase) -> list[MemoryRelationship]:
        """Detect shared dependency relationships between repositories."""
        if not isinstance(source, RepositoryMemory) or not isinstance(target, RepositoryMemory):
            return []
        shared = set(source.dependencies) & set(target.dependencies)
        if not shared:
            return []
        confidence = min(1.0, 0.5 + 0.1 * len(shared))
        return [self._make(source, target, RelationshipType.DEPENDS_ON, round(confidence, 2))]

    def _text_for_similarity(self, memory: MemoryBase) -> str:
        if isinstance(memory, PaperMemory):
            return f"{memory.title} {memory.abstract}"
        if isinstance(memory, RepositoryMemory):
            return f"{memory.repo_name} {memory.architecture_summary}"
        if isinstance(memory, PatchMemory):
            return memory.patch_content[:500]
        if isinstance(memory, MemoryBase):
            return " ".join(memory.tags)
        return ""

    def _token_overlap(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _make(
        self,
        source: MemoryBase,
        target: MemoryBase,
        rel_type: RelationshipType,
        confidence: float,
    ) -> MemoryRelationship:
        return MemoryRelationship(
            source_memory_id=source.memory_id,
            target_memory_id=target.memory_id,
            relationship_type=rel_type,
            confidence=confidence,
            created_at=datetime.now(),
        )

    def _deduplicate(self, rels: list[MemoryRelationship]) -> list[MemoryRelationship]:
        """Deduplicate relationships keeping highest confidence per (src, tgt, type)."""
        best: dict[tuple[str, str, str], MemoryRelationship] = {}
        for r in rels:
            key = (r.source_memory_id, r.target_memory_id, r.relationship_type.value)
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r
        return list(best.values())
