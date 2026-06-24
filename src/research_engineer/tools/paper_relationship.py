"""Paper Relationship Tool for Phase 6 - Literature Intelligence.

Detects relationships between papers: citations, extensions, similarities,
and contradictions. Specialized for cross-paper discovery.
"""

import re
from difflib import SequenceMatcher

from research_engineer.models.literature import (
    PaperRelationship,
    PaperRelationshipInput,
    PaperRelationshipOutput,
    PaperRelationType,
    PaperSummary,
)
from research_engineer.tools.base import Tool, ToolError

ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;\"')\]]+", re.IGNORECASE)

EXTENSION_KEYWORDS = [
    "extends", "builds upon", "builds on", "improves",
    "generalizes", "enhances", "augments", "expands",
]

CONTRADICTION_KEYWORDS = [
    "contradicts", "disproves", "challenges", "refutes",
    "unlike", "in contrast to", "contrary to", "as opposed to",
]

REPRODUCTION_KEYWORDS = [
    "reproduce", "replicate", "reproduction", "replication",
]

SIMILARITY_THRESHOLD = 0.5


class PaperRelationshipTool(Tool[PaperRelationshipInput, PaperRelationshipOutput]):
    """Detect relationships between papers."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: PaperRelationshipInput) -> bool:
        return len(input.papers) >= 1

    async def execute(self, input: PaperRelationshipInput) -> PaperRelationshipOutput:
        try:
            relationships: list[PaperRelationship] = []

            for i, source in enumerate(input.papers):
                for target in input.papers[i + 1:]:
                    if input.detect_citations:
                        relationships.extend(
                            self._detect_citations(source, target)
                        )
                        relationships.extend(
                            self._detect_citations(target, source)
                        )
                    if input.detect_extensions:
                        relationships.extend(
                            self._detect_extensions(source, target)
                        )
                        relationships.extend(
                            self._detect_extensions(target, source)
                        )
                    if input.detect_similarities:
                        rel = self._detect_similarity(source, target)
                        if rel:
                            relationships.append(rel)
                    if input.detect_contradictions:
                        relationships.extend(
                            self._detect_contradictions(source, target)
                        )
                        relationships.extend(
                            self._detect_contradictions(target, source)
                        )

            filtered = [
                r for r in relationships if r.confidence >= input.min_confidence
            ]
            filtered = self._deduplicate(filtered)

            graph = self._build_graph(filtered, input.papers)
            summary = self._build_summary(filtered)

            return PaperRelationshipOutput(
                relationships=filtered,
                relationship_graph=graph,
                summary=summary,
            )
        except Exception as e:
            raise ToolError(f"Relationship detection failed: {e}", input, e)

    def _detect_citations(
        self, source: PaperSummary, target: PaperSummary
    ) -> list[PaperRelationship]:
        """Detect if source cites target via arXiv ID / DOI references."""
        rels: list[PaperRelationship] = []
        text = f"{source.title} {source.abstract} {' '.join(source.key_contributions)}"

        target_ids = {target.paper_id}
        if target.doi:
            target_ids.add(target.doi)

        for ref in ARXIV_ID_PATTERN.findall(text):
            if ref in target_ids:
                rels.append(
                    PaperRelationship(
                        source_paper_id=source.paper_id,
                        target_paper_id=target.paper_id,
                        relationship_type=PaperRelationType.CITES,
                        confidence=0.95,
                        evidence=f"arXiv ID {ref} referenced in text",
                    )
                )
        for ref in DOI_PATTERN.findall(text):
            if ref in target_ids:
                rels.append(
                    PaperRelationship(
                        source_paper_id=source.paper_id,
                        target_paper_id=target.paper_id,
                        relationship_type=PaperRelationType.CITES,
                        confidence=0.9,
                        evidence=f"DOI {ref} referenced in text",
                    )
                )
        return rels

    def _detect_extensions(
        self, source: PaperSummary, target: PaperSummary
    ) -> list[PaperRelationship]:
        """Detect if source extends target via keyword patterns."""
        rels: list[PaperRelationship] = []
        text = f"{source.title} {source.abstract}".lower()

        for kw in EXTENSION_KEYWORDS:
            if kw in text and target.paper_id.lower() in text:
                rels.append(
                    PaperRelationship(
                        source_paper_id=source.paper_id,
                        target_paper_id=target.paper_id,
                        relationship_type=PaperRelationType.EXTENDS,
                        confidence=0.8,
                        evidence=f"Extension keyword '{kw}' with target reference",
                    )
                )
                break

        for kw in EXTENSION_KEYWORDS:
            if kw in text:
                shared = self._topic_overlap(source, target)
                if shared > 0.3:
                    rels.append(
                        PaperRelationship(
                            source_paper_id=source.paper_id,
                            target_paper_id=target.paper_id,
                            relationship_type=PaperRelationType.BUILDS_ON,
                            confidence=0.65,
                            evidence=f"Extension keyword '{kw}' + topic overlap {shared:.2f}",
                        )
                    )
                    break

        for kw in EXTENSION_KEYWORDS:
            if "improve" in kw and kw in text:
                rels.append(
                    PaperRelationship(
                        source_paper_id=source.paper_id,
                        target_paper_id=target.paper_id,
                        relationship_type=PaperRelationType.IMPROVES_UPON,
                        confidence=0.7,
                        evidence=f"Improvement keyword '{kw}' detected",
                    )
                )
                break

        return rels

    def _detect_similarity(
        self, source: PaperSummary, target: PaperSummary
    ) -> PaperRelationship | None:
        """Detect semantic similarity between papers."""
        text_a = f"{source.title} {source.abstract}".lower()
        text_b = f"{target.title} {target.abstract}".lower()
        if not text_a.strip() or not text_b.strip():
            return None

        ratio = SequenceMatcher(None, text_a, text_b).ratio()
        tokens_a = set(text_a.split())
        tokens_b = set(text_b.split())
        if not tokens_a or not tokens_b:
            return None
        overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        score = 0.5 * ratio + 0.5 * overlap

        if score >= SIMILARITY_THRESHOLD:
            return PaperRelationship(
                source_paper_id=source.paper_id,
                target_paper_id=target.paper_id,
                relationship_type=PaperRelationType.SIMILAR_TO,
                confidence=round(score, 3),
                evidence=f"Token overlap {overlap:.2f}, string ratio {ratio:.2f}",
            )
        return None

    def _detect_contradictions(
        self, source: PaperSummary, target: PaperSummary
    ) -> list[PaperRelationship]:
        """Detect contradictions via negation patterns + shared topic."""
        rels: list[PaperRelationship] = []
        text = f"{source.title} {source.abstract}".lower()

        has_contradiction = any(kw in text for kw in CONTRADICTION_KEYWORDS)
        if not has_contradiction:
            return rels

        overlap = self._topic_overlap(source, target)
        if overlap > 0.2:
            rels.append(
                PaperRelationship(
                    source_paper_id=source.paper_id,
                    target_paper_id=target.paper_id,
                    relationship_type=PaperRelationType.CONTRADICTS,
                    confidence=0.6,
                    evidence=f"Contradiction keyword + topic overlap {overlap:.2f}",
                )
            )
        return rels

    def _detect_reproductions(
        self, source: PaperSummary, target: PaperSummary
    ) -> list[PaperRelationship]:
        """Detect reproduction relationships."""
        rels: list[PaperRelationship] = []
        text = f"{source.title} {source.abstract}".lower()
        has_repro = any(kw in text for kw in REPRODUCTION_KEYWORDS)
        if has_repro:
            overlap = self._topic_overlap(source, target)
            if overlap > 0.3:
                rels.append(
                    PaperRelationship(
                        source_paper_id=source.paper_id,
                        target_paper_id=target.paper_id,
                        relationship_type=PaperRelationType.REPRODUCES,
                        confidence=0.75,
                        evidence=f"Reproduction keyword + topic overlap {overlap:.2f}",
                    )
                )
        return rels

    def _topic_overlap(self, a: PaperSummary, b: PaperSummary) -> float:
        text_a = set(f"{a.title} {a.abstract}".lower().split())
        text_b = set(f"{b.title} {b.abstract}".lower().split())
        if not text_a or not text_b:
            return 0.0
        return len(text_a & text_b) / len(text_a | text_b)

    def _deduplicate(
        self, rels: list[PaperRelationship]
    ) -> list[PaperRelationship]:
        """Deduplicate keeping highest confidence."""
        best: dict[tuple[str, str, str], PaperRelationship] = {}
        for r in rels:
            key = (
                r.source_paper_id,
                r.target_paper_id,
                r.relationship_type.value,
            )
            if key not in best or r.confidence > best[key].confidence:
                best[key] = r
        return list(best.values())

    def _build_graph(
        self,
        rels: list[PaperRelationship],
        papers: list[PaperSummary],
    ) -> dict[str, list[str]]:
        """Build adjacency list graph."""
        graph: dict[str, list[str]] = {}
        for p in papers:
            graph[p.paper_id] = []
        for r in rels:
            graph.setdefault(r.source_paper_id, []).append(r.target_paper_id)
            graph.setdefault(r.target_paper_id, []).append(r.source_paper_id)
        return graph

    def _build_summary(
        self, rels: list[PaperRelationship]
    ) -> dict[str, int]:
        """Build counts by relationship type."""
        summary: dict[str, int] = {}
        for r in rels:
            key = r.relationship_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary
