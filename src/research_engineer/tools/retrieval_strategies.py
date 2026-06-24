"""Retrieval strategies for memory search.

Each strategy implements a distinct retrieval pattern:
- DirectLookupStrategy: fetch specific memory by ID
- SemanticSearchStrategy: vector similarity search
- GraphTraversalStrategy: expand via relationship graph
- TagBasedFilterStrategy: filter memories by tags
- TemporalQueryStrategy: filter by date range
- HybridSearchStrategy: combine vector + graph with weighted scoring

All strategies follow a uniform async interface:
    async def retrieve(query: RetrievalQuery, storage, vector_store, graph) -> list[MemoryResult]
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from research_engineer.models.memory import (
    MemoryFilters,
    MemoryResult,
    MemoryType,
)
from research_engineer.tools.base import ToolError

if TYPE_CHECKING:
    from research_engineer.tools.memory_graph import MemoryKnowledgeGraph
    from research_engineer.tools.memory_storage import MemoryStorageTool
    from research_engineer.tools.vector_store import VectorStore


class RetrievalQuery(BaseModel):
    """Unified query for retrieval strategies."""

    query_text: str = Field("", description="Search text")
    memory_id: str | None = Field(None, description="Direct lookup ID")
    memory_types: list[MemoryType] | None = Field(None, description="Type filter")
    tags: list[str] | None = Field(None, description="Tag filter")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Min confidence")
    date_start: datetime | None = Field(None, description="Start date")
    date_end: datetime | None = Field(None, description="End date")
    limit: int = Field(10, ge=1, le=200, description="Max results")
    include_archived: bool = Field(False, description="Include archived memories")
    graph_depth: int = Field(2, ge=1, le=5, description="Graph traversal depth")
    vector_weight: float = Field(0.7, ge=0.0, le=1.0, description="Vector weight (hybrid)")
    graph_weight: float = Field(0.3, ge=0.0, le=1.0, description="Graph weight (hybrid)")

    def to_filters(self) -> MemoryFilters:
        return MemoryFilters(
            memory_types=self.memory_types,
            tags=self.tags,
            min_confidence=self.min_confidence,
            date_start=self.date_start,
            date_end=self.date_end,
            exclude_archived=not self.include_archived,
        )


class RetrievalStrategy:
    """Base class for retrieval strategies."""

    name: str = "base"

    async def retrieve(
        self,
        query: RetrievalQuery,
        storage: MemoryStorageTool | None = None,
        vector_store: VectorStore | None = None,
        graph: MemoryKnowledgeGraph | None = None,
    ) -> list[MemoryResult]:
        raise NotImplementedError


class DirectLookupStrategy(RetrievalStrategy):
    """Retrieve a specific memory by ID."""

    name = "direct_lookup"

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        if not query.memory_id or storage is None:
            return []
        try:
            memory = await storage.get_memory_by_id(query.memory_id)
            if not memory:
                return []
            return [
                MemoryResult(
                    memory=memory.get("content_json", memory),
                    score=1.0,
                    match_type="direct",
                )
            ]
        except Exception as e:
            raise ToolError(f"Direct lookup failed: {e}", None, e)


class SemanticSearchStrategy(RetrievalStrategy):
    """Vector similarity search using embeddings."""

    name = "semantic_search"

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        if vector_store is None or not query.query_text.strip():
            return []
        try:
            from research_engineer.tools.embedding_strategy import (
                EmbeddingConfig,
                EmbeddingStrategy,
            )

            strategy = EmbeddingStrategy(EmbeddingConfig())
            embedding = await strategy.embed(query.query_text, MemoryType.RESEARCH_INSIGHT)

            filters = None
            if query.memory_types:
                filters = {"memory_type": {"$in": [t.value for t in query.memory_types]}}

            results = await vector_store.search(
                query_embedding=embedding,
                limit=query.limit,
                filters=filters,
            )

            return [
                MemoryResult(
                    memory={"memory_id": r.id, "memory_type": r.metadata.get("memory_type", "unknown")},
                    score=r.score,
                    match_type="vector",
                )
                for r in results
            ]
        except ImportError:
            return []
        except Exception as e:
            raise ToolError(f"Semantic search failed: {e}", None, e)


class GraphTraversalStrategy(RetrievalStrategy):
    """Expand results via graph traversal from seed memories."""

    name = "graph_traversal"

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        if graph is None or not query.memory_id:
            return []
        try:
            neighbors = graph.get_neighbors(query.memory_id, limit=query.limit)
            results: list[MemoryResult] = []
            for nid in neighbors:
                if storage:
                    mem = await storage.get_memory_by_id(nid)
                    if mem:
                        results.append(
                            MemoryResult(
                                memory=mem.get("content_json", mem),
                                score=0.7,
                                match_type="graph",
                                relationship_path=[query.memory_id, nid],
                            )
                        )
                else:
                    results.append(
                        MemoryResult(
                            memory={"memory_id": nid},
                            score=0.5,
                            match_type="graph",
                            relationship_path=[query.memory_id, nid],
                        )
                    )
            return results
        except Exception as e:
            raise ToolError(f"Graph traversal failed: {e}", None, e)


class TagBasedFilterStrategy(RetrievalStrategy):
    """Filter memories by tags via storage query."""

    name = "tag_filter"

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        if storage is None or not query.tags:
            return []
        try:
            from research_engineer.tools.memory_storage import MemoryQueryInput

            filters = MemoryFilters(
                tags=query.tags,
                memory_types=query.memory_types,
                min_confidence=query.min_confidence,
                exclude_archived=not query.include_archived,
            )
            output = await storage.execute(MemoryQueryInput(filters=filters, limit=query.limit))
            return [
                MemoryResult(memory=m.get("content_json", m), score=0.8, match_type="tag")
                for m in output.memories
            ]
        except Exception as e:
            raise ToolError(f"Tag filter failed: {e}", None, e)


class TemporalQueryStrategy(RetrievalStrategy):
    """Query memories by date range using storage filters."""

    name = "temporal_query"

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        if storage is None or not (query.date_start or query.date_end):
            return []
        try:
            from research_engineer.tools.memory_storage import MemoryQueryInput

            filters = MemoryFilters(
                memory_types=query.memory_types,
                date_start=query.date_start,
                date_end=query.date_end,
                min_confidence=query.min_confidence,
                exclude_archived=not query.include_archived,
            )
            output = await storage.execute(MemoryQueryInput(filters=filters, limit=query.limit))
            return [
                MemoryResult(memory=m.get("content_json", m), score=0.6, match_type="temporal")
                for m in output.memories
            ]
        except Exception as e:
            raise ToolError(f"Temporal query failed: {e}", None, e)


class HybridSearchStrategy(RetrievalStrategy):
    """Combine vector search + graph expansion with weighted scoring."""

    name = "hybrid_search"

    def __init__(self) -> None:
        self.semantic = SemanticSearchStrategy()
        self.graph = GraphTraversalStrategy()

    async def retrieve(self, query, storage=None, vector_store=None, graph=None):
        try:
            vector_results = await self.semantic.retrieve(
                query, storage, vector_store, graph
            )

            graph_results: list[MemoryResult] = []
            if graph and vector_results:
                for seed in vector_results[:3]:
                    seed_id = seed.memory.get("memory_id") if isinstance(seed.memory, dict) else None
                    if seed_id:
                        sub_query = RetrievalQuery(
                            memory_id=seed_id,
                            limit=max(5, query.limit // 3),
                            graph_depth=query.graph_depth,
                        )
                        graph_results.extend(
                            await self.graph.retrieve(sub_query, storage, vector_store, graph)
                        )

            combined: dict[str, MemoryResult] = {}
            for r in vector_results:
                key = self._key(r)
                r.score = r.score * query.vector_weight
                combined[key] = r
            for r in graph_results:
                key = self._key(r)
                if key in combined:
                    combined[key].score += r.score * query.graph_weight
                    if combined[key].relationship_path is None and r.relationship_path:
                        combined[key].relationship_path = r.relationship_path
                else:
                    r.score = r.score * query.graph_weight
                    combined[key] = r

            ranked = sorted(combined.values(), key=lambda x: x.score, reverse=True)
            return ranked[: query.limit]
        except Exception as e:
            raise ToolError(f"Hybrid search failed: {e}", None, e)

    def _key(self, result: MemoryResult) -> str:
        if isinstance(result.memory, dict):
            return result.memory.get("memory_id", str(id(result.memory)))
        return str(id(result.memory))


STRATEGY_REGISTRY: dict[str, type[RetrievalStrategy]] = {
    "direct_lookup": DirectLookupStrategy,
    "semantic_search": SemanticSearchStrategy,
    "graph_traversal": GraphTraversalStrategy,
    "tag_filter": TagBasedFilterStrategy,
    "temporal_query": TemporalQueryStrategy,
    "hybrid_search": HybridSearchStrategy,
}


def get_strategy(name: str) -> RetrievalStrategy:
    """Get a retrieval strategy instance by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return cls()
