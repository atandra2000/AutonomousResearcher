"""Memory query processing for semantic retrieval."""

from pydantic import BaseModel, Field

from research_engineer.models.memory import MemoryFilters, MemoryResult, MemoryType
from research_engineer.tools.base import Tool, ToolError


class SemanticQuery(BaseModel):
    """Query for memory search."""

    query_text: str = Field(..., description="Search query text")
    memory_types: list[MemoryType] | None = Field(None, description="Filter by memory types")
    tags: list[str] | None = Field(None, description="Filter by tags")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    include_archived: bool = Field(False, description="Include archived memories")

    def to_filters(self) -> MemoryFilters:
        """Convert to MemoryFilters."""
        return MemoryFilters(
            memory_types=self.memory_types,
            tags=self.tags,
            min_confidence=self.min_confidence,
            exclude_archived=not self.include_archived
        )


class QueryIntent(BaseModel):
    """Classified query intent."""

    intent_type: str = Field(..., description="Type of intent")
    confidence: float = Field(..., description="Intent confidence")
    extracted_entities: dict = Field(default_factory=dict, description="Extracted entities")


class QueryProcessor:
    """Process and route memory queries."""

    INTENT_PATTERNS = {
        "find_paper": ["paper", "arxiv", "publication", "research"],
        "find_code": ["code", "implementation", "repository", "repo"],
        "find_plan": ["plan", "experiment", "validation"],
        "find_patch": ["patch", "change", "modification"],
        "find_decision": ["decision", "architecture", "choice"],
        "find_insight": ["insight", "pattern", "finding"],
        "find_failure": ["failure", "error", "problem"],
        "find_success": ["success", "worked", "effective"],
        "find_related": ["related", "similar", "connected"],
    }

    def __init__(self):
        self.intent_cache: dict[str, QueryIntent] = {}

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent based on keywords."""
        query_lower = query.lower()

        best_intent = "general_search"
        best_score = 0.0
        entities = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > best_score:
                best_intent = intent
                best_score = score

        if "arxiv" in query_lower or any(c.isdigit() for c in query_lower.split()):
            entities["may_be_arxiv"] = True

        if "./" in query_lower or "/" in query_lower:
            entities["may_be_path"] = True

        return QueryIntent(
            intent_type=best_intent,
            confidence=min(1.0, best_score / len(patterns)),
            extracted_entities=entities
        )

    async def process(self, query: SemanticQuery, storage=None, vector_store=None, graph=None) -> list[MemoryResult]:
        """Process query and return results."""
        intent = self.classify_intent(query.query_text)

        filters = query.to_filters()

        if vector_store and query.query_text.strip():
            vector_results = await self._vector_search(query, vector_store, filters)
        else:
            vector_results = []

        if graph and vector_results:
            graph_results = await self._graph_expand(vector_results, graph, depth=1)
        else:
            graph_results = []

        if storage:
            direct_results = await self._direct_search(storage, filters, query.max_results)
        else:
            direct_results = []

        all_results = vector_results + graph_results + direct_results

        ranked = self._rerank(all_results, query, intent)

        return ranked[:query.max_results]

    async def _vector_search(self, query: SemanticQuery, vector_store, filters: MemoryFilters) -> list[MemoryResult]:
        """Perform vector similarity search."""
        try:
            from research_engineer.tools.embedding_strategy import EmbeddingStrategy, EmbeddingConfig

            strategy = EmbeddingStrategy(EmbeddingConfig())

            embedding = await strategy.embed(query.query_text, MemoryType.RESEARCH_INSIGHT)

            search_filters = None
            if filters.memory_types:
                search_filters = {"memory_type": {"$in": [t.value for t in filters.memory_types]}}

            results = await vector_store.search(
                query_embedding=embedding,
                limit=query.max_results * 2,
                filters=search_filters
            )

            memory_results = []
            for result in results:
                memory_results.append(
                    MemoryResult(
                        memory={"memory_id": result.id, "memory_type": result.metadata.get("memory_type", "unknown")},
                        score=result.score,
                        match_type="vector"
                    )
                )

            return memory_results

        except ImportError:
            return []
        except Exception as e:
            raise ToolError(f"Vector search failed: {e}", None, e)

    async def _graph_expand(self, results: list[MemoryResult], graph, depth: int = 1) -> list[MemoryResult]:
        """Expand results via graph traversal."""
        try:
            if not results:
                return []

            seed_ids = [r.memory["memory_id"] for r in results[:3]]

            expanded = []
            for memory_id in seed_ids:
                neighbors = await graph.get_neighbors(memory_id)
                for neighbor_id in neighbors[:depth * 2]:
                    expanded.append(
                        MemoryResult(
                            memory={"memory_id": neighbor_id},
                            score=0.5,
                            match_type="graph",
                            relationship_path=[memory_id, neighbor_id]
                        )
                    )

            return expanded

        except Exception:
            return []

    async def _direct_search(self, storage, filters: MemoryFilters, limit: int) -> list[MemoryResult]:
        """Perform direct database search."""
        try:
            from research_engineer.tools.memory_storage import MemoryQueryInput

            input = MemoryQueryInput(
                filters=filters,
                limit=limit
            )

            output = await storage.execute(input)

            results = []
            for memory_data in output.memories:
                results.append(
                    MemoryResult(
                        memory=memory_data["content_json"],
                        score=1.0,
                        match_type="direct"
                    )
                )

            return results

        except Exception:
            return []

    def _rerank(self, results: list[MemoryResult], query: SemanticQuery, intent: QueryIntent) -> list[MemoryResult]:
        """Re-rank results by combined score."""
        if not results:
            return []

        for result in results:
            score = result.score

            if result.match_type == "vector":
                score *= 1.0
            elif result.match_type == "graph":
                score *= 0.7
            elif result.match_type == "direct":
                score *= 0.9

            if intent.intent_type != "general_search":
                memory_type = result.memory.get("memory_type", "")
                if self._intent_matches_type(intent.intent_type, memory_type):
                    score *= 1.2

            result.score = min(1.0, score)

        results.sort(key=lambda x: x.score, reverse=True)

        seen = set()
        unique_results = []
        for result in results:
            if result.memory["memory_id"] not in seen:
                seen.add(result.memory["memory_id"])
                unique_results.append(result)

        return unique_results

    def _intent_matches_type(self, intent: str, memory_type: str) -> bool:
        """Check if intent matches memory type."""
        mapping = {
            "find_paper": "paper",
            "find_code": "repository",
            "find_plan": "experiment_plan",
            "find_patch": "patch",
            "find_decision": "architecture_decision",
            "find_insight": "research_insight",
            "find_failure": "failed_approach",
            "find_success": "successful_approach",
        }

        return mapping.get(intent) == memory_type
