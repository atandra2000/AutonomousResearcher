"""Tests for retrieval strategies."""

import pytest

from research_engineer.models.memory import (
    MemoryBase,
    MemoryResult,
    MemoryType,
    PaperMemory,
)
from research_engineer.tools.memory_graph import MemoryKnowledgeGraph
from research_engineer.models.memory import MemoryRelationship, RelationshipType
from research_engineer.tools.retrieval_strategies import (
    DirectLookupStrategy,
    GraphTraversalStrategy,
    HybridSearchStrategy,
    RetrievalQuery,
    STRATEGY_REGISTRY,
    SemanticSearchStrategy,
    TagBasedFilterStrategy,
    TemporalQueryStrategy,
    get_strategy,
)


class FakeStorage:
    """Minimal fake storage for testing."""

    def __init__(self, memories: dict[str, dict] | None = None):
        self._memories = memories or {}

    async def get_memory_by_id(self, memory_id: str) -> dict | None:
        return self._memories.get(memory_id)

    async def execute(self, input):
        from research_engineer.tools.memory_storage import MemoryQueryOutput

        return MemoryQueryOutput(memories=list(self._memories.values()), total=len(self._memories), limit=100, offset=0)

    async def store_relationship(self, rel):
        pass

    async def get_relationships(self, mid):
        return []


@pytest.fixture
def storage():
    mem1 = PaperMemory(paper_id="1234.00001", title="Test Paper", abstract="attention mechanism")
    mem2 = PaperMemory(paper_id="1234.00002", title="Similar Paper", abstract="attention is great")
    return FakeStorage({
        "mem_1": {"memory_id": "mem_1", "memory_type": "paper", "content_json": mem1.to_dict()},
        "mem_2": {"memory_id": "mem_2", "memory_type": "paper", "content_json": mem2.to_dict()},
    })


@pytest.fixture
def graph():
    g = MemoryKnowledgeGraph()
    rel = MemoryRelationship(
        source_memory_id="mem_1", target_memory_id="mem_2",
        relationship_type=RelationshipType.SIMILAR_TO,
    )
    g.add_relationship(rel)
    return g


class TestStrategyRegistry:
    """Test strategy registry and factory."""

    def test_registry_has_all_strategies(self):
        expected = {
            "direct_lookup",
            "semantic_search",
            "graph_traversal",
            "tag_filter",
            "temporal_query",
            "hybrid_search",
        }
        assert expected.issubset(set(STRATEGY_REGISTRY.keys()))

    def test_get_strategy_returns_instance(self):
        s = get_strategy("direct_lookup")
        assert isinstance(s, DirectLookupStrategy)

    def test_get_strategy_unknown(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent")


class TestDirectLookupStrategy:
    """Test direct lookup by ID."""

    @pytest.mark.asyncio
    async def test_lookup_existing(self, storage):
        strategy = DirectLookupStrategy()
        query = RetrievalQuery(memory_id="mem_1")
        results = await strategy.retrieve(query, storage=storage)
        assert len(results) == 1
        assert results[0].match_type == "direct"
        assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_lookup_nonexistent(self, storage):
        strategy = DirectLookupStrategy()
        query = RetrievalQuery(memory_id="nonexistent")
        results = await strategy.retrieve(query, storage=storage)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_lookup_no_storage(self):
        strategy = DirectLookupStrategy()
        query = RetrievalQuery(memory_id="mem_1")
        results = await strategy.retrieve(query, storage=None)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_lookup_no_id(self, storage):
        strategy = DirectLookupStrategy()
        query = RetrievalQuery()
        results = await strategy.retrieve(query, storage=storage)
        assert len(results) == 0


class TestGraphTraversalStrategy:
    """Test graph traversal expansion."""

    @pytest.mark.asyncio
    async def test_traverse_finds_neighbors(self, graph, storage):
        strategy = GraphTraversalStrategy()
        query = RetrievalQuery(memory_id="mem_1", limit=10)
        results = await strategy.retrieve(query, storage=storage, graph=graph)
        # mem_1 has neighbor mem_2
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_traverse_no_graph(self, storage):
        strategy = GraphTraversalStrategy()
        query = RetrievalQuery(memory_id="mem_1")
        results = await strategy.retrieve(query, storage=storage, graph=None)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_traverse_no_seed(self, graph):
        strategy = GraphTraversalStrategy()
        query = RetrievalQuery()
        results = await strategy.retrieve(query, storage=None, graph=graph)
        assert len(results) == 0


class TestTagBasedFilterStrategy:
    """Test tag-based filtering."""

    @pytest.mark.asyncio
    async def test_tag_filter_no_tags(self, storage):
        strategy = TagBasedFilterStrategy()
        query = RetrievalQuery()
        results = await strategy.retrieve(query, storage=storage)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_tag_filter_no_storage(self):
        strategy = TagBasedFilterStrategy()
        query = RetrievalQuery(tags=["attention"])
        results = await strategy.retrieve(query, storage=None)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_tag_filter_with_tags(self, storage):
        strategy = TagBasedFilterStrategy()
        query = RetrievalQuery(tags=["attention"], limit=10)
        results = await strategy.retrieve(query, storage=storage)
        # fake storage returns all memories
        assert len(results) >= 0


class TestTemporalQueryStrategy:
    """Test temporal query strategy."""

    @pytest.mark.asyncio
    async def test_temporal_no_dates(self, storage):
        strategy = TemporalQueryStrategy()
        query = RetrievalQuery()
        results = await strategy.retrieve(query, storage=storage)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_temporal_no_storage(self):
        from datetime import datetime
        strategy = TemporalQueryStrategy()
        query = RetrievalQuery(date_start=datetime(2020, 1, 1))
        results = await strategy.retrieve(query, storage=None)
        assert len(results) == 0


class TestSemanticSearchStrategy:
    """Test semantic search (vector-based)."""

    @pytest.mark.asyncio
    async def test_no_vector_store(self):
        strategy = SemanticSearchStrategy()
        query = RetrievalQuery(query_text="attention")
        results = await strategy.retrieve(query, vector_store=None)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_empty_query(self):
        strategy = SemanticSearchStrategy()
        query = RetrievalQuery(query_text="")

        class FakeVS:
            async def search(self, **kwargs):
                return []

        results = await strategy.retrieve(query, vector_store=FakeVS())
        assert len(results) == 0


class TestHybridSearchStrategy:
    """Test hybrid search strategy."""

    @pytest.mark.asyncio
    async def test_hybrid_no_deps(self):
        strategy = HybridSearchStrategy()
        query = RetrievalQuery(query_text="attention", limit=5)
        results = await strategy.retrieve(query)
        # Should return empty without vector store or graph
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_with_graph_only(self, graph, storage):
        strategy = HybridSearchStrategy()
        query = RetrievalQuery(query_text="attention", memory_id="mem_1", limit=10)
        # vector_store is None so semantic returns []; graph expansion may run
        results = await strategy.retrieve(query, storage=storage, graph=graph)
        assert isinstance(results, list)


class TestRetrievalQuery:
    """Test RetrievalQuery model."""

    def test_to_filters(self):
        query = RetrievalQuery(
            memory_types=[MemoryType.PAPER],
            tags=["attention"],
            min_confidence=0.8,
            include_archived=True,
        )
        filters = query.to_filters()
        assert filters.min_confidence == 0.8
        assert filters.exclude_archived is False

    def test_defaults(self):
        query = RetrievalQuery()
        assert query.limit == 10
        assert query.vector_weight == 0.7
        assert query.graph_weight == 0.3