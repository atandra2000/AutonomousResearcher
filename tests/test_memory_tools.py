"""Tests for formal memory tools (MemoryQueryTool, MemoryWriteTool, MemoryGraphTool, MemoryRecallTool)."""

import pytest

from research_engineer.models.memory import (
    MemoryBase,
    MemoryRelationship,
    MemoryType,
    PaperMemory,
    RelationshipType,
)
from research_engineer.tools.memory_graph import MemoryKnowledgeGraph
from research_engineer.tools.memory_storage import MemoryStorageTool
from research_engineer.tools.memory_tools import (
    MemoryGraphTool,
    MemoryGraphToolInput,
    MemoryGraphToolOutput,
    MemoryQueryTool,
    MemoryQueryToolInput,
    MemoryQueryToolOutput,
    MemoryRecallTool,
    MemoryRecallToolInput,
    MemoryRecallToolOutput,
    MemoryWriteTool,
    MemoryWriteToolInput,
    MemoryWriteToolOutput,
)
from research_engineer.tools.relationship_detector import RelationshipDetector


@pytest.fixture
def storage(tmp_path):
    return MemoryStorageTool(db_path=str(tmp_path / "test_memory.db"))


@pytest.fixture
def paper_memory():
    return PaperMemory(
        paper_id="2401.00001",
        title="Test Paper",
        abstract="Test abstract about attention",
    )


class TestMemoryQueryTool:
    """Tests for MemoryQueryTool."""

    @pytest.mark.asyncio
    async def test_validate_valid(self, storage):
        tool = MemoryQueryTool(storage=storage)
        inp = MemoryQueryToolInput(query_text="test")
        assert await tool.validate(inp) is True

    @pytest.mark.asyncio
    async def test_validate_invalid(self, storage):
        tool = MemoryQueryTool(storage=storage)
        inp = MemoryQueryToolInput(query_text="", strategy="")
        assert await tool.validate(inp) is False

    @pytest.mark.asyncio
    async def test_execute_unknown_strategy(self, storage):
        tool = MemoryQueryTool(storage=storage)
        inp = MemoryQueryToolInput(query_text="test", strategy="nonexistent")
        with pytest.raises(Exception):
            await tool.execute(inp)

    @pytest.mark.asyncio
    async def test_execute_direct_lookup(self, storage, paper_memory):
        # Store a memory first
        from research_engineer.tools.memory_storage import MemoryStorageInput
        await storage.execute(MemoryStorageInput(memory=paper_memory, operation="store"))

        tool = MemoryQueryTool(storage=storage)
        inp = MemoryQueryToolInput(
            query_text="", strategy="direct_lookup", memory_id=paper_memory.memory_id
        )
        out = await tool.execute(inp)
        assert isinstance(out, MemoryQueryToolOutput)
        assert out.strategy == "direct_lookup"

    @pytest.mark.asyncio
    async def test_execute_tag_filter(self, storage, paper_memory):
        from research_engineer.tools.memory_storage import MemoryStorageInput
        await storage.execute(MemoryStorageInput(memory=paper_memory, operation="store"))

        tool = MemoryQueryTool(storage=storage)
        inp = MemoryQueryToolInput(query_text="test", strategy="tag_filter", tags=["attention"])
        out = await tool.execute(inp)
        assert isinstance(out, MemoryQueryToolOutput)


class TestMemoryWriteTool:
    """Tests for MemoryWriteTool."""

    @pytest.mark.asyncio
    async def test_validate(self, storage, paper_memory):
        tool = MemoryWriteTool(storage=storage)
        assert await tool.validate(MemoryWriteToolInput(memory=paper_memory)) is True

    @pytest.mark.asyncio
    async def test_execute_stores(self, storage, paper_memory):
        tool = MemoryWriteTool(storage=storage)
        inp = MemoryWriteToolInput(memory=paper_memory, detect_relationships=False)
        out = await tool.execute(inp)
        assert isinstance(out, MemoryWriteToolOutput)
        assert out.success is True
        assert out.memory_id == paper_memory.memory_id

    @pytest.mark.asyncio
    async def test_execute_no_storage(self, paper_memory):
        tool = MemoryWriteTool(storage=None)
        inp = MemoryWriteToolInput(memory=paper_memory)
        with pytest.raises(Exception):
            await tool.execute(inp)

    @pytest.mark.asyncio
    async def test_execute_with_relationship_detection(self, storage, paper_memory):
        detector = RelationshipDetector(storage=storage)
        tool = MemoryWriteTool(storage=storage, relationship_detector=detector)
        inp = MemoryWriteToolInput(memory=paper_memory, detect_relationships=True)
        out = await tool.execute(inp)
        assert out.success is True
        # relationships_detected may be 0 if no candidates exist
        assert isinstance(out.relationships_detected, int)


class TestMemoryGraphTool:
    """Tests for MemoryGraphTool."""

    @pytest.fixture
    def graph(self):
        g = MemoryKnowledgeGraph()
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        g.add_relationship(rel)
        return g

    @pytest.mark.asyncio
    async def test_validate(self, graph):
        tool = MemoryGraphTool(graph=graph)
        assert await tool.validate(MemoryGraphToolInput(operation="stats")) is True

    @pytest.mark.asyncio
    async def test_stats(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(MemoryGraphToolInput(operation="stats"))
        assert out.success is True
        assert out.stats is not None
        assert out.stats.node_count == 2
        assert out.stats.edge_count == 1

    @pytest.mark.asyncio
    async def test_neighbors(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(MemoryGraphToolInput(operation="neighbors", memory_id="a"))
        assert out.success is True
        assert "b" in out.neighbors

    @pytest.mark.asyncio
    async def test_neighbors_missing_id(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(MemoryGraphToolInput(operation="neighbors"))
        assert out.success is False

    @pytest.mark.asyncio
    async def test_paths(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(
            MemoryGraphToolInput(operation="paths", source_id="a", target_id="b")
        )
        assert out.success is True
        assert len(out.paths) >= 1

    @pytest.mark.asyncio
    async def test_traverse(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(
            MemoryGraphToolInput(operation="traverse", source_id="a", max_depth=2)
        )
        assert out.success is True
        assert "a" in out.neighbors
        assert "b" in out.neighbors

    @pytest.mark.asyncio
    async def test_unknown_operation(self, graph):
        tool = MemoryGraphTool(graph=graph)
        out = await tool.execute(MemoryGraphToolInput(operation="bogus"))
        assert out.success is False


class TestMemoryRecallTool:
    """Tests for MemoryRecallTool."""

    @pytest.mark.asyncio
    async def test_validate(self, storage):
        tool = MemoryRecallTool(storage=storage)
        assert await tool.validate(MemoryRecallToolInput(current_task="test")) is True

    @pytest.mark.asyncio
    async def test_execute_no_results(self, storage):
        tool = MemoryRecallTool(storage=storage)
        out = await tool.execute(MemoryRecallToolInput(current_task="attention"))
        assert isinstance(out, MemoryRecallToolOutput)
        assert isinstance(out.context_memories, list)

    @pytest.mark.asyncio
    async def test_execute_returns_summary(self, storage):
        tool = MemoryRecallTool(storage=storage)
        out = await tool.execute(MemoryRecallToolInput(current_task="test", limit=3))
        assert isinstance(out.summary, str)