"""Formal memory tools following the Tool[Input, Output] architecture.

Provides:
- MemoryQueryTool: search memories via retrieval strategies
- MemoryWriteTool: store memories
- MemoryGraphTool: graph operations (traversal, neighbors, stats)
- MemoryRecallTool: context injection for current tasks
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from research_engineer.models.memory import (
    MemoryBase,
    MemoryResult,
    MemoryType,
)
from research_engineer.tools.base import Tool, ToolError
from research_engineer.tools.memory_graph import (
    GraphStats,
    MemoryKnowledgeGraph,
)
from research_engineer.tools.retrieval_strategies import (
    RetrievalQuery,
    RetrievalStrategy,
    get_strategy,
)

if TYPE_CHECKING:
    from research_engineer.tools.memory_storage import MemoryStorageTool
    from research_engineer.tools.vector_store import VectorStore


class MemoryQueryToolInput(BaseModel):
    """Input for MemoryQueryTool."""

    query_text: str = Field("", description="Search query text")
    strategy: str = Field("semantic_search", description="Retrieval strategy name")
    memory_id: str | None = Field(None, description="Direct lookup ID")
    memory_types: list[MemoryType] | None = Field(None, description="Type filter")
    tags: list[str] | None = Field(None, description="Tag filter")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence")
    limit: int = Field(10, ge=1, le=200, description="Maximum results")


class MemoryQueryToolOutput(BaseModel):
    """Output for MemoryQueryTool."""

    results: list[MemoryResult] = Field(default_factory=list, description="Search results")
    strategy: str = Field(..., description="Strategy used")
    total: int = Field(0, description="Total results returned")


class MemoryQueryTool(Tool[MemoryQueryToolInput, MemoryQueryToolOutput]):
    """Search memories using pluggable retrieval strategies."""

    def __init__(
        self,
        storage: MemoryStorageTool | None = None,
        vector_store: VectorStore | None = None,
        graph: MemoryKnowledgeGraph | None = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.graph = graph
        self._strategies: dict[str, RetrievalStrategy] = {}

    def _get_strategy(self, name: str) -> RetrievalStrategy:
        if name not in self._strategies:
            self._strategies[name] = get_strategy(name)
        return self._strategies[name]

    async def validate(self, input: MemoryQueryToolInput) -> bool:
        return bool(input.strategy and input.limit > 0)

    async def execute(self, input: MemoryQueryToolInput) -> MemoryQueryToolOutput:
        try:
            strategy = self._get_strategy(input.strategy)
            query = RetrievalQuery(
                query_text=input.query_text,
                memory_id=input.memory_id,
                memory_types=input.memory_types,
                tags=input.tags,
                min_confidence=input.min_confidence,
                limit=input.limit,
            )
            results = await strategy.retrieve(
                query, self.storage, self.vector_store, self.graph
            )
            return MemoryQueryToolOutput(
                results=results, strategy=input.strategy, total=len(results)
            )
        except Exception as e:
            raise ToolError(f"MemoryQueryTool failed: {e}", input, e)


class MemoryWriteToolInput(BaseModel):
    """Input for MemoryWriteTool."""

    memory: MemoryBase = Field(..., description="Memory to store")
    detect_relationships: bool = Field(True, description="Auto-detect relationships after store")


class MemoryWriteToolOutput(BaseModel):
    """Output for MemoryWriteTool."""

    memory_id: str = Field(..., description="Stored memory ID")
    success: bool = Field(..., description="Whether storage succeeded")
    relationships_detected: int = Field(0, description="Relationships detected")
    message: str = Field("", description="Status message")


class MemoryWriteTool(Tool[MemoryWriteToolInput, MemoryWriteToolOutput]):
    """Store a memory and optionally detect relationships."""

    def __init__(
        self,
        storage: MemoryStorageTool | None = None,
        relationship_detector=None,
    ) -> None:
        self.storage = storage
        self.relationship_detector = relationship_detector

    async def validate(self, input: MemoryWriteToolInput) -> bool:
        return bool(input.memory and input.memory.memory_id)

    async def execute(self, input: MemoryWriteToolInput) -> MemoryWriteToolOutput:
        try:
            if self.storage is None:
                raise ToolError("storage is required for MemoryWriteTool", input, None)
            from research_engineer.tools.memory_storage import MemoryStorageInput

            await self.storage.execute(MemoryStorageInput(memory=input.memory, operation="store"))

            rel_count = 0
            if input.detect_relationships and self.relationship_detector is not None:
                from research_engineer.tools.relationship_detector import (
                    RelationshipDetectorInput,
                )

                det_out = await self.relationship_detector.execute(
                    RelationshipDetectorInput(source_memory=input.memory)
                )
                for rel in det_out.relationships:
                    try:
                        await self.storage.store_relationship(rel)
                        rel_count += 1
                    except Exception:
                        pass

            return MemoryWriteToolOutput(
                memory_id=input.memory.memory_id,
                success=True,
                relationships_detected=rel_count,
                message="Memory stored" + (f" with {rel_count} relationships" if rel_count else ""),
            )
        except Exception as e:
            raise ToolError(f"MemoryWriteTool failed: {e}", input, e)


class MemoryGraphToolInput(BaseModel):
    """Input for MemoryGraphTool."""

    operation: str = Field(..., description="neighbors, paths, traverse, stats, add_edge")
    memory_id: str | None = Field(None, description="Memory ID for node ops")
    source_id: str | None = Field(None, description="Source for traversal")
    target_id: str | None = Field(None, description="Target for path finding")
    max_depth: int = Field(2, ge=1, le=6, description="Max depth")
    limit: int = Field(20, description="Result limit")


class MemoryGraphToolOutput(BaseModel):
    """Output for MemoryGraphTool."""

    neighbors: list[str] = Field(default_factory=list, description="Neighbors/reachable nodes")
    paths: list[list[str]] = Field(default_factory=list, description="Paths found")
    stats: GraphStats | None = Field(None, description="Graph statistics")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field("", description="Status message")


class MemoryGraphTool(Tool[MemoryGraphToolInput, MemoryGraphToolOutput]):
    """Operate on the memory knowledge graph."""

    def __init__(self, graph: MemoryKnowledgeGraph | None = None) -> None:
        self.graph = graph or MemoryKnowledgeGraph()

    async def validate(self, input: MemoryGraphToolInput) -> bool:
        return bool(input.operation)

    async def execute(self, input: MemoryGraphToolInput) -> MemoryGraphToolOutput:
        try:
            if input.operation == "neighbors":
                if not input.memory_id:
                    return MemoryGraphToolOutput(success=False, message="memory_id required")
                neighbors = self.graph.get_neighbors(input.memory_id, limit=input.limit)
                return MemoryGraphToolOutput(
                    neighbors=neighbors, success=True, message=f"{len(neighbors)} neighbors"
                )

            if input.operation == "paths":
                if not input.source_id or not input.target_id:
                    return MemoryGraphToolOutput(success=False, message="source_id+target_id required")
                paths = self.graph.find_paths(input.source_id, input.target_id, input.max_depth)
                return MemoryGraphToolOutput(
                    paths=paths, success=True, message=f"{len(paths)} paths"
                )

            if input.operation == "traverse":
                if not input.source_id:
                    return MemoryGraphToolOutput(success=False, message="source_id required")
                nodes = self.graph.traverse(input.source_id, input.max_depth)
                return MemoryGraphToolOutput(
                    neighbors=nodes, success=True, message=f"{len(nodes)} reachable"
                )

            if input.operation == "stats":
                stats = self.graph.get_stats()
                return MemoryGraphToolOutput(stats=stats, success=True, message="Stats computed")

            return MemoryGraphToolOutput(
                success=False, message=f"Unknown operation: {input.operation}"
            )
        except Exception as e:
            raise ToolError(f"MemoryGraphTool failed: {e}", input, e)


class MemoryRecallToolInput(BaseModel):
    """Input for MemoryRecallTool."""

    current_task: str = Field(..., description="Description of current task/context")
    limit: int = Field(5, ge=1, le=50, description="Max context memories")
    strategy: str = Field("semantic_search", description="Retrieval strategy")


class MemoryRecallToolOutput(BaseModel):
    """Output for MemoryRecallTool."""

    context_memories: list[MemoryResult] = Field(
        default_factory=list, description="Relevant memories for context"
    )
    summary: str = Field("", description="Brief summary of recalled context")


class MemoryRecallTool(Tool[MemoryRecallToolInput, MemoryRecallToolOutput]):
    """Inject relevant context memories for a current task."""

    def __init__(
        self,
        storage: MemoryStorageTool | None = None,
        vector_store: VectorStore | None = None,
        graph: MemoryKnowledgeGraph | None = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store
        self.graph = graph
        self.query_tool = MemoryQueryTool(storage, vector_store, graph)

    async def validate(self, input: MemoryRecallToolInput) -> bool:
        return bool(input.current_task)

    async def execute(self, input: MemoryRecallToolInput) -> MemoryRecallToolOutput:
        try:
            query_out = await self.query_tool.execute(
                MemoryQueryToolInput(
                    query_text=input.current_task,
                    strategy=input.strategy,
                    limit=input.limit,
                )
            )
            memories = query_out.results

            summary_parts: list[str] = []
            for mem in memories[:3]:
                if isinstance(mem.memory, dict):
                    title = mem.memory.get("title") or mem.memory.get("description") or mem.memory.get("paper_id", "")
                    if title:
                        summary_parts.append(str(title)[:100])
            summary = " | ".join(summary_parts) if summary_parts else "No context found"

            return MemoryRecallToolOutput(context_memories=memories, summary=summary)
        except Exception as e:
            raise ToolError(f"MemoryRecallTool failed: {e}", input, e)
