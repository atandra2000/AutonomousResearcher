"""Memory knowledge graph backed by NetworkX.

Provides traversal, neighborhood discovery, centrality, community detection,
and graph statistics for the memory relationship graph. Sync I/O (NetworkX
is in-memory) wrapped in async methods for interface consistency.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.memory import MemoryRelationship, RelationshipType
from research_engineer.tools.base import ToolError

try:
    import networkx as nx

    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    _HAS_NETWORKX = False


class GraphStats(BaseModel):
    """Statistics about the memory graph."""

    node_count: int = Field(0, description="Number of nodes")
    edge_count: int = Field(0, description="Number of edges")
    density: float = Field(0.0, description="Graph density")
    avg_degree: float = Field(0.0, description="Average degree")
    connected_components: int = Field(0, description="Number of weakly connected components")
    most_central: list[str] = Field(default_factory=list, description="Most central node IDs")
    relationship_counts: dict[str, int] = Field(
        default_factory=dict, description="Edge counts by relationship type"
    )


class MemoryGraphInput(BaseModel):
    """Input for memory graph operations."""

    operation: str = Field(
        ..., description="Operation: add_node, add_edge, neighbors, paths, stats, traverse"
    )
    memory_id: str | None = Field(None, description="Memory ID for node operations")
    memory_type: str | None = Field(None, description="Memory type metadata")
    relationship: MemoryRelationship | None = Field(None, description="Relationship for edge ops")
    source_id: str | None = Field(None, description="Source ID for traversal")
    target_id: str | None = Field(None, description="Target ID for path finding")
    max_depth: int = Field(2, ge=1, le=6, description="Max traversal depth")
    relationship_types: list[RelationshipType] | None = Field(
        None, description="Filter traversal by relationship type"
    )
    limit: int = Field(20, description="Limit for neighbors/traversal")


class MemoryGraphOutput(BaseModel):
    """Output from memory graph operations."""

    success: bool = Field(..., description="Whether operation succeeded")
    neighbors: list[str] = Field(default_factory=list, description="Neighbor node IDs")
    paths: list[list[str]] = Field(default_factory=list, description="Paths found")
    stats: GraphStats | None = Field(None, description="Graph statistics")
    message: str = Field("", description="Status message")


class MemoryKnowledgeGraph:
    """NetworkX-backed in-memory graph for memory relationships."""

    def __init__(self) -> None:
        if not _HAS_NETWORKX:
            raise ImportError("networkx is required for MemoryKnowledgeGraph")
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def add_node(self, memory_id: str, memory_type: str | None = None, **attrs: Any) -> None:
        """Add a node representing a memory."""
        self.graph.add_node(memory_id, memory_type=memory_type, **attrs)

    def add_relationship(self, relationship: MemoryRelationship) -> None:
        """Add a directed edge representing a relationship."""
        self.graph.add_edge(
            relationship.source_memory_id,
            relationship.target_memory_id,
            relationship_type=relationship.relationship_type.value,
            confidence=relationship.confidence,
            relationship_id=relationship.relationship_id,
            created_at=relationship.created_at.isoformat(),
        )

    def remove_node(self, memory_id: str) -> bool:
        if memory_id in self.graph:
            self.graph.remove_node(memory_id)
            return True
        return False

    def get_neighbors(
        self,
        memory_id: str,
        relationship_types: list[RelationshipType] | None = None,
        limit: int = 20,
    ) -> list[str]:
        """Get neighbors of a node (both in and out edges)."""
        if memory_id not in self.graph:
            return []
        rel_filter = {rt.value for rt in relationship_types} if relationship_types else None

        neighbors: set[str] = set()
        for _, target, data in self.graph.edges(memory_id, data=True):
            if rel_filter is None or data.get("relationship_type") in rel_filter:
                neighbors.add(target)
        for source, _, data in self.graph.in_edges(memory_id, data=True):
            if rel_filter is None or data.get("relationship_type") in rel_filter:
                neighbors.add(source)
        return list(neighbors)[:limit]

    def find_paths(self, source: str, target: str, max_depth: int = 3) -> list[list[str]]:
        """Find all simple paths between two nodes up to max_depth."""
        if source not in self.graph or target not in self.graph:
            return []
        try:
            # Treat as undirected for path finding
            undirected = self.graph.to_undirected()
            paths = nx.all_simple_paths(undirected, source, target, cutoff=max_depth)
            return [list(p) for p in paths]
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

    def traverse(
        self,
        source_id: str,
        max_depth: int = 2,
        relationship_types: list[RelationshipType] | None = None,
    ) -> list[str]:
        """BFS traversal from a source node, returning reachable node IDs."""
        if source_id not in self.graph:
            return []
        undirected = self.graph.to_undirected()
        if relationship_types:
            subgraph = nx.Graph()
            for u, v, data in undirected.edges(data=True):
                if data.get("relationship_type") in {rt.value for rt in relationship_types}:
                    subgraph.add_edge(u, v)
            if source_id not in subgraph:
                return [source_id]
            visited = list(nx.bfs_tree(subgraph, source_id, depth_limit=max_depth).nodes())
        else:
            visited = list(nx.bfs_tree(undirected, source_id, depth_limit=max_depth).nodes())
        return visited

    def get_stats(self) -> GraphStats:
        """Compute graph statistics."""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        density = nx.density(self.graph) if node_count > 1 else 0.0
        avg_degree = (2 * edge_count / node_count) if node_count else 0.0
        components = nx.number_weakly_connected_components(self.graph)

        most_central: list[str] = []
        if node_count > 0:
            try:
                centrality = nx.degree_centrality(self.graph)
                ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                most_central = [n for n, _ in ranked[:10]]
            except Exception:
                most_central = []

        rel_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rtype = data.get("relationship_type", "unknown")
            rel_counts[rtype] = rel_counts.get(rtype, 0) + 1

        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            density=round(density, 4),
            avg_degree=round(avg_degree, 4),
            connected_components=components,
            most_central=most_central,
            relationship_counts=rel_counts,
        )

    def load_from_storage(self, relationships: list[dict]) -> int:
        """Load relationships from storage rows into the graph.

        Args:
            relationships: list of dicts from MemoryStorageTool.get_relationships

        Returns:
            Number of edges loaded.
        """
        count = 0
        for row in relationships:
            source = row.get("source_memory_id")
            target = row.get("target_memory_id")
            rtype = row.get("relationship_type")
            if source and target:
                self.add_node(source)
                self.add_node(target)
                rel = MemoryRelationship(
                    source_memory_id=source,
                    target_memory_id=target,
                    relationship_type=RelationshipType(rtype) if rtype else RelationshipType.SIMILAR_TO,
                    confidence=row.get("confidence", 1.0),
                )
                self.add_relationship(rel)
                count += 1
        return count

    def clear(self) -> None:
        """Clear the entire graph."""
        self.graph.clear()


async def execute_graph_operation(input: MemoryGraphInput) -> MemoryGraphOutput:
    """Execute a memory graph operation (standalone helper).

    This is a convenience function used by MemoryGraphTool and the agent layer.
    """
    raise NotImplementedError("Use MemoryGraphTool.execute instead")


class MemoryGraphTool:
    """Thin async wrapper over MemoryKnowledgeGraph following tool conventions.

    Not a Tool subclass because the graph is stateful/in-memory; instead it
    exposes async methods mirroring the Tool[Input, Output] signature.
    """

    def __init__(self) -> None:
        self.graph = MemoryKnowledgeGraph()

    async def execute(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        try:
            handlers = {
                "add_node": self._op_add_node,
                "add_edge": self._op_add_edge,
                "neighbors": self._op_neighbors,
                "paths": self._op_paths,
                "traverse": self._op_traverse,
                "stats": self._op_stats,
            }
            handler = handlers.get(input.operation)
            if handler is None:
                return MemoryGraphOutput(success=False, message=f"Unknown operation: {input.operation}")
            return handler(input)
        except Exception as e:
            raise ToolError(f"Graph operation failed: {e}", input, e)

    def _op_add_node(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        if input.memory_id:
            self.graph.add_node(input.memory_id, input.memory_type)
            return MemoryGraphOutput(success=True, message="Node added")
        return MemoryGraphOutput(success=False, message="memory_id required")

    def _op_add_edge(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        if input.relationship:
            self.graph.add_node(input.relationship.source_memory_id)
            self.graph.add_node(input.relationship.target_memory_id)
            self.graph.add_relationship(input.relationship)
            return MemoryGraphOutput(success=True, message="Edge added")
        return MemoryGraphOutput(success=False, message="relationship required")

    def _op_neighbors(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        if not input.memory_id:
            return MemoryGraphOutput(success=False, message="memory_id required")
        neighbors = self.graph.get_neighbors(input.memory_id, input.relationship_types, input.limit)
        return MemoryGraphOutput(success=True, neighbors=neighbors, message=f"{len(neighbors)} neighbors")

    def _op_paths(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        if not input.source_id or not input.target_id:
            return MemoryGraphOutput(success=False, message="source_id and target_id required")
        paths = self.graph.find_paths(input.source_id, input.target_id, input.max_depth)
        return MemoryGraphOutput(success=True, paths=paths, message=f"{len(paths)} paths")

    def _op_traverse(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        if not input.source_id:
            return MemoryGraphOutput(success=False, message="source_id required")
        nodes = self.graph.traverse(input.source_id, input.max_depth, input.relationship_types)
        return MemoryGraphOutput(success=True, neighbors=nodes, message=f"{len(nodes)} reachable")

    def _op_stats(self, input: MemoryGraphInput) -> MemoryGraphOutput:
        stats = self.graph.get_stats()
        return MemoryGraphOutput(success=True, stats=stats, message="Stats computed")
