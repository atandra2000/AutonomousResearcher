"""Tests for MemoryKnowledgeGraph."""

import pytest

from research_engineer.models.memory import (
    MemoryRelationship,
    RelationshipType,
)
from research_engineer.tools.memory_graph import (
    GraphStats,
    MemoryGraphInput,
    MemoryGraphOutput,
    MemoryKnowledgeGraph,
)


@pytest.fixture
def graph():
    return MemoryKnowledgeGraph()


@pytest.fixture
def relationship():
    return MemoryRelationship(
        source_memory_id="mem_1",
        target_memory_id="mem_2",
        relationship_type=RelationshipType.CITES,
        confidence=0.9,
    )


class TestNodeOperations:
    """Test node add/remove operations."""

    def test_add_node(self, graph):
        graph.add_node("mem_1", "paper")
        assert "mem_1" in graph.graph
        assert graph.graph.nodes["mem_1"]["memory_type"] == "paper"

    def test_remove_existing_node(self, graph):
        graph.add_node("mem_1")
        assert graph.remove_node("mem_1") is True
        assert "mem_1" not in graph.graph

    def test_remove_nonexistent_node(self, graph):
        assert graph.remove_node("nonexistent") is False


class TestEdgeOperations:
    """Test relationship/edge operations."""

    def test_add_relationship(self, graph, relationship):
        graph.add_relationship(relationship)
        assert graph.graph.has_edge("mem_1", "mem_2")
        assert graph.graph.number_of_edges() == 1

    def test_add_relationship_creates_nodes(self, graph, relationship):
        graph.add_relationship(relationship)
        assert "mem_1" in graph.graph
        assert "mem_2" in graph.graph

    def test_multiple_relationship_types(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.SIMILAR_TO,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        # MultiDiGraph allows parallel edges
        assert graph.graph.number_of_edges() == 2


class TestNeighbors:
    """Test neighbor discovery."""

    def test_get_neighbors_outgoing(self, graph):
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel)
        neighbors = graph.get_neighbors("a")
        assert "b" in neighbors

    def test_get_neighbors_incoming(self, graph):
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel)
        neighbors = graph.get_neighbors("b")
        assert "a" in neighbors

    def test_get_neighbors_filtered_by_type(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="a", target_memory_id="c",
            relationship_type=RelationshipType.SIMILAR_TO,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)

        neighbors = graph.get_neighbors("a", relationship_types=[RelationshipType.CITES])
        assert "b" in neighbors
        assert "c" not in neighbors

    def test_get_neighbors_nonexistent(self, graph):
        assert graph.get_neighbors("nonexistent") == []

    def test_get_neighbors_limit(self, graph):
        for i in range(10):
            rel = MemoryRelationship(
                source_memory_id="a", target_memory_id=f"b_{i}",
                relationship_type=RelationshipType.CITES,
            )
            graph.add_relationship(rel)
        neighbors = graph.get_neighbors("a", limit=5)
        assert len(neighbors) <= 5


class TestPathFinding:
    """Test path finding."""

    def test_direct_path(self, graph):
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel)
        paths = graph.find_paths("a", "b")
        assert len(paths) >= 1
        assert paths[0] == ["a", "b"]

    def test_indirect_path(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="b", target_memory_id="c",
            relationship_type=RelationshipType.SIMILAR_TO,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        paths = graph.find_paths("a", "c", max_depth=3)
        assert len(paths) >= 1
        assert paths[0] == ["a", "b", "c"]

    def test_no_path(self, graph):
        graph.add_node("a")
        graph.add_node("b")
        paths = graph.find_paths("a", "b")
        assert paths == []

    def test_nonexistent_nodes(self, graph):
        assert graph.find_paths("x", "y") == []


class TestTraversal:
    """Test BFS traversal."""

    def test_traverse_simple(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="b", target_memory_id="c",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        nodes = graph.traverse("a", max_depth=2)
        assert "a" in nodes
        assert "b" in nodes
        assert "c" in nodes

    def test_traverse_depth_limit(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="b", target_memory_id="c",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)
        nodes = graph.traverse("a", max_depth=1)
        assert "a" in nodes
        assert "b" in nodes
        assert "c" not in nodes

    def test_traverse_nonexistent(self, graph):
        assert graph.traverse("nonexistent") == []


class TestStats:
    """Test graph statistics."""

    def test_empty_graph_stats(self, graph):
        stats = graph.get_stats()
        assert isinstance(stats, GraphStats)
        assert stats.node_count == 0
        assert stats.edge_count == 0

    def test_stats_with_nodes(self, graph):
        rel1 = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        rel2 = MemoryRelationship(
            source_memory_id="b", target_memory_id="c",
            relationship_type=RelationshipType.SIMILAR_TO,
        )
        graph.add_relationship(rel1)
        graph.add_relationship(rel2)

        stats = graph.get_stats()
        assert stats.node_count == 3
        assert stats.edge_count == 2
        assert stats.relationship_counts.get("cites") == 1
        assert stats.relationship_counts.get("similar_to") == 1
        assert stats.connected_components >= 1

    def test_density(self, graph):
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel)
        stats = graph.get_stats()
        assert stats.density > 0


class TestLoadFromStorage:
    """Test loading relationships from storage dict format."""

    def test_load_from_storage(self, graph):
        relationships = [
            {
                "source_memory_id": "a",
                "target_memory_id": "b",
                "relationship_type": "cites",
                "confidence": 0.9,
            },
            {
                "source_memory_id": "b",
                "target_memory_id": "c",
                "relationship_type": "similar_to",
                "confidence": 0.8,
            },
        ]
        count = graph.load_from_storage(relationships)
        assert count == 2
        assert graph.graph.number_of_nodes() == 3
        assert graph.graph.number_of_edges() == 2

    def test_clear(self, graph):
        rel = MemoryRelationship(
            source_memory_id="a", target_memory_id="b",
            relationship_type=RelationshipType.CITES,
        )
        graph.add_relationship(rel)
        graph.clear()
        assert graph.graph.number_of_nodes() == 0