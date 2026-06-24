"""Phase 12 - Symbol graph.

In-memory directed graph over indexed symbols supporting the queries
required by the hybrid retriever:

* dependencies (what does X depend on?)
* dependents (who depends on X?)
* callers (who calls X?)
* callees (what does X call?)
* related symbols (co-occurrence / shared neighbors)

Built from :class:`~research_engineer.memory.models.SymbolEdge` lists
emitted by the indexer. The graph is adjacency-list based (dicts of
sets) for O(1) neighbor lookup and low memory footprint, suitable for
100k+ LOC repositories.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from research_engineer.memory.models import (
    RelationKind,
    Symbol,
    SymbolEdge,
)


class SymbolGraph:
    """Directed multigraph of symbols keyed by ``symbol_id``.

    Maintains forward and reverse adjacency maps per relation kind so
    both directions (dependents vs dependencies, callers vs callees) are
    O(1).
    """

    def __init__(self) -> None:
        # forward[src][relation] -> set[target]
        self._forward: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # reverse[target][relation] -> set[src]
        self._reverse: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._symbols: dict[str, Symbol] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_symbol(self, symbol: Symbol) -> None:
        self._symbols[symbol.symbol_id] = symbol

    def add_edge(self, edge: SymbolEdge) -> None:
        self._forward[edge.source_id][edge.relation.value].add(edge.target_id)
        self._reverse[edge.target_id][edge.relation.value].add(edge.source_id)

    def build(
        self,
        symbols: Iterable[Symbol],
        edges: Iterable[SymbolEdge],
    ) -> None:
        for sym in symbols:
            self.add_symbol(sym)
        for edge in edges:
            self.add_edge(edge)

    def clear(self) -> None:
        self._forward.clear()
        self._reverse.clear()
        self._symbols.clear()

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_symbol(self, symbol_id: str) -> Symbol | None:
        return self._symbols.get(symbol_id)

    def all_symbols(self) -> list[Symbol]:
        return list(self._symbols.values())

    def dependencies(self, symbol_id: str) -> list[str]:
        """What does ``symbol_id`` depend on?"""
        return list(self._forward[symbol_id][RelationKind.DEPENDS_ON.value])

    def dependents(self, symbol_id: str) -> list[str]:
        """Who depends on ``symbol_id``?"""
        return list(self._reverse[symbol_id][RelationKind.DEPENDS_ON.value])

    def callees(self, symbol_id: str) -> list[str]:
        """What does ``symbol_id`` call?"""
        return list(self._forward[symbol_id][RelationKind.CALLS.value])

    def callers(self, symbol_id: str) -> list[str]:
        """Who calls ``symbol_id``?"""
        return list(self._reverse[symbol_id][RelationKind.CALLS.value])

    def tests_for(self, symbol_id: str) -> list[str]:
        """Test symbols that test ``symbol_id``."""
        return list(self._reverse[symbol_id][RelationKind.TESTS.value])

    def tested_symbols(self, symbol_id: str) -> list[str]:
        """Symbols that ``symbol_id`` tests."""
        return list(self._forward[symbol_id][RelationKind.TESTS.value])

    def related(self, symbol_id: str, max_depth: int = 2) -> list[str]:
        """Symbols related to ``symbol_id`` via any relation.

        Performs a bounded BFS over all edge kinds, excluding the seed
        itself. Useful for graph-score boosting in hybrid retrieval.
        """
        if symbol_id not in self._forward and symbol_id not in self._reverse:
            return []
        seen: set[str] = set()
        frontier: set[str] = {symbol_id}
        for _ in range(max_depth):
            nxt: set[str] = set()
            for node in frontier:
                for targets in self._forward[node].values():
                    nxt.update(targets)
                for targets in self._reverse[node].values():
                    nxt.update(targets)
            nxt -= seen
            nxt.discard(symbol_id)
            if not nxt:
                break
            seen |= nxt
            frontier = nxt
        return list(seen)

    def neighborhood(self, symbol_id: str, limit: int = 50) -> list[str]:
        """Direct neighbors (depth-1) of ``symbol_id`` across all relations."""
        if symbol_id not in self._symbols:
            return []
        neighbors: set[str] = set()
        for targets in self._forward[symbol_id].values():
            neighbors.update(targets)
        for targets in self._reverse[symbol_id].values():
            neighbors.update(targets)
        neighbors.discard(symbol_id)
        return list(neighbors)[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        edge_count = sum(
            len(targets)
            for rels in self._forward.values()
            for targets in rels.values()
        )
        return {
            "nodes": len(self._symbols),
            "edges": edge_count,
        }

    def all_edges(self) -> list[SymbolEdge]:
        """Return all edges as :class:`SymbolEdge` objects (for persistence)."""
        edges: list[SymbolEdge] = []
        for src, rels in self._forward.items():
            for rel, targets in rels.items():
                for tgt in targets:
                    edges.append(
                        SymbolEdge(
                            source_id=src,
                            target_id=tgt,
                            relation=RelationKind(rel),
                        )
                    )
        return edges


__all__ = ["SymbolGraph"]
