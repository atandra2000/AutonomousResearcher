"""Phase 12 - Hybrid retriever.

Combines three signals to rank code chunks for a natural-language query:

1. **Semantic similarity** — cosine similarity between the query embedding
   and chunk embeddings (lexical via :class:`HashingEmbedder` by default;
   swap for a real embedder in production).
2. **Graph proximity** — for each semantic hit, boost chunks whose parent
   symbol is graph-adjacent (dependencies, callers, callees, related).
3. **Metadata boosts** — heuristic boosts for test symbols, entry points,
   and kind-based relevance (e.g. classes/functions weighted higher).

Scores are normalized to [0, 1] and combined with configurable weights
before a final re-ranking. This avoids the brittleness of pure vector
search: a query like "EMA checkpoint support" will surface the training
loop, checkpoint logic, optimizer state handling, related configs, and
related tests through graph expansion even when lexical overlap is low.
"""

from __future__ import annotations

from typing import Any

from research_engineer.memory.embeddings import EmbedderBackend
from research_engineer.memory.models import (
    CodeChunk,
    RetrievalResult,
    Symbol,
    SymbolKind,
)
from research_engineer.memory.symbol_graph import SymbolGraph
from research_engineer.memory.vector_store import VectorBackend


class HybridRetriever:
    """Hybrid (semantic + graph + metadata) code retriever.

    Parameters
    ----------
    vector_backend:
        Pre-populated vector index of chunk embeddings.
    embedder:
        Embedding backend used to embed the query (must match the one
        used at index time).
    graph:
        Symbol graph for proximity scoring.
    symbols:
        Mapping ``symbol_id -> Symbol`` for resolving hits.
    chunks:
        Mapping ``chunk_id -> CodeChunk`` for resolving vector hits to
        source text.
    weights:
        Tuple ``(semantic, graph, metadata)`` weights for the final
        combined score. Defaults to ``(0.6, 0.3, 0.1)``.
    """

    def __init__(
        self,
        vector_backend: VectorBackend,
        embedder: EmbedderBackend,
        graph: SymbolGraph,
        symbols: dict[str, Symbol],
        chunks: dict[str, CodeChunk],
        *,
        weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    ) -> None:
        self.vector = vector_backend
        self.embedder = embedder
        self.graph = graph
        self.symbols = symbols
        self.chunks = chunks
        self.w_sem, self.w_graph, self.w_meta = weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_limit: int = 50,
        include_related: bool = True,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Run a hybrid retrieval for ``query``.

        Args:
            query: Natural-language query (e.g. "EMA checkpoint support").
            limit: Final number of results to return.
            candidate_limit: How many semantic candidates to consider
                before graph/metadata re-ranking.
            include_related: Attach graph-related symbols to each result.
            filter: Optional metadata filter passed to the vector backend
                (e.g. ``{"kind": "function"}``).

        Returns:
            Ranked list of :class:`RetrievalResult`.
        """
        # 1. Semantic candidates.
        qvec = self.embedder.embed_query(query)
        hits = self.vector.search(qvec, limit=candidate_limit, filter=filter)
        if not hits:
            return []

        # Normalize semantic scores to [0,1].
        max_sem = max((h[1] for h in hits), default=1.0) or 1.0

        scored: list[RetrievalResult] = []
        for chunk_id, sem_score, payload in hits:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            sym = self.symbols.get(chunk.symbol_id)
            sem = sem_score / max_sem if max_sem > 0 else 0.0

            # 2. Graph score: proximity of this symbol to other top hits.
            graph_score = self._graph_score(chunk.symbol_id, hits)

            # 3. Metadata score.
            meta_score = self._metadata_score(sym, chunk, query)

            combined = (
                self.w_sem * sem
                + self.w_graph * graph_score
                + self.w_meta * meta_score
            )
            related: list[Symbol] = []
            if include_related and sym is not None:
                related = self._related_symbols(sym.symbol_id)
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    symbol=sym,
                    semantic_score=round(sem, 4),
                    graph_score=round(graph_score, 4),
                    metadata_score=round(meta_score, 4),
                    combined_score=round(combined, 4),
                    related_symbols=related,
                )
            )

        scored.sort(key=lambda r: r.combined_score, reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _graph_score(
        self,
        symbol_id: str,
        hits: list[tuple[str, float, dict[str, Any]]],
    ) -> float:
        """Score graph proximity: how many other hits share an edge."""
        neighbor_ids = set(self.graph.neighborhood(symbol_id, limit=100))
        if not neighbor_ids:
            return 0.0
        # Count how many other hit-chunks belong to neighboring symbols.
        shared = 0
        total = 0
        for _, _, payload in hits:
            other_sym = payload.get("symbol_id")
            if other_sym is None or other_sym == symbol_id:
                continue
            total += 1
            if other_sym in neighbor_ids:
                shared += 1
        if total == 0:
            return 0.0
        return shared / total

    @staticmethod
    def _metadata_score(
        symbol: Symbol | None, chunk: CodeChunk, query: str
    ) -> float:
        """Heuristic metadata boosts in [0, 1]."""
        score = 0.0
        ql = query.lower()
        if symbol is not None:
            # Name overlap with query tokens.
            name_l = symbol.name.lower()
            qtokens = set(ql.split())
            name_tokens = set(name_l.replace("_", " ").split())
            if qtokens & name_tokens:
                score += 0.3
            # Kind-based boosts.
            if symbol.kind == SymbolKind.FUNCTION:
                score += 0.1
            elif symbol.kind == SymbolKind.CLASS:
                score += 0.15
            elif symbol.kind == SymbolKind.METHOD:
                score += 0.05
            # Test symbols get a mild boost when query mentions "test".
            if symbol.is_test and "test" in ql:
                score += 0.2
            # Entry points boosted when query mentions "main"/"cli"/"entry".
            if symbol.is_entry_point and any(
                w in ql for w in ("main", "cli", "entry", "command")
            ):
                score += 0.2
        # Docstring presence is a mild quality signal.
        if symbol is not None and symbol.docstring:
            score += 0.05
        return min(score, 1.0)

    def _related_symbols(self, symbol_id: str, limit: int = 5) -> list[Symbol]:
        """Return up to ``limit`` graph-related symbols."""
        related_ids = self.graph.related(symbol_id, max_depth=2)
        out: list[Symbol] = []
        for rid in related_ids:
            sym = self.symbols.get(rid)
            if sym is not None:
                out.append(sym)
            if len(out) >= limit:
                break
        return out


__all__ = ["HybridRetriever"]
