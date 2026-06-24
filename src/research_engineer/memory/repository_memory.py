"""Phase 12 - RepositoryMemory facade.

The single entry point for repository intelligence. Wires together the
indexer, symbol graph, embedder, vector backend, retriever, and
persistent store. Exposes a small, high-level API used by the
:class:`~research_engineer.agents.task_agent.TaskAgent` and the CLI:

* :meth:`build`      — full index of a repository.
* :meth:`refresh`    — incremental re-index (changed files only).
* :meth:`query`      — hybrid retrieval of relevant code + context.
* :meth:`graph`      — symbol-graph neighborhood for a symbol name.
* :meth:`stats`      — index statistics.
* :meth:`get_context` — assemble a compact context string for injection
  into an LLM planning prompt (relevant symbols, files, deps, tests).

The facade is repository-scoped: instantiate one per repository. The
underlying SQLite store is shared across instances (multi-repo safe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_engineer.memory.embeddings import EmbedderBackend, HashingEmbedder
from research_engineer.memory.indexer import IndexResult, RepositoryIndexer
from research_engineer.memory.models import (
    CodeChunk,
    IndexStats,
    RetrievalResult,
    Symbol,
    SymbolEdge,
    SymbolKind,
)
from research_engineer.memory.retriever import HybridRetriever
from research_engineer.memory.storage import RepositoryMemoryStore
from research_engineer.memory.symbol_graph import SymbolGraph
from research_engineer.memory.vector_store import (
    InMemoryVectorBackend,
    VectorBackend,
    chunk_payload,
)


class RepositoryMemory:
    """Repository-scoped semantic memory with hybrid retrieval.

    Parameters
    ----------
    repo_path:
        Repository root to index/serve.
    store:
        Persistent SQLite store (shared across repos). If ``None``, a
        default store at ``data/repo_memory.db`` is created.
    embedder:
        Embedding backend. Defaults to :class:`HashingEmbedder` (offline).
    vector_backend:
        Vector index backend. Defaults to :class:`InMemoryVectorBackend`.
    auto_load:
        If True (default), load any existing index for ``repo_path`` from
        the store on construction.
    """

    def __init__(
        self,
        repo_path: str,
        *,
        store: RepositoryMemoryStore | None = None,
        embedder: EmbedderBackend | None = None,
        vector_backend: VectorBackend | None = None,
        auto_load: bool = True,
    ) -> None:
        self.repo_path = str(Path(repo_path).resolve())
        self.store = store or RepositoryMemoryStore()
        self.embedder = embedder or HashingEmbedder()
        self.vector = vector_backend or InMemoryVectorBackend()
        self._symbol_graph = SymbolGraph()
        self._symbols: dict[str, Symbol] = {}
        self._chunks: dict[str, CodeChunk] = {}
        self._indexer = RepositoryIndexer(self.repo_path)
        self._retriever: HybridRetriever | None = None
        if auto_load and self.store.has_index(self.repo_path):
            self.load()

    # ------------------------------------------------------------------
    # Build / refresh
    # ------------------------------------------------------------------

    def build(self) -> IndexStats:
        """Full index of the repository. Replaces any existing index."""
        result = self._indexer.index()
        stats = self._compute_stats(result)
        self._hydrate(result)
        self.store.replace_repo(
            self.repo_path,
            list(self._symbols.values()),
            list(self._chunks.values()),
            self._symbol_graph.all_edges(),
            result.file_hashes,
            stats,
        )
        return stats

    def refresh(self) -> tuple[IndexStats, list[str]]:
        """Incremental re-index of changed files only.

        Returns the updated stats and the list of changed file paths.
        """
        known = self.store.load_hashes(self.repo_path)
        result, changed = self._indexer.index_incremental(known)
        if not changed:
            stats = self.store.load_stats(self.repo_path) or self._compute_stats(result)
            return stats, []
        # Merge: update symbols/chunks for changed files, rebuild graph.
        self._merge_incremental(result)
        stats = self._compute_stats_from_current()
        self.store.update_files(
            self.repo_path,
            list(self._symbols.values()),
            list(self._chunks.values()),
            self._collect_edges(),
            result.file_hashes,
            stats,
        )
        return stats, changed

    # ------------------------------------------------------------------
    # Load from store
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Hydrate symbols, chunks, graph, and vector index from the store."""
        symbols = self.store.load_symbols(self.repo_path)
        chunks = self.store.load_chunks(self.repo_path)
        edges = self.store.load_edges(self.repo_path)
        if not symbols:
            return False
        self._symbols = {s.symbol_id: s for s in symbols}
        self._chunks = {c.chunk_id: c for c in chunks}
        self._symbol_graph = SymbolGraph()
        self._symbol_graph.build(symbols, edges)
        self._rebuild_vector_index()
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        limit: int = 10,
        include_related: bool = True,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Hybrid retrieval of relevant code chunks + context."""
        if self._retriever is None:
            self._rebuild_vector_index()
        if self._retriever is None or self.vector.count() == 0:
            return []
        return self._retriever.retrieve(
            query,
            limit=limit,
            include_related=include_related,
            filter=filter,
        )

    def graph(self, symbol_name: str, limit: int = 20) -> dict[str, Any]:
        """Return the symbol-graph neighborhood for a symbol name.

        Matches by name (case-insensitive) and returns dependencies,
        dependents, callers, callees, and related symbols.
        """
        sym = self._find_symbol_by_name(symbol_name)
        if sym is None:
            return {"found": False, "query": symbol_name}
        sid = sym.symbol_id
        return {
            "found": True,
            "symbol": sym.model_dump(),
            "dependencies": [self._sym_brief(i) for i in self._symbol_graph.dependencies(sid)],
            "dependents": [self._sym_brief(i) for i in self._symbol_graph.dependents(sid)],
            "callers": [self._sym_brief(i) for i in self._symbol_graph.callers(sid)],
            "callees": [self._sym_brief(i) for i in self._symbol_graph.callees(sid)],
            "related": [self._sym_brief(i) for i in self._symbol_graph.related(sid, max_depth=2)],
            "tests": [self._sym_brief(i) for i in self._symbol_graph.tests_for(sid)],
        }

    def stats(self) -> IndexStats:
        """Return index statistics (from store if loaded, else compute)."""
        s = self.store.load_stats(self.repo_path)
        if s is not None:
            return s
        return self._compute_stats_from_current()

    # ------------------------------------------------------------------
    # Context assembly for TaskAgent injection
    # ------------------------------------------------------------------

    def get_context(self, goal: str, *, limit: int = 8) -> str:
        """Build a compact context string for an LLM planning prompt.

        Retrieves relevant symbols, their files, dependency context, and
        related tests, then formats them into a concise markdown block
        suitable for injection into a planning or coding prompt.
        """
        results = self.query(goal, limit=limit, include_related=True)
        if not results:
            return ""
        lines = [f"## Repository Memory Context for: {goal}", ""]
        seen_files: set[str] = set()
        seen_syms: set[str] = set()
        for r in results:
            sym = r.symbol
            chunk = r.chunk
            if sym and sym.symbol_id not in seen_syms:
                seen_syms.add(sym.symbol_id)
                self._format_symbol_context(sym, chunk, lines)
            seen_files.add(chunk.file_path)
        if seen_files:
            lines.append("### Relevant files")
            for f in sorted(seen_files):
                lines.append(f"- `{f}`")
        return "\n".join(lines)

    def _format_symbol_context(
        self, sym: Symbol, chunk: CodeChunk, lines: list[str]
    ) -> None:
        """Format a single symbol's context into ``lines``."""
        lines.append(
            f"### {sym.kind.value}: `{sym.qualified_name}` "
            f"({chunk.file_path}:{chunk.line_start}-{chunk.line_end})"
        )
        if sym.docstring:
            lines.append(f"> {sym.docstring.splitlines()[0]}")
        if sym.signature:
            lines.append(f"```python\n{sym.signature}\n```")
        self._format_graph_context(sym.symbol_id, lines)
        lines.append("")

    def _format_graph_context(
        self, symbol_id: str, lines: list[str]
    ) -> None:
        """Format dependency/caller/test context for a symbol."""
        deps = self._symbol_graph.dependencies(symbol_id)
        if deps:
            dep_names = self._resolve_qualified_names(deps, 5)
            if dep_names:
                lines.append(f"- Depends on: {', '.join(dep_names)}")
        callers = self._symbol_graph.callers(symbol_id)
        if callers:
            caller_names = self._resolve_qualified_names(callers, 5)
            if caller_names:
                lines.append(f"- Called by: {', '.join(caller_names)}")
        tests = self._symbol_graph.tests_for(symbol_id)
        if tests:
            test_names = self._resolve_file_paths(tests, 5)
            if test_names:
                lines.append(f"- Related tests: {', '.join(test_names)}")

    def _resolve_qualified_names(
        self, symbol_ids: list[str], limit: int
    ) -> list[str]:
        """Resolve symbol ids to qualified names (up to ``limit``)."""
        return [
            self._symbols[sid].qualified_name
            for sid in symbol_ids[:limit]
            if sid in self._symbols
        ]

    def _resolve_file_paths(
        self, symbol_ids: list[str], limit: int
    ) -> list[str]:
        """Resolve symbol ids to file paths (up to ``limit``)."""
        return [
            self._symbols[sid].file_path
            for sid in symbol_ids[:limit]
            if sid in self._symbols
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hydrate(self, result: IndexResult) -> None:
        """Populate in-memory structures from an index result."""
        self._symbols = {s.symbol_id: s for s in result.symbols}
        self._chunks = {c.chunk_id: c for c in result.chunks}
        self._symbol_graph = SymbolGraph()
        self._symbol_graph.build(result.symbols, result.edges)
        self._rebuild_vector_index()

    def _merge_incremental(self, result: IndexResult) -> None:
        """Merge an incremental index result into current state."""
        # Update symbols.
        for sym in result.symbols:
            self._symbols[sym.symbol_id] = sym
        # Update chunks.
        for chunk in result.chunks:
            self._chunks[chunk.chunk_id] = chunk
        # Rebuild graph from scratch (cheap, in-memory).
        self._symbol_graph = SymbolGraph()
        self._symbol_graph.build(self._symbols.values(), result.edges)

    def _rebuild_vector_index(self) -> None:
        """Rebuild the vector index from current chunks."""
        self.vector.clear()
        if not self._chunks:
            self._retriever = None
            return
        chunk_list = list(self._chunks.values())
        vectors = self.embedder.embed_chunks(chunk_list)  # type: ignore[attr-defined]
        ids = [c.chunk_id for c in chunk_list]
        payloads = [chunk_payload(c) for c in chunk_list]
        self.vector.add(ids, vectors, payloads)
        self._retriever = HybridRetriever(
            self.vector,
            self.embedder,
            self._symbol_graph,
            self._symbols,
            self._chunks,
        )

    def _collect_edges(self) -> list[SymbolEdge]:
        """Collect all edges from the graph (for persistence)."""
        from research_engineer.memory.models import RelationKind

        edges: list[SymbolEdge] = []
        for src, rels in self._symbol_graph._forward.items():  # type: ignore[attr-defined]
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

    def _compute_stats(self, result: IndexResult) -> IndexStats:
        by_kind: dict[str, int] = {}
        for s in result.symbols:
            by_kind[s.kind.value] = by_kind.get(s.kind.value, 0) + 1
        return IndexStats(
            repo_path=self.repo_path,
            total_files=len(result.file_hashes),
            total_symbols=len(result.symbols),
            total_chunks=len(result.chunks),
            total_edges=len(result.edges),
            symbols_by_kind=by_kind,
            index_time_seconds=getattr(result, "index_time_seconds", 0.0),
        )

    def _compute_stats_from_current(self) -> IndexStats:
        by_kind: dict[str, int] = {}
        for s in self._symbols.values():
            by_kind[s.kind.value] = by_kind.get(s.kind.value, 0) + 1
        return IndexStats(
            repo_path=self.repo_path,
            total_files=len({s.file_path for s in self._symbols.values()}),
            total_symbols=len(self._symbols),
            total_chunks=len(self._chunks),
            total_edges=self._symbol_graph.stats()["edges"],
            symbols_by_kind=by_kind,
        )

    def _find_symbol_by_name(self, name: str) -> Symbol | None:
        """Case-insensitive name match, preferring non-test symbols."""
        name_l = name.lower()
        candidates = [
            s
            for s in self._symbols.values()
            if s.name.lower() == name_l or s.qualified_name.lower() == name_l
        ]
        if not candidates:
            # Substring match fallback.
            candidates = [
                s for s in self._symbols.values() if name_l in s.name.lower()
            ]
        if not candidates:
            return None
        # Prefer non-test, then classes/functions over modules/configs.
        candidates.sort(
            key=lambda s: (
                s.is_test,
                s.kind == SymbolKind.MODULE,
                s.kind == SymbolKind.CONFIG,
            )
        )
        return candidates[0]

    def _sym_brief(self, symbol_id: str) -> dict[str, Any]:
        s = self._symbols.get(symbol_id)
        if s is None:
            return {"symbol_id": symbol_id}
        return {
            "symbol_id": s.symbol_id,
            "name": s.name,
            "kind": s.kind.value,
            "qualified_name": s.qualified_name,
            "file_path": s.file_path,
        }


__all__ = ["RepositoryMemory"]
