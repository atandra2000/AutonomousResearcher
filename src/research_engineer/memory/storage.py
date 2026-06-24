"""Phase 12 - Persistent storage for repository memory.

SQLite-backed persistence for symbols, code chunks, symbol-graph edges,
and file hashes (for incremental re-indexing). All tables are keyed by
``repo_path`` so multiple repositories can coexist in one database.

The store is the single source of truth that survives process restarts.
On startup, :class:`RepositoryMemory` loads symbols/chunks/edges from
here into memory, rebuilds the :class:`SymbolGraph`, and hydrates the
vector index.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_engineer.memory.models import (
    CodeChunk,
    IndexStats,
    Symbol,
    SymbolEdge,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repo_symbols (
    repo_path TEXT NOT NULL,
    symbol_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    docstring TEXT,
    signature TEXT,
    decorators TEXT,
    bases TEXT,
    is_test INTEGER DEFAULT 0,
    is_entry_point INTEGER DEFAULT 0,
    metadata TEXT,
    PRIMARY KEY (repo_path, symbol_id)
);

CREATE TABLE IF NOT EXISTS repo_chunks (
    repo_path TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    symbol_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    text TEXT NOT NULL,
    line_start INTEGER DEFAULT 0,
    line_end INTEGER DEFAULT 0,
    language TEXT DEFAULT 'python',
    PRIMARY KEY (repo_path, chunk_id)
);

CREATE TABLE IF NOT EXISTS repo_edges (
    repo_path TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    context TEXT
);
CREATE INDEX IF NOT EXISTS idx_repo_edges_source
    ON repo_edges (repo_path, source_id);
CREATE INDEX IF NOT EXISTS idx_repo_edges_target
    ON repo_edges (repo_path, target_id);

CREATE TABLE IF NOT EXISTS repo_file_hashes (
    repo_path TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (repo_path, file_path)
);

CREATE TABLE IF NOT EXISTS repo_index_stats (
    repo_path TEXT PRIMARY KEY,
    stats_json TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
"""


class RepositoryMemoryStore:
    """SQLite persistence for repository memory."""

    def __init__(self, db_path: str | Path = "data/repo_memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def replace_repo(
        self,
        repo_path: str,
        symbols: list[Symbol],
        chunks: list[CodeChunk],
        edges: list[SymbolEdge],
        file_hashes: dict[str, str],
        stats: IndexStats,
    ) -> None:
        """Atomically replace all data for ``repo_path``."""
        with self._conn() as conn:
            self._clear_repo(conn, repo_path)
            self._insert_symbols(conn, repo_path, symbols)
            self._insert_chunks(conn, repo_path, chunks)
            self._insert_edges(conn, repo_path, edges)
            self._insert_hashes(conn, repo_path, file_hashes)
            self._insert_stats(conn, repo_path, stats)
            conn.commit()

    def update_files(
        self,
        repo_path: str,
        symbols: list[Symbol],
        chunks: list[CodeChunk],
        edges: list[SymbolEdge],
        file_hashes: dict[str, str],
        stats: IndexStats,
    ) -> None:
        """Incremental update: upsert symbols/chunks/hashes for changed files.

        Edges are rebuilt for the whole repo (cheap) to stay consistent.
        """
        with self._conn() as conn:
            # Upsert symbols.
            self._insert_symbols(conn, repo_path, symbols)
            # Upsert chunks (delete-then-insert per chunk_id).
            self._insert_chunks(conn, repo_path, chunks)
            # Replace hashes.
            self._insert_hashes(conn, repo_path, file_hashes)
            # Rebuild edges for the repo.
            conn.execute(
                "DELETE FROM repo_edges WHERE repo_path = ?", (repo_path,)
            )
            self._insert_edges(conn, repo_path, edges)
            self._insert_stats(conn, repo_path, stats)
            conn.commit()

    def _clear_repo(self, conn: sqlite3.Connection, repo_path: str) -> None:
        for tbl in (
            "repo_symbols",
            "repo_chunks",
            "repo_edges",
            "repo_file_hashes",
        ):
            conn.execute(
                f"DELETE FROM {tbl} WHERE repo_path = ?", (repo_path,)
            )

    def _insert_symbols(
        self, conn: sqlite3.Connection, repo_path: str, symbols: list[Symbol]
    ) -> None:
        rows = [
            (
                repo_path,
                s.symbol_id,
                s.name,
                s.kind.value,
                s.qualified_name,
                s.file_path,
                s.line_start,
                s.line_end,
                s.docstring,
                s.signature,
                json.dumps(s.decorators),
                json.dumps(s.bases),
                int(s.is_test),
                int(s.is_entry_point),
                json.dumps(s.metadata),
            )
            for s in symbols
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO repo_symbols
            (repo_path, symbol_id, name, kind, qualified_name, file_path,
             line_start, line_end, docstring, signature, decorators, bases,
             is_test, is_entry_point, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _insert_chunks(
        self, conn: sqlite3.Connection, repo_path: str, chunks: list[CodeChunk]
    ) -> None:
        rows = [
            (
                repo_path,
                c.chunk_id,
                c.symbol_id,
                c.file_path,
                c.kind.value,
                c.name,
                c.text,
                c.line_start,
                c.line_end,
                c.language,
            )
            for c in chunks
        ]
        # Delete-then-insert per chunk to handle updates cleanly.
        for r in rows:
            conn.execute(
                "DELETE FROM repo_chunks WHERE repo_path=? AND chunk_id=?",
                (repo_path, r[1]),
            )
        conn.executemany(
            """INSERT INTO repo_chunks
            (repo_path, chunk_id, symbol_id, file_path, kind, name, text,
             line_start, line_end, language)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _insert_edges(
        self, conn: sqlite3.Connection, repo_path: str, edges: list[SymbolEdge]
    ) -> None:
        rows = [
            (repo_path, e.source_id, e.target_id, e.relation.value, e.weight, e.context)
            for e in edges
        ]
        conn.executemany(
            """INSERT INTO repo_edges
            (repo_path, source_id, target_id, relation, weight, context)
            VALUES (?,?,?,?,?,?)""",
            rows,
        )

    def _insert_hashes(
        self,
        conn: sqlite3.Connection,
        repo_path: str,
        hashes: dict[str, str],
    ) -> None:
        rows = [(repo_path, fp, h) for fp, h in hashes.items()]
        conn.executemany(
            """INSERT OR REPLACE INTO repo_file_hashes
            (repo_path, file_path, content_hash) VALUES (?,?,?)""",
            rows,
        )

    def _insert_stats(
        self, conn: sqlite3.Connection, repo_path: str, stats: IndexStats
    ) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO repo_index_stats
            (repo_path, stats_json, indexed_at) VALUES (?,?,?)""",
            (repo_path, stats.model_dump_json(), stats.indexed_at.isoformat()),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_symbols(self, repo_path: str) -> list[Symbol]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM repo_symbols WHERE repo_path = ?", (repo_path,)
            ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def load_chunks(self, repo_path: str) -> list[CodeChunk]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM repo_chunks WHERE repo_path = ?", (repo_path,)
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def load_edges(self, repo_path: str) -> list[SymbolEdge]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM repo_edges WHERE repo_path = ?", (repo_path,)
            ).fetchall()
        return [
            SymbolEdge(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation=__import__(
                    "research_engineer.memory.models", fromlist=["RelationKind"]
                ).RelationKind(r["relation"]),
                weight=r["weight"],
                context=r["context"],
            )
            for r in rows
        ]

    def load_hashes(self, repo_path: str) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash FROM repo_file_hashes WHERE repo_path = ?",
                (repo_path,),
            ).fetchall()
        return {r["file_path"]: r["content_hash"] for r in rows}

    def load_stats(self, repo_path: str) -> IndexStats | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT stats_json FROM repo_index_stats WHERE repo_path = ?",
                (repo_path,),
            ).fetchone()
        if row is None:
            return None
        return IndexStats.model_validate_json(row["stats_json"])

    def has_index(self, repo_path: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM repo_index_stats WHERE repo_path = ? LIMIT 1",
                (repo_path,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Row -> model
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
        from research_engineer.memory.models import SymbolKind

        return Symbol(
            symbol_id=row["symbol_id"],
            name=row["name"],
            kind=SymbolKind(row["kind"]),
            qualified_name=row["qualified_name"],
            file_path=row["file_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            docstring=row["docstring"],
            signature=row["signature"],
            decorators=json.loads(row["decorators"] or "[]"),
            bases=json.loads(row["bases"] or "[]"),
            is_test=bool(row["is_test"]),
            is_entry_point=bool(row["is_entry_point"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> CodeChunk:
        from research_engineer.memory.models import SymbolKind

        return CodeChunk(
            chunk_id=row["chunk_id"],
            symbol_id=row["symbol_id"],
            file_path=row["file_path"],
            kind=SymbolKind(row["kind"]),
            name=row["name"],
            text=row["text"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            language=row["language"],
        )


__all__ = ["RepositoryMemoryStore"]
