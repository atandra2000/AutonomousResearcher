"""Persistent vector backend backed by ChromaDB (optional, D4).

Provides a :class:`ChromaDBBackend` that implements the same
:class:`~research_engineer.memory.vector_store.VectorBackend` interface as
the in-memory default, but persists to disk via ChromaDB. This lets
repository memory survive across runs and scale beyond the brute-force
in-memory index.

The ``chromadb`` dependency is imported lazily on first use so the module
imports cleanly even when ChromaDB is not installed; construction raises a
clear :class:`ImportError` only when the backend is actually instantiated.
"""

from __future__ import annotations

from typing import Any

from research_engineer.memory.vector_store import VectorBackend


class ChromaDBBackend(VectorBackend):
    """Persistent vector index backed by ChromaDB.

    Implements the ``add`` / ``search`` / ``delete`` / ``count`` / ``clear``
    interface of :class:`VectorBackend`. Vectors and payloads are persisted
    to a ChromaDB collection on disk (or in-memory when ``persist_path`` is
    ``None``). Cosine similarity is approximated by ChromaDB's ``cosine``
    distance metric, and results are returned as ``(id, score, payload)``
    triples ranked by descending similarity.

    Parameters
    ----------
    collection_name:
        Name of the ChromaDB collection to use.
    persist_path:
        Directory for the persistent DuckDB backend. ``None`` uses an
        in-memory client (lost when the process exits).
    """

    name = "chromadb"

    def __init__(
        self,
        collection_name: str = "repository_memory",
        persist_path: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "ChromaDBBackend requires the 'chromadb' package. "
                "Install it with: uv pip install chromadb"
            ) from e
        self._collection_name = collection_name
        self._persist_path = persist_path
        if persist_path is not None:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.Client()
        # Use cosine distance; ChromaDB stores L2-normalized vectors when
        # distance="cosine" so the returned distances are 1 - cosine_sim.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        # ChromaDB requires non-empty metadata dicts with non-None values.
        # Coerce None -> "" and ensure each dict has at least one key.
        clean_meta: list[dict[str, Any]] = []
        for p in payloads:
            meta = {k: ("" if v is None else v) for k, v in p.items()}
            if not meta:
                meta = {"_placeholder": ""}
            clean_meta.append(meta)
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=clean_meta,
        )

    def search(
        self, query: list[float], limit: int = 10, filter: dict[str, Any] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if self.count() == 0:
            return []
        where = filter or None
        res = self._collection.query(
            query_embeddings=[query],
            n_results=limit,
            where=where,
        )
        out: list[tuple[str, float, dict[str, Any]]] = []
        ids_batch = res.get("ids", [[]])
        dists_batch = res.get("distances", [[]])
        metas_batch = res.get("metadatas", [[]])
        if not ids_batch:
            return []
        ids = ids_batch[0]
        dists = dists_batch[0] if dists_batch else [0.0] * len(ids)
        metas = metas_batch[0] if metas_batch else [{}] * len(ids)
        for cid, dist, meta in zip(ids, dists, metas):
            # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
            # Convert to a similarity score in [-1, 1], then clamp.
            sim = max(-1.0, min(1.0, 1.0 - float(dist)))
            out.append((str(cid), sim, dict(meta) if meta else {}))
        return out

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=ids)

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def is_chromadb_available() -> bool:
    """Return True if ``chromadb`` can be imported."""
    try:
        import chromadb  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["ChromaDBBackend", "is_chromadb_available"]
