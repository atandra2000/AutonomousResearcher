"""Phase 12 - Vector storage abstraction.

A minimal, dependency-free in-memory vector index with cosine
similarity search. Designed as the default backend for repository
memory so the system works offline and in CI without ChromaDB or FAISS.

Production deployments can register a heavier backend (ChromaDB, FAISS,
pgvector) by subclassing :class:`VectorBackend` and passing it to
:class:`~research_engineer.memory.repository_memory.RepositoryMemory`.
The interface is intentionally small: ``add``, ``search``, ``delete``,
``count``.
"""

from __future__ import annotations

import math
from typing import Any

from research_engineer.memory.models import CodeChunk


class VectorBackend:
    """Abstract vector storage backend."""

    name: str = "abstract"

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def search(
        self, query: list[float], limit: int = 10, filter: dict[str, Any] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Return ``(id, score, payload)`` triples ranked by cosine similarity."""
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryVectorBackend(VectorBackend):
    """Brute-force cosine similarity over in-memory vectors.

    Suitable for repositories up to ~100k chunks. For larger corpora,
    swap in an ANN-backed backend (FAISS/HNSW). The brute-force scan is
    O(n*d) per query which is fine for interactive use at this scale.
    """

    name = "inmemory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._payloads: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        for cid, vec, payload in zip(ids, vectors, payloads):
            if cid in self._index:
                idx = self._index[cid]
                self._vectors[idx] = vec
                self._payloads[idx] = payload
            else:
                self._index[cid] = len(self._ids)
                self._ids.append(cid)
                self._vectors.append(vec)
                self._payloads.append(payload)

    def search(
        self, query: list[float], limit: int = 10, filter: dict[str, Any] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._vectors:
            return []
        scored: list[tuple[float, int]] = []
        qnorm = math.sqrt(sum(x * x for x in query)) or 1.0
        for i, vec in enumerate(self._vectors):
            if filter and not self._matches(self._payloads[i], filter):
                continue
            dot = sum(a * b for a, b in zip(query, vec))
            vnorm = math.sqrt(sum(x * x for x in vec)) or 1.0
            cos = dot / (qnorm * vnorm)
            scored.append((cos, i))
        scored.sort(reverse=True)
        return [
            (self._ids[i], score, self._payloads[i])
            for score, i in scored[:limit]
        ]

    def delete(self, ids: list[str]) -> None:
        keep_ids = set(ids)
        new_ids, new_vecs, new_payloads, new_index = [], [], [], {}
        for i, cid in enumerate(self._ids):
            if cid in keep_ids:
                continue
            new_index[cid] = len(new_ids)
            new_ids.append(cid)
            new_vecs.append(self._vectors[i])
            new_payloads.append(self._payloads[i])
        self._ids, self._vectors, self._payloads, self._index = (
            new_ids,
            new_vecs,
            new_payloads,
            new_index,
        )

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._vectors.clear()
        self._payloads.clear()
        self._index.clear()

    @staticmethod
    def _matches(payload: dict[str, Any], filter: dict[str, Any]) -> bool:
        for k, v in filter.items():
            if payload.get(k) != v:
                return False
        return True


def chunk_payload(chunk: CodeChunk) -> dict[str, Any]:
    """Build a searchable payload dict for a code chunk."""
    return {
        "chunk_id": chunk.chunk_id,
        "symbol_id": chunk.symbol_id,
        "file_path": chunk.file_path,
        "kind": chunk.kind.value,
        "name": chunk.name,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "language": chunk.language,
    }


__all__ = ["VectorBackend", "InMemoryVectorBackend", "chunk_payload"]
