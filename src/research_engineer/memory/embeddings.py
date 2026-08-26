"""Phase 12 - Embedding abstraction.

Pluggable embedding backend for repository memory. The default
:class:`HashingEmbedder` is a deterministic, dependency-free TF-IDF-style
embedder suitable for tests and offline use. Production deployments can
swap in a sentence-transformers or API-backed embedder by registering a
new :class:`EmbedderBackend` subclass.

All embedders produce L2-normalized float vectors of a fixed dimension
so they are backend-agnostic and interchangeable with the vector store.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable

from research_engineer.memory.models import CodeChunk


class EmbedderBackend:
    """Abstract embedding backend."""

    name: str = "abstract"
    dim: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder(EmbedderBackend):
    """Deterministic hashing + TF embedding (no external deps).

    Uses feature hashing into ``dim`` buckets with term-frequency
    weighting, then L2-normalizes. This is *not* a semantic embedder but
    gives reasonable lexical overlap for hybrid retrieval ranking and
    works offline in CI. Swap for a real embedder in production.
    """

    name = "hashing"
    dim = 256

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        # Simple, language-agnostic tokenizer: split on non-alphanumeric.
        tokens: list[str] = []
        cur: list[str] = []
        for ch in text.lower():
            if ch.isalnum() or ch == "_":
                cur.append(ch)
            else:
                if cur:
                    tokens.append("".join(cur))
                    cur = []
        if cur:
            tokens.append("".join(cur))
        return tokens

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            counts = Counter(self._tokenize(text))
            for token, tf in counts.items():
                # Hash token to a bucket; sign alternates to reduce collisions.
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                bucket = h % self.dim
                sign = 1.0 if (h >> self.dim.bit_length()) & 1 else -1.0
                vec[bucket] += sign * math.log1p(tf)
            # L2 normalize.
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vecs.append([v / norm for v in vec])
        return vecs

    def embed_chunks(self, chunks: Iterable[CodeChunk]) -> list[list[float]]:
        """Embed chunks using name + docstring + text for richer context."""
        texts = [
            f"{c.name} {c.kind.value} {c.text[:1000]}" for c in chunks
        ]
        return self.embed(texts)


class SentenceTransformerEmbedder(EmbedderBackend):
    """Semantic embedder backed by ``sentence-transformers`` (optional, D4).

    Loads a real transformer model (default: ``all-MiniLM-L6-v2``) to produce
    semantically meaningful embeddings, unlike the lexical :class:`HashingEmbedder`.
    The dependency is imported lazily on first use so the module imports cleanly
    even when ``sentence-transformers`` is not installed; construction raises a
    clear :class:`ImportError` only when the embedder is actually instantiated.

    The output dimension matches the loaded model (384 for MiniLM). Vectors are
    L2-normalized by the underlying model so they interoperate directly with the
    cosine-similarity :class:`~research_engineer.memory.vector_store.VectorBackend`.
    """

    name = "sentence_transformer"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'sentence-transformers' "
                "package. Install it with: uv pip install sentence-transformers"
            ) from e
        self._model_name = model_name
        self._normalize = normalize_embeddings
        # Load eagerly so failures surface at construction time. ``device=None``
        # lets the library auto-select CPU/GPU.
        self._model = SentenceTransformer(model_name, device=device)
        # Cache the dimension. Prefer the non-deprecated method name; fall
        # back to the legacy one for older sentence-transformers versions.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = getattr(self._model, "get_sentence_embedding_dimension")
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return [list(map(float, row)) for row in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_chunks(self, chunks: Iterable[CodeChunk]) -> list[list[float]]:
        texts = [f"{c.name} {c.kind.value} {c.text[:1000]}" for c in chunks]
        return self.embed(texts)


def is_sentence_transformer_available() -> bool:
    """Return True if ``sentence-transformers`` can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


__all__ = [
    "EmbedderBackend",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "is_sentence_transformer_available",
]
