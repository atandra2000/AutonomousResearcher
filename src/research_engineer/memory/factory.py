"""Config-driven construction of repository memory backends (D4).

Reads the optional ``embedding`` / ``vector_store`` sections of
``llm_config.yaml`` (resolved via
:func:`research_engineer.llm.factory.load_config`, which honors the
``RE_LLM_CONFIG`` environment variable) and builds a
:class:`~research_engineer.memory.repository_memory.RepositoryMemory`
wired to the configured backends.

Missing sections keep the dependency-free defaults
(:class:`HashingEmbedder` + :class:`InMemoryVectorBackend`) so the system
works offline and in CI. Configured backends whose optional dependencies
are not installed (or whose model fails to load) fall back to the
defaults instead of raising, so a bad config never breaks construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_engineer.llm.factory import load_config
from research_engineer.memory.embeddings import (
    EmbedderBackend,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    is_sentence_transformer_available,
)
from research_engineer.memory.repository_memory import RepositoryMemory
from research_engineer.memory.storage import RepositoryMemoryStore
from research_engineer.memory.vector_backend import (
    ChromaDBBackend,
    is_chromadb_available,
)
from research_engineer.memory.vector_store import (
    InMemoryVectorBackend,
    VectorBackend,
)

_DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

__all__ = ["build_repository_memory"]


def _build_embedder(cfg: dict[str, Any]) -> EmbedderBackend:
    """Build the embedder requested by the ``embedding`` config section."""
    backend = str(cfg.get("backend", "hashing")).lower()
    if backend == "sentence_transformer":
        if is_sentence_transformer_available():
            model = str(cfg.get("model") or _DEFAULT_ST_MODEL)
            try:
                return SentenceTransformerEmbedder(model)
            except Exception:
                # Model download/load failure: graceful fallback below.
                pass
        # Dependency missing or model failed to load: use the offline default.
    return HashingEmbedder()


def _build_vector_backend(cfg: dict[str, Any]) -> VectorBackend:
    """Build the backend requested by the ``vector_store`` config section."""
    backend = str(cfg.get("backend", "inmemory")).lower()
    if backend == "chromadb" and is_chromadb_available():
        collection = str(cfg.get("collection") or "repository_memory")
        persist = cfg.get("persist_path")
        return ChromaDBBackend(
            collection_name=collection,
            persist_path=str(persist) if persist else None,
        )
    return InMemoryVectorBackend()


def build_repository_memory(
    repo_path: str,
    *,
    config_path: str | Path | None = None,
    store: RepositoryMemoryStore | None = None,
) -> RepositoryMemory:
    """Construct a :class:`RepositoryMemory` from ``llm_config.yaml``.

    Parameters
    ----------
    repo_path:
        Repository root to index/serve.
    config_path:
        Explicit path to a YAML config. ``None`` resolves the default
        (``RE_LLM_CONFIG`` env var, then ``llm_config.yaml`` at the repo
        root or CWD).
    store:
        Optional shared SQLite store; defaults to the facade's standard
        ``data/repo_memory.db``.

    Returns
    -------
    RepositoryMemory
        Wired with the configured embedder and vector backend, falling
        back to the dependency-free defaults for any missing/unavailable
        option.
    """
    cfg = load_config(config_path)
    embedding_cfg = cfg.get("embedding")
    vector_cfg = cfg.get("vector_store")
    if not isinstance(embedding_cfg, dict):
        embedding_cfg = {}
    if not isinstance(vector_cfg, dict):
        vector_cfg = {}
    return RepositoryMemory(
        repo_path,
        store=store,
        embedder=_build_embedder(embedding_cfg),
        vector_backend=_build_vector_backend(vector_cfg),
    )
