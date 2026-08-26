"""Tests for D4 - Better memory (SentenceTransformerEmbedder + ChromaDBBackend).

Backend-agnostic tests that verify the new optional backends implement the
same interface as the defaults and interoperate with HybridRetriever. The
heavy sentence-transformers test is gated on model availability so the suite
passes offline in CI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from research_engineer.memory import (
    ChromaDBBackend,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorBackend,
    SymbolGraph,
    VectorBackend,
    is_chromadb_available,
    is_sentence_transformer_available,
)
from research_engineer.memory.embeddings import (
    EmbedderBackend,
    SentenceTransformerEmbedder,
)


class TestExports:
    def test_new_classes_exported(self):
        assert ChromaDBBackend is not None
        assert SentenceTransformerEmbedder is not None

    def test_availability_helpers(self):
        assert isinstance(is_chromadb_available(), bool)
        assert isinstance(is_sentence_transformer_available(), bool)


class TestChromaDBBackendInterface:
    def test_is_vector_backend(self):
        assert issubclass(ChromaDBBackend, VectorBackend)

    def test_has_required_methods(self):
        for m in ("add", "search", "delete", "count", "clear"):
            assert hasattr(ChromaDBBackend, m)

    def test_name_attr(self):
        assert ChromaDBBackend.name == "chromadb"


class TestChromaDBBackendFunctional:
    """Functional tests using an in-memory ChromaDB client (no disk I/O)."""

    @pytest.fixture
    def backend(self):
        if not is_chromadb_available():
            pytest.skip("chromadb not installed")
        b = ChromaDBBackend(collection_name="test_d4", persist_path=None)
        yield b
        b.clear()

    def test_add_and_count(self, backend):
        backend.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [{"name": "a"}, {"name": "b"}])
        assert backend.count() == 2

    def test_search_returns_similar(self, backend):
        backend.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"k": "a"}, {"k": "b"}])
        results = backend.search([1.0, 0.0], limit=2)
        assert len(results) == 2
        top_id, top_score, top_meta = results[0]
        assert top_id == "a"
        assert top_score > 0.5
        assert top_meta["k"] == "a"

    def test_search_empty_backend(self, backend):
        assert backend.search([1.0, 0.0], limit=5) == []

    def test_delete(self, backend):
        backend.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{}, {}])
        backend.delete(["a"])
        assert backend.count() == 1

    def test_clear(self, backend):
        backend.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{}, {}])
        assert backend.count() == 2
        backend.clear()
        assert backend.count() == 0

    def test_none_payload_values_coerced(self, backend):
        backend.add(["a"], [[1.0, 0.0]], [{"name": "a", "doc": None}])
        results = backend.search([1.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0][2]["name"] == "a"


class TestSentenceTransformerEmbedderInterface:
    def test_is_embedder_backend(self):
        assert issubclass(SentenceTransformerEmbedder, EmbedderBackend)

    def test_name_attr(self):
        assert SentenceTransformerEmbedder.name == "sentence_transformer"

    def test_import_error_message_when_unavailable(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerEmbedder()


class TestSentenceTransformerEmbedderFunctional:
    """Gated on the model being loadable; skipped in offline CI."""

    @pytest.fixture
    def embedder(self):
        if not is_sentence_transformer_available():
            pytest.skip("sentence-transformers not installed")
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pytest.skip("could not load sentence-transformer model (offline)")

    def test_dim_matches_model(self, embedder):
        assert embedder.dim > 0

    def test_embed_produces_normalized_vectors(self, embedder):
        vecs = embedder.embed(["hello world", "code function"])
        assert len(vecs) == 2
        assert len(vecs[0]) == embedder.dim
        for v in vecs:
            norm = math.sqrt(sum(x * x for x in v))
            assert 0.9 < norm <= 1.1

    def test_embed_empty_list(self, embedder):
        assert embedder.embed([]) == []

    def test_embed_query_single(self, embedder):
        v = embedder.embed_query("test query")
        assert len(v) == embedder.dim


class TestHybridRetrieverWithChromaDB:
    """Verify HybridRetriever works across the ChromaDB backend."""

    def test_retriever_uses_chromadb_backend(self):
        if not is_chromadb_available():
            pytest.skip("chromadb not installed")
        from research_engineer.memory import CodeChunk, Symbol, SymbolKind
        from research_engineer.memory.vector_store import chunk_payload

        backend = ChromaDBBackend(collection_name="hybrid_test", persist_path=None)
        embedder = HashingEmbedder(dim=256)
        graph = SymbolGraph()
        sym = Symbol(
            symbol_id="s1", name="train", kind=SymbolKind.FUNCTION,
            qualified_name="pkg.train", file_path="a.py",
            line_start=1, line_end=10,
            docstring="trains the model", is_test=False, is_entry_point=False,
        )
        chunk = CodeChunk(
            chunk_id="c1", symbol_id="s1", file_path="a.py", name="train",
            kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
            text="def train(): pass", language="python",
        )
        graph.add_symbol(sym)
        vec = embedder.embed(["train model function"])[0]
        backend.add(["c1"], [vec], [chunk_payload(chunk)])
        retriever = HybridRetriever(
            backend, embedder, graph, {"s1": sym}, {"c1": chunk}
        )
        results = retriever.retrieve("train model", limit=5)
        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c1"


class TestChromaDBPersistence:
    def test_persist_path_creates_db(self, tmp_path: Path):
        if not is_chromadb_available():
            pytest.skip("chromadb not installed")
        path = tmp_path / "chroma"
        backend = ChromaDBBackend(collection_name="persist_test", persist_path=str(path))
        backend.add(["x"], [[1.0, 0.0]], [{"name": "x"}])
        assert backend.count() == 1
        backend2 = ChromaDBBackend(collection_name="persist_test", persist_path=str(path))
        assert backend2.count() == 1


class TestBuildRepositoryMemoryFactory:
    """Config-driven construction (llm_config.yaml embedding/vector_store)."""

    def _write_config(self, path: Path, text: str) -> str:
        path.write_text(text)
        return str(path)

    def _make(self, tmp_path: Path, cfg_text: str) -> Any:
        from research_engineer.memory.factory import build_repository_memory
        from research_engineer.memory.storage import RepositoryMemoryStore

        cfg = self._write_config(tmp_path / "cfg.yaml", cfg_text)
        store = RepositoryMemoryStore(str(tmp_path / "store.db"))
        return build_repository_memory(".", config_path=cfg, store=store)

    def test_defaults_when_no_sections(self, tmp_path: Path):
        mem = self._make(tmp_path, "default_provider: ollama\n")
        assert isinstance(mem.embedder, HashingEmbedder)
        assert isinstance(mem.vector, InMemoryVectorBackend)

    def test_empty_file_uses_defaults(self, tmp_path: Path):
        mem = self._make(tmp_path, "")
        assert isinstance(mem.embedder, HashingEmbedder)
        assert isinstance(mem.vector, InMemoryVectorBackend)

    def test_unknown_backend_names_fall_back(self, tmp_path: Path):
        mem = self._make(
            tmp_path,
            "embedding:\n  backend: bogus\nvector_store:\n  backend: also_bogus\n",
        )
        assert isinstance(mem.embedder, HashingEmbedder)
        assert isinstance(mem.vector, InMemoryVectorBackend)

    def test_chromadb_backend_selected(self, tmp_path: Path):
        if not is_chromadb_available():
            pytest.skip("chromadb not installed")
        persist = tmp_path / "vstore"
        mem = self._make(
            tmp_path,
            "vector_store:\n"
            "  backend: chromadb\n"
            f"  persist_path: {persist}\n"
            "  collection: cfg_test\n",
        )
        assert isinstance(mem.vector, ChromaDBBackend)
        mem.vector.clear()

    def test_sentence_transformer_requested_but_missing(self, tmp_path: Path):
        import sys as _sys

        monkey = pytest.MonkeyPatch()
        try:
            monkey.setitem(_sys.modules, "sentence_transformers", None)
            mem = self._make(
                tmp_path, "embedding:\n  backend: sentence_transformer\n"
            )
        finally:
            monkey.undo()
        # Graceful fallback to the offline embedder.
        assert isinstance(mem.embedder, HashingEmbedder)

    def test_build_and_query_end_to_end(self, tmp_path: Path):
        from research_engineer.memory.factory import build_repository_memory
        from research_engineer.memory.storage import RepositoryMemoryStore

        repo = tmp_path / "repo"
        (repo / "pkg").mkdir(parents=True)
        (repo / "pkg" / "mod.py").write_text(
            "def train_model():\n    '''Trains the model.'''\n    pass\n"
        )
        store = RepositoryMemoryStore(str(tmp_path / "store.db"))
        cfg = self._write_config(tmp_path / "cfg.yaml", "")
        mem = build_repository_memory(str(repo), config_path=cfg, store=store)
        stats = mem.build()
        assert stats.total_symbols > 0
        results = mem.query("train model", limit=3)
        assert isinstance(results, list)
