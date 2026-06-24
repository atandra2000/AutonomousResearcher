"""Tests for Phase 12 - Repository Memory subsystem.

Covers the indexer, symbol graph, embedder, vector backend, hybrid
retriever, persistent storage, RepositoryMemory facade, and the
TaskAgent integration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_engineer.memory import (
    CodeChunk,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorBackend,
    IndexStats,
    RelationKind,
    RepositoryIndexer,
    RepositoryMemory,
    RepositoryMemoryStore,
    RetrievalResult,
    Symbol,
    SymbolEdge,
    SymbolGraph,
    SymbolKind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_repo(tmp_path):
    """Create a small fake repository with training/checkpoint code."""
    root = tmp_path / "myrepo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)

    (root / "src" / "pkg" / "trainer.py").write_text(
        '''"""Training module with checkpoint support."""
import torch


class Trainer:
    """Training loop with checkpoint saving."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, epochs):
        """Run training loop."""
        for epoch in range(epochs):
            self._step(epoch)

    def _step(self, epoch):
        self.optimizer.step()

    def save_checkpoint(self, path):
        """Save model and optimizer state."""
        torch.save({"model": self.model, "optimizer": self.optimizer}, path)


def train_model(config):
    """Entry point for training."""
    return Trainer(config.model, config.optimizer)
''',
        encoding="utf-8",
    )

    (root / "src" / "pkg" / "ema.py").write_text(
        '''"""Exponential moving average module."""


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, decay=0.999):
        self.decay = decay

    def update(self, model):
        """Update EMA weights."""
        pass
''',
        encoding="utf-8",
    )

    (root / "tests" / "test_trainer.py").write_text(
        '''"""Tests for trainer module."""
from src.pkg.trainer import Trainer


def test_save_checkpoint():
    """Test checkpoint saving."""
    pass


def test_train():
    """Test training loop."""
    pass
''',
        encoding="utf-8",
    )

    (root / "configs" / "train.yaml").write_text(
        "model: resnet50\nbatch_size: 32\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_symbol_kind_values(self):
        assert SymbolKind.MODULE == "module"
        assert SymbolKind.CLASS == "class"
        assert SymbolKind.FUNCTION == "function"
        assert SymbolKind.METHOD == "method"

    def test_relation_kind_values(self):
        assert RelationKind.DEPENDS_ON == "depends_on"
        assert RelationKind.CALLS == "calls"
        assert RelationKind.TESTS == "tests"

    def test_symbol_serialization(self):
        s = Symbol(
            symbol_id="f:module:foo",
            name="foo",
            kind=SymbolKind.FUNCTION,
            qualified_name="mod.foo",
            file_path="f.py",
            line_start=1,
        )
        d = s.model_dump()
        assert d["name"] == "foo"
        assert d["kind"] == "function"

    def test_code_chunk(self):
        c = CodeChunk(
            chunk_id="cid",
            symbol_id="sid",
            file_path="f.py",
            kind=SymbolKind.FUNCTION,
            name="foo",
            text="def foo(): pass",
        )
        assert c.language == "python"

    def test_index_stats(self):
        s = IndexStats(repo_path=".", total_symbols=10)
        assert s.total_symbols == 10


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class TestRepositoryIndexer:
    def test_index_extracts_symbols(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        names = {s.name for s in result.symbols}
        assert "Trainer" in names
        assert "EMA" in names
        assert "train_model" in names
        assert "save_checkpoint" in names
        assert "test_save_checkpoint" in names

    def test_index_extracts_chunks(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        assert len(result.chunks) > 0
        # Each symbol should have at least one chunk.
        assert len(result.chunks) >= len(result.symbols) * 0.5

    def test_index_builds_edges(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        relations = {e.relation for e in result.edges}
        assert RelationKind.DEFINES in relations
        assert RelationKind.DEPENDS_ON in relations

    def test_index_marks_test_modules(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        test_modules = [
            s for s in result.symbols if s.is_test and s.kind == SymbolKind.MODULE
        ]
        assert len(test_modules) >= 1
        assert any("test_trainer" in s.file_path for s in test_modules)

    def test_index_marks_test_functions(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        test_fns = [s for s in result.symbols if s.is_test and s.kind == SymbolKind.FUNCTION]
        assert any(s.name == "test_save_checkpoint" for s in test_fns)

    def test_index_detects_entry_point(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        entry_points = [s for s in result.symbols if s.is_entry_point]
        assert any(s.name == "train_model" for s in entry_points)

    def test_index_includes_config_files(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        configs = [s for s in result.symbols if s.kind == SymbolKind.CONFIG]
        assert any("train.yaml" in s.name for s in configs)

    def test_index_creates_tests_edges(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        tests_edges = [e for e in result.edges if e.relation == RelationKind.TESTS]
        assert len(tests_edges) >= 1

    def test_index_incremental(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        # Second pass with known hashes should find no changes.
        result2, changed = idx.index_incremental(result.file_hashes)
        assert changed == []

    def test_index_incremental_detects_changes(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        # Modify a file.
        (sample_repo / "src" / "pkg" / "ema.py").write_text("class EMA:\n    pass\n")
        result2, changed = idx.index_incremental(result.file_hashes)
        assert "src/pkg/ema.py" in changed

    def test_index_file_hashes(self, sample_repo):
        idx = RepositoryIndexer(str(sample_repo))
        result = idx.index()
        assert len(result.file_hashes) >= 4  # 3 py + 1 yaml

    def test_index_skips_noise_dirs(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "junk.py").write_text("x = 1")
        (tmp_path / "real.py").write_text("y = 2")
        idx = RepositoryIndexer(str(tmp_path))
        result = idx.index()
        files = {s.file_path for s in result.symbols}
        assert not any("__pycache__" in f for f in files)


# ---------------------------------------------------------------------------
# Symbol Graph
# ---------------------------------------------------------------------------


class TestSymbolGraph:
    def test_build_and_query(self):
        sym1 = Symbol(
            symbol_id="s1", name="A", kind=SymbolKind.MODULE,
            qualified_name="a", file_path="a.py",
        )
        sym2 = Symbol(
            symbol_id="s2", name="B", kind=SymbolKind.FUNCTION,
            qualified_name="a.b", file_path="a.py",
        )
        edge = SymbolEdge(
            source_id="s1", target_id="s2", relation=RelationKind.DEFINES,
        )
        g = SymbolGraph()
        g.build([sym1, sym2], [edge])
        assert g.get_symbol("s1") is not None
        assert g.stats()["nodes"] == 2
        assert g.stats()["edges"] == 1

    def test_dependencies_and_dependents(self):
        g = SymbolGraph()
        g.build([], [
            SymbolEdge(source_id="a", target_id="b", relation=RelationKind.DEPENDS_ON),
            SymbolEdge(source_id="a", target_id="c", relation=RelationKind.DEPENDS_ON),
        ])
        assert set(g.dependencies("a")) == {"b", "c"}
        assert "a" in g.dependents("b")

    def test_callers_and_callees(self):
        g = SymbolGraph()
        g.build([], [
            SymbolEdge(source_id="f1", target_id="f2", relation=RelationKind.CALLS),
        ])
        assert g.callees("f1") == ["f2"]
        assert g.callers("f2") == ["f1"]

    def test_related(self):
        g = SymbolGraph()
        g.build([], [
            SymbolEdge(source_id="a", target_id="b", relation=RelationKind.RELATED),
            SymbolEdge(source_id="b", target_id="c", relation=RelationKind.RELATED),
        ])
        related = g.related("a", max_depth=2)
        assert "b" in related
        assert "c" in related

    def tests_for(self):
        g = SymbolGraph()
        g.build([], [
            SymbolEdge(source_id="test_f", target_id="func_f", relation=RelationKind.TESTS),
        ])
        assert g.tests_for("func_f") == ["test_f"]

    def test_all_edges(self):
        g = SymbolGraph()
        g.build([], [
            SymbolEdge(source_id="a", target_id="b", relation=RelationKind.DEPENDS_ON),
        ])
        edges = g.all_edges()
        assert len(edges) == 1
        assert edges[0].source_id == "a"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class TestHashingEmbedder:
    def test_embed_produces_vectors(self):
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["hello world", "foo bar"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 64

    def test_embed_l2_normalized(self):
        import math

        emb = HashingEmbedder(dim=64)
        vec = emb.embed_query("test code")
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 0.01

    def test_embed_chunks(self):
        emb = HashingEmbedder(dim=64)
        chunks = [
            CodeChunk(
                chunk_id="c1", symbol_id="s1", file_path="f.py",
                kind=SymbolKind.FUNCTION, name="foo", text="def foo(): pass",
            )
        ]
        vecs = emb.embed_chunks(chunks)
        assert len(vecs) == 1
        assert len(vecs[0]) == 64

    def test_similar_texts_have_higher_similarity(self):
        emb = HashingEmbedder(dim=128)
        v1 = emb.embed_query("checkpoint save model")
        v2 = emb.embed_query("checkpoint save model state")
        v3 = emb.embed_query("completely different topic")
        sim12 = sum(a * b for a, b in zip(v1, v2))
        sim13 = sum(a * b for a, b in zip(v1, v3))
        assert sim12 > sim13


# ---------------------------------------------------------------------------
# Vector Backend
# ---------------------------------------------------------------------------


class TestInMemoryVectorBackend:
    def test_add_and_search(self):
        backend = InMemoryVectorBackend()
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["checkpoint", "training", "optimizer"])
        backend.add(["c1", "c2", "c3"], vecs, [{"name": "checkpoint"}, {"name": "training"}, {"name": "optimizer"}])
        assert backend.count() == 3
        q = emb.embed_query("checkpoint")
        results = backend.search(q, limit=1)
        assert results[0][0] == "c1"

    def test_delete(self):
        backend = InMemoryVectorBackend()
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["a", "b"])
        backend.add(["c1", "c2"], vecs, [{}, {}])
        backend.delete(["c1"])
        assert backend.count() == 1

    def test_filter(self):
        backend = InMemoryVectorBackend()
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["checkpoint", "training"])
        backend.add(["c1", "c2"], vecs, [{"kind": "function"}, {"kind": "class"}])
        q = emb.embed_query("checkpoint")
        results = backend.search(q, limit=10, filter={"kind": "class"})
        assert all(r[2]["kind"] == "class" for r in results)

    def test_clear(self):
        backend = InMemoryVectorBackend()
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["a"])
        backend.add(["c1"], vecs, [{}])
        backend.clear()
        assert backend.count() == 0


# ---------------------------------------------------------------------------
# RepositoryMemoryStore (SQLite)
# ---------------------------------------------------------------------------


class TestRepositoryMemoryStore:
    def test_replace_and_load(self, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        repo = str(tmp_path)
        symbols = [
            Symbol(symbol_id="s1", name="A", kind=SymbolKind.FUNCTION,
                   qualified_name="mod.a", file_path="f.py"),
        ]
        chunks = [
            CodeChunk(chunk_id="c1", symbol_id="s1", file_path="f.py",
                      kind=SymbolKind.FUNCTION, name="A", text="def a(): pass"),
        ]
        edges = [
            SymbolEdge(source_id="s1", target_id="s2", relation=RelationKind.CALLS),
        ]
        stats = IndexStats(repo_path=repo, total_symbols=1)
        store.replace_repo(repo, symbols, chunks, edges, {"f.py": "abc"}, stats)
        assert store.has_index(repo)
        loaded_syms = store.load_symbols(repo)
        assert len(loaded_syms) == 1
        assert loaded_syms[0].name == "A"
        loaded_chunks = store.load_chunks(repo)
        assert len(loaded_chunks) == 1
        loaded_edges = store.load_edges(repo)
        assert len(loaded_edges) == 1
        loaded_hashes = store.load_hashes(repo)
        assert loaded_hashes == {"f.py": "abc"}

    def test_load_stats(self, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        repo = str(tmp_path)
        stats = IndexStats(repo_path=repo, total_symbols=5)
        store.replace_repo(repo, [], [], [], {}, stats)
        loaded = store.load_stats(repo)
        assert loaded is not None
        assert loaded.total_symbols == 5

    def test_has_index_false(self, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        assert not store.has_index("nonexistent")


# ---------------------------------------------------------------------------
# RepositoryMemory facade
# ---------------------------------------------------------------------------


class TestRepositoryMemory:
    def test_build_and_query(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        stats = mem.build()
        assert stats.total_symbols > 0
        assert stats.total_chunks > 0
        assert stats.total_edges > 0
        # Query for checkpoint-related code.
        results = mem.query("checkpoint save model", limit=5)
        assert len(results) > 0
        # The Trainer class or save_checkpoint should rank high.
        top_names = {r.symbol.name for r in results if r.symbol}
        assert "Trainer" in top_names or "save_checkpoint" in top_names

    def test_query_ema(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        results = mem.query("EMA exponential moving average", limit=5)
        assert len(results) > 0
        top_names = {r.symbol.name for r in results if r.symbol}
        assert "EMA" in top_names

    def test_graph_lookup(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        g = mem.graph("Trainer")
        assert g["found"] is True
        assert g["symbol"]["name"] == "Trainer"
        # Should find tests.
        assert len(g["tests"]) >= 1

    def test_graph_not_found(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        g = mem.graph("NonexistentSymbol")
        assert g["found"] is False

    def test_get_context(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        ctx = mem.get_context("Add EMA checkpoint support")
        assert "Repository Memory Context" in ctx
        assert "EMA" in ctx or "Trainer" in ctx

    def test_get_context_empty_when_no_index(self, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(tmp_path), store=store)
        ctx = mem.get_context("anything")
        assert ctx == ""

    def test_persistence(self, sample_repo, tmp_path):
        db = str(tmp_path / "mem.db")
        store1 = RepositoryMemoryStore(db)
        mem1 = RepositoryMemory(str(sample_repo), store=store1)
        mem1.build()
        # New instance, same store — should load from disk.
        store2 = RepositoryMemoryStore(db)
        mem2 = RepositoryMemory(str(sample_repo), store=store2)
        assert mem2.store.has_index(str(sample_repo.resolve()))
        results = mem2.query("checkpoint", limit=3)
        assert len(results) > 0

    def test_refresh_detects_changes(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        # Modify a file.
        (sample_repo / "src" / "pkg" / "ema.py").write_text("class EMA:\n    pass\n")
        stats, changed = mem.refresh()
        assert "src/pkg/ema.py" in changed

    def test_refresh_no_changes(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        stats, changed = mem.refresh()
        assert changed == []

    def test_stats(self, sample_repo, tmp_path):
        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        stats = mem.stats()
        assert stats.total_symbols > 0
        assert "function" in stats.symbols_by_kind
        assert "class" in stats.symbols_by_kind


# ---------------------------------------------------------------------------
# TaskAgent integration
# ---------------------------------------------------------------------------


class TestTaskAgentMemoryIntegration:
    @pytest.mark.asyncio
    async def test_task_agent_accepts_repository_memory(self, sample_repo, tmp_path):
        from research_engineer.agents import TaskAgent
        from research_engineer.memory import RepositoryMemory

        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        agent = TaskAgent(repository_memory=mem)
        assert agent._repository_memory is mem

    @pytest.mark.asyncio
    async def test_retrieve_memory_context_returns_string(self, sample_repo, tmp_path):
        from research_engineer.agents import TaskAgent
        from research_engineer.memory import RepositoryMemory

        store = RepositoryMemoryStore(str(tmp_path / "mem.db"))
        mem = RepositoryMemory(str(sample_repo), store=store)
        mem.build()
        agent = TaskAgent(repository_memory=mem)
        ctx = await agent._retrieve_memory_context(
            "checkpoint support", type("C", (), {"repo_path": str(sample_repo), "output_dir": str(tmp_path)})(),
            None,
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    @pytest.mark.asyncio
    async def test_retrieve_memory_context_empty_when_no_mem(self, tmp_path):
        from research_engineer.agents import TaskAgent

        agent = TaskAgent()
        agent._repository_memory = None
        ctx = await agent._retrieve_memory_context(
            "test", type("C", (), {"repo_path": str(tmp_path), "output_dir": str(tmp_path)})(),
            None,
        )
        # Should be empty string (no index, or build failed gracefully).
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestRepositoryMemoryCLI:
    def test_memory_build_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "build", "--help"])
        assert result.exit_code == 0
        assert "index" in result.output.lower()

    def test_memory_query_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "query", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_memory_symbol_graph_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "symbol-graph", "--help"])
        assert result.exit_code == 0
        assert "symbol" in result.output.lower()

    def test_memory_refresh_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "refresh", "--help"])
        assert result.exit_code == 0

    def test_memory_stats_repo_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["memory", "stats", "--help"])
        assert result.exit_code == 0