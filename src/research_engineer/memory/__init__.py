"""Phase 12 - Repository Memory subsystem.

Persistent semantic memory for repository understanding. Upgrades the
system from file scanning to a symbol-graph + vector-index hybrid that
automatically surfaces relevant code, dependencies, callers, and tests
before planning or coding.

Public API::

    from research_engineer.memory import (
        RepositoryMemory,
        RepositoryIndexer,
        SymbolGraph,
        HybridRetriever,
        RepositoryMemoryStore,
        HashingEmbedder,
        InMemoryVectorBackend,
    )

Typical usage::

    mem = RepositoryMemory("./my_repo")
    mem.build()                       # full index
    results = mem.query("EMA checkpoint support")
    context = mem.get_context("Add EMA checkpoint support")
"""

from research_engineer.memory.embeddings import EmbedderBackend, HashingEmbedder
from research_engineer.memory.indexer import IndexResult, RepositoryIndexer
from research_engineer.memory.models import (
    CodeChunk,
    IndexStats,
    RelationKind,
    RetrievalResult,
    Symbol,
    SymbolEdge,
    SymbolKind,
)
from research_engineer.memory.repository_memory import RepositoryMemory
from research_engineer.memory.retriever import HybridRetriever
from research_engineer.memory.storage import RepositoryMemoryStore
from research_engineer.memory.symbol_graph import SymbolGraph
from research_engineer.memory.vector_store import (
    InMemoryVectorBackend,
    VectorBackend,
    chunk_payload,
)

__all__ = [
    # Facade
    "RepositoryMemory",
    # Components
    "RepositoryIndexer",
    "IndexResult",
    "SymbolGraph",
    "HybridRetriever",
    "RepositoryMemoryStore",
    "EmbedderBackend",
    "HashingEmbedder",
    "VectorBackend",
    "InMemoryVectorBackend",
    "chunk_payload",
    # Models
    "Symbol",
    "SymbolKind",
    "SymbolEdge",
    "RelationKind",
    "CodeChunk",
    "RetrievalResult",
    "IndexStats",
]
