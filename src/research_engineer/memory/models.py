"""Phase 12 - Repository Memory domain models.

Typed Pydantic models for repository indexing: symbols, code chunks,
symbol-graph edges, and retrieval results. These models are the
serialization contract for the persistent SQLite store and the vector
index.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SymbolKind(StrEnum):
    """Kind of indexed symbol."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    AGENT = "agent"
    TOOL = "tool"
    CLI_COMMAND = "cli_command"
    CONFIG = "config"
    VARIABLE = "variable"


class RelationKind(StrEnum):
    """Edge kind in the symbol graph."""

    DEPENDS_ON = "depends_on"        # A imports/requires B
    DEPENDENT_OF = "dependent_of"     # inverse of depends_on
    CALLS = "calls"                   # A calls function B
    CALLED_BY = "called_by"           # inverse of calls
    DEFINES = "defines"               # module defines class/function
    DEFINED_IN = "defined_in"         # inverse of defines
    INHERITS = "inherits"             # class inherits from class
    RELATED = "related"               # co-occurrence / heuristic
    TESTS = "tests"                   # test file tests source file


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Symbol(BaseModel):
    """A single indexed symbol (file, class, function, method, ...)."""

    symbol_id: str = Field(..., description="Stable unique id: repo:rel_path:kind:name")
    name: str = Field(..., description="Symbol name")
    kind: SymbolKind = Field(..., description="Symbol kind")
    qualified_name: str = Field(
        ..., description="Fully-qualified name (module.path.name)"
    )
    file_path: str = Field(..., description="Repository-relative file path")
    line_start: int = Field(default=0, description="Start line (1-indexed)")
    line_end: int = Field(default=0, description="End line (inclusive)")
    docstring: str | None = Field(default=None, description="Docstring if any")
    signature: str | None = Field(default=None, description="Function/class signature")
    decorators: list[str] = Field(default_factory=list, description="Decorators")
    bases: list[str] = Field(default_factory=list, description="Base classes")
    is_test: bool = Field(default=False, description="Whether this is a test symbol")
    is_entry_point: bool = Field(default=False, description="CLI/main entry point")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra metadata (e.g. complexity)"
    )


class CodeChunk(BaseModel):
    """A semantically meaningful code region for embedding.

    Chunks are the unit of vector retrieval. Each chunk references its
    parent symbol and carries the source text plus enough metadata to
    reconstruct context at retrieval time.
    """

    chunk_id: str = Field(..., description="Stable unique id: symbol_id:#n")
    symbol_id: str = Field(..., description="Parent symbol id")
    file_path: str = Field(..., description="Repository-relative file path")
    kind: SymbolKind = Field(..., description="Inherited symbol kind")
    name: str = Field(..., description="Symbol name")
    text: str = Field(..., description="Source text of the chunk")
    line_start: int = Field(default=0, description="Start line (1-indexed)")
    line_end: int = Field(default=0, description="End line (inclusive)")
    language: str = Field(default="python", description="Source language")


class SymbolEdge(BaseModel):
    """A directed edge in the symbol graph."""

    source_id: str = Field(..., description="Source symbol id")
    target_id: str = Field(..., description="Target symbol id")
    relation: RelationKind = Field(..., description="Edge kind")
    weight: float = Field(default=1.0, description="Edge weight / confidence")
    context: str | None = Field(default=None, description="Where the edge arises")


class RetrievalResult(BaseModel):
    """A single hybrid-retrieval hit."""

    chunk: CodeChunk = Field(..., description="Matched code chunk")
    symbol: Symbol | None = Field(default=None, description="Parent symbol")
    semantic_score: float = Field(default=0.0, description="Vector similarity score")
    graph_score: float = Field(default=0.0, description="Graph proximity score")
    metadata_score: float = Field(default=0.0, description="Metadata boost score")
    combined_score: float = Field(default=0.0, description="Weighted hybrid score")
    related_symbols: list[Symbol] = Field(
        default_factory=list, description="Graph-expanded related symbols"
    )


class IndexStats(BaseModel):
    """Statistics about a repository index."""

    repo_path: str = Field(..., description="Indexed repository path")
    total_files: int = Field(default=0, description="Files indexed")
    total_symbols: int = Field(default=0, description="Symbols indexed")
    total_chunks: int = Field(default=0, description="Code chunks indexed")
    total_edges: int = Field(default=0, description="Graph edges")
    symbols_by_kind: dict[str, int] = Field(
        default_factory=dict, description="Symbol counts by kind"
    )
    indexed_at: datetime = Field(default_factory=datetime.now)
    index_time_seconds: float = Field(default=0.0, description="Last index duration")


__all__ = [
    "SymbolKind",
    "RelationKind",
    "Symbol",
    "CodeChunk",
    "SymbolEdge",
    "RetrievalResult",
    "IndexStats",
]
