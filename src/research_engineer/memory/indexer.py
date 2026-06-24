"""Phase 12 - Repository indexer.

Walks a repository, parses every Python file with :mod:`ast`, extracts
symbols (modules, classes, functions, methods, imports), produces
semantic code chunks, and emits symbol-graph edges. Designed for
incremental updates: files are fingerprinted by mtime+size so unchanged
files can be skipped on re-index.

Non-Python config files (yaml/toml/json) are indexed as ``CONFIG``
symbols so they participate in retrieval and the graph.
"""

from __future__ import annotations

import ast
import hashlib
import os
import time
from pathlib import Path

from research_engineer.memory.models import (
    CodeChunk,
    RelationKind,
    Symbol,
    SymbolEdge,
    SymbolKind,
)

# Directories that are never indexed.
_NOISE_DIRS = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "data",
    "output",
})

# File extensions we index beyond Python.
_CONFIG_EXTS = frozenset({".yaml", ".yml", ".toml", ".json", ".cfg", ".ini"})


class IndexResult:
    """Container for a single indexing pass."""

    def __init__(self) -> None:
        self.symbols: list[Symbol] = []
        self.chunks: list[CodeChunk] = []
        self.edges: list[SymbolEdge] = []
        self.file_hashes: dict[str, str] = {}
        self.errors: list[str] = []


class RepositoryIndexer:
    """AST-based repository indexer producing symbols, chunks, and edges.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to index.
    max_file_bytes:
        Skip files larger than this (default 1 MiB) to bound memory.
    """

    def __init__(
        self,
        repo_path: str,
        *,
        max_file_bytes: int = 1_048_576,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.max_file_bytes = max_file_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self) -> IndexResult:
        """Index the entire repository (full pass)."""
        start = time.time()
        result = IndexResult()
        for file_path in self._walk():
            self._index_file(file_path, result)
        result.index_time_seconds = round(time.time() - start, 3)  # type: ignore[attr-defined]
        # Build cross-file edges (imports, calls, tests).
        self._build_cross_file_edges(result)
        return result

    def index_incremental(
        self, known_hashes: dict[str, str]
    ) -> tuple[IndexResult, list[str]]:
        """Index only files whose hash changed.

        Returns the :class:`IndexResult` for changed files plus the set
        of file paths that were (re)indexed.
        """
        start = time.time()
        result = IndexResult()
        changed: list[str] = []
        for file_path in self._walk():
            rel = self._rel(file_path)
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            digest = self._hash_content(content)
            if known_hashes.get(rel) == digest:
                result.file_hashes[rel] = digest
                continue
            changed.append(rel)
            self._index_file(file_path, result)
        result.index_time_seconds = round(time.time() - start, 3)  # type: ignore[attr-defined]
        self._build_cross_file_edges(result)
        return result, changed

    # ------------------------------------------------------------------
    # Walking
    # ------------------------------------------------------------------

    def _walk(self) -> list[Path]:
        """Yield files to index, skipping noise directories."""
        files: list[Path] = []
        for root, dirs, filenames in os.walk(self.repo_path):
            # Mutate dirs in-place to prune the walk.
            dirs[:] = [d for d in dirs if d not in _NOISE_DIRS]
            for fname in filenames:
                fp = Path(root) / fname
                ext = fp.suffix
                if ext == ".py" or ext in _CONFIG_EXTS:
                    try:
                        if fp.stat().st_size <= self.max_file_bytes:
                            files.append(fp)
                    except OSError:
                        continue
        return files

    # ------------------------------------------------------------------
    # Per-file indexing
    # ------------------------------------------------------------------

    def _index_file(self, file_path: Path, result: IndexResult) -> None:
        rel = self._rel(file_path)
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            result.errors.append(f"{rel}: {e}")
            return
        result.file_hashes[rel] = self._hash_content(content)

        ext = file_path.suffix
        if ext == ".py":
            self._index_python(file_path, rel, content, result)
        else:
            self._index_config(file_path, rel, content, result)

    def _index_python(
        self,
        file_path: Path,
        rel: str,
        content: str,
        result: IndexResult,
    ) -> None:
        module_sym = self._module_symbol(rel)
        # Mark module as test if it lives under a tests/ dir or is named test_*.
        module_sym.is_test = (
            "/tests/" in f"/{rel}/"
            or rel.startswith("tests/")
            or file_path.name.startswith("test_")
        )
        result.symbols.append(module_sym)
        result.chunks.append(self._module_chunk(module_sym, content))

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            result.errors.append(f"{rel}: syntax error: {e}")
            return

        # Walk top-level statements to preserve order and nesting.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_function(node, rel, content, result, module_sym)
            elif isinstance(node, ast.ClassDef):
                self._index_class(node, rel, content, result, module_sym)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._index_import_edge(node, rel, module_sym, result)

        # Detect entry points (__main__ block).
        if self._has_main_block(tree):
            module_sym.is_entry_point = True

    def _index_config(
        self,
        file_path: Path,
        rel: str,
        content: str,
        result: IndexResult,
    ) -> None:
        sym = Symbol(
            symbol_id=self._sym_id(rel, SymbolKind.CONFIG, file_path.name),
            name=file_path.name,
            kind=SymbolKind.CONFIG,
            qualified_name=rel,
            file_path=rel,
            line_start=1,
            line_end=content.count("\n") + 1,
            metadata={"ext": file_path.suffix},
        )
        result.symbols.append(sym)
        result.chunks.append(
            CodeChunk(
                chunk_id=f"{sym.symbol_id}:#0",
                symbol_id=sym.symbol_id,
                file_path=rel,
                kind=SymbolKind.CONFIG,
                name=sym.name,
                text=content[:8000],
                line_start=1,
                line_end=sym.line_end,
                language="config",
            )
        )

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _module_symbol(self, rel: str) -> Symbol:
        mod_name = rel.replace("/", ".").removesuffix(".py")
        return Symbol(
            symbol_id=self._sym_id(rel, SymbolKind.MODULE, mod_name),
            name=Path(rel).stem,
            kind=SymbolKind.MODULE,
            qualified_name=mod_name,
            file_path=rel,
            line_start=1,
        )

    def _module_chunk(self, module_sym: Symbol, content: str) -> CodeChunk:
        return CodeChunk(
            chunk_id=f"{module_sym.symbol_id}:#0",
            symbol_id=module_sym.symbol_id,
            file_path=module_sym.file_path,
            kind=SymbolKind.MODULE,
            name=module_sym.name,
            text=content[:8000],
            line_start=1,
            line_end=content.count("\n") + 1,
        )

    def _index_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        rel: str,
        content: str,
        result: IndexResult,
        module_sym: Symbol,
    ) -> None:
        kind = SymbolKind.FUNCTION
        is_test = node.name.startswith("test_") or rel.startswith("tests/")
        qname = f"{module_sym.qualified_name}.{node.name}"
        sym = Symbol(
            symbol_id=self._sym_id(rel, kind, node.name, node.lineno),
            name=node.name,
            kind=kind,
            qualified_name=qname,
            file_path=rel,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            docstring=ast.get_docstring(node),
            signature=self._signature(node),
            decorators=self._decorator_names(node),
            is_test=is_test,
            is_entry_point=node.name in ("main", "train_model", "run", "main_train"),
        )
        result.symbols.append(sym)
        result.chunks.append(self._symbol_chunk(sym, content, node))
        # defines edge: module -> function
        result.edges.append(
            SymbolEdge(
                source_id=module_sym.symbol_id,
                target_id=sym.symbol_id,
                relation=RelationKind.DEFINES,
            )
        )
        result.edges.append(
            SymbolEdge(
                source_id=sym.symbol_id,
                target_id=module_sym.symbol_id,
                relation=RelationKind.DEFINED_IN,
            )
        )
        # Calls edges (intra-file resolved later; raw calls recorded here).
        for callee in self._extract_calls(node):
            result.edges.append(
                SymbolEdge(
                    source_id=sym.symbol_id,
                    target_id=callee,
                    relation=RelationKind.CALLS,
                    context=rel,
                )
            )

    def _index_class(
        self,
        node: ast.ClassDef,
        rel: str,
        content: str,
        result: IndexResult,
        module_sym: Symbol,
    ) -> None:
        qname = f"{module_sym.qualified_name}.{node.name}"
        bases = [self._base_name(b) for b in node.bases]
        sym = Symbol(
            symbol_id=self._sym_id(rel, SymbolKind.CLASS, node.name, node.lineno),
            name=node.name,
            kind=SymbolKind.CLASS,
            qualified_name=qname,
            file_path=rel,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            docstring=ast.get_docstring(node),
            decorators=self._decorator_names(node),
            bases=bases,
        )
        result.symbols.append(sym)
        result.chunks.append(self._symbol_chunk(sym, content, node))
        result.edges.append(
            SymbolEdge(
                source_id=module_sym.symbol_id,
                target_id=sym.symbol_id,
                relation=RelationKind.DEFINES,
            )
        )
        # Methods
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_method(item, rel, content, result, sym)

    def _index_method(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        rel: str,
        content: str,
        result: IndexResult,
        class_sym: Symbol,
    ) -> None:
        qname = f"{class_sym.qualified_name}.{node.name}"
        is_test = node.name.startswith("test_")
        sym = Symbol(
            symbol_id=self._sym_id(rel, SymbolKind.METHOD, node.name, node.lineno),
            name=node.name,
            kind=SymbolKind.METHOD,
            qualified_name=qname,
            file_path=rel,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            docstring=ast.get_docstring(node),
            signature=self._signature(node),
            decorators=self._decorator_names(node),
            is_test=is_test,
        )
        result.symbols.append(sym)
        result.chunks.append(self._symbol_chunk(sym, content, node))
        result.edges.append(
            SymbolEdge(
                source_id=class_sym.symbol_id,
                target_id=sym.symbol_id,
                relation=RelationKind.DEFINES,
            )
        )

    def _index_import_edge(
        self,
        node: ast.Import | ast.ImportFrom,
        rel: str,
        module_sym: Symbol,
        result: IndexResult,
    ) -> None:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                target_qname = f"{mod}.{alias.name}" if mod else alias.name
                result.edges.append(
                    SymbolEdge(
                        source_id=module_sym.symbol_id,
                        target_id=target_qname,
                        relation=RelationKind.DEPENDS_ON,
                        context=f"from {mod} import {alias.name}",
                    )
                )
        else:
            for alias in node.names:
                result.edges.append(
                    SymbolEdge(
                        source_id=module_sym.symbol_id,
                        target_id=alias.name,
                        relation=RelationKind.DEPENDS_ON,
                        context=f"import {alias.name}",
                    )
                )

    # ------------------------------------------------------------------
    # Cross-file edge resolution
    # ------------------------------------------------------------------

    def _build_cross_file_edges(self, result: IndexResult) -> None:
        """Resolve DEPENDS_ON / CALLS edges to concrete symbol ids.

        Unresolved edges (external imports, stdlib) are dropped. We also
        add TESTS edges: files under tests/ that import a source module
        get a ``TESTS`` edge to that module's symbol.
        """
        qname_to_id: dict[str, str] = {}
        name_to_ids: dict[str, list[str]] = {}
        for sym in result.symbols:
            qname_to_id[sym.qualified_name] = sym.symbol_id
            name_to_ids.setdefault(sym.name, []).append(sym.symbol_id)

        resolved: list[SymbolEdge] = []
        for edge in result.edges:
            if edge.relation == RelationKind.DEPENDS_ON:
                self._resolve_dep_edge(edge, qname_to_id, name_to_ids, resolved)
            elif edge.relation == RelationKind.CALLS:
                self._resolve_call_edge(edge, name_to_ids, resolved)
            else:
                resolved.append(edge)

        self._add_tests_edges(result, resolved)
        result.edges = resolved

    @staticmethod
    def _resolve_dep_edge(
        edge: SymbolEdge,
        qname_to_id: dict[str, str],
        name_to_ids: dict[str, list[str]],
        resolved: list[SymbolEdge],
    ) -> None:
        """Resolve a DEPENDS_ON edge to a concrete symbol id."""
        target = qname_to_id.get(edge.target_id)
        if target is None:
            candidates = name_to_ids.get(edge.target_id.split(".")[-1], [])
            if len(candidates) == 1:
                target = candidates[0]
        if target is None:
            return
        resolved.append(
            SymbolEdge(
                source_id=edge.source_id,
                target_id=target,
                relation=RelationKind.DEPENDS_ON,
                context=edge.context,
            )
        )
        resolved.append(
            SymbolEdge(
                source_id=target,
                target_id=edge.source_id,
                relation=RelationKind.DEPENDENT_OF,
            )
        )

    @staticmethod
    def _resolve_call_edge(
        edge: SymbolEdge,
        name_to_ids: dict[str, list[str]],
        resolved: list[SymbolEdge],
    ) -> None:
        """Resolve a CALLS edge to function symbols."""
        candidates = name_to_ids.get(edge.target_id, [])
        for cand in candidates:
            resolved.append(
                SymbolEdge(
                    source_id=edge.source_id,
                    target_id=cand,
                    relation=RelationKind.CALLS,
                    context=edge.context,
                )
            )
            resolved.append(
                SymbolEdge(
                    source_id=cand,
                    target_id=edge.source_id,
                    relation=RelationKind.CALLED_BY,
                )
            )

    @staticmethod
    def _add_tests_edges(result: IndexResult, resolved: list[SymbolEdge]) -> None:
        """Add TESTS edges from test modules to their imported symbols."""
        for sym in result.symbols:
            if sym.is_test and sym.kind == SymbolKind.MODULE:
                for edge in resolved:
                    if (
                        edge.source_id == sym.symbol_id
                        and edge.relation == RelationKind.DEPENDS_ON
                    ):
                        resolved.append(
                            SymbolEdge(
                                source_id=sym.symbol_id,
                                target_id=edge.target_id,
                                relation=RelationKind.TESTS,
                            )
                        )

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract called function names (bare names only)."""
        names: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    names.append(func.id)
                elif isinstance(func, ast.Attribute):
                    names.append(func.attr)
        return names

    def _symbol_chunk(
        self, sym: Symbol, content: str, node: ast.AST
    ) -> CodeChunk:
        lines = content.splitlines()
        start = max(0, sym.line_start - 1)
        end = sym.line_end or start + 1
        text = "\n".join(lines[start:end])
        return CodeChunk(
            chunk_id=f"{sym.symbol_id}:#0",
            symbol_id=sym.symbol_id,
            file_path=sym.file_path,
            kind=sym.kind,
            name=sym.name,
            text=text[:8000],
            line_start=sym.line_start,
            line_end=end,
        )

    @staticmethod
    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        try:
            return ast.unparse(node.args)  # type: ignore[union-attr]
        except Exception:
            return ""

    @staticmethod
    def _decorator_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                names.append(dec.attr)
            elif isinstance(dec, ast.Call):
                f = dec.func
                if isinstance(f, ast.Name):
                    names.append(f.id)
                elif isinstance(f, ast.Attribute):
                    names.append(f.attr)
        return names

    @staticmethod
    def _base_name(base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        try:
            return ast.unparse(base)  # type: ignore[union-attr]
        except Exception:
            return ""

    @staticmethod
    def _has_main_block(tree: ast.Module) -> bool:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    return True
        return False

    # ------------------------------------------------------------------
    # Hashing / ids
    # ------------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_path))
        except ValueError:
            return str(path)

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        h.update(f"{path.stat().st_mtime}:{path.stat().st_size}".encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _sym_id(self, rel: str, kind: SymbolKind, name: str, line: int = 0) -> str:
        parts = [rel, kind.value, name]
        if line:
            parts.append(str(line))
        return ":".join(parts)


__all__ = ["RepositoryIndexer", "IndexResult"]
