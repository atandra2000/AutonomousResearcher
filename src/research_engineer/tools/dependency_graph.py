"""Dependency Graph Tool for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.ast_models import (
    ASTOutput,
    GraphEdge,
    GraphNode,
    ImportInfo,
)
from research_engineer.tools.base import Tool, ToolError


class DependencyInput(BaseModel):
    """Input for dependency graph tool."""

    files: list[ASTOutput] = Field(..., description="AST outputs from all files")
    project_root: str = Field(..., description="Project root directory")
    include_external_imports: bool = Field(default=False, description="Include external imports")


class DependencyOutput(BaseModel):
    """Output from dependency graph tool."""

    file_level_graph: dict[str, list[str]] = Field(default_factory=dict, description="file -> [dependencies]")
    module_level_graph: dict[str, list[str]] = Field(default_factory=dict, description="module -> [dependencies]")
    cross_file_references: list[dict[str, Any]] = Field(default_factory=list, description="Cross-file references")
    circular_dependencies: list[list[str]] = Field(default_factory=list, description="Detected circular deps")
    import_frequency: dict[str, int] = Field(default_factory=dict, description="Import frequency")
    dependency_tree: dict[str, list[str]] = Field(default_factory=dict, description="Dependency tree")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class DependencyGraphTool(Tool[DependencyInput, DependencyOutput]):
    """Build dependency relationships between files and modules."""

    def __init__(self):
        self._import_cache: dict[str, set[str]] = {}
        self._node_cache: dict[str, GraphNode] = {}
        self._edge_cache: dict[str, GraphEdge] = {}

    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        filename = file_path.split('/')[-1]
        if filename.endswith('.py'):
            filename = filename[:-3]
        return filename

    def _calculate_distance(self, file1: str, file2: str, project_root: str) -> int:
        """Calculate relative distance between two files."""
        try:
            rel1 = file1.replace(project_root + '/', '').replace(project_root, '')
            rel2 = file2.replace(project_root + '/', '').replace(project_root, '')

            depth1 = rel1.count('/')
            depth2 = rel2.count('/')

            return abs(depth1 - depth2)
        except:
            return 0

    def _resolve_import(self, import_info: ImportInfo, current_file: str, project_root: str) -> list[str]:
        """Resolve import to actual file paths."""
        resolved = []

        if import_info.is_relative:
            current_dir = '/'.join(current_file.split('/')[:-1])
            depth = import_info.level

            if depth == 1:
                resolved.append(f"{current_dir}/__init__.py")
            elif depth > 1:
                parent_dir = current_dir
                for _ in range(depth - 1):
                    parent_dir = '/'.join(parent_dir.split('/')[:-1])
                if import_info.names:
                    for name in import_info.names:
                        resolved.append(f"{parent_dir}/{name}.py")
                else:
                    resolved.append(f"{parent_dir}/__init__.py")
        else:
            for name in import_info.names:
                module_path = name.replace('.', '/')
                potential_paths = [
                    f"{project_root}/{module_path}.py",
                    f"{project_root}/{module_path}/__init__.py",
                ]
                for path in potential_paths:
                    if 'site-packages' not in path and '.venv' not in path:
                        resolved.append(path)

        return resolved

    def _build_file_dependency(self, ast_output: ASTOutput, project_root: str) -> list[tuple[str, str]]:
        """Build file-level dependencies from AST output."""
        dependencies = []

        for import_info in ast_output.imports:
            resolved = self._resolve_import(import_info, ast_output.file_path, project_root)
            for target in resolved:
                dependencies.append((ast_output.file_path, target))

        return dependencies

    def _build_module_dependency(self, ast_output: ASTOutput) -> list[tuple[str, str]]:
        """Build module-level dependencies."""
        dependencies = []

        for import_info in ast_output.imports:
            if import_info.module:
                dependencies.append((ast_output.file_path, import_info.module))

        return dependencies

    def _detect_circular_dependencies(self, file_deps: list[tuple[str, str]]) -> list[list[str]]:
        """Detect circular dependencies using DFS."""
        graph: dict[str, list[str]] = {}
        for source, target in file_deps:
            if source not in graph:
                graph[source] = []
            graph[source].append(target)

        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path + [neighbor]):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor) if neighbor in path else len(path)
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) > 1:
                        cycles.append(cycle)

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node, [node])

        return cycles

    async def execute(self, input: DependencyInput) -> DependencyOutput:
        """Execute dependency graph construction."""
        try:
            file_level_graph: dict[str, list[str]] = {}
            module_level_graph: dict[str, list[str]] = {}
            file_deps: list[tuple[str, str]] = []
            module_deps: list[tuple[str, str]] = []
            cross_references: list[dict[str, Any]] = []
            import_frequency: dict[str, int] = {}
            dependency_tree: dict[str, list[str]] = {}

            # Process each file
            for ast_output in input.files:
                # Get file dependencies
                file_deps.extend(self._build_file_dependency(ast_output, input.project_root))

                # Get module dependencies
                module_deps.extend(self._build_module_dependency(ast_output))

                # Track import frequency
                for import_info in ast_output.imports:
                    if import_info.module:
                        import_frequency[import_info.module] = import_frequency.get(import_info.module, 0) + 1
                    for name in import_info.names:
                        import_frequency[name] = import_frequency.get(name, 0) + 1

            # Build file-level graph
            for source, target in file_deps:
                if source not in file_level_graph:
                    file_level_graph[source] = []
                if target not in file_level_graph[source]:
                    file_level_graph[source].append(target)

            # Build module-level graph
            for source, target in module_deps:
                if source not in module_level_graph:
                    module_level_graph[source] = []
                if target not in module_level_graph[source]:
                    module_level_graph[source].append(target)

            # Build dependency tree
            for source, target in file_deps:
                if source not in dependency_tree:
                    dependency_tree[source] = []
                if target not in dependency_tree[source]:
                    dependency_tree[source].append(target)

            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(file_deps)

            # Generate dependency nodes and edges for knowledge graph
            nodes: list[GraphNode] = []
            edges: list[GraphEdge] = []

            # Create nodes for each file
            for ast_output in input.files:
                if ast_output.file_path not in self._node_cache:
                    node = GraphNode(
                        id=ast_output.file_path,
                        type="module",
                        name=self._extract_module_name(ast_output.file_path),
                        file_path=ast_output.file_path,
                        line_number=ast_output.classes[0].line_number if ast_output.classes else 1,
                        properties={
                            "line_count": ast_output.line_count,
                            "complexity": ast_output.complexity_metrics.cyclomatic_complexity,
                        },
                        description=f"Module with {len(ast_output.classes)} classes and {len(ast_output.functions)} functions",
                    )
                    self._node_cache[ast_output.file_path] = node
                nodes.append(self._node_cache[ast_output.file_path])

            # Create edges for dependencies
            for source, target in file_deps:
                edge_key = f"{source}:{target}"
                if edge_key not in self._edge_cache:
                    edge = GraphEdge(
                        source=source,
                        target=target,
                        relationship="imports",
                        context="Import statement",
                    )
                    self._edge_cache[edge_key] = edge
                edges.append(self._edge_cache[edge_key])

            return DependencyOutput(
                file_level_graph=file_level_graph,
                module_level_graph=module_level_graph,
                cross_file_references=cross_references,
                circular_dependencies=circular_deps,
                import_frequency=import_frequency,
                dependency_tree=dependency_tree,
            )

        except Exception as e:
            raise ToolError(f"Failed to build dependency graph: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
