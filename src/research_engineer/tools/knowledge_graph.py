"""Knowledge Graph Tool for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.ast_models import ASTOutput, GraphEdge, GraphNode
from research_engineer.tools.base import Tool, ToolError
from research_engineer.tools.config_analyzer import ConfigOutput
from research_engineer.tools.dependency_graph import DependencyOutput
from research_engineer.tools.training_pipeline import TrainingPipelineOutput


class KnowledgeGraphInput(BaseModel):
    """Input for knowledge graph tool."""

    repo_scan: dict[str, Any] = Field(..., description="Repository scan results")
    ast_results: list[ASTOutput] = Field(default_factory=list, description="AST analysis results")
    dependencies: DependencyOutput = Field(default_factory=DependencyOutput, description="Dependency graph")
    training_pipeline: TrainingPipelineOutput | None = Field(default=None, description="Training pipeline analysis")
    config_analysis: ConfigOutput | None = Field(default=None, description="Configuration analysis")


class KnowledgeGraphOutput(BaseModel):
    """Output from knowledge graph analysis."""

    nodes: list[GraphNode] = Field(default_factory=list, description="Graph nodes")
    edges: list[GraphEdge] = Field(default_factory=list, description="Graph edges")
    central_nodes: list[str] = Field(default_factory=list, description="Most important nodes")
    communities: list[list[str]] = Field(default_factory=list, description="Related node groups")
    relationships_by_type: dict[str, int] = Field(default_factory=dict, description="Edge type counts")
    node_types: dict[str, int] = Field(default_factory=dict, description="Node type counts")
    graph_statistics: dict[str, float] = Field(default_factory=dict, description="Graph statistics")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class KnowledgeGraphTool(Tool[KnowledgeGraphInput, KnowledgeGraphOutput]):
    """Build comprehensive knowledge graph from analysis results."""

    def __init__(self):
        self._node_cache: dict[str, GraphNode] = {}
        self._edge_cache: dict[str, GraphEdge] = {}

    def _create_node(self, node_id: str, node_type: str, name: str, **kwargs) -> GraphNode:
        """Create or retrieve cached node."""
        if node_id not in self._node_cache:
            self._node_cache[node_id] = GraphNode(
                id=node_id,
                type=node_type,
                name=name,
                **kwargs
            )
        return self._node_cache[node_id]

    def _create_edge(self, source: str, target: str, relationship: str, **kwargs) -> GraphEdge:
        """Create or retrieve cached edge."""
        edge_key = f"{source}:{target}:{relationship}"
        if edge_key not in self._edge_cache:
            self._edge_cache[edge_key] = GraphEdge(
                source=source,
                target=target,
                relationship=relationship,
                **kwargs
            )
        return self._edge_cache[edge_key]

    def _extract_classes_from_ast(self, ast_output: ASTOutput) -> list[GraphNode]:
        """Extract class nodes from AST output."""
        nodes = []
        for cls in ast_output.classes:
            node = GraphNode(
                id=f"{ast_output.file_path}#{cls.name}",
                type="class",
                name=cls.name,
                file_path=ast_output.file_path,
                line_number=cls.line_number,
                metadata={
                    "line_count": 1 if not cls.methods else len(cls.methods),
                    "complexity": cls.complexity_score,
                    "inherits": cls.inherits,
                },
                description=f"Class {cls.name}",
            )
            nodes.append(node)
        return nodes

    def _extract_functions_from_ast(self, ast_output: ASTOutput) -> list[GraphNode]:
        """Extract function nodes from AST output."""
        nodes = []
        for func in ast_output.functions:
            node = GraphNode(
                id=f"{ast_output.file_path}#{func.name}",
                type="function",
                name=func.name,
                file_path=ast_output.file_path,
                line_number=func.line_number,
                metadata={
                    "line_count": 1,
                    "complexity": func.complexity_score,
                    "is_async": func.is_async,
                },
                description=f"Function {func.name}",
            )
            nodes.append(node)
        return nodes

    def _extract_imports_as_nodes(self, ast_output: ASTOutput) -> list[GraphNode]:
        """Extract import relationships as nodes."""
        nodes = []
        for imp in ast_output.imports:
            if imp.module:
                node = GraphNode(
                    id=f"import:{imp.module}",
                    type="module",
                    name=imp.module,
                    file_path=ast_output.file_path,
                    line_number=imp.line_number,
                    metadata={
                        "is_relative": imp.is_relative,
                    },
                    description=f"Import: {imp.module}",
                )
                nodes.append(node)
        return nodes

    def _create_edges_from_ast(self, ast_output: ASTOutput) -> list[GraphEdge]:
        """Create edges from AST data."""
        edges = []

        # Class inheritance edges
        for cls in ast_output.classes:
            for base in cls.inherits:
                edge = GraphEdge(
                    source=f"{ast_output.file_path}#{cls.name}",
                    target=f"{ast_output.file_path}#{base}",
                    relationship="inherits",
                    context=f"{cls.name} inherits from {base}",
                )
                edges.append(edge)

        # Method call edges (simplified)
        for func in ast_output.functions:
            # Check for method calls in docstring or comments
            for other_func in ast_output.functions:
                if func.name != other_func.name:
                    # Simple check - if function name appears in docstring
                    if other_func.docstring and func.name in other_func.docstring:
                        edge = GraphEdge(
                            source=f"{ast_output.file_path}#{func.name}",
                            target=f"{ast_output.file_path}#{other_func.name}",
                            relationship="calls",
                            context=f"{func.name} calls {other_func.name}",
                        )
                        edges.append(edge)

        return edges

    def _create_edges_from_dependencies(self, dependencies: DependencyOutput) -> list[GraphEdge]:
        """Create edges from dependency graph."""
        edges = []

        for source_file, targets in dependencies.file_level_graph.items():
            for target_file in targets:
                edge = GraphEdge(
                    source=source_file,
                    target=target_file,
                    relationship="imports",
                    context=f"Imports from {target_file}",
                )
                edges.append(edge)

        return edges

    def _create_edges_from_training_pipeline(self, pipeline: TrainingPipelineOutput) -> list[GraphEdge]:
        """Create edges from training pipeline analysis."""
        edges = []

        # Pipeline stage edges
        prev_stage = None
        for stage in pipeline.full_pipeline:
            if prev_stage:
                edge = GraphEdge(
                    source=f"stage:{prev_stage.stage_name}",
                    target=f"stage:{stage.stage_name}",
                    relationship="follows",
                    context=f"{prev_stage.stage_name} -> {stage.stage_name}",
                )
                edges.append(edge)
            prev_stage = stage

        # Dataset to model edges
        if pipeline.dataset_loader and pipeline.model_class:
            edge = GraphEdge(
                source=pipeline.dataset_loader.class_path,
                target=pipeline.model_class,
                relationship="feeds",
                context=f"{pipeline.dataset_loader.class_path} -> {pipeline.model_class}",
            )
            edges.append(edge)

        # Model to loss edges
        for loss in pipeline.loss_functions:
            if pipeline.model_class:
                edge = GraphEdge(
                    source=pipeline.model_class,
                    target=loss,
                    relationship="uses",
                    context=f"{pipeline.model_class} uses {loss}",
                )
                edges.append(edge)

        return edges

    def _calculate_central_nodes(self) -> list[str]:
        """Calculate central nodes based on degree centrality."""
        # Simple centrality calculation based on node occurrences
        node_counts: dict[str, int] = {}

        for edge in self._edge_cache.values():
            node_counts[edge.source] = node_counts.get(edge.source, 0) + 1
            node_counts[edge.target] = node_counts.get(edge.target, 0) + 1

        # Sort by count and return top 10
        sorted_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)
        return [node for node, count in sorted_nodes[:10]]

    def _detect_communities(self) -> list[list[str]]:
        """Detect communities in the graph (simple version)."""
        communities: list[list[str]] = []

        # Group nodes by type
        type_groups: dict[str, list[str]] = {}
        for node in self._node_cache.values():
            if node.type not in type_groups:
                type_groups[node.type] = []
            type_groups[node.type].append(node.id)

        # Each type group is a community
        for group in type_groups.values():
            if len(group) > 1:
                communities.append(group)

        return communities

    def _build_graph_from_ast(self, ast_outputs: list[ASTOutput]) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Build graph from AST outputs."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for ast_output in ast_outputs:
            # Extract classes
            class_nodes = self._extract_classes_from_ast(ast_output)
            nodes.extend(class_nodes)

            # Extract functions
            func_nodes = self._extract_functions_from_ast(ast_output)
            nodes.extend(func_nodes)

            # Extract imports
            import_nodes = self._extract_imports_as_nodes(ast_output)
            nodes.extend(import_nodes)

            # Create edges from AST
            ast_edges = self._create_edges_from_ast(ast_output)
            edges.extend(ast_edges)

        return nodes, edges

    def _build_graph_from_dependencies(self, dependencies: DependencyOutput) -> list[GraphEdge]:
        """Build edges from dependency graph."""
        return self._create_edges_from_dependencies(dependencies)

    def _build_graph_from_training(self, pipeline: TrainingPipelineOutput) -> list[GraphEdge]:
        """Build edges from training pipeline."""
        return self._create_edges_from_training_pipeline(pipeline)

    def _detect_node_types(self) -> dict[str, int]:
        """Count nodes by type."""
        counts: dict[str, int] = {}
        for node in self._node_cache.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return counts

    def _detect_relationship_types(self) -> dict[str, int]:
        """Count edges by relationship type."""
        counts: dict[str, int] = {}
        for edge in self._edge_cache.values():
            counts[edge.relationship] = counts.get(edge.relationship, 0) + 1
        return counts

    async def execute(self, input: KnowledgeGraphInput) -> KnowledgeGraphOutput:
        """Execute knowledge graph construction."""
        try:
            # Reset caches
            self._node_cache = {}
            self._edge_cache = {}

            # Build nodes and edges from AST
            ast_nodes, ast_edges = self._build_graph_from_ast(input.ast_results)

            # Build edges from dependencies
            dependency_edges = self._build_graph_from_dependencies(input.dependencies)

            # Build edges from training pipeline
            training_edges: list[GraphEdge] = []
            if input.training_pipeline:
                training_edges = self._build_graph_from_training(input.training_pipeline)

            # Combine all edges
            all_edges = ast_edges + dependency_edges + training_edges

            # Calculate centrality
            central_nodes = self._calculate_central_nodes()

            # Detect communities
            communities = self._detect_communities()

            # Detect node types
            node_types = self._detect_node_types()

            # Detect relationship types
            relationships = self._detect_relationship_types()

            # Calculate graph statistics
            total_nodes = len(self._node_cache)
            total_edges = len(all_edges)

            avgDegree = total_edges / total_nodes if total_nodes > 0 else 0

            return KnowledgeGraphOutput(
                nodes=list(self._node_cache.values()),
                edges=all_edges,
                central_nodes=central_nodes,
                communities=communities,
                relationships_by_type=relationships,
                node_types=node_types,
                graph_statistics={
                    "total_nodes": total_nodes,
                    "total_edges": total_edges,
                    "avg_degree": avgDegree,
                    "density": total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0,
                },
            )

        except Exception as e:
            raise ToolError(f"Failed to build knowledge graph: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
