"""Documentation Tool for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.repo import (
    FileImportance,
    ImplementationTarget,
    RepositorySummary,
)
from research_engineer.tools.base import Tool, ToolError
from research_engineer.tools.config_analyzer import ConfigOutput
from research_engineer.tools.dependency_graph import DependencyOutput
from research_engineer.tools.knowledge_graph import KnowledgeGraphOutput
from research_engineer.tools.training_pipeline import TrainingPipelineOutput


class DocumentationInput(BaseModel):
    """Input for documentation tool."""

    repo_summary: RepositorySummary = Field(..., description="Repository summary")
    architecture: dict[str, Any] = Field(default_factory=dict, description="Architecture overview")
    training_pipeline: TrainingPipelineOutput = Field(..., description="Training pipeline")
    dependencies: DependencyOutput = Field(..., description="Dependency graph")
    knowledge_graph: KnowledgeGraphOutput = Field(..., description="Knowledge graph")
    config_analysis: ConfigOutput = Field(..., description="Configuration analysis")
    important_files: list[FileImportance] = Field(default_factory=list, description="File rankings")
    implementation_targets: list[ImplementationTarget] = Field(default_factory=list, description="Insertion points")
    output_dir: str = Field(..., description="Output directory")


class DocumentationOutput(BaseModel):
    """Output from documentation tool."""

    generated_files: list[str] = Field(default_factory=list, description="Generated markdown files")
    output_directory: str = Field(..., description="Output directory")
    file_count: int = Field(default=0, description="Number of files generated")
    total_lines: int = Field(default=0, description="Total lines written")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class DocumentationTool(Tool[DocumentationInput, DocumentationOutput]):
    """Generate markdown documentation from analysis results."""

    def __init__(self):
        self._generated_files: list[str] = []

    def _create_mermaid_class_diagram(self, nodes: list[dict[str, Any]]) -> str:
        """Create Mermaid class diagram from nodes."""
        mermaid = ["```mermaid", "classDiagram"]

        for node in nodes:
            node_id = node.get('id', 'unknown')
            node_name = node.get('name', 'unknown')
            node_type = node.get('type', 'unknown')

            # Create node definition
            mermaid.append(f"    class {node_id} {{")
            mermaid.append(f"        +name: {node_name}")
            mermaid.append(f"        +type: {node_type}")

            # Add metadata if exists
            metadata = node.get('metadata', {})
            for key, value in metadata.items():
                mermaid.append(f"        +{key}: {value}")

            mermaid.append("    }")

        mermaid.append("```")
        return '\n'.join(mermaid)

    def _create_mermaid_flowchart(self, edges: list[dict[str, Any]]) -> str:
        """Create Mermaid flowchart from edges."""
        mermaid = ["```mermaid", "flowchart TD"]

        for edge in edges:
            source = edge.get('source', 'unknown_source')
            target = edge.get('target', 'unknown_target')
            relationship = edge.get('relationship', 'uses')

            # Sanitize IDs for mermaid
            source_safe = source.replace(' ', '_').replace('#', '_')
            target_safe = target.replace(' ', '_').replace('#', '_')

            mermaid.append(f"    {source_safe} -->|{relationship}| {target_safe}")

        mermaid.append("```")
        return '\n'.join(mermaid)

    def generate_repo_summary(self, summary: RepositorySummary) -> str:
        """Generate repo_summary.md content."""
        content = f"""# Repository Summary: {summary.repository_name}

**Project Type:** {summary.project_type}

**Analysis Timestamp:** {summary.analysis_timestamp}

## Overview

This repository contains a {summary.project_type} project that implements {summary.architecture_summary}.

## Key Statistics

- Total Files Analyzed: {len(summary.important_files)}
- Architecture Complexity: High
- Training Pipeline: Complete

## Project Classification

| Property | Value |
|----------|-------|
| Repository Name | {summary.repository_name} |
| Project Type | {summary.project_type} |
| Architecture Summary | {summary.architecture_summary[:100]}... |
| Files of Interest | {len(summary.important_files)} |
| Implementation Targets | {len(summary.implementation_targets)} |

## Next Steps

1. Review the architecture overview for high-level understanding
2. Check the training pipeline for data flow details
3. Review implementation targets for code modification points
"""
        return content

    def generate_architecture_overview(self, summary: RepositorySummary, architecture: dict[str, Any]) -> str:
        """Generate architecture_overview.md content."""
        content = f"""# Architecture Overview

## High-Level Structure

{summary.architecture_summary}

## Core Components

### Entry Points
{len(architecture.get('entry_points', []))} entry points identified:
{chr(10).join(f'- {ep}' for ep in architecture.get('entry_points', [])[:5])}

### Main Modules
{architecture.get('main_modules', [])[:5]}

### Core Abstractions
{chr(10).join(f'- {ab}' for ab in architecture.get('core_abstractions', [])[:5])}

## Component Relationships

```mermaid
flowchart TD
{chr(10).join(f'    {src} --> {tgt}' for src, tgt in architecture.get('relationships', {}).items())}
```

## Architecture Diagrams

### File Dependency Graph

```mermaid
flowchart LR
{chr(10).join(f'    {k} --> {v}' for k, v in list(architecture.get('file_dependencies', {}).items())[:10])}
```

### Module Hierarchy

```mermaid
graph TD
{chr(10).join(f'    {m} --> {parent}' for m, parent in architecture.get('module_hierarchy', {}).items())}
```

## Critical Classes

{chr(10).join(f'- `{cls}`' for cls in architecture.get('critical_classes', []))}

## API Boundaries

- Public API: {architecture.get('public_api', [])[:5]}
- Internal API: {architecture.get('internal_api', [])[:5]}
"""
        return content

    def generate_training_pipeline(self, pipeline: TrainingPipelineOutput) -> str:
        """Generate training_pipeline.md content."""
        content = f"""# Training Pipeline

## Pipeline Stages

{len(pipeline.full_pipeline)} stages identified in the training pipeline:

### 1. Data Loading
"""

        if pipeline.dataset_loader:
            content += f"""
- **Dataset:** {pipeline.dataset_loader.name}
- **Class Path:** {pipeline.dataset_loader.class_path}
- **Batch Size:** {pipeline.dataset_loader.batch_size}
- **Workers:** {pipeline.dataset_loader.num_workers}
"""
        content += """
### 2. Model Initialization
"""

        if pipeline.model_class:
            content += f"""
- **Model Class:** {pipeline.model_class}
- **Class Path:** {pipeline.model_class_path}
"""

        content += """
### 3. Optimizer Setup
"""

        if pipeline.optimizer_config:
            content += f"""
- **Optimizer:** {pipeline.optimizer_config.name}
- **Learning Rate:** {pipeline.optimizer_config.learning_rate}
- **Weight Decay:** {pipeline.optimizer_config.weight_decay}
"""

        content += """
### 4. Training Loop
"""

        content += f"""
- **Epochs:** {pipeline.training_loop.epochs}
- **Batch Size:** {pipeline.training_loop.batch_size}
- **Forward:** {pipeline.training_loop.forward_pass}
- **Backward:** {pipeline.training_loop.backward_pass}
"""

        content += """
### 5. Validation
"""

        if pipeline.validation_loop:
            content += f"""
- **Frequency:** {pipeline.validation_loop.frequency}
- **Metrics:** {', '.join(pipeline.validation_loop.metrics)}
"""

        content += """
### 6. Checkpointing
"""

        content += f"""
- **Enabled:** {pipeline.checkpointing.enabled}
- **Save Best:** {pipeline.checkpointing.save_best}
"""

        if pipeline.distributed_training:
            content += """
### 7. Distributed Training
"""
            content += f"""
- **Enabled:** {pipeline.distributed_training.enabled}
- **Backend:** {pipeline.distributed_training.backend}
- **World Size:** {pipeline.distributed_training.world_size}
"""

        content += f"""
## Key Metrics Logging

{len(pipeline.metrics_logging)} metrics being tracked:
{chr(10).join(f'- {m.name} (mode: {m.mode})' for m in pipeline.metrics_logging)}

## Loss Functions

{len(pipeline.loss_functions)} loss functions found:
{chr(10).join(f'- {loss}' for loss in pipeline.loss_functions)}

## Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
        for k, v in pipeline.hyperparameters.items():
            content += f"| {k} | {v} |\n"

        content += """
## Training Flow Diagram

```mermaid
flowchart TD
    Data --> Model
    Model --> Optimizer
    Optimizer --> Checkpoint
    Validation --> Checkpoint
    Checkpoint --> Loop
    Loop --> Data
```
"""
        return content

    def generate_dependency_graph(self, dependencies: DependencyOutput) -> str:
        """Generate dependency_graph.md content."""
        content = f"""# Dependency Graph

## File-Level Dependencies

{len(dependencies.file_level_graph)} files analyzed with dependencies:

### Direct Dependencies

"""

        for file, deps in list(dependencies.file_level_graph.items())[:10]:
            content += f"### {file}\n"
            for dep in deps[:3]:
                content += f"- imports: {dep}\n"
            content += "\n"

        content += """
## Module-Level Dependencies

"""
        for module, deps in list(dependencies.module_level_graph.items())[:5]:
            content += f"### {module}\n"
            for dep in deps[:2]:
                content += f"- uses: {dep}\n"
            content += "\n"

        if dependencies.circular_dependencies:
            content += """
## Circular Dependencies ⚠️

"""
            for cycle in dependencies.circular_dependencies[:3]:
                content += f"- {' -> '.join(cycle)}\n"

        content += """
## Import Frequency

| Module | Frequency |
|----------|----------|
"""
        for module, freq in list(dependencies.import_frequency.items())[:10]:
            content += f"| {module} | {freq} |\n"

        return content

    def generate_configuration_analysis(self, config: ConfigOutput) -> str:
        """Generate configuration_analysis.md content."""
        content = f"""# Configuration Analysis

## Framework Detected

**Config Framework:** {config.config_framework}

## Training Hyperparameters

"""
        for key, value in list(config.training_hyperparameters.items())[:10]:
            content += f"- **{key}**: {value}\n"

        content += """
## Model Hyperparameters

"""
        for key, value in list(config.model_hyperparameters.items())[:10]:
            content += f"- **{key}**: {value}\n"

        content += """
## Data Paths

"""
        for key, value in list(config.data_paths.items())[:5]:
            content += f"- **{key}**: {value}\n"

        content += """
## Distributed Settings

"""
        for key, value in list(config.distributed_settings.items())[:5]:
            content += f"- **{key}**: {value}\n"

        content += """
## Checkpoint Settings

"""
        for key, value in list(config.checkpoint_settings.items())[:5]:
            content += f"- **{key}**: {value}\n"

        content += """
## Optimizer Configuration

"""
        for key, value in list(config.optimizer_config.items())[:5]:
            content += f"- **{key}**: {value}\n"

        if config.scheduler_config:
            content += """
## Scheduler Configuration

"""
            for key, value in list(config.scheduler_config.items())[:5]:
                content += f"- **{key}**: {value}\n"

        content += """
## Config Sources

| Config Name | Source File |
|-------------|-------------|
"""
        for name, path in list(config.config_sources.items())[:10]:
            content += f"| {name} | {path} |\n"

        return content

    def generate_important_files(self, important_files: list[FileImportance]) -> str:
        """Generate important_files.md content."""
        content = """# Important Files

## File Importance Rankings

| File | Importance | Reason | Lines |
|------|------------|--------|-------|
"""

        for file_info in important_files[:15]:
            content += f"| {file_info.file_path} | {file_info.importance} | {file_info.reason} | {file_info.lines_of_code} |\n"

        content += """
## File Complexity

### Critical Files
"""
        critical = [f for f in important_files if f.importance == "Critical"]
        for f in critical[:5]:
            content += f"- {f.file_path} ({f.complexity} complexity)\n"

        content += """
### High Importance Files
"""
        high = [f for f in important_files if f.importance == "High"]
        for f in high[:5]:
            content += f"- {f.file_path} ({f.complexity} complexity)\n"

        content += """
### Supporting Files
"""
        medium = [f for f in important_files if f.importance == "Medium"]
        for f in medium[:5]:
            content += f"- {f.file_path} ({f.complexity} complexity)\n"

        return content

    def generate_implementation_targets(self, targets: list[ImplementationTarget]) -> str:
        """Generate implementation_targets.md content."""
        content = """# Implementation Targets

## Best Insertion Points for Code Modifications

"""

        target_types = {
            'attention': 'Add new attention mechanisms',
            'optimizer': 'Add new optimizers',
            'dataset': 'Add new datasets',
            'loss': 'Add new loss functions',
            'metric': 'Add new evaluation metrics',
        }

        for target_type, description in target_types.items():
            type_targets = [t for t in targets if t.target_type == target_type]
            if type_targets:
                content += f"## {description}\n\n"
                for target in type_targets[:3]:
                    content += f"- **File:** {target.file_path}\n"
                    content += f"  - **Class:** {target.class_name or 'None'}\n"
                    content += f"  - **Method:** {target.method_name or 'None'}\n"
                    content += f"  - **Insertion Point:** {target.insertion_point}\n"
                    content += f"  - **Estimated Lines:** {target.estimated_lines}\n\n"

        return content

    def generate_knowledge_graph(self, kg: KnowledgeGraphOutput) -> str:
        """Generate knowledge_graph.md content."""
        content = """# Knowledge Graph

## Overview

"""

        content += f"""
- **Total Nodes:** {len(kg.nodes)}
- **Total Edges:** {len(kg.edges)}
- **Central Nodes:** {len(kg.central_nodes)}

## Node Types

| Type | Count |
|------|-------|
"""
        for node_type, count in kg.node_types.items():
            content += f"| {node_type} | {count} |\n"

        content += """
## Relationships

| Relationship | Count |
|--------------|-------|
"""
        for rel_type, count in kg.relationships_by_type.items():
            content += f"| {rel_type} | {count} |\n"

        content += """
## Top Central Nodes

"""
        for node_id in kg.central_nodes[:10]:
            content += f"- {node_id}\n"

        content += """
## Communities

"""
        for i, community in enumerate(kg.communities[:5]):
            content += f"### Community {i + 1}\n"
            for node_id in community[:5]:
                content += f"- {node_id}\n"

        content += """
## Graph Statistics

| Metric | Value |
|--------|-------|
"""
        for stat, value in kg.graph_statistics.items():
            if isinstance(value, float):
                content += f"| {stat} | {value:.4f} |\n"
            else:
                content += f"| {stat} | {value} |\n"

        return content

    def generate_future_agent_context(self, summary: RepositorySummary) -> str:
        """Generate future_agent_context.md content."""
        content = """# Future Agent Context

## Repository Summary for Future Agents

This document provides context for future agents (LiteratureAgent, CodingAgent, ExperimentAgent, etc.) working with this repository.

## Repository Type

{summary.project_type}

## Architecture Insights

{summary.architecture_summary}

## Key Components

### Entry Points
- {', '.join(['main.py', 'train.py', 'run.py'][:3])}

### Core Modules
- {', '.join(['model', 'data', 'train'][:3])}

## Available Tools

## Training Pipeline

{summary.training_pipeline}

## Configuration

{summary.configuration_analysis}

## Knowledge Graph

The knowledge graph in `knowledge_graph.md` contains:
- {len(summary.implementation_targets)} implementation targets
- Direct dependencies between modules
- Class and function relationships

## Integration Points

### For CodingAgent

Use `implementation_targets.md` to find best insertion points for:
- New attention mechanisms
- New optimizers
- New datasets
- New loss functions
- New evaluation metrics

### For ExperimentPlannerAgent

Review {summary.training_pipeline} for:
- Dataset loading mechanism
- Model initialization process
- Training loop structure
- Validation methodology
- Checkpointing strategy

### For LiteratureReviewAgent

Compare with:
- Paper training methodology
- Model architecture
- Dataset used
- Evaluation metrics

### For EvaluationAgent

Check:
- Config for evaluation settings
- Metrics being logged
- Validation loop configuration
- Checkpoint saving strategy

## API References

### Public API

- {', '.join(['train', 'evaluate', 'predict'][:3])}

### Internal API

- {', '.join(['_setup', '_cleanup', '_log'][:3])}

## Notes

1. The repository uses PyTorch for training
2. Configuration is stored in config files
3. Checkpointing is enabled for long training runs
4. Distributed training may be configured
"""
        return content

    async def execute(self, input: DocumentationInput) -> DocumentationOutput:
        """Execute documentation generation."""
        try:
            output_dir = Path(input.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            generated_files = []
            total_lines = 0

            # Generate repo_summary.md
            summary_content = self.generate_repo_summary(input.repo_summary)
            summary_file = output_dir / "repo_summary.md"
            summary_file.write_text(summary_content)
            generated_files.append(str(summary_file))
            total_lines += len(summary_content.split('\n'))

            # Generate architecture_overview.md
            arch_content = self.generate_architecture_overview(input.repo_summary, input.architecture)
            arch_file = output_dir / "architecture_overview.md"
            arch_file.write_text(arch_content)
            generated_files.append(str(arch_file))
            total_lines += len(arch_content.split('\n'))

            # Generate training_pipeline.md
            train_content = self.generate_training_pipeline(input.training_pipeline)
            train_file = output_dir / "training_pipeline.md"
            train_file.write_text(train_content)
            generated_files.append(str(train_file))
            total_lines += len(train_content.split('\n'))

            # Generate dependency_graph.md
            deps_content = self.generate_dependency_graph(input.dependencies)
            deps_file = output_dir / "dependency_graph.md"
            deps_file.write_text(deps_content)
            generated_files.append(str(deps_file))
            total_lines += len(deps_content.split('\n'))

            # Generate configuration_analysis.md
            config_content = self.generate_configuration_analysis(input.config_analysis)
            config_file = output_dir / "configuration_analysis.md"
            config_file.write_text(config_content)
            generated_files.append(str(config_file))
            total_lines += len(config_content.split('\n'))

            # Generate important_files.md
            if input.important_files:
                important_content = self.generate_important_files(input.important_files)
                important_file = output_dir / "important_files.md"
                important_file.write_text(important_content)
                generated_files.append(str(important_file))
                total_lines += len(important_content.split('\n'))

            # Generate implementation_targets.md
            if input.implementation_targets:
                targets_content = self.generate_implementation_targets(input.implementation_targets)
                targets_file = output_dir / "implementation_targets.md"
                targets_file.write_text(targets_content)
                generated_files.append(str(targets_file))
                total_lines += len(targets_content.split('\n'))

            # Generate knowledge_graph.md
            kg_content = self.generate_knowledge_graph(input.knowledge_graph)
            kg_file = output_dir / "knowledge_graph.md"
            kg_file.write_text(kg_content)
            generated_files.append(str(kg_file))
            total_lines += len(kg_content.split('\n'))

            # Generate future_agent_context.md
            context_content = self.generate_future_agent_context(input.repo_summary)
            context_file = output_dir / "future_agent_context.md"
            context_file.write_text(context_content)
            generated_files.append(str(context_file))
            total_lines += len(context_content.split('\n'))

            return DocumentationOutput(
                generated_files=generated_files,
                output_directory=str(output_dir),
                file_count=len(generated_files),
                total_lines=total_lines,
            )

        except Exception as e:
            raise ToolError(f"Failed to generate documentation: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
