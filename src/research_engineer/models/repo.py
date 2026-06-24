"""Repository domain models for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RepositoryType:
    """Enum for repository types."""
    LLM_TRAINING = "LLMTrainingFramework"
    CV_TRAINING = "CVTrainingPipeline"
    INFERENCE = "InferenceServer"
    RAG = "RAGSystem"
    AGENT = "AgentFramework"
    FINE_TUNING = "FineTuningSystem"
    EVALUATION = "EvaluationFramework"
    MULTIMODAL = "MultimodalSystem"
    DIFFUSION = "DiffusionModel"
    RL = "ReinforcementLearning"
    UNKNOWN = "Unknown"


class Repository(BaseModel):
    """Represents a research repository."""

    name: str = Field(..., description="Repository name")
    path: str = Field(..., description="File system path to repository")
    repo_type: str = Field(..., description="Type of repository")
    main_language: str = Field(default="Python", description="Primary programming language")
    entry_points: list[str] = Field(default_factory=list, description="Entry point files")
    dependencies: list[str] = Field(default_factory=list, description="Python dependencies")
    frameworks: list[str] = Field(default_factory=list, description="ML frameworks used")
    total_files: int = Field(default=0, description="Total files in repository")
    total_lines: int = Field(default=0, description="Total lines of code")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FileImportance(BaseModel):
    """Represents importance of a file in the repository."""

    file_path: str = Field(..., description="Path to file")
    importance: str = Field(..., description="Critical, High, Medium, Low")
    reason: str = Field(..., description="Explanation of why this file matters")
    complexity: str = Field(default="Medium", description="Low, Medium, High")
    lines_of_code: int = Field(default=0, description="Lines of code in file")
    dependencies_count: int = Field(default=0, description="Number of outgoing dependencies")


class ImplementationTarget(BaseModel):
    """Represents a target location for code insertion/modification."""

    file_path: str = Field(..., description="Path to file")
    class_name: str | None = Field(None, description="Class name if applicable")
    method_name: str | None = Field(None, description="Method name if applicable")
    target_type: str = Field(..., description="attention, optimizer, dataset, loss, metric")
    insertion_point: str = Field(..., description="Detailed location info")
    complexity: str = Field(default="Medium", description="Low, Medium, High")
    estimated_lines: int = Field(default=50, description="Estimated lines to add")


class KnowledgeGraph(BaseModel):
    """Represents knowledge graph of repository."""

    nodes: list[dict[str, Any]] = Field(default_factory=list, description="Graph nodes")
    edges: list[dict[str, Any]] = Field(default_factory=list, description="Graph edges")
    communities: list[list[str]] = Field(default_factory=list, description="Related node groups")
    central_nodes: list[str] = Field(default_factory=list, description="Most important nodes")
    relationships_by_type: dict[str, int] = Field(default_factory=dict, description="Edge type counts")


class TrainingPipeline(BaseModel):
    """Represents training pipeline analysis."""

    dataset_loading: str = Field(..., description="Dataset loading mechanism")
    preprocessing: str = Field(..., description="Preprocessing steps")
    tokenization: str | None = Field(None, description="Tokenization approach")
    batch_construction: str = Field(..., description="Batch building")
    model_initialization: str = Field(..., description="Model initialization")
    optimizer_creation: str = Field(..., description="Optimizer setup")
    training_loop: str = Field(..., description="Training loop structure")
    validation_loop: str | None = Field(None, description="Validation loop")
    checkpoint_saving: str = Field(..., description="Checkpoint mechanism")
    metric_logging: str = Field(..., description="Metrics tracking")
    distributed_training: str | None = Field(None, description="Distributed setup")


class ConfigurationAnalysis(BaseModel):
    """Represents configuration system analysis."""

    config_files: list[str] = Field(default_factory=list, description="Config file paths")
    training_hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Training config")
    model_hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Model config")
    data_paths: dict[str, str] = Field(default_factory=dict, description="Data locations")
    distributed_settings: dict[str, Any] = Field(default_factory=dict, description="Distributed config")
    checkpoint_settings: dict[str, Any] = Field(default_factory=dict, description="Checkpoint config")
    config_framework: str = Field(default="unknown", description="YAML/JSON/TOML/Python")


class RepositorySummary(BaseModel):
    """Summary of repository analysis."""

    repository_name: str = Field(..., description="Repository name")
    project_type: str = Field(..., description="Project type classification")
    architecture_summary: str = Field(..., description="High-level architecture description")
    important_files: list[FileImportance] = Field(..., description="File importance rankings")
    training_pipeline: str = Field(..., description="Training pipeline description")
    knowledge_graph: KnowledgeGraph = Field(..., description="Knowledge graph")
    implementation_targets: list[ImplementationTarget] = Field(
        default_factory=list,
        description="Best insertion points"
    )
    configuration_analysis: ConfigurationAnalysis = Field(
        ...,
        description="Configuration analysis"
    )
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis time")


class ArchitectureOverview(BaseModel):
    """Detailed architecture overview."""

    high_level_structure: str = Field(..., description="High-level architecture")
    entry_points: list[str] = Field(default_factory=list, description="Entry point files")
    main_modules: list[str] = Field(default_factory=list, description="Main modules")
    core_abstractions: list[str] = Field(default_factory=list, description="Key abstractions")
    critical_classes: list[str] = Field(default_factory=list, description="Important classes")
    interfaces: list[str] = Field(default_factory=list, description="Interface definitions")
    package_boundaries: list[str] = Field(default_factory=list, description="Module boundaries")
    mermaid_diagram: str = Field(..., description="Mermaid architecture diagram")


class DependencyGraph(BaseModel):
    """Dependency relationship analysis."""

    file_level_edges: list[dict[str, str]] = Field(
        default_factory=list,
        description="File-to-file dependencies"
    )
    module_level_edges: list[dict[str, str]] = Field(
        default_factory=list,
        description="Module-to-module dependencies"
    )
    circular_dependencies: list[list[str]] = Field(
        default_factory=list,
        description="Detected circular dependencies"
    )
    dependency_tree: dict[str, list[str]] = Field(
        default_factory=dict,
        description="File dependency tree"
    )
