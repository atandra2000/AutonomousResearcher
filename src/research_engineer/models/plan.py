"""Engineering report and implementation plan models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ComplexityMetrics(BaseModel):
    """Metrics for assessing implementation complexity."""

    code_complexity: str = Field(
        default="Medium",
        description="Code complexity: Low, Medium, High"
    )
    data_requirements: str = Field(
        default="Medium",
        description="Data requirements: Low, Medium, High"
    )
    compute_requirements: str = Field(
        default="Medium",
        description="Compute requirements: Low, Medium, High"
    )
    inference_complexity: str = Field(
        default="Low",
        description="Inference complexity: Low, Medium, High"
    )
    training_time: str = Field(
        default="1-2 days on 2x A100",
        description="Estimated training time and hardware"
    )
    deployment_complexity: str = Field(
        default="Medium",
        description="Deployment complexity: Low, Medium, High"
    )
    memory_requirements: str = Field(
        default="16GB RAM, 8GB VRAM",
        description="Memory requirements"
    )
    model_size: str = Field(
        default="100MB - 10GB",
        description="Model file size range"
    )


class FileRequirement(BaseModel):
    """Description of a required file for implementation."""

    filename: str = Field(..., description="File name (e.g., model.py)")
    path: str = Field(default="", description="File path within project")
    purpose: str = Field(..., description="Purpose of the file")
    dependencies: list[str] = Field(default_factory=list, description="Dependencies")
    complexity: str = Field(default="Medium", description="Complexity: Low, Medium, High")
    estimated_lines: int = Field(default=100, description="Estimated lines of code")


class EngineeringReport(BaseModel):
    """Engineering-focused analysis and implementation plan."""

    paper_id: str = Field(..., description="Paper ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Report creation time")

    complexity_analysis: ComplexityMetrics = Field(
        ...,
        description="Complexity metrics"
    )
    step_by_step_implementation: str = Field(
        ...,
        description="Step-by-step implementation guide"
    )
    files_required: list[FileRequirement] = Field(
        ...,
        min_length=1,
        description="List of required files"
    )
    development_effort: str = Field(
        ...,
        description="Estimated development effort (e.g., '3-5 days for experienced ML engineer')"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Required dependencies"
    )
    pytorch_modules: list[str] = Field(
        default_factory=list,
        description="PyTorch modules to use"
    )
    test_coverage: list[str] = Field(
        default_factory=list,
        description="Test coverage requirements"
    )
    benchmark_targets: list[str] = Field(
        default_factory=list,
        description="Benchmark targets to compare against"
    )
    deployment_options: list[str] = Field(
        default_factory=list,
        description="Deployment options"
    )
    cost_estimation: str = Field(
        default="Varies based on hardware and cloud provider",
        description="Estimated costs"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
