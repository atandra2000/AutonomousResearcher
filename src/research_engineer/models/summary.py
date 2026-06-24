"""Research summary domain models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    """Structured summary of an ML research paper."""

    paper_id: str = Field(..., description="Paper ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Summary creation time")

    # Core research sections
    executive_summary: str = Field(
        ...,
        description="2-3 sentence summary of the paper's main achievement"
    )
    problem_statement: str = Field(
        ...,
        description="Specific problem addressed in the paper"
    )
    core_contributions: list[str] = Field(
        ...,
        min_length=1,
        description="List of core contributions"
    )
    model_architecture: str = Field(
        ...,
        description="Main model architecture description"
    )
    training_methodology: str = Field(
        ...,
        description="Training approach, loss functions, optimization"
    )
    dataset_information: str = Field(
        ...,
        description="Dataset names, sizes, preprocessing"
    )
    evaluation_methodology: str = Field(
        ...,
        description="Evaluation metrics, baselines, experimental setup"
    )
    key_results: list[str] = Field(
        ...,
        min_length=1,
        description="Key performance metrics and results"
    )
    limitations: list[str] = Field(
        ...,
        min_length=1,
        description="Paper's acknowledged limitations"
    )
    reproduction_challenges: list[str] = Field(
        ...,
        min_length=1,
        description="Challenges in reproducing the work"
    )

    # Engineering-focused sections
    engineering_complexity: str = Field(
        default="Medium",
        description="Overall complexity: Low, Medium, High"
    )
    implementation_difficulty: str = Field(
        default="Medium",
        description="Implementation difficulty: Low, Medium, High"
    )
    compute_requirements: str = Field(
        default="High",
        description="Compute requirements: Low, Medium, High"
    )
    hardware_requirements: str = Field(
        default="2x GPU",
        description="Required hardware specifications"
    )
    expected_training_time: str = Field(
        default="1-2 days",
        description="Estimated training time"
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()
