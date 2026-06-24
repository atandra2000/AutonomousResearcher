"""Storage models for persistent data."""

from datetime import datetime

from pydantic import BaseModel, Field


class StoredPaper(BaseModel):
    """Represents a paper stored in the database."""

    id: int | None = Field(None, description="Database record ID")
    paper_id: str = Field(..., description="Paper ID (arXiv or local hash)")
    title: str = Field(..., description="Paper title")
    authors_json: str = Field(..., description="Serialized authors list")
    summary_json: str = Field(..., description="Serialized research summary")
    plan_json: str = Field(..., description="Serialized engineering report")
    created_at: datetime = Field(default_factory=datetime.now, description="Storage timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    @classmethod
    def from_data(
        cls,
        paper_id: str,
        title: str,
        authors_json: str,
        summary_json: str,
        plan_json: str,
        id: int | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None
    ) -> "StoredPaper":
        """Create a stored paper from component data."""
        return cls(
            id=id,
            paper_id=paper_id,
            title=title,
            authors_json=authors_json,
            summary_json=summary_json,
            plan_json=plan_json,
            created_at=created_at or datetime.now(),
            updated_at=updated_at
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
