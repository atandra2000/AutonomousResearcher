"""E7 - Production deployment API/service models.

Typed Pydantic v2 request/response models for the production run service.
The wire surface deliberately exposes only identifiers, statuses, goals,
and results — never internal secrets, prompts, credentials, or tool
interfaces.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(StrEnum):
    """Lifecycle status of a submitted agent run.

    State machine::

        queued -> running -> completed
                          | failed
                          | cancelled

    A ``running`` run whose worker dies becomes ``resumable`` (a transient
    status set during recovery) and may be resumed back into ``running``.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESUMABLE = "resumable"

    @classmethod
    def terminal(cls) -> tuple[RunStatus, ...]:
        """Statuses from which a run cannot transition."""
        return (cls.COMPLETED, cls.FAILED, cls.CANCELLED)

    def is_terminal(self) -> bool:
        return self in RunStatus.terminal()


class CreateRunRequest(BaseModel):
    """Body of ``POST /runs``."""

    goal: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="High-level goal for the autonomous run.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional caller-supplied metadata (scalar values only). "
            "Stored with the run record."
        ),
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Optional per-run step budget override.",
    )
    max_runtime_seconds: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional per-run wall-clock budget override.",
    )


class ArtifactReference(BaseModel):
    """Reference to an artifact produced by a run.

    Mirrors the project-wide artifact convention: an artifact is a file on
    the artifact volume, identified by a checksum-bearing manifest entry.
    """

    name: str = Field(..., description="Artifact file name within the run directory")
    path: str = Field(..., description="Run-relative path of the artifact")
    sha256: str = Field(default="", description="SHA-256 checksum when computed")
    size_bytes: int = Field(default=0, ge=0)
    content_type: str = Field(default="application/octet-stream")


class RunRecord(BaseModel):
    """Persistent record of one submitted run."""

    run_id: str
    goal: str
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worker_id: str | None = None
    claim_count: int = 0
    cancel_requested: bool = False
    error: str | None = None
    termination_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    budget_overrides: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class RunCreatedResponse(BaseModel):
    """Response body of successful ``POST /runs``."""

    run_id: str
    status: RunStatus
    created_at: datetime


class RunStatusResponse(BaseModel):
    """Response body of ``GET /runs/{run_id}``."""

    run_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    termination_reason: str | None = None
    has_checkpoint: bool = False
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class CancelResponse(BaseModel):
    """Response body of ``POST /runs/{run_id}/cancel``."""

    run_id: str
    status: RunStatus
    cancelled: bool


class ResumeResponse(BaseModel):
    """Response body of ``POST /runs/{run_id}/resume``."""

    run_id: str
    status: RunStatus
    resumed: bool


class ResultResponse(BaseModel):
    """Response body of ``GET /runs/{run_id}/result``."""

    run_id: str
    status: RunStatus
    output: Any = None
    termination_reason: str | None = None
    duration_seconds: float | None = None
    steps: int = 0
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    available: bool = Field(
        default=False, description="True when the run has reached a terminal state"
    )


class HealthResponse(BaseModel):
    """Liveness probe response: the process itself is alive."""

    status: str = "ok"
    service: str = "research-engineer"


class ReadinessResponse(BaseModel):
    """Readiness probe response: dependencies are reachable."""

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


__all__ = [
    "ArtifactReference",
    "CancelResponse",
    "CreateRunRequest",
    "HealthResponse",
    "ReadinessResponse",
    "ResultResponse",
    "ResumeResponse",
    "RunCreatedResponse",
    "RunRecord",
    "RunStatus",
    "RunStatusResponse",
]
