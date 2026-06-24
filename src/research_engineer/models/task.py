"""Phase 11 - Terminal-first autonomous coding agent models.

Typed Pydantic models for the ``research-engineer task`` command and the
:class:`~research_engineer.agents.task_agent.TaskAgent` orchestrator.

The task agent runs a single autonomous coding turn:
analyze repository -> reason about goal -> implement patch -> show diff
-> optionally execute tests.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle status of a task run."""

    CREATED = "created"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    DIFFING = "diffing"
    TESTING = "testing"
    REVIEWING = "reviewing"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStepType(StrEnum):
    """Kind of step executed within a task run."""

    ANALYZE_REPO = "analyze_repo"
    RESEARCH = "research"
    PLAN = "plan"
    IMPLEMENT = "implement"
    REVIEW = "review"
    DIFF = "diff"
    TEST = "test"
    REPAIR = "repair"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TaskConfig(BaseModel):
    """Configuration for a single task run."""

    goal: str = Field(..., description="Natural-language coding goal")
    repo_path: str = Field(..., description="Repository path to operate on")
    paper_input: str | None = Field(
        default=None,
        description="Optional paper ID/URL/PDF for research-grounded tasks",
    )
    run_tests: bool = Field(
        default=False,
        description="If True, execute the test command after patching",
    )
    test_command: str = Field(
        default="uv run pytest",
        description="Command to run for the optional test step",
    )
    dry_run: bool = Field(
        default=True,
        description="If True, generate patches without applying them",
    )
    stream: bool = Field(
        default=True,
        description="Stream LLM tokens to stdout during interactive phases",
    )
    output_dir: str = Field(
        default="output/tasks",
        description="Directory to save task artifacts",
    )
    timeout_seconds: int = Field(
        default=600,
        ge=1,
        description="Timeout for the optional test command",
    )
    delegate: bool = Field(
        default=False,
        description="If True, use multi-agent delegation pipeline (Phase 13)",
    )
    max_repair_iterations: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Max review/test repair iterations in delegation mode",
    )


# ---------------------------------------------------------------------------
# Step + result
# ---------------------------------------------------------------------------


class TaskStep(BaseModel):
    """A single step executed within a task run."""

    step_type: TaskStepType = Field(..., description="Kind of step")
    status: TaskStatus = Field(..., description="Step outcome status")
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(
        default=None, description="When the step finished"
    )
    duration_seconds: float = Field(default=0.0, description="Step duration")
    summary: str = Field(default="", description="Human-readable step summary")
    artifacts: list[str] = Field(
        default_factory=list,
        description="File paths produced by this step",
    )
    error: str | None = Field(default=None, description="Error message if failed")


class TaskResult(BaseModel):
    """Final result of a task run."""

    task_id: str = Field(..., description="Unique task identifier")
    goal: str = Field(..., description="Original coding goal")
    repo_path: str = Field(..., description="Repository operated on")
    status: TaskStatus = Field(..., description="Final task status")
    steps: list[TaskStep] = Field(
        default_factory=list, description="Executed steps in order"
    )
    implementation_id: str | None = Field(
        default=None, description="CodingAgent implementation ID"
    )
    patches_generated: int = Field(
        default=0, description="Number of patches produced"
    )
    diff: str = Field(default="", description="Unified diff of changes")
    test_exit_code: int | None = Field(
        default=None, description="Exit code of the test command (if run)"
    )
    test_stdout: str = Field(default="", description="Test command stdout")
    test_stderr: str = Field(default="", description="Test command stderr")
    generated_files: list[str] = Field(
        default_factory=list, description="All files produced"
    )
    processing_time_seconds: float = Field(
        default=0.0, description="Total task wall-clock time"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    error: str | None = Field(default=None, description="Top-level error")
    # Phase 13: Delegation metadata.
    delegated: bool = Field(
        default=False, description="Whether delegation mode was used"
    )
    repair_iterations: int = Field(
        default=0, description="Number of repair iterations executed"
    )
    review_feedback: str = Field(
        default="", description="Review feedback summary"
    )
    review_issues: list[str] = Field(
        default_factory=list, description="Issues found in review"
    )
    test_failures: list[str] = Field(
        default_factory=list, description="Parsed test failure messages"
    )


def new_task_id() -> str:
    """Generate a unique task identifier."""
    return f"task_{uuid4().hex[:12]}"


__all__ = [
    "TaskStatus",
    "TaskStepType",
    "TaskConfig",
    "TaskStep",
    "TaskResult",
    "new_task_id",
]
