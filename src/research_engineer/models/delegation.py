"""Phase 13 - Multi-agent delegation models.

Typed Pydantic models for the delegation framework: agent roles,
capabilities, shared task context (structured inter-agent communication),
delegation steps, and feedback loop records.

These models are the serialization contract between the TaskAgent
coordinator and its specialized sub-agents. They enable role/capability-
based routing rather than hardcoded task logic.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentRole(StrEnum):
    """Specialized role an agent plays in the delegation pipeline."""

    COORDINATOR = "coordinator"
    REPOSITORY_ANALYZER = "repository_analyzer"
    RESEARCHER = "researcher"
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"
    TERMINAL = "terminal"


class AgentCapability(StrEnum):
    """Capability an agent provides, used for routing."""

    REPOSITORY_ANALYSIS = "repository_analysis"
    RESEARCH = "research"
    ARCHITECTURE = "architecture"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TEST_EXECUTION = "test_execution"
    TERMINAL_OPS = "terminal_ops"
    REPAIR = "repair"


class DelegationStatus(StrEnum):
    """Status of a single delegation step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FeedbackType(StrEnum):
    """Kind of feedback from review/test loops."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    TEST_FAILURE = "test_failure"
    TEST_PASS = "test_pass"


# ---------------------------------------------------------------------------
# Shared task context
# ---------------------------------------------------------------------------


class SharedTaskContext(BaseModel):
    """Structured context shared across all agents in a delegation pipeline.

    This is the single communication channel between agents. Each agent
    reads the fields it needs and writes its outputs back, enabling
    inter-agent communication without tight coupling.

    The context accumulates as the pipeline progresses: the repository
    analysis adds repo_summary, the architect adds the plan, the coder
    adds patches, the reviewer adds feedback, etc.
    """

    goal: str = Field(..., description="Natural-language coding goal")
    repo_path: str = Field(..., description="Repository path")
    paper_input: str | None = Field(
        default=None, description="Optional paper ID/URL/PDF"
    )
    output_dir: str = Field(default="output/tasks", description="Output directory")

    # Accumulated context from each phase.
    repo_summary: dict[str, Any] = Field(
        default_factory=dict, description="Repository analysis result"
    )
    memory_context: str = Field(
        default="", description="Repository memory context (Phase 12)"
    )
    research_context: str = Field(
        default="", description="Research findings about the approach"
    )
    implementation_plan: str = Field(
        default="", description="Architectural plan from ArchitectAgent"
    )
    implementation_id: str | None = Field(
        default=None, description="CodingAgent implementation ID"
    )
    patches_generated: int = Field(
        default=0, description="Number of patches produced"
    )
    diff: str = Field(default="", description="Unified diff of changes")
    review_feedback: str = Field(
        default="", description="Review feedback from ReviewerAgent"
    )
    review_issues: list[str] = Field(
        default_factory=list, description="Specific issues found in review"
    )
    test_exit_code: int | None = Field(
        default=None, description="Test command exit code"
    )
    test_stdout: str = Field(default="", description="Test command stdout")
    test_stderr: str = Field(default="", description="Test command stderr")
    test_failures: list[str] = Field(
        default_factory=list, description="Parsed test failure messages"
    )
    generated_files: list[str] = Field(
        default_factory=list, description="All files produced across phases"
    )

    # Metadata.
    repair_iteration: int = Field(
        default=0, description="Current repair iteration (0 = first pass)"
    )
    max_repair_iterations: int = Field(
        default=2, description="Max repair iterations before giving up"
    )
    # Phase 14: Self-repair history.
    repair_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of repair cycles (FailureReport + RepairStrategy)",
    )
    last_failure_category: str = Field(
        default="", description="Category of the last failure (for stagnation)"
    )

    def add_files(self, files: list[str]) -> None:
        """Add file paths to generated_files (deduplicated)."""
        for f in files:
            if f not in self.generated_files:
                self.generated_files.append(f)


# ---------------------------------------------------------------------------
# Delegation step
# ---------------------------------------------------------------------------


class DelegationStep(BaseModel):
    """Record of a single agent invocation within the pipeline."""

    step_id: str = Field(..., description="Unique step identifier")
    role: AgentRole = Field(..., description="Agent role invoked")
    capability: AgentCapability = Field(..., description="Capability exercised")
    agent_name: str = Field(..., description="Agent class name")
    status: DelegationStatus = Field(
        default=DelegationStatus.PENDING, description="Step status"
    )
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0)
    summary: str = Field(default="", description="Human-readable summary")
    output: dict[str, Any] = Field(
        default_factory=dict, description="Structured output from the agent"
    )
    error: str | None = Field(default=None, description="Error if failed")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class Feedback(BaseModel):
    """Feedback from a review or test step."""

    feedback_type: FeedbackType = Field(..., description="Kind of feedback")
    approved: bool = Field(..., description="Whether the step passed")
    issues: list[str] = Field(
        default_factory=list, description="Issues found (if any)"
    )
    summary: str = Field(default="", description="Human-readable summary")
    repair_needed: bool = Field(
        default=False, description="Whether a repair iteration is needed"
    )


# ---------------------------------------------------------------------------
# Agent descriptor (for capability routing)
# ---------------------------------------------------------------------------


class AgentDescriptor(BaseModel):
    """Describes an agent's role and capabilities for routing."""

    agent_name: str = Field(..., description="Agent class name")
    role: AgentRole = Field(..., description="Agent role")
    capabilities: list[AgentCapability] = Field(
        default_factory=list, description="What this agent can do"
    )
    agent: Any | None = Field(
        default=None,
        description="The agent instance (excluded from serialization)",
        exclude=True,
    )

    def has_capability(self, cap: AgentCapability) -> bool:
        return cap in self.capabilities

    def can_role(self, role: AgentRole) -> bool:
        return self.role == role


__all__ = [
    "AgentRole",
    "AgentCapability",
    "DelegationStatus",
    "FeedbackType",
    "SharedTaskContext",
    "DelegationStep",
    "Feedback",
    "AgentDescriptor",
]
