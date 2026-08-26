"""E1 - Production Agent Runtime models.

Typed Pydantic models for the generic, async-first :class:`AgentRuntime`
orchestration layer. These models describe the runtime's lifecycle, the
serializable execution state (supporting future checkpointing), budgets,
termination reasons, and the recoverable-vs-fatal error taxonomy.

The runtime is intentionally generic: it does not know about any specific
agent. It drives a ``plan -> act -> observe -> evaluate`` loop through
injected async callables and enforces budgets/termination uniformly.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentPhase(StrEnum):
    """Phase within a single runtime step."""

    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    EVALUATING = "evaluating"


class AgentState(StrEnum):
    """Lifecycle state of a runtime execution.

    The runtime is a deterministic state machine:

        CREATED -> RUNNING -> TERMINATED

    ``RUNNING`` is the only non-terminal state; every other state is
    terminal. The precise reason for termination is captured by
    :class:`AgentTermination`.
    """

    CREATED = "created"
    RUNNING = "running"
    TERMINATED = "terminated"


class AgentTermination(StrEnum):
    """Why a runtime execution stopped.

    Mirrors the required termination reasons from the E1 spec: success,
    budget exceeded, timeout, cancelled, error, and no-progress.
    """

    SUCCESS = "success"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"
    NO_PROGRESS = "no_progress"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class AgentBudget(BaseModel):
    """Resource budgets enforced by the runtime.

    All fields are optional; ``None`` means "unbounded". The runtime checks
    these between phases and terminates with ``BUDGET_EXCEEDED`` (or
    ``TIMEOUT`` for the wall-clock budget) when any is exceeded.
    """

    max_steps: int | None = Field(
        default=None, ge=1, description="Maximum plan/act/observe/evaluate steps"
    )
    max_tool_calls: int | None = Field(
        default=None, ge=1, description="Maximum total tool invocations"
    )
    max_runtime_seconds: float | None = Field(
        default=None, gt=0.0, description="Maximum wall-clock runtime in seconds"
    )
    max_cost_usd: float | None = Field(
        default=None, gt=0.0, description="Maximum cumulative USD cost"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum cumulative token usage"
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class AgentPolicy(BaseModel):
    """Runtime policy: budgets plus error-recovery behaviour.

    ``max_recoverable_errors`` caps how many recoverable errors the runtime
    tolerates before terminating with ``ERROR``. A fatal error always
    terminates immediately regardless of this cap.
    """

    budget: AgentBudget = Field(
        default_factory=AgentBudget, description="Resource budgets"
    )
    max_recoverable_errors: int = Field(
        default=3, ge=0, description="Recoverable errors tolerated before ERROR"
    )
    stagnation_window: int = Field(
        default=3, ge=1, description="Steps without progress before NO_PROGRESS"
    )
    progress_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description="Min |delta| in evaluation score to count as progress",
    )


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class AgentError(BaseModel):
    """A classified error raised during a runtime step.

    ``recoverable`` distinguishes transient failures (network blips, tool
    timeouts) that the runtime may retry from fatal failures (invalid
    configuration, permanent provider errors) that terminate the run.
    """

    message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Exception class name")
    recoverable: bool = Field(
        default=False, description="True when the runtime may retry this error"
    )
    step: int = Field(default=0, ge=0, description="Step number at which it occurred")
    phase: AgentPhase | None = Field(
        default=None, description="Phase at which it occurred"
    )


# ---------------------------------------------------------------------------
# Step + context + execution
# ---------------------------------------------------------------------------


class AgentStep(BaseModel):
    """A single plan/act/observe/evaluate iteration of the runtime."""

    step: int = Field(..., ge=1, description="One-based step number")
    plan: Any = Field(default=None, description="Plan output for this step")
    action: Any = Field(default=None, description="Action output for this step")
    observation: Any = Field(default=None, description="Observation output for this step")
    evaluation: Any = Field(default=None, description="Evaluation output for this step")
    score: float | None = Field(
        default=None, description="Numeric evaluation score (for progress tracking)"
    )
    tool_calls: int = Field(default=0, ge=0, description="Tool calls made this step")
    tokens: int = Field(default=0, ge=0, description="Tokens consumed this step")
    cost_usd: float = Field(default=0.0, ge=0.0, description="USD cost this step")
    error: AgentError | None = Field(
        default=None, description="Error raised during this step, if any"
    )
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0, ge=0.0)

class AgentContext(BaseModel):
    """Serializable execution state of a runtime run.

    This is the single source of truth for a run's progress. It is fully
    JSON-serializable (``model_dump_json``) so it can be persisted for
    future checkpointing (E2) without modification.
    """

    execution_id: str = Field(
        default_factory=lambda: f"exec_{uuid4().hex[:12]}",
        description="Unique execution identifier",
    )
    goal: str = Field(..., description="High-level goal for this run")
    state: AgentState = Field(
        default=AgentState.CREATED, description="Current lifecycle state"
    )
    phase: AgentPhase | None = Field(
        default=None, description="Current phase within the loop"
    )
    termination: AgentTermination | None = Field(
        default=None, description="Why the run terminated (when terminal)"
    )
    termination_reason: str = Field(
        default="", description="Human-readable termination reason"
    )
    steps: list[AgentStep] = Field(
        default_factory=list, description="Completed steps in order"
    )
    current_step: int = Field(default=0, ge=0, description="Steps completed")
    tool_calls: int = Field(default=0, ge=0, description="Total tool calls made")
    tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Total USD cost")
    recoverable_errors: int = Field(
        default=0, ge=0, description="Recoverable errors tolerated so far"
    )
    stagnation_count: int = Field(
        default=0, ge=0, description="Consecutive steps without progress"
    )
    best_score: float | None = Field(
        default=None, description="Best evaluation score observed"
    )
    output: Any = Field(default=None, description="Final output of the run")
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary caller metadata"
    )

    def is_terminal(self) -> bool:
        """True when the run has reached a terminal state."""
        return self.state == AgentState.TERMINATED

    def is_success(self) -> bool:
        """True when the run terminated successfully."""
        return self.termination == AgentTermination.SUCCESS


class AgentExecution(BaseModel):
    """The outcome of a runtime run."""

    context: AgentContext = Field(..., description="Final execution state")
    termination: AgentTermination = Field(..., description="Why the run stopped")
    reason: str = Field(default="", description="Human-readable termination reason")
    output: Any = Field(default=None, description="Final output")
    timestamp: datetime = Field(default_factory=datetime.now)


__all__ = [
    "AgentPhase",
    "AgentState",
    "AgentTermination",
    "AgentBudget",
    "AgentPolicy",
    "AgentError",
    "AgentStep",
    "AgentContext",
    "AgentExecution",
]

