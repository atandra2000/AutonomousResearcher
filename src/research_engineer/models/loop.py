"""Phase 9 - Autonomous Research Loop models.

Typed Pydantic models for the autonomous research loop orchestrator,
including loop state, iteration records, stopping conditions, approval
workflow, and storage I/O.
"""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LoopStatus(StrEnum):
    """Lifecycle status of a research loop."""

    CREATED = "created"
    RUNNING = "running"
    ITERATING = "iterating"
    AWAITING_APPROVAL = "awaiting_approval"
    EVALUATED = "evaluated"
    STOPPED = "stopped"
    FAILED = "failed"


class IterationPhase(StrEnum):
    """Phase within a single iteration."""

    LITERATURE = "literature"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    EXPERIMENT = "experiment"
    EVALUATION = "evaluation"
    DECISION = "decision"


class StoppingCondition(StrEnum):
    """Reason the loop stopped."""

    TARGET_ACHIEVED = "target_achieved"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_IMPROVEMENT = "no_improvement"


class ApprovalGate(StrEnum):
    """Approval gate within an iteration."""

    PLAN = "plan"
    IMPLEMENTATION = "implementation"
    NEXT_ITERATION = "next_iteration"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    """Configuration for the autonomous research loop."""

    goal: str = Field(..., description="High-level research goal")
    repo_path: str = Field(..., description="Repository path")
    max_iterations: int = Field(
        5, ge=1, le=100, description="Maximum iterations"
    )
    target_metric_name: str | None = Field(
        None, description="Metric name to optimize"
    )
    target_metric_value: float | None = Field(
        None, description="Target metric value to achieve"
    )
    higher_is_better: bool = Field(
        False, description="True for accuracy, False for loss"
    )
    budget_hours: float | None = Field(
        None, gt=0.0, description="GPU-hour budget"
    )
    budget_cost: float | None = Field(
        None, gt=0.0, description="USD budget"
    )
    approval_mode: bool = Field(
        False, description="Enable human-approval gates"
    )
    skip_literature_after_first: bool = Field(
        True, description="Skip literature discovery after first iteration"
    )
    stagnation_window: int = Field(
        3, ge=2, description="Iterations without improvement before stopping"
    )
    improvement_threshold: float = Field(
        1e-4, gt=0.0, description="Minimum metric improvement to count"
    )
    dry_run: bool = Field(
        True, description="Dry-run experiments (no actual execution)"
    )
    stop_on_error: bool = Field(
        False, description="Stop loop on iteration error"
    )
    output_dir: str = Field(
        "output/loops", description="Default output directory"
    )
    primary_metric: str | None = Field(
        None, description="Override primary metric for evaluation"
    )
    experiment_command: str | None = Field(
        None, description="Seed experiment command for iteration 1"
    )
    cost_per_gpu_hour: float = Field(
        2.0, gt=0.0, description="USD per GPU-hour for budget tracking"
    )


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    """A human-approval gate request."""

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request ID",
    )
    loop_id: str = Field(..., description="Parent loop ID")
    iteration_number: int = Field(..., description="Current iteration number")
    gate: ApprovalGate = Field(..., description="Which gate")
    summary: str = Field("", description="Human-readable summary")
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Related artifact IDs"
    )
    options: list[str] = Field(
        default_factory=lambda: ["approve", "modify", "stop"],
        description="Available decisions",
    )


# ---------------------------------------------------------------------------
# Loop State
# ---------------------------------------------------------------------------


class LoopState(BaseModel):
    """Mutable state of a running loop."""

    loop_id: str = Field(..., description="Loop ID")
    goal: str = Field(..., description="Research goal")
    status: LoopStatus = Field(
        LoopStatus.CREATED, description="Current status"
    )
    current_iteration: int = Field(
        0, ge=0, description="Current iteration number"
    )
    iterations: list["LoopIteration"] = Field(
        default_factory=list, description="Completed iterations"
    )
    best_metric_value: float | None = Field(
        None, description="Best metric value achieved"
    )
    cumulative_cost_hours: float = Field(
        0.0, ge=0.0, description="Cumulative GPU-hours used"
    )
    cumulative_cost_usd: float = Field(
        0.0, ge=0.0, description="Cumulative cost in USD"
    )
    pending_approval: ApprovalRequest | None = Field(
        None, description="Pending approval request, if paused"
    )
    next_command: str | None = Field(
        None, description="Next experiment command derived from evaluation"
    )
    started_at: datetime = Field(
        default_factory=datetime.now, description="Loop start time"
    )
    updated_at: datetime | None = Field(
        None, description="Last update time"
    )
    error: str | None = Field(None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class LoopIteration(BaseModel):
    """A single iteration record within the loop."""

    iteration_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique iteration ID",
    )
    loop_id: str = Field(..., description="Parent loop ID")
    iteration_number: int = Field(..., description="1-based iteration number")
    phase: IterationPhase = Field(
        IterationPhase.LITERATURE, description="Last phase reached"
    )
    paper_id: str | None = Field(None, description="Paper ID used")
    paper_title: str | None = Field(None, description="Paper title")
    plan_id: str | None = Field(None, description="Plan ID from planner")
    implementation_id: str | None = Field(
        None, description="Implementation ID from coding agent"
    )
    experiment_id: str | None = Field(
        None, description="Experiment ID from experiment agent"
    )
    evaluation_id: str | None = Field(
        None, description="Evaluation ID from evaluation agent"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Metrics from this iteration"
    )
    primary_metric_name: str | None = Field(
        None, description="Tracked primary metric name"
    )
    primary_metric_value: float | None = Field(
        None, description="Primary metric value this iteration"
    )
    best_metric_value: float | None = Field(
        None, description="Running best metric value"
    )
    improvement: float | None = Field(
        None, description="Improvement vs previous best"
    )
    decision: StoppingCondition | None = Field(
        None, description="Stopping decision, if any"
    )
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    error: str | None = Field(None, description="Error message if failed")
    status: LoopStatus = Field(
        LoopStatus.CREATED, description="Iteration outcome status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Record timestamp"
    )

    def is_success(self) -> bool:
        """Return True if the iteration completed without error."""
        return self.error is None and self.status != LoopStatus.FAILED


# ---------------------------------------------------------------------------
# Iteration Storage I/O
# ---------------------------------------------------------------------------


class IterationStorageInput(BaseModel):
    """Input for iteration storage operations."""

    iteration: LoopIteration = Field(..., description="Iteration to store")
    operation: str = Field("store", description="store, update")


class IterationStorageOutput(BaseModel):
    """Output from iteration storage operations."""

    iteration_id: str = Field(..., description="Iteration ID")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field("", description="Status message")


class IterationQueryInput(BaseModel):
    """Input for querying iterations."""

    iteration_id: str | None = Field(None, description="Specific iteration ID")
    loop_id: str | None = Field(None, description="Filter by loop ID")
    paper_id: str | None = Field(None, description="Filter by paper ID")
    status: LoopStatus | None = Field(None, description="Filter by status")
    search_text: str | None = Field(None, description="Text search")
    limit: int = Field(100, ge=1, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class IterationQueryOutput(BaseModel):
    """Output from querying iterations."""

    iterations: list[LoopIteration] = Field(
        default_factory=list, description="Matching iterations"
    )
    total: int = Field(0, description="Total matching count")


# ---------------------------------------------------------------------------
# Loop Storage I/O
# ---------------------------------------------------------------------------


class LoopRecord(BaseModel):
    """A persisted research loop record."""

    loop_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique loop ID",
    )
    goal: str = Field(..., description="Research goal")
    config_json: str = Field(..., description="Serialized LoopConfig")
    status: LoopStatus = Field(
        LoopStatus.CREATED, description="Loop status"
    )
    iteration_count: int = Field(0, description="Iterations completed")
    best_metric_value: float | None = Field(
        None, description="Best metric achieved"
    )
    primary_metric_name: str | None = Field(
        None, description="Tracked metric name"
    )
    stopping_condition: StoppingCondition | None = Field(
        None, description="Why the loop stopped"
    )
    stopping_reason: str = Field("", description="Human-readable stop reason")
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime | None = Field(
        None, description="Last update timestamp"
    )


class LoopStorageInput(BaseModel):
    """Input for loop storage operations."""

    loop: LoopRecord = Field(..., description="Loop to store")
    operation: str = Field("store", description="store, update")


class LoopStorageOutput(BaseModel):
    """Output from loop storage operations."""

    loop_id: str = Field(..., description="Loop ID")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field("", description="Status message")


class LoopQueryInput(BaseModel):
    """Input for querying loops."""

    loop_id: str | None = Field(None, description="Specific loop ID")
    status: LoopStatus | None = Field(None, description="Filter by status")
    search_text: str | None = Field(None, description="Text search")
    limit: int = Field(100, ge=1, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class LoopQueryOutput(BaseModel):
    """Output from querying loops."""

    loops: list[LoopRecord] = Field(
        default_factory=list, description="Matching loops"
    )
    total: int = Field(0, description="Total matching count")


# ---------------------------------------------------------------------------
# Stopping Condition Checker
# ---------------------------------------------------------------------------


class StoppingCheckInput(BaseModel):
    """Input for stopping condition evaluation."""

    state: LoopState = Field(..., description="Current loop state")
    config: LoopConfig = Field(..., description="Loop configuration")
    history: list[LoopIteration] = Field(
        default_factory=list, description="Past iterations"
    )


class StoppingCheckOutput(BaseModel):
    """Output from stopping condition evaluation."""

    should_stop: bool = Field(False, description="Whether to stop")
    condition: StoppingCondition | None = Field(
        None, description="Stop condition"
    )
    reason: str = Field("", description="Human-readable reason")
    metric_value: float | None = Field(
        None, description="Current best metric value"
    )
    target_value: float | None = Field(
        None, description="Target metric value"
    )


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------


class ReportInput(BaseModel):
    """Input for report generation."""

    loop: LoopRecord = Field(..., description="Loop record")
    iterations: list[LoopIteration] = Field(
        default_factory=list, description="All iterations"
    )
    config: LoopConfig = Field(..., description="Loop config")
    output_dir: str = Field(..., description="Output directory")


class ReportOutput(BaseModel):
    """Output from report generation."""

    report_path: str = Field(..., description="Markdown report path")
    json_path: str = Field(..., description="JSON report path")
    summary: str = Field("", description="Report summary")


# ---------------------------------------------------------------------------
# Top-Level Result
# ---------------------------------------------------------------------------


class LoopResult(BaseModel):
    """Top-level result from the autonomous research loop."""

    loop_id: str = Field(..., description="Loop ID")
    goal: str = Field(..., description="Research goal")
    status: LoopStatus = Field(..., description="Final status")
    iterations: list[LoopIteration] = Field(
        default_factory=list, description="All iterations"
    )
    iteration_count: int = Field(0, description="Iterations completed")
    best_metric_value: float | None = Field(
        None, description="Best metric achieved"
    )
    primary_metric_name: str | None = Field(
        None, description="Tracked metric name"
    )
    stopping_condition: StoppingCondition | None = Field(
        None, description="Why the loop stopped"
    )
    stopping_reason: str = Field("", description="Human-readable reason")
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    generated_files: list[str] = Field(
        default_factory=list, description="Generated file paths"
    )
    output_dir: str | None = Field(None, description="Output directory")
    processing_time_seconds: float = Field(0.0, description="Total time")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Result timestamp"
    )

    def is_complete(self) -> bool:
        """Return True if the loop reached a terminal state."""
        return self.status in (LoopStatus.STOPPED, LoopStatus.FAILED)

    def is_success(self) -> bool:
        """Return True if the loop achieved its target or ran successfully."""
        if self.stopping_condition == StoppingCondition.TARGET_ACHIEVED:
            return True
        if self.status == LoopStatus.STOPPED and self.iteration_count > 0:
            return True
        return False


# Rebuild forward references
LoopState.model_rebuild()
