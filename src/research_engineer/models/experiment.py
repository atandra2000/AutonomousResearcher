"""Phase 7 - Experiment execution models.

Typed Pydantic models for experiment execution, monitoring, metric
collection, artifact collection, failure detection, and storage.
"""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExperimentType(StrEnum):
    """Type of experiment to execute."""

    TRAINING = "training"
    EVALUATION = "evaluation"
    VALIDATION = "validation"
    DRY_RUN = "dry_run"


class ExperimentStatus(StrEnum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    CRASHED = "crashed"


class MetricType(StrEnum):
    """Type of metric value."""

    SCALAR = "scalar"
    LOSS = "loss"
    ACCURACY = "accuracy"
    LATENCY = "latency"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


class ArtifactType(StrEnum):
    """Type of experiment artifact."""

    CHECKPOINT = "checkpoint"
    LOG = "log"
    METRIC_FILE = "metric_file"
    PLOT = "plot"
    CONFIG = "config"
    OUTPUT = "output"
    OTHER = "other"


class FailureSeverity(StrEnum):
    """Severity of a detected failure."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------


class StatusTransition(BaseModel):
    """A status transition in an experiment run."""

    from_status: ExperimentStatus = Field(..., description="Previous status")
    to_status: ExperimentStatus = Field(..., description="New status")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When transition occurred"
    )
    reason: str = Field("", description="Reason for transition")


class ExperimentRun(BaseModel):
    """A single experiment run record."""

    experiment_id: str = Field(..., description="Unique experiment ID")
    command: list[str] = Field(..., description="Command list")
    working_dir: str = Field(..., description="Working directory")
    experiment_type: ExperimentType = Field(..., description="Experiment type")
    status: ExperimentStatus = Field(
        ExperimentStatus.PENDING, description="Current status"
    )
    pid: int | None = Field(None, description="Process ID")
    start_time: datetime = Field(
        default_factory=datetime.now, description="Start time"
    )
    end_time: datetime | None = Field(None, description="End time")
    exit_code: int | None = Field(None, description="Process exit code")
    stdout: str = Field("", description="Captured stdout")
    stderr: str = Field("", description="Captured stderr")
    timeout_seconds: int = Field(3600, description="Timeout in seconds")
    status_history: list[StatusTransition] = Field(
        default_factory=list, description="Status transition history"
    )
    duration_seconds: float = Field(0.0, description="Run duration in seconds")
    error_message: str | None = Field(None, description="Error message if any")
    killed: bool = Field(False, description="Whether process was killed")

    def is_success(self) -> bool:
        """Return True if the experiment completed successfully."""
        return self.status == ExperimentStatus.COMPLETED

    def is_terminal(self) -> bool:
        """Return True if the experiment is in a terminal state."""
        return self.status in {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.TIMEOUT,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.CRASHED,
        }


class ExperimentRunnerInput(BaseModel):
    """Input for launching an experiment."""

    command: list[str] = Field(..., description="Command to execute")
    working_dir: str = Field(..., description="Working directory")
    experiment_id: str = Field(..., description="Experiment ID")
    experiment_type: ExperimentType = Field(
        ExperimentType.TRAINING, description="Experiment type"
    )
    timeout_seconds: int = Field(3600, ge=1, description="Max runtime seconds")
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Additional env vars"
    )
    dry_run: bool = Field(True, description="If True, do not execute")
    capture_output: bool = Field(True, description="Capture stdout/stderr")
    max_output_bytes: int = Field(
        10_000_000, ge=1024, description="Max output bytes"
    )


class ExperimentRunnerOutput(BaseModel):
    """Output from launching an experiment."""

    run: ExperimentRun = Field(..., description="The experiment run")
    launched: bool = Field(..., description="Whether process was launched")
    message: str = Field("", description="Status message")


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


class MonitoringInput(BaseModel):
    """Input for monitoring an experiment.

    The monitoring tool analyzes a completed (or in-progress) run's output
    and scans the working directory for checkpoints.
    """

    experiment_id: str = Field(..., description="Experiment ID to monitor")
    stdout: str = Field("", description="Captured stdout to analyze")
    stderr: str = Field("", description="Captured stderr to analyze")
    working_dir: str | None = Field(
        None, description="Working dir for checkpoint scan"
    )
    log_tail_lines: int = Field(
        100, ge=1, description="Lines to keep in tail buffer"
    )
    status: ExperimentStatus = Field(
        ExperimentStatus.COMPLETED, description="Current run status"
    )


class MonitoringOutput(BaseModel):
    """Output from monitoring an experiment."""

    experiment_id: str = Field(..., description="Experiment ID")
    final_status: ExperimentStatus = Field(..., description="Final status")
    stdout_tail: str = Field("", description="Last N lines of stdout")
    stderr_tail: str = Field("", description="Last N lines of stderr")
    total_stdout_lines: int = Field(0, description="Total stdout lines")
    total_stderr_lines: int = Field(0, description="Total stderr lines")
    elapsed_seconds: float = Field(0.0, description="Elapsed seconds")
    metrics_detected: list["MetricReading"] = Field(
        default_factory=list, description="Metrics found"
    )
    checkpoints_found: list[str] = Field(
        default_factory=list, description="Checkpoint paths"
    )
    anomalies: list[str] = Field(
        default_factory=list, description="Detected anomalies"
    )
    poll_count: int = Field(1, description="Number of polls performed")


# ---------------------------------------------------------------------------
# Metric Collection
# ---------------------------------------------------------------------------


class MetricPattern(BaseModel):
    """A pattern for parsing a metric from logs."""

    name: str = Field(..., description="Metric name")
    regex: str = Field(..., description="Regex with one capture group")
    metric_type: MetricType = Field(
        MetricType.SCALAR, description="Metric type"
    )
    aggregate: str = Field("last", description="last, max, min, mean")


class MetricReading(BaseModel):
    """A single metric reading."""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    step: int | None = Field(None, description="Step number")
    timestamp: datetime | None = Field(
        None, description="When the metric was recorded"
    )
    source: str = Field("stdout", description="Where the metric came from")
    metric_type: MetricType = Field(
        MetricType.SCALAR, description="Metric type"
    )


class MetricSeries(BaseModel):
    """A time series of a metric."""

    name: str = Field(..., description="Metric name")
    values: list[float] = Field(default_factory=list, description="Values")
    steps: list[int] = Field(default_factory=list, description="Step numbers")
    source: str = Field("stdout", description="Source of the series")
    best_value: float = Field(0.0, description="Best value")
    best_step: int | None = Field(None, description="Step of best value")
    final_value: float = Field(0.0, description="Final value")


class MetricCollectorInput(BaseModel):
    """Input for metric collection."""

    experiment_id: str = Field(..., description="Experiment ID")
    stdout: str = Field("", description="Stdout to parse")
    stderr: str = Field("", description="Stderr to parse")
    output_dir: str | None = Field(
        None, description="Directory with metric files"
    )
    metric_patterns: list[MetricPattern] | None = Field(
        None, description="Custom patterns (overrides defaults)"
    )
    expected_metrics: list[str] | None = Field(
        None, description="Expected metric names"
    )


class MetricCollectorOutput(BaseModel):
    """Output from metric collection."""

    metrics: list[MetricReading] = Field(
        default_factory=list, description="All parsed metrics"
    )
    metric_series: list[MetricSeries] = Field(
        default_factory=list, description="Time-series metrics"
    )
    summary_metrics: dict[str, float] = Field(
        default_factory=dict, description="Final/best values"
    )
    parsing_errors: list[str] = Field(
        default_factory=list, description="Parsing errors"
    )
    sources: list[str] = Field(
        default_factory=list, description="Where metrics came from"
    )


# ---------------------------------------------------------------------------
# Artifact Collection
# ---------------------------------------------------------------------------


class ArtifactPattern(BaseModel):
    """A pattern for discovering artifacts."""

    name: str = Field(..., description="Pattern name")
    glob_pattern: str = Field(..., description="Glob pattern")
    artifact_type: ArtifactType = Field(..., description="Artifact type")
    max_files: int = Field(10, ge=1, description="Max files to collect")


class ExperimentArtifact(BaseModel):
    """A single experiment artifact."""

    artifact_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Artifact ID"
    )
    name: str = Field(..., description="Artifact name")
    original_path: str = Field(..., description="Original file path")
    stored_path: str | None = Field(
        None, description="Path where artifact was copied"
    )
    artifact_type: ArtifactType = Field(..., description="Artifact type")
    size_bytes: int = Field(0, description="File size in bytes")
    checksum: str | None = Field(None, description="SHA-256 checksum")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Extra metadata"
    )


class ArtifactCollectorInput(BaseModel):
    """Input for artifact collection."""

    experiment_id: str = Field(..., description="Experiment ID")
    output_dir: str = Field(..., description="Where to store collected artifacts")
    working_dir: str = Field(..., description="Where the experiment ran")
    artifact_patterns: list[ArtifactPattern] | None = Field(
        None, description="Custom patterns (overrides defaults)"
    )
    max_artifact_size_mb: float = Field(
        500.0, ge=0.0, description="Max artifact size in MB"
    )
    copy_artifacts: bool = Field(
        True, description="Copy artifacts into output_dir"
    )


class ArtifactCollectorOutput(BaseModel):
    """Output from artifact collection."""

    artifacts: list[ExperimentArtifact] = Field(
        default_factory=list, description="Collected artifacts"
    )
    total_size_mb: float = Field(0.0, description="Total size in MB")
    output_dir: str = Field(..., description="Output directory")
    collection_errors: list[str] = Field(
        default_factory=list, description="Collection errors"
    )


# ---------------------------------------------------------------------------
# Failure Detection
# ---------------------------------------------------------------------------


class AnomalyIndicator(BaseModel):
    """An anomaly detected during or after an experiment."""

    indicator: str = Field(..., description="Anomaly identifier")
    description: str = Field(..., description="Description")
    confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence 0-1"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence"
    )


class FailureDetectorInput(BaseModel):
    """Input for failure detection."""

    run: ExperimentRun = Field(..., description="The experiment run")
    metrics: list[MetricReading] = Field(
        default_factory=list, description="Collected metrics"
    )
    artifacts: list[ExperimentArtifact] = Field(
        default_factory=list, description="Collected artifacts"
    )
    expected_metrics: list[str] | None = Field(
        None, description="Expected metric names"
    )


class FailureDetectorOutput(BaseModel):
    """Output from failure detection."""

    detected_failure: bool = Field(
        False, description="Whether a failure was detected"
    )
    failure_mode: str | None = Field(
        None, description="Failure mode (FailureMode value)"
    )
    severity: FailureSeverity = Field(
        FailureSeverity.NONE, description="Failure severity"
    )
    root_cause_hypothesis: str = Field(
        "", description="Hypothesized root cause"
    )
    error_snippets: list[str] = Field(
        default_factory=list, description="Relevant log snippets"
    )
    anomaly_indicators: list[AnomalyIndicator] = Field(
        default_factory=list, description="Anomaly indicators"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    lessons_learned: list[str] = Field(
        default_factory=list, description="Lessons for memory"
    )


# ---------------------------------------------------------------------------
# Experiment Storage
# ---------------------------------------------------------------------------


class ExperimentRecord(BaseModel):
    """A persisted experiment record."""

    experiment_id: str = Field(..., description="Experiment ID")
    paper_id: str | None = Field(None, description="Associated paper ID")
    plan_id: str | None = Field(None, description="Associated plan ID")
    patch_id: str | None = Field(None, description="Associated patch ID")
    implementation_id: str | None = Field(
        None, description="Associated implementation ID"
    )
    repo_path: str = Field(..., description="Repository path")
    command: list[str] = Field(..., description="Command list")
    experiment_type: ExperimentType = Field(..., description="Experiment type")
    status: ExperimentStatus = Field(..., description="Final status")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime | None = Field(None, description="End time")
    duration_seconds: float = Field(0.0, description="Duration seconds")
    exit_code: int | None = Field(None, description="Exit code")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Summary metrics"
    )
    metric_series: list[MetricSeries] = Field(
        default_factory=list, description="Metric series"
    )
    artifacts: list[ExperimentArtifact] = Field(
        default_factory=list, description="Artifacts"
    )
    failure_mode: str | None = Field(None, description="Failure mode")
    failure_severity: FailureSeverity = Field(
        FailureSeverity.NONE, description="Failure severity"
    )
    root_cause: str | None = Field(None, description="Root cause")
    lessons_learned: list[str] = Field(
        default_factory=list, description="Lessons learned"
    )
    output_dir: str | None = Field(None, description="Output directory")
    memory_id: str | None = Field(None, description="Linked memory ID")
    tags: list[str] = Field(default_factory=list, description="Tags")
    notes: str = Field("", description="Free-form notes")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: datetime | None = Field(
        None, description="Last update timestamp"
    )

    def is_success(self) -> bool:
        """Return True if the experiment succeeded."""
        return self.status == ExperimentStatus.COMPLETED


class ExperimentStorageInput(BaseModel):
    """Input for experiment storage operations."""

    experiment: ExperimentRecord = Field(..., description="Experiment to store")
    operation: str = Field("store", description="store, update")


class ExperimentStorageOutput(BaseModel):
    """Output from experiment storage operations."""

    experiment_id: str = Field(..., description="Experiment ID")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field("", description="Status message")


class ExperimentQueryInput(BaseModel):
    """Input for querying experiments."""

    experiment_id: str | None = Field(None, description="Specific experiment ID")
    paper_id: str | None = Field(None, description="Filter by paper ID")
    repo_path: str | None = Field(None, description="Filter by repo path")
    status: ExperimentStatus | None = Field(None, description="Filter by status")
    experiment_type: ExperimentType | None = Field(
        None, description="Filter by type"
    )
    search_text: str | None = Field(None, description="Text search")
    limit: int = Field(100, ge=1, description="Max results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class ExperimentQueryOutput(BaseModel):
    """Output from querying experiments."""

    experiments: list[ExperimentRecord] = Field(
        default_factory=list, description="Matching experiments"
    )
    total: int = Field(0, description="Total matching count")


# ---------------------------------------------------------------------------
# Top-Level Result
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """Top-level result from the full experiment workflow."""

    result_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Result ID"
    )
    experiment_id: str = Field(..., description="Experiment ID")
    paper_id: str | None = Field(None, description="Paper ID")
    plan_id: str | None = Field(None, description="Plan ID")
    patch_id: str | None = Field(None, description="Patch ID")
    implementation_id: str | None = Field(
        None, description="Implementation ID"
    )
    repo_path: str = Field(..., description="Repository path")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Result timestamp"
    )
    run: ExperimentRun | None = Field(None, description="The run record")
    monitoring: MonitoringOutput | None = Field(
        None, description="Monitoring output"
    )
    metrics: MetricCollectorOutput | None = Field(
        None, description="Collected metrics"
    )
    artifacts: ArtifactCollectorOutput | None = Field(
        None, description="Collected artifacts"
    )
    failure: FailureDetectorOutput | None = Field(
        None, description="Failure analysis"
    )
    record: ExperimentRecord | None = Field(
        None, description="Stored record"
    )
    memory_ids: list[str] = Field(
        default_factory=list, description="Memory IDs created"
    )
    output_dir: str | None = Field(None, description="Output directory")
    generated_files: list[str] = Field(
        default_factory=list, description="Generated file paths"
    )
    processing_time_seconds: float = Field(0.0, description="Total time")

    def is_success(self) -> bool:
        """Return True if the experiment succeeded."""
        if self.run:
            return self.run.is_success()
        if self.record:
            return self.record.is_success()
        return False


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Configuration for Experiment Agent."""

    default_timeout_seconds: int = Field(
        3600, ge=1, description="Default timeout"
    )
    default_poll_interval: float = Field(
        5.0, ge=0.1, description="Default poll interval"
    )
    default_log_tail_lines: int = Field(
        100, ge=1, description="Default log tail lines"
    )
    max_artifact_size_mb: float = Field(
        500.0, ge=0.0, description="Max artifact size MB"
    )
    store_results: bool = Field(
        True, description="Store results in memory"
    )
    update_graph: bool = Field(
        True, description="Auto-update knowledge graph"
    )
    output_dir: str = Field(
        "output/experiments", description="Default output directory"
    )
    dry_run_default: bool = Field(
        True, description="Dry run by default"
    )


# Resolve forward references
MonitoringOutput.model_rebuild()
