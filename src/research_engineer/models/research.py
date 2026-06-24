"""Phase 15 - Autonomous research workflow models.

Typed Pydantic models for the research orchestration framework:
research context (structured artifacts between stages), research stages,
hypotheses, experiment designs, and the final research result.

These models replace prompt chaining with structured artifacts that flow
through the pipeline, enabling full traceability from research goal to
final conclusions.
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


class ResearchStageType(StrEnum):
    """Stages in the autonomous research workflow."""

    LITERATURE_DISCOVERY = "literature_discovery"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_PLANNING = "experiment_planning"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULT_ANALYSIS = "result_analysis"
    REPORT_GENERATION = "report_generation"


class ResearchStageStatus(StrEnum):
    """Status of a single research stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchWorkflowStatus(StrEnum):
    """Overall status of the research workflow."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HypothesisStatus(StrEnum):
    """Status of a research hypothesis."""

    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Shared research context
# ---------------------------------------------------------------------------


class PaperFinding(BaseModel):
    """A single paper discovered during literature review."""

    paper_id: str = Field(..., description="Paper identifier (arXiv ID, DOI)")
    title: str = Field(default="", description="Paper title")
    authors: list[str] = Field(default_factory=list, description="Authors")
    year: int | None = Field(default=None, description="Publication year")
    abstract: str = Field(default="", description="Abstract")
    key_findings: list[str] = Field(
        default_factory=list, description="Key findings extracted"
    )
    relevance_score: float = Field(
        default=0.0, description="Relevance to the research goal (0-1)"
    )
    citation_count: int = Field(default=0, description="Citation count")


class KnowledgeSynthesis(BaseModel):
    """Synthesized knowledge from the literature review."""

    key_findings: list[str] = Field(
        default_factory=list, description="Key findings across all papers"
    )
    research_gaps: list[str] = Field(
        default_factory=list, description="Identified research gaps"
    )
    consensus_points: list[str] = Field(
        default_factory=list, description="Points of agreement in the literature"
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Contradictions between papers"
    )
    emerging_trends: list[str] = Field(
        default_factory=list, description="Emerging research trends"
    )
    summary: str = Field(default="", description="Narrative synthesis summary")


class Hypothesis(BaseModel):
    """A testable research hypothesis."""

    hypothesis_id: str = Field(..., description="Unique hypothesis ID")
    statement: str = Field(..., description="The hypothesis statement")
    rationale: str = Field(
        default="", description="Why this hypothesis is worth testing"
    )
    based_on_papers: list[str] = Field(
        default_factory=list, description="Paper IDs supporting this hypothesis"
    )
    based_on_gaps: list[str] = Field(
        default_factory=list, description="Research gaps this addresses"
    )
    expected_outcome: str = Field(
        default="", description="Expected outcome if hypothesis is true"
    )
    testability: str = Field(
        default="medium",
        description="How testable: high, medium, low",
    )
    novelty: str = Field(
        default="medium",
        description="How novel: high, medium, low",
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority (10=highest)"
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED, description="Current status"
    )


class ExperimentDesign(BaseModel):
    """Design for a single experiment to test a hypothesis."""

    experiment_id: str = Field(..., description="Unique experiment ID")
    hypothesis_id: str = Field(..., description="Hypothesis being tested")
    title: str = Field(default="", description="Experiment title")
    description: str = Field(default="", description="What the experiment does")
    command: str = Field(
        default="python train.py", description="Command to execute"
    )
    expected_duration_hours: float = Field(
        default=1.0, description="Expected GPU hours"
    )
    metrics: list[str] = Field(
        default_factory=list, description="Metrics to track"
    )
    resources: list[str] = Field(
        default_factory=list, description="Required resources"
    )


class ExperimentOutcome(BaseModel):
    """Outcome of a single experiment execution."""

    experiment_id: str = Field(..., description="Experiment ID")
    status: str = Field(default="unknown", description="Execution status")
    exit_code: int | None = Field(default=None, description="Process exit code")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Collected metrics"
    )
    duration_seconds: float = Field(default=0.0)
    stdout: str = Field(default="", description="Command stdout (truncated)")
    stderr: str = Field(default="", description="Command stderr (truncated)")
    error: str | None = Field(default=None, description="Error if failed")


class ResultAnalysis(BaseModel):
    """Analysis of experiment results."""

    experiment_id: str = Field(..., description="Experiment analyzed")
    hypothesis_id: str = Field(..., description="Hypothesis being evaluated")
    key_metrics: dict[str, float] = Field(
        default_factory=dict, description="Key metric values"
    )
    comparison: str = Field(
        default="", description="Comparison with baseline/expected"
    )
    significance: str = Field(
        default="inconclusive",
        description="Statistical significance assessment",
    )
    conclusion: str = Field(
        default="", description="Conclusion about the hypothesis"
    )
    hypothesis_status: HypothesisStatus = Field(
        default=HypothesisStatus.INCONCLUSIVE, description="Updated hypothesis status"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for next steps"
    )


class SharedResearchContext(BaseModel):
    """Structured context shared across all research workflow stages.

    This is the single communication channel between stages. Each stage
    reads the fields it needs and writes its outputs back, enabling
    inter-stage communication without prompt chaining.
    """

    research_goal: str = Field(..., description="The research objective")
    repo_path: str = Field(default=".", description="Repository for experiments")
    output_dir: str = Field(
        default="output/research", description="Output directory"
    )
    workflow_id: str = Field(
        default_factory=lambda: f"rw_{uuid4().hex[:12]}",
        description="Unique workflow ID",
    )

    # Stage 1: Literature discovery.
    papers: list[PaperFinding] = Field(
        default_factory=list, description="Discovered papers"
    )
    literature_review: str = Field(
        default="", description="Generated literature review markdown"
    )

    # Stage 2: Knowledge synthesis.
    synthesis: KnowledgeSynthesis | None = Field(
        default=None, description="Synthesized knowledge"
    )

    # Stage 3: Hypothesis generation.
    hypotheses: list[Hypothesis] = Field(
        default_factory=list, description="Generated hypotheses"
    )

    # Stage 4: Experiment planning.
    experiment_designs: list[ExperimentDesign] = Field(
        default_factory=list, description="Planned experiments"
    )

    # Stage 5: Experiment execution.
    experiment_outcomes: list[ExperimentOutcome] = Field(
        default_factory=list, description="Experiment results"
    )

    # Stage 6: Result analysis.
    analyses: list[ResultAnalysis] = Field(
        default_factory=list, description="Result analyses"
    )

    # Stage 7: Report generation.
    final_report: str = Field(
        default="", description="Final research report (markdown)"
    )
    report_path: str = Field(
        default="", description="Path to saved report file"
    )

    # Metadata.
    max_papers: int = Field(default=20, description="Max papers to discover")
    max_hypotheses: int = Field(default=5, description="Max hypotheses to generate")
    dry_run_experiments: bool = Field(
        default=True, description="If True, don't execute experiment commands"
    )
    experiment_timeout: int = Field(
        default=3600, description="Experiment timeout in seconds"
    )
    stream: bool = Field(
        default=True, description="Stream LLM tokens during stages"
    )
    skip_stages: list[ResearchStageType] = Field(
        default_factory=list, description="Stages to skip"
    )

    def add_paper(self, paper: PaperFinding) -> None:
        if paper.paper_id not in {p.paper_id for p in self.papers}:
            self.papers.append(paper)

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        if hyp.hypothesis_id not in {h.hypothesis_id for h in self.hypotheses}:
            self.hypotheses.append(hyp)


# ---------------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------------


class ResearchStageRecord(BaseModel):
    """Record of a single stage execution within the research workflow."""

    stage_id: str = Field(..., description="Unique stage ID")
    stage_type: ResearchStageType = Field(..., description="Which stage")
    status: ResearchStageStatus = Field(
        default=ResearchStageStatus.PENDING, description="Stage status"
    )
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0)
    summary: str = Field(default="", description="Human-readable summary")
    output: dict[str, Any] = Field(
        default_factory=dict, description="Structured output from the stage"
    )
    error: str | None = Field(default=None, description="Error if failed")


# ---------------------------------------------------------------------------
# Research result
# ---------------------------------------------------------------------------


class ResearchResult(BaseModel):
    """Final result of the autonomous research workflow."""

    workflow_id: str = Field(..., description="Unique workflow ID")
    research_goal: str = Field(..., description="Original research goal")
    status: ResearchWorkflowStatus = Field(
        default=ResearchWorkflowStatus.CREATED, description="Final status"
    )
    stages: list[ResearchStageRecord] = Field(
        default_factory=list, description="All stage records"
    )
    papers_found: int = Field(default=0, description="Number of papers discovered")
    hypotheses_generated: int = Field(
        default=0, description="Number of hypotheses generated"
    )
    experiments_run: int = Field(
        default=0, description="Number of experiments executed"
    )
    final_report: str = Field(default="", description="Final research report")
    report_path: str = Field(default="", description="Path to saved report")
    generated_files: list[str] = Field(
        default_factory=list, description="All files produced"
    )
    processing_time_seconds: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    error: str | None = Field(default=None, description="Top-level error")


__all__ = [
    "ResearchStageType",
    "ResearchStageStatus",
    "ResearchWorkflowStatus",
    "HypothesisStatus",
    "PaperFinding",
    "KnowledgeSynthesis",
    "Hypothesis",
    "ExperimentDesign",
    "ExperimentOutcome",
    "ResultAnalysis",
    "SharedResearchContext",
    "ResearchStageRecord",
    "ResearchResult",
]
