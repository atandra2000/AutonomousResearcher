"""Phase 4 - Coding Agent & Patch Generation models.

All Pydantic models for the Coding Agent, including:
- Code changes and patches
- Test specifications
- Code review results
- Migration planning
- Rollback strategies
- Implementation tracking
"""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class ChangeType(StrEnum):
    """Type of code change."""

    NEW_FILE = "new_file"
    MODIFICATION = "modification"
    DELETION = "deletion"
    RENAME = "rename"
    CONFIG_UPDATE = "config_update"
    TEST_ADDITION = "test_addition"
    DOC_UPDATE = "documentation_update"


class PatchStatus(StrEnum):
    """Status of a generated patch."""

    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


class ReviewStatus(StrEnum):
    """Status of code review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


class RiskLevel(StrEnum):
    """Risk level for implementation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplexityLevel(StrEnum):
    """Complexity level of code change."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TestType(StrEnum):
    """Type of test."""

    UNIT = "unit"
    INTEGRATION = "integration"
    REGRESSION = "regression"
    NUMERICAL = "numerical_equivalence"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


# --- Code Change Models ---


class CodeChange(BaseModel):
    """Represents a single code change."""

    file_path: str = Field(..., description="Path to the file being changed")
    change_type: ChangeType = Field(..., description="Type of change")
    description: str = Field(..., description="Human-readable description")
    reason: str = Field(..., description="Why this change is needed")
    impact: str = Field(default="medium", description="Impact level: low/medium/high")
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of other changes this depends on",
    )
    side_effects: list[str] = Field(
        default_factory=list,
        description="Potential side effects",
    )
    estimated_lines_added: int = Field(default=0, description="Estimated lines added")
    estimated_lines_removed: int = Field(default=0, description="Estimated lines removed")
    complexity: ComplexityLevel = Field(
        default=ComplexityLevel.MODERATE,
        description="Complexity level",
    )


class GeneratedPatch(BaseModel):
    """A generated patch/diff for code changes."""

    patch_id: str = Field(..., description="Unique patch identifier")
    file_path: str = Field(..., description="Target file path")
    change_type: ChangeType = Field(..., description="Type of change")
    diff: str = Field(..., description="Unified diff content")
    explanation: str = Field(..., description="Explanation of changes")
    reason: str = Field(..., description="Why this change is needed")
    impact: str = Field(default="medium", description="Impact assessment")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Patch dependencies",
    )
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk level")
    status: PatchStatus = Field(
        default=PatchStatus.GENERATED,
        description="Current patch status",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Patch creation timestamp",
    )
    review_comments: list[str] = Field(
        default_factory=list,
        description="Review comments",
    )
    approval_required: bool = Field(
        default=True,
        description="Whether human approval is required",
    )
    estimated_lines_added: int = Field(default=0, description="Estimated lines added")
    estimated_lines_removed: int = Field(default=0, description="Estimated lines removed")


# --- Test Models ---


class TestSpecification(BaseModel):
    """Specification for a test to be generated."""

    test_id: str = Field(..., description="Unique test identifier")
    test_name: str = Field(..., description="Test function/method name")
    test_type: TestType = Field(..., description="Type of test")
    target_file: str = Field(..., description="File containing code to test")
    target_function: str | None = Field(
        default=None,
        description="Specific function/class being tested",
    )
    description: str = Field(..., description="What this test validates")
    test_code: str = Field(..., description="Generated test code")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Test dependencies/imports",
    )
    expected_behavior: str = Field(..., description="Expected behavior being tested")
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Edge cases covered",
    )
    priority: str = Field(default="medium", description="Test priority")


class TestSuite(BaseModel):
    """Collection of generated tests."""

    suite_id: str = Field(..., description="Unique suite identifier")
    suite_name: str = Field(..., description="Test suite name")
    description: str = Field(..., description="Suite description")
    tests: list[TestSpecification] = Field(
        default_factory=list,
        description="Tests in this suite",
    )
    total_tests: int = Field(default=0, description="Total number of tests")
    coverage_estimate: str = Field(
        default="unknown",
        description="Estimated code coverage",
    )
    execution_time_estimate: str = Field(
        default="unknown",
        description="Estimated execution time",
    )


# --- Code Review Models ---


class ReviewComment(BaseModel):
    """A single code review comment."""

    comment_id: str = Field(..., description="Unique comment identifier")
    file_path: str = Field(..., description="File being reviewed")
    line_number: int | None = Field(
        default=None,
        description="Specific line number (if applicable)",
    )
    comment_type: str = Field(
        default="general",
        description="Type: bug/performance/style/security/maintainability",
    )
    severity: str = Field(
        default="medium",
        description="Severity: critical/high/medium/low",
    )
    comment: str = Field(..., description="Review comment text")
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix or improvement",
    )


class ReviewResult(BaseModel):
    """Result of code review."""

    review_id: str = Field(..., description="Unique review identifier")
    patch_id: str = Field(..., description="Patch being reviewed")
    status: ReviewStatus = Field(..., description="Review status")
    overall_assessment: str = Field(..., description="Overall assessment summary")
    correctness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Correctness score (0-10)",
    )
    architecture_consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Architecture consistency score (0-10)",
    )
    code_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Code quality score (0-10)",
    )
    test_coverage_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Test coverage score (0-10)",
    )
    maintainability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Maintainability score (0-10)",
    )
    comments: list[ReviewComment] = Field(
        default_factory=list,
        description="Detailed review comments",
    )
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="Issues that must be fixed before approval",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Non-blocking recommendations",
    )
    reviewed_at: datetime = Field(
        default_factory=datetime.now,
        description="Review timestamp",
    )


# --- Migration Models ---


class MigrationStep(BaseModel):
    """A single step in a migration plan."""

    step_number: int = Field(..., description="Step order")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed description")
    action_type: str = Field(
        default="code_change",
        description="Type: code_change/config_update/data_migration/checkpoint_migration",
    )
    files_affected: list[str] = Field(
        default_factory=list,
        description="Files affected by this step",
    )
    prerequisites: list[int] = Field(
        default_factory=list,
        description="Prerequisite step numbers",
    )
    estimated_duration: str = Field(
        default="unknown",
        description="Estimated time to complete",
    )
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk level")
    rollback_instructions: str = Field(
        default="Revert changes",
        description="How to rollback this step",
    )
    validation_steps: list[str] = Field(
        default_factory=list,
        description="Steps to validate after completion",
    )


class MigrationPlan(BaseModel):
    """Complete migration plan for major changes."""

    plan_id: str = Field(..., description="Unique plan identifier")
    title: str = Field(..., description="Migration plan title")
    description: str = Field(..., description="Overall description")
    reason: str = Field(..., description="Why migration is needed")
    steps: list[MigrationStep] = Field(
        default_factory=list,
        description="Migration steps in order",
    )
    total_steps: int = Field(default=0, description="Total number of steps")
    estimated_total_duration: str = Field(
        default="unknown",
        description="Estimated total duration",
    )
    backward_compatibility: str = Field(
        default="unknown",
        description="Backward compatibility notes",
    )
    checkpoint_migration_notes: str = Field(
        default="N/A",
        description="Checkpoint migration guidance",
    )
    config_migration_notes: str = Field(
        default="N/A",
        description="Configuration migration guidance",
    )
    versioning_recommendations: str = Field(
        default="N/A",
        description="Versioning strategy recommendations",
    )
    risks: list[str] = Field(default_factory=list, description="Migration risks")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Plan creation timestamp",
    )


# --- Rollback Models ---


class RollbackStep(BaseModel):
    """A single step in a rollback plan."""

    step_number: int = Field(..., description="Step order")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed description")
    action: str = Field(..., description="Action to perform")
    files_affected: list[str] = Field(
        default_factory=list,
        description="Files affected",
    )
    estimated_duration: str = Field(
        default="unknown",
        description="Estimated time to complete",
    )
    validation_check: str = Field(
        default="Verify change reverted",
        description="How to validate rollback",
    )


class RollbackPlan(BaseModel):
    """Rollback strategy for a change."""

    plan_id: str = Field(..., description="Unique plan identifier")
    patch_id: str = Field(..., description="Associated patch ID")
    title: str = Field(..., description="Rollback plan title")
    trigger_conditions: list[str] = Field(
        default_factory=list,
        description="When to trigger rollback",
    )
    steps: list[RollbackStep] = Field(
        default_factory=list,
        description="Rollback steps in order",
    )
    total_steps: int = Field(default=0, description="Total number of steps")
    estimated_total_duration: str = Field(
        default="unknown",
        description="Estimated total rollback time",
    )
    failure_scenarios: list[str] = Field(
        default_factory=list,
        description="Potential failure scenarios",
    )
    recovery_procedures: list[str] = Field(
        default_factory=list,
        description="Recovery procedures for failures",
    )
    data_loss_risk: str = Field(
        default="unknown",
        description="Data loss risk assessment",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Plan creation timestamp",
    )


# --- Implementation Tracking Models ---


class ImplementationRequest(BaseModel):
    """Request for implementation."""

    request_id: str = Field(..., description="Unique request identifier")
    paper_id: str | None = Field(
        default=None,
        description="Associated paper ID (if applicable)",
    )
    repo_path: str = Field(..., description="Target repository path")
    task_description: str = Field(..., description="What to implement")
    implementation_plan_path: str | None = Field(
        default=None,
        description="Path to implementation plan file",
    )
    priority: str = Field(default="medium", description="Implementation priority")
    constraints: list[str] = Field(
        default_factory=list,
        description="Implementation constraints",
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Specific requirements",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Request creation timestamp",
    )


class ImplementationResult(BaseModel):
    """Result of implementation process."""

    implementation_id: str = Field(..., description="Unique implementation identifier")
    request_id: str = Field(..., description="Associated request ID")
    paper_id: str | None = Field(default=None, description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    task_description: str = Field(..., description="What was implemented")
    status: str = Field(default="completed", description="Implementation status")
    patches_generated: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Generated patches",
    )
    tests_generated: list[TestSuite] = Field(
        default_factory=list,
        description="Generated test suites",
    )
    review_result: ReviewResult | None = Field(
        default=None,
        description="Self-review result",
    )
    migration_plan: MigrationPlan | None = Field(
        default=None,
        description="Migration plan (if applicable)",
    )
    rollback_plan: RollbackPlan | None = Field(
        default=None,
        description="Rollback plan",
    )
    implementation_report_md: str = Field(
        default="",
        description="Implementation report in markdown",
    )
    code_change_summary_md: str = Field(
        default="",
        description="Code change summary in markdown",
    )
    patch_review_md: str = Field(
        default="",
        description="Patch review in markdown",
    )
    migration_plan_md: str = Field(
        default="",
        description="Migration plan in markdown",
    )
    test_plan_md: str = Field(
        default="",
        description="Test plan in markdown",
    )
    rollback_plan_md: str = Field(
        default="",
        description="Rollback plan in markdown",
    )
    generated_files: list[str] = Field(
        default_factory=list,
        description="List of generated files",
    )
    generated_diffs_dir: str | None = Field(
        default=None,
        description="Path to generated diffs directory",
    )
    generated_tests_dir: str | None = Field(
        default=None,
        description="Path to generated tests directory",
    )
    implementation_time_seconds: float = Field(
        default=0.0,
        description="Implementation duration",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Result creation timestamp",
    )

    def to_summary_dict(self) -> dict:
        """Convert to summary dictionary."""
        return {
            "implementation_id": self.implementation_id,
            "request_id": self.request_id,
            "paper_id": self.paper_id,
            "repo_path": self.repo_path,
            "task_description": self.task_description,
            "status": self.status,
            "patches_count": len(self.patches_generated),
            "tests_count": sum(len(ts.tests) for ts in self.tests_generated),
            "review_status": self.review_result.status.value if self.review_result else "pending",
            "implementation_time_seconds": self.implementation_time_seconds,
            "generated_files_count": len(self.generated_files),
            "created_at": self.created_at.isoformat(),
        }
