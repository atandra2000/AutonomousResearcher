"""Implementation Report Tool for Phase 4.

Generates comprehensive implementation reports.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    ImplementationResult,
    MigrationPlan,
    ReviewResult,
    RollbackPlan,
)
from research_engineer.tools.base import Tool, ToolError


class ImplementationReportInput(BaseModel):
    """Input for implementation report generation."""

    implementation_result: ImplementationResult = Field(
        ...,
        description="Implementation result",
    )
    output_dir: str = Field(default="output", description="Output directory")
    include_diffs: bool = Field(
        default=True,
        description="Include diff files",
    )
    include_tests: bool = Field(
        default=True,
        description="Include test files",
    )
    format: str = Field(default="markdown", description="Output format")


class ImplementationReportOutput(BaseModel):
    """Output from implementation report generation."""

    report_md: str = Field(..., description="Markdown report")
    change_summary_md: str = Field(..., description="Change summary")
    patch_review_md: str = Field(..., description="Patch review")
    migration_plan_md: str = Field(..., description="Migration plan")
    test_plan_md: str = Field(..., description="Test plan")
    rollback_plan_md: str = Field(..., description="Rollback plan")
    generated_files: list[str] = Field(
        default_factory=list,
        description="Generated report files",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Generation duration",
    )


class ImplementationReportTool(Tool[ImplementationReportInput, ImplementationReportOutput]):
    """
    Tool for generating implementation reports.

    This tool:
    1. Generates implementation summary
    2. Creates change documentation
    3. Documents patch reviews
    4. Creates migration documentation
    5. Documents test plans
    6. Documents rollback plans
    """

    async def execute(self, input: ImplementationReportInput) -> ImplementationReportOutput:
        """Generate implementation report."""
        import time
        start_time = time.time()

        try:
            # Create output directory
            output_path = Path(input.output_dir) / "implementations" / input.implementation_result.implementation_id
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate reports
            report_md = self._generate_main_report(input.implementation_result)
            change_summary_md = self._generate_change_summary(input.implementation_result)
            patch_review_md = self._generate_patch_review(input.implementation_result)
            migration_plan_md = self._generate_migration_doc(input.implementation_result)
            test_plan_md = self._generate_test_plan(input.implementation_result)
            rollback_plan_md = self._generate_rollback_doc(input.implementation_result)

            # Write files
            generated_files = []

            files_to_write = {
                "implementation_report.md": report_md,
                "code_change_summary.md": change_summary_md,
                "patch_review.md": patch_review_md,
                "migration_plan.md": migration_plan_md,
                "test_plan.md": test_plan_md,
                "rollback_plan.md": rollback_plan_md,
            }

            for filename, content in files_to_write.items():
                filepath = output_path / filename
                filepath.write_text(content, encoding="utf-8")
                generated_files.append(str(filepath))

            elapsed = time.time() - start_time

            return ImplementationReportOutput(
                report_md=report_md,
                change_summary_md=change_summary_md,
                patch_review_md=patch_review_md,
                migration_plan_md=migration_plan_md,
                test_plan_md=test_plan_md,
                rollback_plan_md=rollback_plan_md,
                generated_files=generated_files,
                generation_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Report generation failed: {e}", input, e)

    def _generate_main_report(self, result: ImplementationResult) -> str:
        """Generate main implementation report."""
        lines = [
            f"# Implementation Report: {result.implementation_id}",
            "",
            f"**Request ID**: {result.request_id}",
            f"**Paper ID**: {result.paper_id or 'N/A'}",
            f"**Repository**: {result.repo_path}",
            f"**Status**: {result.status}",
            f"**Generated**: {result.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Implementation Time**: {result.implementation_time_seconds:.1f}s",
            "",
            "## Task Description",
            "",
            result.task_description,
            "",
            "## Summary",
            "",
            f"- **Patches Generated**: {len(result.patches_generated)}",
            f"- **Test Suites**: {len(result.tests_generated)}",
            f"- **Total Tests**: {sum(len(ts.tests) for ts in result.tests_generated)}",
            f"- **Review Status**: {result.review_result.status.value if result.review_result else 'pending'}",
            "",
        ]

        # Add review scores if available
        if result.review_result:
            lines.extend([
                "## Review Scores",
                "",
                f"- **Correctness**: {result.review_result.correctness_score}/10",
                f"- **Architecture**: {result.review_result.architecture_consistency_score}/10",
                f"- **Code Quality**: {result.review_result.code_quality_score}/10",
                f"- **Test Coverage**: {result.review_result.test_coverage_score}/10",
                f"- **Maintainability**: {result.review_result.maintainability_score}/10",
                "",
            ])

        # Add generated files
        if result.generated_files:
            lines.extend([
                "## Generated Files",
                "",
            ])
            for f in result.generated_files:
                lines.append(f"- `{f}`")
            lines.append("")

        # Add recommendations
        if result.review_result and result.review_result.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in result.review_result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    def _generate_change_summary(self, result: ImplementationResult) -> str:
        """Generate code change summary."""
        lines = [
            "# Code Change Summary",
            "",
            f"**Implementation**: {result.implementation_id}",
            "",
            "## Changes Overview",
            "",
        ]

        for patch in result.patches_generated:
            lines.extend([
                f"### {patch.file_path}",
                "",
                f"- **Type**: {patch.change_type.value}",
                f"- **Impact**: {patch.impact}",
                f"- **Risk**: {patch.risk_level.value}",
                f"- **Status**: {patch.status.value}",
                "",
                f"**Description**: {patch.explanation}",
                "",
                f"**Reason**: {patch.reason}",
                "",
            ])

            if patch.dependencies:
                lines.append(f"**Dependencies**: {', '.join(patch.dependencies)}")
                lines.append("")

        return "\n".join(lines)

    def _generate_patch_review(self, result: ImplementationResult) -> str:
        """Generate patch review documentation."""
        lines = [
            "# Patch Review",
            "",
            f"**Implementation**: {result.implementation_id}",
            "",
        ]

        if result.review_result:
            review = result.review_result
            lines.extend([
                "## Review Summary",
                "",
                f"**Status**: {review.status.value}",
                "",
                f"**Overall Assessment**: {review.overall_assessment}",
                "",
                "### Scores",
                "",
                f"- Correctness: {review.correctness_score}/10",
                f"- Architecture: {review.architecture_consistency_score}/10",
                f"- Code Quality: {review.code_quality_score}/10",
                f"- Test Coverage: {review.test_coverage_score}/10",
                f"- Maintainability: {review.maintainability_score}/10",
                "",
            ])

            if review.blocking_issues:
                lines.extend([
                    "### Blocking Issues",
                    "",
                ])
                for issue in review.blocking_issues:
                    lines.append(f"- ❌ {issue}")
                lines.append("")

            if review.comments:
                lines.extend([
                    "### Review Comments",
                    "",
                ])
                for comment in review.comments[:10]:  # Limit to first 10
                    lines.extend([
                        f"- **{comment.comment_type}** ({comment.severity}): {comment.comment}",
                        f"  - File: `{comment.file_path}`",
                        f"  - Suggestion: {comment.suggestion or 'N/A'}",
                        "",
                    ])

            if review.recommendations:
                lines.extend([
                    "### Recommendations",
                    "",
                ])
                for rec in review.recommendations:
                    lines.append(f"- {rec}")
                lines.append("")
        else:
            lines.append("*No review completed yet.*")
            lines.append("")

        return "\n".join(lines)

    def _generate_migration_doc(self, result: ImplementationResult) -> str:
        """Generate migration plan documentation."""
        lines = [
            "# Migration Plan",
            "",
            f"**Implementation**: {result.implementation_id}",
            "",
        ]

        if result.migration_plan:
            plan = result.migration_plan
            lines.extend([
                f"## {plan.title}",
                "",
                f"**Description**: {plan.description}",
                "",
                f"**Reason**: {plan.reason}",
                "",
                f"**Total Steps**: {plan.total_steps}",
                f"**Estimated Duration**: {plan.estimated_total_duration}",
                f"**Backward Compatibility**: {plan.backward_compatibility}",
                "",
                "## Migration Steps",
                "",
            ])

            for step in plan.steps:
                lines.extend([
                    f"### Step {step.step_number}: {step.title}",
                    "",
                    step.description,
                    "",
                    f"- **Type**: {step.action_type}",
                    f"- **Duration**: {step.estimated_duration}",
                    f"- **Risk**: {step.risk_level.value}",
                    "",
                    f"**Files Affected**: {', '.join(step.files_affected) if step.files_affected else 'None'}",
                    "",
                    f"**Validation**: {'; '.join(step.validation_steps)}",
                    "",
                    f"**Rollback**: {step.rollback_instructions}",
                    "",
                ])

            if plan.risks:
                lines.extend([
                    "## Risks",
                    "",
                ])
                for risk in plan.risks:
                    lines.append(f"- ⚠️ {risk}")
                lines.append("")
        else:
            lines.append("*No migration plan required.*")
            lines.append("")

        return "\n".join(lines)

    def _generate_test_plan(self, result: ImplementationResult) -> str:
        """Generate test plan documentation."""
        lines = [
            "# Test Plan",
            "",
            f"**Implementation**: {result.implementation_id}",
            "",
        ]

        if result.tests_generated:
            for suite in result.tests_generated:
                lines.extend([
                    f"## {suite.suite_name}",
                    "",
                    f"**Description**: {suite.description}",
                    "",
                    f"**Total Tests**: {suite.total_tests}",
                    f"**Estimated Coverage**: {suite.coverage_estimate}",
                    f"**Estimated Execution Time**: {suite.execution_time_estimate}",
                    "",
                    "### Tests",
                    "",
                ])

                for test in suite.tests:
                    lines.extend([
                        f"#### {test.test_name}",
                        "",
                        f"- **Type**: {test.test_type.value}",
                        f"- **Target**: `{test.target_file}`",
                        f"- **Priority**: {test.priority}",
                        "",
                        f"**Description**: {test.description}",
                        "",
                        f"**Expected Behavior**: {test.expected_behavior}",
                        "",
                    ])

                    if test.edge_cases:
                        lines.append(f"**Edge Cases**: {', '.join(test.edge_cases)}")
                        lines.append("")
        else:
            lines.append("*No tests generated.*")
            lines.append("")

        return "\n".join(lines)

    def _generate_rollback_doc(self, result: ImplementationResult) -> str:
        """Generate rollback plan documentation."""
        lines = [
            "# Rollback Plan",
            "",
            f"**Implementation**: {result.implementation_id}",
            "",
        ]

        if result.rollback_plan:
            plan = result.rollback_plan
            lines.extend([
                f"## {plan.title}",
                "",
                "## Trigger Conditions",
                "",
            ])
            for condition in plan.trigger_conditions:
                lines.append(f"- {condition}")
            lines.append("")

            lines.extend([
                "## Rollback Steps",
                "",
            ])

            for step in plan.steps:
                lines.extend([
                    f"### Step {step.step_number}: {step.title}",
                    "",
                    step.description,
                    "",
                    f"**Action**: {step.action}",
                    f"**Duration**: {step.estimated_duration}",
                    "",
                    f"**Files Affected**: {', '.join(step.files_affected) if step.files_affected else 'None'}",
                    "",
                    f"**Validation**: {step.validation_check}",
                    "",
                ])

            lines.extend([
                "## Failure Scenarios",
                "",
            ])
            for scenario in plan.failure_scenarios:
                lines.append(f"- ⚠️ {scenario}")
            lines.append("")

            lines.extend([
                "## Recovery Procedures",
                "",
            ])
            for proc in plan.recovery_procedures:
                lines.append(f"- {proc}")
            lines.append("")

            lines.extend([
                "## Summary",
                "",
                f"- **Total Steps**: {plan.total_steps}",
                f"- **Estimated Duration**: {plan.estimated_total_duration}",
                f"- **Data Loss Risk**: {plan.data_loss_risk}",
                "",
            ])
        else:
            lines.append("*No rollback plan generated.*")
            lines.append("")

        return "\n".join(lines)
