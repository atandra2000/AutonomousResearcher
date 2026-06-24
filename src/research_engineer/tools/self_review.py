"""Self Review Tool for Phase 4.

Performs automated code review on generated patches.
"""

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    GeneratedPatch,
    ReviewComment,
    ReviewResult,
    ReviewStatus,
)
from research_engineer.tools.base import Tool, ToolError


class SelfReviewInput(BaseModel):
    """Input for self review."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Patches to review",
    )
    repo_path: str = Field(..., description="Repository path")
    review_criteria: list[str] = Field(
        default_factory=list,
        description="Specific review criteria",
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict review mode",
    )
    check_architecture: bool = Field(
        default=True,
        description="Check architecture consistency",
    )
    check_style: bool = Field(
        default=True,
        description="Check coding style",
    )
    check_performance: bool = Field(
        default=True,
        description="Check performance implications",
    )
    check_security: bool = Field(
        default=False,
        description="Check security implications",
    )


class SelfReviewOutput(BaseModel):
    """Output from self review."""

    review_result: ReviewResult = Field(..., description="Review result")
    total_comments: int = Field(default=0, description="Total review comments")
    blocking_issues: int = Field(default=0, description="Number of blocking issues")
    recommendations: int = Field(default=0, description="Number of recommendations")
    review_time_seconds: float = Field(default=0.0, description="Review duration")


class SelfReviewTool(Tool[SelfReviewInput, SelfReviewOutput]):
    """
    Tool for automated code review.

    This tool:
    1. Reviews code correctness
    2. Checks architecture consistency
    3. Validates coding style
    4. Identifies performance concerns
    5. Checks for security issues
    6. Generates review comments
    7. Provides overall assessment
    """

    async def execute(self, input: SelfReviewInput) -> SelfReviewOutput:
        """Perform automated code review."""
        import time
        start_time = time.time()

        try:
            # Initialize review
            review_id = f"review_{len(input.patches)}_patches"

            # Review each patch
            all_comments = []
            blocking_issues = []
            recommendations = []

            scores = {
                "correctness": [],
                "architecture": [],
                "quality": [],
                "coverage": [],
                "maintainability": [],
            }

            for patch in input.patches:
                patch_review = await self._review_patch(patch, input)
                all_comments.extend(patch_review.comments)
                blocking_issues.extend(patch_review.blocking_issues)
                recommendations.extend(patch_review.recommendations)

                # Collect scores
                scores["correctness"].append(patch_review.correctness_score)
                scores["architecture"].append(patch_review.architecture_consistency_score)
                scores["quality"].append(patch_review.code_quality_score)
                scores["coverage"].append(patch_review.test_coverage_score)
                scores["maintainability"].append(patch_review.maintainability_score)

            # Calculate average scores
            avg_scores = {
                key: sum(values) / len(values) if values else 5.0
                for key, values in scores.items()
            }

            # Determine overall status
            if blocking_issues:
                status = ReviewStatus.CHANGES_REQUESTED
            elif avg_scores["correctness"] < 6.0:
                status = ReviewStatus.CHANGES_REQUESTED
            elif avg_scores["quality"] < 6.0:
                status = ReviewStatus.CHANGES_REQUESTED
            else:
                status = ReviewStatus.APPROVED

            # Build overall assessment
            overall = self._build_overall_assessment(avg_scores, blocking_issues, recommendations)

            review_result = ReviewResult(
                review_id=review_id,
                patch_id=",".join(p.patch_id for p in input.patches),
                status=status,
                overall_assessment=overall,
                correctness_score=round(avg_scores["correctness"], 1),
                architecture_consistency_score=round(avg_scores["architecture"], 1),
                code_quality_score=round(avg_scores["quality"], 1),
                test_coverage_score=round(avg_scores["coverage"], 1),
                maintainability_score=round(avg_scores["maintainability"], 1),
                comments=all_comments,
                blocking_issues=blocking_issues,
                recommendations=recommendations,
            )

            elapsed = time.time() - start_time

            return SelfReviewOutput(
                review_result=review_result,
                total_comments=len(all_comments),
                blocking_issues=len(blocking_issues),
                recommendations=len(recommendations),
                review_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Self review failed: {e}", input, e)

    async def _review_patch(self, patch: GeneratedPatch, input: SelfReviewInput) -> ReviewResult:
        """Review a single patch."""
        comments = []
        blocking = []
        recommendations = []

        # Check 1: Correctness
        correctness_score, correctness_comments = self._check_correctness(patch)
        comments.extend(correctness_comments)
        if correctness_score < 5.0:
            blocking.append(f"Correctness issues in {patch.file_path}")

        # Check 2: Architecture consistency
        arch_score = 7.0
        if input.check_architecture:
            arch_score, arch_comments = self._check_architecture(patch, input)
            comments.extend(arch_comments)
            if arch_score < 5.0:
                blocking.append(f"Architecture inconsistency in {patch.file_path}")

        # Check 3: Code quality
        quality_score = 7.0
        if input.check_style:
            quality_score, quality_comments = self._check_code_quality(patch)
            comments.extend(quality_comments)

        # Check 4: Performance
        perf_score = 7.0
        if input.check_performance:
            perf_score, perf_comments = self._check_performance(patch)
            comments.extend(perf_comments)
            if perf_score < 5.0:
                recommendations.append(f"Review performance implications in {patch.file_path}")

        # Check 5: Security
        security_score = 7.0
        if input.check_security:
            security_score, security_comments = self._check_security(patch)
            comments.extend(security_comments)
            if security_score < 5.0:
                blocking.append(f"Security concerns in {patch.file_path}")

        # Calculate test coverage score (placeholder)
        coverage_score = 5.0

        # Calculate maintainability
        maintainability_score = (quality_score + arch_score) / 2

        return ReviewResult(
            review_id=f"review_{patch.patch_id}",
            patch_id=patch.patch_id,
            status=ReviewStatus.IN_PROGRESS,
            overall_assessment=f"Review of {patch.file_path}",
            correctness_score=correctness_score,
            architecture_consistency_score=arch_score,
            code_quality_score=quality_score,
            test_coverage_score=coverage_score,
            maintainability_score=maintainability_score,
            comments=comments,
            blocking_issues=blocking,
            recommendations=recommendations,
        )

    def _check_correctness(self, patch: GeneratedPatch) -> tuple[float, list[ReviewComment]]:
        """Check code correctness."""
        comments = []
        score = 8.0

        # Check for common issues in diff
        diff_lower = patch.diff.lower()

        if "todo" in diff_lower or "fixme" in diff_lower:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_todo",
                file_path=patch.file_path,
                comment_type="maintainability",
                severity="low",
                comment="Contains TODO/FIXME comments",
                suggestion="Complete or remove TODO comments before merging",
            ))
            score -= 1.0

        if "hack" in diff_lower or "workaround" in diff_lower:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_hack",
                file_path=patch.file_path,
                comment_type="quality",
                severity="medium",
                comment="Contains hack/workaround",
                suggestion="Consider a more robust solution",
            ))
            score -= 2.0

        return score, comments

    def _check_architecture(self, patch: GeneratedPatch, input: SelfReviewInput) -> tuple[float, list[ReviewComment]]:
        """Check architecture consistency."""
        comments = []
        score = 7.5

        # Check file location
        file_path = patch.file_path.lower()

        if "model" in file_path and "models/" not in file_path:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_arch1",
                file_path=patch.file_path,
                comment_type="architecture",
                severity="medium",
                comment="Model file not in models/ directory",
                suggestion="Move to models/ directory for consistency",
            ))
            score -= 1.5

        if "test" in file_path and "tests/" not in file_path:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_arch2",
                file_path=patch.file_path,
                comment_type="architecture",
                severity="low",
                comment="Test file not in tests/ directory",
                suggestion="Move to tests/ directory",
            ))
            score -= 1.0

        return score, comments

    def _check_code_quality(self, patch: GeneratedPatch) -> tuple[float, list[ReviewComment]]:
        """Check code quality and style."""
        comments = []
        score = 8.0

        # Check for docstrings
        if '"""' not in patch.diff and "'''" not in patch.diff:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_doc",
                file_path=patch.file_path,
                comment_type="style",
                severity="low",
                comment="Missing docstrings",
                suggestion="Add docstrings to functions and classes",
            ))
            score -= 1.0

        # Check for type hints (simplified check)
        if ": " not in patch.diff and "->" not in patch.diff:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_type",
                file_path=patch.file_path,
                comment_type="style",
                severity="low",
                comment="Missing type hints",
                suggestion="Add type hints for better maintainability",
            ))
            score -= 0.5

        return score, comments

    def _check_performance(self, patch: GeneratedPatch) -> tuple[float, list[ReviewComment]]:
        """Check performance implications."""
        comments = []
        score = 7.5

        diff_lower = patch.diff.lower()

        # Check for potential performance issues
        if "for " in diff_lower and "range(" in diff_lower:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_perf1",
                file_path=patch.file_path,
                comment_type="performance",
                severity="low",
                comment="Contains loop with range()",
                suggestion="Consider using list comprehension or vectorization",
            ))
            score -= 0.5

        return score, comments

    def _check_security(self, patch: GeneratedPatch) -> tuple[float, list[ReviewComment]]:
        """Check security implications."""
        comments = []
        score = 8.0

        diff_lower = patch.diff.lower()

        # Check for hardcoded credentials
        if "password" in diff_lower or "secret" in diff_lower or "api_key" in diff_lower:
            comments.append(ReviewComment(
                comment_id=f"comment_{patch.patch_id}_sec1",
                file_path=patch.file_path,
                comment_type="security",
                severity="high",
                comment="Potential hardcoded credentials",
                suggestion="Use environment variables or secrets manager",
            ))
            score -= 3.0

        return score, comments

    def _build_overall_assessment(
        self,
        scores: dict,
        blocking: list[str],
        recommendations: list[str],
    ) -> str:
        """Build overall assessment summary."""
        avg_score = sum(scores.values()) / len(scores)

        assessment_parts = [
            f"Overall score: {avg_score:.1f}/10",
            f"Correctness: {scores['correctness']:.1f}/10",
            f"Architecture: {scores['architecture']:.1f}/10",
            f"Quality: {scores['quality']:.1f}/10",
        ]

        if blocking:
            assessment_parts.append(f"Blocking issues: {len(blocking)}")

        if recommendations:
            assessment_parts.append(f"Recommendations: {len(recommendations)}")

        return ". ".join(assessment_parts)
