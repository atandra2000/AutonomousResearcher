"""Patch Generation Tool for Phase 4.

Generates structured diffs and patch files from code changes.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.models.coding import ChangeType, GeneratedPatch, PatchStatus
from research_engineer.tools.base import Tool, ToolError
from research_engineer.tools.code_generation import CodeChange


class PatchGenerationInput(BaseModel):
    """Input for patch generation."""

    changes: list[CodeChange] = Field(
        default_factory=list,
        description="List of code changes",
    )
    repo_path: str = Field(..., description="Repository path")
    patch_id_prefix: str = Field(default="patch", description="Patch ID prefix")
    include_explanation: bool = Field(
        default=True,
        description="Include detailed explanations",
    )
    generate_metadata: bool = Field(
        default=True,
        description="Generate patch metadata",
    )


class PatchGenerationOutput(BaseModel):
    """Output from patch generation."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Generated patches",
    )
    total_patches: int = Field(default=0, description="Total number of patches")
    patches_by_type: dict = Field(
        default_factory=dict,
        description="Patches grouped by change type",
    )
    total_lines_added: int = Field(default=0, description="Total lines added")
    total_lines_removed: int = Field(default=0, description="Total lines removed")
    files_affected: list[str] = Field(
        default_factory=list,
        description="All affected files",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Generation duration",
    )


class PatchGenerationTool(Tool[PatchGenerationInput, PatchGenerationOutput]):
    """
    Tool for generating structured patches from code changes.

    This tool:
    1. Converts code changes to unified diff format
    2. Adds metadata and explanations
    3. Groups patches by type
    4. Validates patch structure
    5. Prepares patches for review
    """

    async def execute(self, input: PatchGenerationInput) -> PatchGenerationOutput:
        """Generate patches from code changes."""
        import time
        start_time = time.time()

        try:
            # Validate repository
            repo_path = Path(input.repo_path)
            if not repo_path.exists():
                raise ToolError(f"Repository does not exist: {input.repo_path}", input)

            # Generate patches
            patches = []
            for i, change in enumerate(input.changes):
                patch = await self._generate_patch(change, i, input)
                patches.append(patch)

            # Group by type
            patches_by_type = {}
            for patch in patches:
                type_key = patch.change_type.value
                if type_key not in patches_by_type:
                    patches_by_type[type_key] = []
                patches_by_type[type_key].append(patch)

            # Calculate statistics
            total_added = sum(p.estimated_lines_added for p in patches)
            total_removed = sum(p.estimated_lines_removed for p in patches)
            files_affected = list(set(p.file_path for p in patches))

            elapsed = time.time() - start_time

            return PatchGenerationOutput(
                patches=patches,
                total_patches=len(patches),
                patches_by_type=patches_by_type,
                total_lines_added=total_added,
                total_lines_removed=total_removed,
                files_affected=files_affected,
                generation_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Patch generation failed: {e}", input, e)

    async def _generate_patch(
        self,
        change: CodeChange,
        index: int,
        input: PatchGenerationInput,
    ) -> GeneratedPatch:
        """Generate a single patch from a code change."""
        patch_id = f"{input.patch_id_prefix}_{index:04d}"

        # Generate diff content
        diff_content = await self._create_unified_diff(change, input)

        # Build explanation
        explanation = self._build_explanation(change, input.include_explanation)

        # Determine risk level
        risk_level = self._assess_risk(change)

        patch = GeneratedPatch(
            patch_id=patch_id,
            file_path=change.file_path,
            change_type=change.change_type,
            diff=diff_content,
            explanation=explanation,
            reason=change.reason,
            impact=change.impact,
            dependencies=change.dependencies,
            risk_level=risk_level,
            status=PatchStatus.GENERATED,
            approval_required=True,
            estimated_lines_added=change.estimated_lines_added,
            estimated_lines_removed=change.estimated_lines_removed,
        )

        return patch

    async def _create_unified_diff(self, change: CodeChange, input: PatchGenerationInput) -> str:
        """Create unified diff format from code change."""
        repo_path = Path(input.repo_path)

        if change.change_type == ChangeType.NEW_FILE:
            return await self._create_new_file_diff(change, repo_path)
        elif change.change_type == ChangeType.MODIFICATION:
            return await self._create_modification_diff(change, repo_path)
        elif change.change_type == ChangeType.DELETION:
            return await self._create_deletion_diff(change, repo_path)
        else:
            return self._create_generic_diff(change)

    async def _create_new_file_diff(self, change: CodeChange, repo_path: Path) -> str:
        """Create diff for new file."""
        file_path = repo_path / change.file_path

        # Check if file already exists
        if file_path.exists():
            # Read existing content
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            line_count = len(lines)
        else:
            lines = []
            line_count = change.estimated_lines_added

        diff_lines = [
            "--- /dev/null",
            f"+++ b/{change.file_path}",
            f"@@ -0,0 +1,{line_count} @@",
        ]

        # Add file header comment
        diff_lines.extend([
            f"+# File: {change.file_path}",
            f"+# Change: {change.description}",
            f"+# Reason: {change.reason}",
            "+",
        ])

        # Placeholder for actual content
        if change.estimated_lines_added > 0:
            diff_lines.append(f"+# [Generated code: {change.estimated_lines_added} lines]")

        return "\n".join(diff_lines)

    async def _create_modification_diff(self, change: CodeChange, repo_path: Path) -> str:
        """Create diff for file modification."""
        file_path = repo_path / change.file_path

        if not file_path.exists():
            # File doesn't exist, treat as new file
            return await self._create_new_file_diff(change, repo_path)

        # Read existing content
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        original_line_count = len(lines)

        # Calculate new line count
        new_line_count = original_line_count + change.estimated_lines_added - change.estimated_lines_removed

        diff_lines = [
            f"--- a/{change.file_path}",
            f"+++ b/{change.file_path}",
            f"@@ -1,{original_line_count} +1,{new_line_count} @@",
        ]

        # Add change metadata
        diff_lines.extend([
            f" # Change: {change.description}",
            f" # Reason: {change.reason}",
            f" # Impact: {change.impact}",
        ])

        if change.estimated_lines_added > 0:
            diff_lines.append(f"+# [Added: {change.estimated_lines_added} lines]")
        if change.estimated_lines_removed > 0:
            diff_lines.append(f"-# [Removed: {change.estimated_lines_removed} lines]")

        return "\n".join(diff_lines)

    async def _create_deletion_diff(self, change: CodeChange, repo_path: Path) -> str:
        """Create diff for file deletion."""
        file_path = repo_path / change.file_path

        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            line_count = len(lines)
        else:
            line_count = change.estimated_lines_removed

        diff_lines = [
            f"--- a/{change.file_path}",
            "+++ /dev/null",
            f"@@ -1,{line_count} +0,0 @@",
            f"-# Deleted: {change.file_path}",
            f"-# Reason: {change.reason}",
        ]

        return "\n".join(diff_lines)

    def _create_generic_diff(self, change: CodeChange) -> str:
        """Create generic diff for other change types."""
        return f"""--- a/{change.file_path}
+++ b/{change.file_path}
@@ -0,0 +1 @@
+# {change.change_type.value}: {change.description}
# Reason: {change.reason}
"""

    def _build_explanation(self, change: CodeChange, include_detail: bool) -> str:
        """Build explanation for patch."""
        explanation = [
            f"**File**: {change.file_path}",
            f"**Change Type**: {change.change_type.value}",
            f"**Description**: {change.description}",
            f"**Reason**: {change.reason}",
            f"**Impact**: {change.impact}",
        ]

        if include_detail:
            explanation.extend([
                f"**Complexity**: {change.complexity.value}",
                f"**Lines Added**: {change.estimated_lines_added}",
                f"**Lines Removed**: {change.estimated_lines_removed}",
            ])

            if change.dependencies:
                explanation.append(f"**Dependencies**: {', '.join(change.dependencies)}")

            if change.side_effects:
                explanation.append(f"**Side Effects**: {', '.join(change.side_effects)}")

        return "\n".join(explanation)

    def _assess_risk(self, change: CodeChange) -> str:
        """Assess risk level for patch."""
        from research_engineer.models.coding import ComplexityLevel, RiskLevel

        # Base risk on complexity
        complexity_risk = {
            ComplexityLevel.TRIVIAL: RiskLevel.LOW,
            ComplexityLevel.SIMPLE: RiskLevel.LOW,
            ComplexityLevel.MODERATE: RiskLevel.MEDIUM,
            ComplexityLevel.COMPLEX: RiskLevel.HIGH,
            ComplexityLevel.VERY_COMPLEX: RiskLevel.CRITICAL,
        }

        base_risk = complexity_risk.get(change.complexity, RiskLevel.MEDIUM)

        # Adjust based on change type
        if change.change_type == ChangeType.DELETION:
            # Deletions are higher risk
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH

        # Adjust based on impact
        if change.impact == "high":
            if base_risk in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                base_risk = RiskLevel.HIGH

        return base_risk.value
