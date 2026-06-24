"""Code Generation Tool for Phase 4.

Generates code changes based on implementation plans and repository context.
"""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    ChangeType,
    CodeChange,
    ComplexityLevel,
    GeneratedPatch,
)
from research_engineer.models.planner import ImplementationPlan, ImplementationStep
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class CodeGenerationInput(BaseModel):
    """Input for code generation."""

    paper_id: str | None = Field(default=None, description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary | None = Field(
        default=None,
        description="Paper summary",
    )
    repo_summary: RepositorySummary | None = Field(
        default=None,
        description="Repository summary",
    )
    implementation_plan: ImplementationPlan | None = Field(
        default=None,
        description="Implementation plan",
    )
    implementation_step: ImplementationStep | None = Field(
        default=None,
        description="Specific step to implement",
    )
    task_description: str = Field(..., description="What to implement")
    target_files: list[str] = Field(
        default_factory=list,
        description="Target files for modification",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Implementation constraints",
    )


class CodeGenerationOutput(BaseModel):
    """Output from code generation."""

    changes: list[CodeChange] = Field(
        default_factory=list,
        description="List of code changes",
    )
    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Generated patches",
    )
    total_changes: int = Field(default=0, description="Total number of changes")
    files_to_modify: list[str] = Field(
        default_factory=list,
        description="Files that need modification",
    )
    new_files: list[str] = Field(default_factory=list, description="New files to create")
    estimated_lines_added: int = Field(default=0, description="Estimated lines added")
    estimated_lines_removed: int = Field(default=0, description="Estimated lines removed")
    complexity_assessment: str = Field(
        default="unknown",
        description="Overall complexity assessment",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Generation duration",
    )


class CodeGenerationTool(Tool[CodeGenerationInput, CodeGenerationOutput]):
    """
    Tool for generating code changes.

    This tool:
    1. Analyzes implementation requirements
    2. Identifies modification targets
    3. Generates code changes
    4. Creates patch proposals
    5. Never directly modifies files (patch-first philosophy)
    """

    async def execute(self, input: CodeGenerationInput) -> CodeGenerationOutput:
        """Generate code changes based on implementation requirements."""
        import time
        start_time = time.time()

        try:
            # Validate repository exists
            repo_path = Path(input.repo_path)
            if not repo_path.exists():
                raise ToolError(f"Repository does not exist: {input.repo_path}", input)

            # Generate code changes
            changes = await self._generate_changes(input)

            # Generate patches
            patches = await self._generate_patches(input, changes)

            # Calculate statistics
            total_lines_added = sum(c.estimated_lines_added for c in changes)
            total_lines_removed = sum(c.estimated_lines_removed for c in changes)

            files_to_modify = list(set(c.file_path for c in changes if c.change_type == ChangeType.MODIFICATION))
            new_files = list(set(c.file_path for c in changes if c.change_type == ChangeType.NEW_FILE))

            # Assess complexity
            complexity = self._assess_complexity(changes)

            elapsed = time.time() - start_time

            return CodeGenerationOutput(
                changes=changes,
                patches=patches,
                total_changes=len(changes),
                files_to_modify=files_to_modify,
                new_files=new_files,
                estimated_lines_added=total_lines_added,
                estimated_lines_removed=total_lines_removed,
                complexity_assessment=complexity,
                generation_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Code generation failed: {e}", input, e)

    async def _generate_changes(self, input: CodeGenerationInput) -> list[CodeChange]:
        """Generate code changes based on requirements."""
        changes = []

        # If implementation plan is provided, use it
        if input.implementation_plan:
            changes.extend(await self._generate_from_plan(input))
        else:
            # Generate changes from task description
            changes.extend(await self._generate_from_task(input))

        return changes

    async def _generate_from_plan(self, input: CodeGenerationInput) -> list[CodeChange]:
        """Generate changes from implementation plan."""
        changes = []
        plan = input.implementation_plan

        # Process implementation targets
        for target in plan.targets:
            change = CodeChange(
                file_path=target.file_path,
                change_type=ChangeType.MODIFICATION,
                description=target.description,
                reason=f"Implementation of {input.paper_id or 'task'}",
                impact="medium",
                complexity=ComplexityLevel.MODERATE,
                estimated_lines_added=target.estimated_lines or 10,
                estimated_lines_removed=0,
            )
            changes.append(change)

        # If specific step is provided, focus on it
        if input.implementation_step:
            step = input.implementation_step
            for target in plan.targets:
                if target.file_path in step.dependencies or target.file_path in step.targets:
                    change = CodeChange(
                        file_path=target.file_path,
                        change_type=ChangeType.MODIFICATION,
                        description=f"Step {step.step_number}: {step.title}",
                        reason=step.description,
                        impact="high",
                        complexity=self._difficulty_to_complexity(step.difficulty),
                        dependencies=[f"step_{d}" for d in step.dependencies],
                    )
                    changes.append(change)

        return changes

    async def _generate_from_task(self, input: CodeGenerationInput) -> list[CodeChange]:
        """Generate changes from task description."""
        changes = []

        # Analyze task description to determine change type
        task_lower = input.task_description.lower()

        # Detect common ML implementation patterns
        if "add" in task_lower and "model" in task_lower:
            change = CodeChange(
                file_path="models/new_model.py",
                change_type=ChangeType.NEW_FILE,
                description=f"Add new model: {input.task_description}",
                reason="New model implementation requested",
                impact="medium",
                complexity=ComplexityLevel.COMPLEX,
                estimated_lines_added=200,
            )
            changes.append(change)

        elif "add" in task_lower and "layer" in task_lower:
            change = CodeChange(
                file_path="models/layers.py",
                change_type=ChangeType.MODIFICATION,
                description=f"Add new layer: {input.task_description}",
                reason="New layer implementation requested",
                impact="low",
                complexity=ComplexityLevel.MODERATE,
                estimated_lines_added=50,
            )
            changes.append(change)

        elif "update" in task_lower or "modify" in task_lower:
            # Try to identify target file from repository summary
            target_file = self._identify_target_file(input.task_description, input.repo_summary)
            change = CodeChange(
                file_path=target_file,
                change_type=ChangeType.MODIFICATION,
                description=f"Update: {input.task_description}",
                reason="Modification requested",
                impact="medium",
                complexity=ComplexityLevel.MODERATE,
            )
            changes.append(change)

        elif "config" in task_lower or "configuration" in task_lower:
            change = CodeChange(
                file_path="config.yaml",
                change_type=ChangeType.CONFIG_UPDATE,
                description=f"Configuration update: {input.task_description}",
                reason="Configuration change requested",
                impact="low",
                complexity=ComplexityLevel.SIMPLE,
            )
            changes.append(change)

        else:
            # Generic change
            change = CodeChange(
                file_path="src/implementation.py",
                change_type=ChangeType.MODIFICATION,
                description=input.task_description,
                reason="Implementation requested",
                impact="medium",
                complexity=ComplexityLevel.MODERATE,
            )
            changes.append(change)

        return changes

    def _identify_target_file(self, task: str, repo_summary: RepositorySummary | None) -> str:
        """Identify target file from task description and repository context."""
        if not repo_summary:
            return "src/implementation.py"

        task_lower = task.lower()

        # Search in important files
        if repo_summary.important_files:
            for file_imp in repo_summary.important_files:
                file_path = file_imp.file_path if hasattr(file_imp, 'file_path') else str(file_imp)
                if any(keyword in file_path.lower() for keyword in task_lower.split()):
                    return file_path

        # Default based on task keywords
        if "model" in task_lower:
            return "models/model.py"
        elif "train" in task_lower:
            return "train.py"
        elif "data" in task_lower:
            return "data/dataset.py"
        elif "eval" in task_lower:
            return "eval.py"

        return "src/implementation.py"

    def _difficulty_to_complexity(self, difficulty) -> ComplexityLevel:
        """Convert difficulty level to complexity level."""
        from research_engineer.models.planner import DifficultyLevel

        mapping = {
            DifficultyLevel.TRIVIAL: ComplexityLevel.TRIVIAL,
            DifficultyLevel.EASY: ComplexityLevel.SIMPLE,
            DifficultyLevel.MODERATE: ComplexityLevel.MODERATE,
            DifficultyLevel.HARD: ComplexityLevel.COMPLEX,
            DifficultyLevel.VERY_HARD: ComplexityLevel.VERY_COMPLEX,
        }
        return mapping.get(difficulty, ComplexityLevel.MODERATE)

    def _assess_complexity(self, changes: list[CodeChange]) -> str:
        """Assess overall complexity of changes."""
        if not changes:
            return "unknown"

        complexity_scores = {
            ComplexityLevel.TRIVIAL: 1,
            ComplexityLevel.SIMPLE: 2,
            ComplexityLevel.MODERATE: 3,
            ComplexityLevel.COMPLEX: 4,
            ComplexityLevel.VERY_COMPLEX: 5,
        }

        avg_score = sum(complexity_scores.get(c.complexity, 3) for c in changes) / len(changes)

        if avg_score <= 1.5:
            return "trivial"
        elif avg_score <= 2.5:
            return "simple"
        elif avg_score <= 3.5:
            return "moderate"
        elif avg_score <= 4.5:
            return "complex"
        else:
            return "very complex"

    async def _generate_patches(self, input: CodeGenerationInput, changes: list[CodeChange]) -> list[GeneratedPatch]:
        """Generate patches from code changes."""
        patches = []

        for i, change in enumerate(changes):
            # Generate a placeholder diff
            # In a real implementation, this would generate actual unified diffs
            diff_content = self._generate_placeholder_diff(change)

            patch = GeneratedPatch(
                patch_id=f"patch_{i:04d}",
                file_path=change.file_path,
                change_type=change.change_type,
                diff=diff_content,
                explanation=change.description,
                reason=change.reason,
                impact=change.impact,
                dependencies=change.dependencies,
                risk_level=self._complexity_to_risk(change.complexity),
                approval_required=True,
            )
            patches.append(patch)

        return patches

    def _generate_placeholder_diff(self, change: CodeChange) -> str:
        """Generate placeholder diff content."""
        if change.change_type == ChangeType.NEW_FILE:
            return f"""--- /dev/null
+++ b/{change.file_path}
@@ -0,0 +1,{change.estimated_lines_added} @@
+# New file: {change.file_path}
+# Description: {change.description}
+# Reason: {change.reason}
+#
+# [Code generation would insert actual implementation here]
+# Estimated lines: {change.estimated_lines_added}
"""
        elif change.change_type == ChangeType.MODIFICATION:
            return f"""--- a/{change.file_path}
+++ b/{change.file_path}
@@ -1,5 +1,{change.estimated_lines_added} @@
 # Modified file: {change.file_path}
 # Description: {change.description}
 # Reason: {change.reason}
+#
+# [Code generation would insert actual changes here]
+# Lines added: {change.estimated_lines_added}, removed: {change.estimated_lines_removed}
"""
        else:
            return f"# {change.change_type.value}: {change.file_path}\n# {change.description}\n"

    def _complexity_to_risk(self, complexity: ComplexityLevel) -> str:
        """Convert complexity level to risk level."""
        from research_engineer.models.coding import RiskLevel

        mapping = {
            ComplexityLevel.TRIVIAL: RiskLevel.LOW,
            ComplexityLevel.SIMPLE: RiskLevel.LOW,
            ComplexityLevel.MODERATE: RiskLevel.MEDIUM,
            ComplexityLevel.COMPLEX: RiskLevel.HIGH,
            ComplexityLevel.VERY_COMPLEX: RiskLevel.CRITICAL,
        }
        return mapping.get(complexity, RiskLevel.MEDIUM)
