"""Code Generation Tool for Phase 4.

Generates code changes based on implementation plans and repository context.
"""

import ast
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.llm import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMRole,
)
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

logger = logging.getLogger(__name__)


def _is_valid_python(file_path: str, content: str) -> bool:
    """True when *content* parses as Python (non-.py files pass through)."""
    if not file_path.endswith(".py"):
        return True
    try:
        ast.parse(content)
    except SyntaxError as e:
        logger.debug("Syntax check failed for %s: %s", file_path, e)
        return False
    return True


def _strip_code_fences(text: str) -> str:
    """Strip a single surrounding markdown code fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return text
    body = stripped[first_newline + 1 :]
    end = body.rfind("```")
    if end != -1:
        body = body[:end]
    return body.rstrip() + "\n"


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
    memory_context: str = Field(
        default="",
        description="Recalled memory context (insights, successes, failures)",
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
    3. Generates code changes (LLM-backed when a provider is available,
       rule-based target identification otherwise)
    4. Creates patch proposals
    5. Never directly modifies files (patch-first philosophy)
    """

    #: Maximum characters of existing file content sent as LLM context.
    MAX_CONTEXT_CHARS = 12_000

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm

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

            # Refine changes with concrete file content via the LLM
            if self.llm is not None:
                await self._refine_changes_with_llm(input, changes)

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

    async def _refine_changes_with_llm(
        self, input: CodeGenerationInput, changes: list[CodeChange]
    ) -> None:
        """Fill in ``proposed_content`` for each change via the LLM.

        The LLM produces *what* the resulting file should look like and,
        when the rule-based target does not exist in the repo, also
        picks a real target file from the repository listing. Failures
        degrade per-change to placeholder diffs and are logged, never
        raised.
        """
        assert self.llm is not None
        repo_path = Path(input.repo_path)
        candidates = self._candidate_files(repo_path)
        for change in changes:
            file_path = repo_path / change.file_path
            if change.change_type == ChangeType.MODIFICATION and not file_path.exists():
                resolved = await self._resolve_target_file(
                    input, change, candidates, repo_path
                )
                if resolved is None:
                    continue
                file_path = repo_path / resolved
                change.file_path = resolved
            existing = ""
            if file_path.exists():
                try:
                    existing = file_path.read_text(encoding="utf-8")
                except OSError as e:
                    logger.warning(
                        "Cannot read %s for LLM refinement: %s",
                        change.file_path,
                        e,
                    )
                    continue

            if len(existing) > self.MAX_CONTEXT_CHARS:
                head = existing[: self.MAX_CONTEXT_CHARS]
                existing = (
                    f"{head}\n\n# ... [truncated, "
                    f"{len(existing) - self.MAX_CONTEXT_CHARS} more chars]"
                )
            prompt = self._build_refinement_prompt(input, change, existing)
            request = LLMRequest(
                messages=[
                    LLMMessage(
                        role=LLMRole.SYSTEM,
                        content=(
                            "You are an expert ML engineer. Return ONLY the "
                            "complete final content of the target file as raw "
                            "source code. No markdown fences, no commentary."
                        ),
                    ),
                    LLMMessage(role=LLMRole.USER, content=prompt),
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            try:
                resp = await self.llm.complete(request)
            except Exception as e:
                logger.warning(
                    "LLM code generation failed for %s: %s",
                    change.file_path,
                    e,
                )
                continue
            content = _strip_code_fences(resp.content)
            if not content.strip():
                continue
            if not _is_valid_python(change.file_path, content):
                logger.warning(
                    "Rejected LLM proposal for %s: generated content "
                    "does not parse as Python",
                    change.file_path,
                )
                continue
            change.proposed_content = content

    def _build_refinement_prompt(
        self,
        input: CodeGenerationInput,
        change: CodeChange,
        existing: str,
    ) -> str:
        """Build the user prompt for one file's proposed content."""
        parts = [
            f"Task: {input.task_description}",
            f"Target file: {change.file_path}",
            f"Change type: {change.change_type.value}",
            f"Description: {change.description}",
        ]
        if input.constraints:
            parts.append(f"Constraints: {'; '.join(input.constraints)}")
        if input.memory_context:
            parts.append(
                f"Recalled memory context:\n{input.memory_context}"
            )
        if input.implementation_plan is not None:
            plan_summary = getattr(
                input.implementation_plan, "overview", ""
            ) or ""
            if plan_summary:
                parts.append(f"Plan overview: {plan_summary}")
        if existing:
            parts.append(
                f"\n## Current file content\n```\n{existing}\n```"
            )
        else:
            parts.append(
                "\nThe file does not exist yet; create it from scratch "
                "following the repository's conventions."
            )
        return "\n".join(parts)

    @staticmethod
    def _candidate_files(repo_path: Path, limit: int = 60) -> list[str]:
        """List repository Python files (relative paths) as LLM candidates."""
        candidates: list[str] = []
        for p in sorted(repo_path.rglob("*.py")):
            if any(part in {".git", "__pycache__", ".venv", "node_modules"} for part in p.parts):
                continue
            candidates.append(str(p.relative_to(repo_path)))
            if len(candidates) >= limit:
                break
        return candidates

    async def _resolve_target_file(
        self,
        input: CodeGenerationInput,
        change: CodeChange,
        candidates: list[str],
        repo_path: Path,
    ) -> str | None:
        """Ask the LLM to pick a real existing file for a modification.

        Returns the resolved repo-relative path, or None when resolution
        fails (the change is then skipped with a warning).
        """
        assert self.llm is not None
        listing = "\n".join(f"- {c}" for c in candidates)
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=(
                        "You are a software engineering assistant. Reply "
                        "with EXACTLY one line containing a single file "
                        "path chosen from the provided list and nothing else."
                    ),
                ),
                LLMMessage(
                    role=LLMRole.USER,
                    content=(
                        f"Task: {input.task_description}\n\n"
                        f"Proposed (nonexistent) target: {change.file_path}\n"
                        f"Description: {change.description}\n\n"
                        f"Choose the best existing file to modify:\n{listing}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=512,
        )
        try:
            resp = await self.llm.complete(request)
        except Exception as e:
            logger.warning(
                "LLM target resolution failed for %s: %s",
                change.file_path,
                e,
            )
            return None
        lines = resp.content.strip().strip("`'\"").splitlines()
        candidate = lines[0].strip() if lines else ""
        if candidate in candidates and (repo_path / candidate).exists():
            logger.info(
                "Resolved change target %s -> %s", change.file_path, candidate
            )
            return candidate
        logger.warning(
            "LLM resolved target %r is not an existing repo file", candidate
        )
        return None
