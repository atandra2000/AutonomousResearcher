"""Patch Application Tool for Phase 4.

Applies generated patches to repository with approval workflow.
"""
import ast
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.models.coding import GeneratedPatch, PatchStatus
from research_engineer.tools.base import Tool, ToolError

logger = logging.getLogger(__name__)


def _is_valid_python(file_path: str, content: str) -> bool:
    """True when *content* parses as Python (non-.py files pass through)."""
    if not file_path.endswith(".py"):
        return True
    try:
        ast.parse(content)
    except SyntaxError:
        return False
    return True


class PatchApplicationInput(BaseModel):
    """Input for patch application."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Patches to apply",
    )
    repo_path: str = Field(..., description="Repository path")
    dry_run: bool = Field(
        default=True,
        description="If True, only simulate application",
    )
    require_approval: bool = Field(
        default=True,
        description="Require approval before applying",
    )
    approved: bool = Field(
        default=False,
        description="Whether patches are approved",
    )
    backup_enabled: bool = Field(
        default=True,
        description="Create backups before applying",
    )


class PatchApplicationOutput(BaseModel):
    """Output from patch application."""

    applied_patches: list[str] = Field(
        default_factory=list,
        description="IDs of successfully applied patches",
    )
    failed_patches: list[str] = Field(
        default_factory=list,
        description="IDs of failed patches",
    )
    skipped_patches: list[str] = Field(
        default_factory=list,
        description="IDs of skipped patches",
    )
    backup_files: list[str] = Field(
        default_factory=list,
        description="Backup files created",
    )
    files_modified: list[str] = Field(
        default_factory=list,
        description="Files that were modified",
    )
    application_status: str = Field(
        default="pending",
        description="Overall application status",
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Error messages if any",
    )
    application_time_seconds: float = Field(
        default=0.0,
        description="Application duration",
    )
    rollback_info: dict = Field(
        default_factory=dict,
        description="Information for rollback if needed",
    )


class PatchApplicationTool(Tool[PatchApplicationInput, PatchApplicationOutput]):
    """
    Tool for applying patches to repository.

    This tool:
    1. Validates patches before application
    2. Creates backups of modified files
    3. Applies patches using system patch command
    4. Handles failures gracefully
    5. Supports dry-run mode
    6. Requires approval for safety
    7. Generates rollback information
    """

    async def validate(self, input: PatchApplicationInput) -> bool:
        """Validate patch application input."""
        if not input.patches:
            return False
        repo_path = Path(input.repo_path)
        if not repo_path.exists():
            return False
        if not repo_path.is_dir():
            return False
        for patch in input.patches:
            if not patch.diff:
                return False
        return True

    async def execute(self, input: PatchApplicationInput) -> PatchApplicationOutput:
        """Apply patches to repository."""
        start_time = time.time()

        try:
            repo_path = Path(input.repo_path)
            if not repo_path.exists():
                raise ToolError(f"Repository does not exist: {input.repo_path}", input)
            if not repo_path.is_dir():
                raise ToolError(f"Path is not a directory: {input.repo_path}", input)

            if input.require_approval and not input.approved:
                return PatchApplicationOutput(
                    application_status="rejected",
                    error_messages=["Patches require approval before application"],
                    application_time_seconds=round(time.time() - start_time, 2),
                )

            applied = []
            failed = []
            skipped = []
            backup_files = []
            files_modified = []
            errors = []
            rollback_data = {"backups": [], "patches_applied": [], "timestamp": datetime.now().isoformat()}

            sorted_patches = self._sort_by_dependencies(input.patches)

            for patch in sorted_patches:
                try:
                    if patch.approval_required and not input.approved:
                        skipped.append(patch.patch_id)
                        continue

                    is_valid, error_msg = await self.validate_patch(patch, repo_path)
                    if not is_valid:
                        failed.append(patch.patch_id)
                        errors.append(f"Validation failed for {patch.patch_id}: {error_msg}")
                        continue

                    if input.backup_enabled and not input.dry_run:
                        backup_path = await self._create_backup(patch, repo_path)
                        if backup_path:
                            backup_files.append(backup_path)
                            rollback_data["backups"].append(backup_path)

                    if input.dry_run:
                        files_modified.append(patch.file_path)
                        applied.append(patch.patch_id)
                    else:
                        success = await self._apply_patch(patch, repo_path)
                        if success:
                            applied.append(patch.patch_id)
                            files_modified.append(patch.file_path)
                            patch.status = PatchStatus.APPLIED
                            rollback_data["patches_applied"].append(patch.patch_id)
                        else:
                            failed.append(patch.patch_id)
                            errors.append(f"Failed to apply patch {patch.patch_id}")

                except Exception as e:
                    failed.append(patch.patch_id)
                    errors.append(f"Error applying {patch.patch_id}: {e}")

            if failed:
                status = "partial_success" if applied else "failed"
            elif skipped:
                status = "partial_success" if applied else "skipped"
            else:
                status = "success" if applied else "no_op"

            elapsed = time.time() - start_time

            return PatchApplicationOutput(
                applied_patches=applied,
                failed_patches=failed,
                skipped_patches=skipped,
                backup_files=backup_files,
                files_modified=files_modified,
                application_status=status,
                error_messages=errors,
                application_time_seconds=round(elapsed, 2),
                rollback_info=rollback_data,
            )

        except Exception as e:
            raise ToolError(f"Patch application failed: {e}", input, e)

    async def validate_patch(self, patch: GeneratedPatch, repo_path: Path) -> tuple[bool, str]:
        """Validate a single patch before application."""
        if not patch.diff:
            return False, "Patch diff is empty"

        target_file = repo_path / patch.file_path

        if patch.change_type == "new_file":
            if target_file.exists():
                return False, f"File already exists: {patch.file_path}"
        elif patch.change_type == "deletion":
            if not target_file.exists():
                return False, f"File does not exist: {patch.file_path}"
        elif patch.change_type == "modification":
            if not target_file.exists():
                return False, f"File does not exist: {patch.file_path}"

            if patch.diff.startswith("diff --git"):
                result = subprocess.run(
                    ["patch", "--dry-run", "-p1"],
                    input=patch.diff,
                    cwd=str(repo_path),
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    return False, f"Patch validation failed: {result.stderr}"

        return True, ""

    def _sort_by_dependencies(self, patches: list[GeneratedPatch]) -> list[GeneratedPatch]:
        """Sort patches by dependencies to apply in correct order."""
        if not patches:
            return patches

        patch_map = {p.patch_id: p for p in patches}
        visited = set()
        result = []

        def visit(patch_id: str):
            if patch_id in visited:
                return
            visited.add(patch_id)
            patch = patch_map.get(patch_id)
            if patch:
                for dep in patch.dependencies:
                    if dep in patch_map:
                        visit(dep)
                result.append(patch)

        for patch in patches:
            visit(patch.patch_id)

        return result

    async def _create_backup(self, patch: GeneratedPatch, repo_path: Path) -> str | None:
        """Create backup of file before modification."""
        import shutil

        file_path = repo_path / patch.file_path

        if not file_path.exists():
            return None

        backup_dir = repo_path / ".patch_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_path = backup_dir / Path(patch.file_path).name
        shutil.copy2(file_path, backup_path)

        return str(backup_path)

    async def _apply_patch(self, patch: GeneratedPatch, repo_path: Path) -> bool:
        """Apply a single patch to repository."""
        try:
            if patch.change_type == "new_file":
                return await self._apply_new_file(patch, repo_path)
            elif patch.change_type == "modification":
                return await self._apply_modification(patch, repo_path)
            elif patch.change_type == "deletion":
                return await self._apply_deletion(patch, repo_path)
            else:
                return False

        except Exception as e:
            logger.error("Error applying patch %s: %s", patch.patch_id, e)
            return False

    async def _apply_new_file(self, patch: GeneratedPatch, repo_path: Path) -> bool:
        """Apply new file patch."""
        file_path = repo_path / patch.file_path
        content = self._extract_content_from_diff(patch.diff)
        if not _is_valid_python(patch.file_path, content):
            logger.warning(
                "Rejected new-file patch for %s: content does not parse",
                patch.file_path,
            )
            return False
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return True

    async def _apply_modification(self, patch: GeneratedPatch, repo_path: Path) -> bool:
        """Apply modification patch."""
        file_path = repo_path / patch.file_path

        if not file_path.exists():
            return await self._apply_new_file(patch, repo_path)

        original = file_path.read_text(encoding="utf-8")

        if patch.diff.startswith("diff --git"):
            result = subprocess.run(
                ["patch", "-p1"],
                input=patch.diff,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if "Reversed (or previously applied) patch detected" in result.stderr:
                    return True
                if result.returncode == 1:
                    conflicts = self._parse_conflicts(result.output)
                    if conflicts:
                        raise ToolError(f"Patch conflicts detected in {patch.file_path}: {conflicts}", None)
                return False
            # Defense in depth: an external `patch(1)` run may still
            # produce non-parsing Python; roll back when it does.
            applied = file_path.read_text(encoding="utf-8")
            if not _is_valid_python(patch.file_path, applied):
                logger.warning(
                    "Rolled back %s: patched content does not parse",
                    patch.file_path,
                )
                file_path.write_text(original, encoding="utf-8")
                return False
        else:
            content = file_path.read_text(encoding="utf-8")
            modified_content = self._apply_diff_to_content(content, patch.diff)
            if not _is_valid_python(patch.file_path, modified_content):
                logger.warning(
                    "Rejected patch for %s: resulting content does not parse",
                    patch.file_path,
                )
                return False
            file_path.write_text(modified_content, encoding="utf-8")

        return True

    async def _apply_deletion(self, patch: GeneratedPatch, repo_path: Path) -> bool:
        """Apply deletion patch."""
        file_path = repo_path / patch.file_path
        if file_path.exists():
            file_path.unlink()
        return True

    def _extract_content_from_diff(self, diff: str) -> str:
        """Extract file content from diff."""
        lines = diff.splitlines()
        content_lines = []
        in_hunk = False

        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                content_lines.append(line[1:])
                in_hunk = True
            elif line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith(" "):
                content_lines.append(line[1:])

        return "\n".join(content_lines) if content_lines else "# Generated file\n"

    def _apply_diff_to_content(self, content: str, diff: str) -> str:
        """Apply a unified diff to *content* (hunk-based).

        Honors ``@@ -start,count +start,count @@`` headers so changes
        land at their intended positions instead of being appended to
        the end of the file. Context lines must match; on mismatch the
        diff is treated as non-applicable and *content* is returned
        unchanged (callers reject/roll back via syntax validation or
        explicit failure handling).
        """
        original = content.splitlines(keepends=True)

        def _with_eol(line: str) -> str:
            return line if line.endswith("\n") else line + "\n"

        result: list[str] = []
        src_idx = 0  # 0-based cursor into `original`
        saw_change = False

        for tag, payload, old_start in _parse_diff_ops(diff):
            if tag == "hunk":
                # Copy untouched lines before this hunk.
                result.extend(
                    original[src_idx : max(old_start - 1, src_idx)]
                )
                src_idx = max(old_start - 1, src_idx)
                continue
            if tag in ("context", "del"):
                if not self._line_matches(original, src_idx, payload):
                    return content  # mismatch — refuse the diff
                if tag == "context":
                    result.append(original[src_idx])
                else:
                    saw_change = True
                src_idx += 1
            elif tag == "add":
                result.append(_with_eol(payload))
                saw_change = True
        result.extend(original[src_idx:])
        if not saw_change:
            return content
        return "".join(result)

    @staticmethod
    def _line_matches(
        original: list[str], src_idx: int, payload: str
    ) -> bool:
        """True when original[src_idx] equals the diff line payload."""
        if src_idx >= len(original):
            return False
        return original[src_idx].rstrip("\n") == payload.rstrip("\n")

    def _parse_conflicts(self, output: str) -> list[str]:
        """Parse patch output for conflict information."""
        conflicts = []
        for line in output.splitlines():
            if "conflict" in line.lower() or "reject" in line.lower():
                conflicts.append(line.strip())
        return conflicts

    async def generate_rollback_plan(self, application_output: PatchApplicationOutput) -> dict:
        """Generate rollback plan from application output."""
        return {
            "rollback_available": bool(application_output.backup_files),
            "backup_files": application_output.backup_files,
            "patches_to_revert": application_output.applied_patches,
            "timestamp": application_output.application_time_seconds,
        }


def _parse_diff_ops(diff: str) -> list[tuple[str, str, int]]:
    """Tokenize a unified diff into (tag, payload, old_start) ops.

    Tags: ``hunk`` (payload "", old_start = 1-based source line),
    ``context``/``del``/``add`` (payload = line text), and
    ``blank_context`` for empty context lines. Parsing stops at any
    unrecognized line (e.g. the next file header).
    """
    ops: list[tuple[str, str, int]] = []
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
            if not match:
                break
            ops.append(("hunk", "", int(match.group(1))))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        if line.startswith(" "):
            ops.append(("context", line[1:], 0))
        elif line.startswith("-") and not line.startswith("---"):
            ops.append(("del", line[1:], 0))
        elif line.startswith("+") and not line.startswith("+++"):
            ops.append(("add", line[1:], 0))
        else:
            break  # end of hunks
    return ops
