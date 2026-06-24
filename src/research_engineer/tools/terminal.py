"""Terminal Tool for Phase 11 - Terminal-first autonomous coding agent.

Provides safe, sandboxed filesystem and shell operations for the
:class:`~research_engineer.agents.task_agent.TaskAgent`:

- ``run_command``  : execute an allowlisted shell command in a workdir
- ``read_file``    : read a file's text content (size-capped)
- ``write_file``   : write text content to a file (creates parents)
- ``search_code``  : ripgrep-style content search (pure-Python fallback)
- ``apply_patch``  : apply a unified diff via the system ``patch`` command
- ``git_status``   : return ``git status --porcelain`` output
- ``git_diff``      : return ``git diff`` (optionally cached/staged)

Design mirrors :class:`~research_engineer.tools.experiment_runner.ExperimentRunnerTool`:
command allowlist, dry-run default, timeout, working-directory confinement,
and a typed ``Tool[Input, Output]`` interface.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.tools.base import Tool, ToolError

# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

#: Commands permitted by ``run_command``. Kept deliberately small; the
#: task agent's test step uses these to run test suites and linters.
ALLOWED_COMMAND_PREFIXES = frozenset({
    "python",
    "python3",
    "pytest",
    "uv",
    "ruff",
    "mypy",
    "bash",
    "sh",
    "make",
    "git",
    "pip",
    "echo",
    "cat",
    "ls",
    "rg",
    "grep",
    "find",
})

#: Maximum bytes returned for file reads / command stdout to bound memory.
MAX_OUTPUT_BYTES = 1_000_000


def _command_allowed(command: list[str]) -> bool:
    """Return True if the command starts with an allowed prefix."""
    if not command:
        return False
    base = os.path.basename(command[0])
    return base in ALLOWED_COMMAND_PREFIXES


def _split_command(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return command
    return command.split()


# ---------------------------------------------------------------------------
# Input / Output models
# ---------------------------------------------------------------------------


class TerminalInput(BaseModel):
    """Input for :class:`TerminalTool`.

    Exactly one ``operation`` is set per call; the remaining fields are
    routed to that operation.
    """

    operation: str = Field(
        ...,
        description=(
            "One of: run_command, read_file, write_file, search_code, "
            "apply_patch, git_status, git_diff"
        ),
    )
    repo_path: str = Field(..., description="Working directory / repo root")

    # run_command
    command: str | list[str] | None = Field(
        default=None, description="Command to execute"
    )
    timeout_seconds: int = Field(
        default=300, ge=1, description="Command timeout"
    )
    dry_run: bool = Field(
        default=False, description="If True, return the command without executing"
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict, description="Extra environment variables"
    )

    # read_file / write_file
    file_path: str | None = Field(default=None, description="Target file path")
    content: str | None = Field(
        default=None, description="Content to write (write_file) or search (search_code)"
    )
    max_bytes: int = Field(
        default=MAX_OUTPUT_BYTES, ge=1, description="Cap on bytes read"
    )

    # search_code
    pattern: str | None = Field(
        default=None, description="Regex pattern (search_code)"
    )
    file_glob: str | None = Field(
        default=None, description="Optional glob filter (search_code)"
    )
    max_matches: int = Field(
        default=200, ge=1, description="Cap on search matches"
    )

    # apply_patch
    patch_diff: str | None = Field(
        default=None, description="Unified diff to apply (apply_patch)"
    )

    # git_diff
    staged: bool = Field(
        default=False, description="If True, show staged diff (--cached)"
    )


class TerminalOutput(BaseModel):
    """Output from :class:`TerminalTool`."""

    operation: str = Field(..., description="Operation that was executed")
    success: bool = Field(..., description="Whether the operation succeeded")
    exit_code: int | None = Field(default=None, description="Process exit code")
    stdout: str = Field(default="", description="Captured stdout")
    stderr: str = Field(default="", description="Captured stderr")
    content: str | None = Field(
        default=None, description="File content (read_file) or diff (git_diff)"
    )
    matches: list[dict[str, Any]] = Field(
        default_factory=list, description="Search matches (search_code)"
    )
    files_modified: list[str] = Field(
        default_factory=list, description="Files changed (apply_patch)"
    )
    duration_seconds: float = Field(default=0.0, description="Operation duration")
    error: str | None = Field(default=None, description="Error message if any")


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class TerminalTool(Tool[TerminalInput, TerminalOutput]):
    """Safe terminal / filesystem operations for the autonomous coding agent.

    All shell execution is confined to ``repo_path`` and restricted to an
    allowlist of command prefixes. File reads are size-capped. Patch
    application uses the system ``patch`` utility with ``--dry-run``
    validation first.
    """

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def validate(self, input: TerminalInput) -> bool:
        if not input.operation:
            return False
        if not input.repo_path:
            return False
        repo = Path(input.repo_path)
        if not repo.exists() or not repo.is_dir():
            return False
        return True

    async def execute(self, input: TerminalInput) -> TerminalOutput:
        try:
            op = input.operation
            if op == "run_command":
                return await self._run_command(input)
            if op == "read_file":
                return await self._read_file(input)
            if op == "write_file":
                return await self._write_file(input)
            if op == "search_code":
                return await self._search_code(input)
            if op == "apply_patch":
                return await self._apply_patch(input)
            if op == "git_status":
                return await self._git_status(input)
            if op == "git_diff":
                return await self._git_diff(input)
            raise ToolError(
                f"Unknown terminal operation: {op!r}",
                input,
                None,
            )
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Terminal tool failed: {e}", input, e)

    # ------------------------------------------------------------------
    # run_command
    # ------------------------------------------------------------------

    async def _run_command(self, input: TerminalInput) -> TerminalOutput:
        import time

        start = time.time()
        cmd = _split_command(input.command or [])
        if not cmd:
            raise ToolError("run_command requires `command`", input, None)
        if not _command_allowed(cmd):
            raise ToolError(
                f"Command not allowed: {cmd[0]}. "
                f"Allowed prefixes: {sorted(ALLOWED_COMMAND_PREFIXES)}",
                input,
                None,
            )

        if input.dry_run:
            return TerminalOutput(
                operation="run_command",
                success=True,
                stdout=" ".join(cmd),
                duration_seconds=round(time.time() - start, 3),
            )

        env = os.environ.copy()
        env.update(input.env_vars)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=input.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, NotADirectoryError, OSError) as e:
            return TerminalOutput(
                operation="run_command",
                success=False,
                error=str(e),
                duration_seconds=round(time.time() - start, 3),
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=input.timeout_seconds
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return TerminalOutput(
                operation="run_command",
                success=False,
                exit_code=-1,
                stderr=f"Command timed out after {input.timeout_seconds}s",
                duration_seconds=round(time.time() - start, 3),
            )

        exit_code = proc.returncode if proc.returncode is not None else -1
        return TerminalOutput(
            operation="run_command",
            success=exit_code == 0,
            exit_code=exit_code,
            stdout=self._decode(stdout_b, input.max_bytes),
            stderr=self._decode(stderr_b, input.max_bytes),
            duration_seconds=round(time.time() - start, 3),
        )

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    async def _read_file(self, input: TerminalInput) -> TerminalOutput:
        if not input.file_path:
            raise ToolError("read_file requires `file_path`", input, None)
        path = self._resolve(input.repo_path, input.file_path)
        if not path.exists():
            return TerminalOutput(
                operation="read_file",
                success=False,
                error=f"File not found: {input.file_path}",
            )
        if not path.is_file():
            return TerminalOutput(
                operation="read_file",
                success=False,
                error=f"Not a file: {input.file_path}",
            )
        try:
            raw = path.read_bytes()
        except OSError as e:
            return TerminalOutput(
                operation="read_file",
                success=False,
                error=str(e),
            )
        text = self._decode(raw, input.max_bytes)
        return TerminalOutput(
            operation="read_file",
            success=True,
            content=text,
        )

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    async def _write_file(self, input: TerminalInput) -> TerminalOutput:
        if not input.file_path:
            raise ToolError("write_file requires `file_path`", input, None)
        if input.content is None:
            raise ToolError("write_file requires `content`", input, None)
        path = self._resolve(input.repo_path, input.file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input.content, encoding="utf-8")
        except OSError as e:
            return TerminalOutput(
                operation="write_file",
                success=False,
                error=str(e),
            )
        return TerminalOutput(
            operation="write_file",
            success=True,
            files_modified=[str(path.relative_to(Path(input.repo_path).resolve()))]
            if path.is_relative_to(Path(input.repo_path).resolve())
            else [str(path)],
        )

    # ------------------------------------------------------------------
    # search_code
    # ------------------------------------------------------------------

    async def _search_code(self, input: TerminalInput) -> TerminalOutput:
        pattern = input.pattern or input.content
        if not pattern:
            raise ToolError(
                "search_code requires `pattern` (or `content`)", input, None
            )
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return TerminalOutput(
                operation="search_code",
                success=False,
                error=f"Invalid regex: {e}",
            )

        root = Path(input.repo_path).resolve()
        glob_pat = input.file_glob or "*.py"
        matches: list[dict[str, Any]] = []
        try:
            for file_path in sorted(root.rglob(glob_pat)):
                if self._is_noise(file_path):
                    continue
                if not file_path.is_file():
                    continue
                self._scan_file(file_path, root, regex, input.max_matches, matches)
                if len(matches) >= input.max_matches:
                    return TerminalOutput(
                        operation="search_code",
                        success=True,
                        matches=matches,
                    )
        except OSError as e:
            return TerminalOutput(
                operation="search_code",
                success=False,
                error=str(e),
            )
        return TerminalOutput(
            operation="search_code",
            success=True,
            matches=matches,
        )

    @staticmethod
    def _is_noise(file_path: Path) -> bool:
        """True if ``file_path`` is inside a noise directory."""
        noise = {".git", "__pycache__", ".venv", "node_modules"}
        return any(part in noise for part in file_path.parts)

    @staticmethod
    def _scan_file(
        file_path: Path,
        root: Path,
        regex: re.Pattern[str],
        max_matches: int,
        matches: list[dict[str, Any]],
    ) -> None:
        """Scan a single file for regex matches, appending to ``matches``."""
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        rel = (
            str(file_path.relative_to(root))
            if file_path.is_relative_to(root)
            else str(file_path)
        )
        for lineno, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(
                    {"path": rel, "line": lineno, "text": line[:500]}
                )
                if len(matches) >= max_matches:
                    return

    # ------------------------------------------------------------------
    # apply_patch
    # ------------------------------------------------------------------

    async def _apply_patch(self, input: TerminalInput) -> TerminalOutput:
        import time

        start = time.time()
        if not input.patch_diff:
            raise ToolError("apply_patch requires `patch_diff`", input, None)
        repo = Path(input.repo_path)
        if not repo.is_dir():
            return TerminalOutput(
                operation="apply_patch",
                success=False,
                error=f"Not a directory: {input.repo_path}",
            )

        # Validate first with --dry-run.
        try:
            check = subprocess.run(
                ["patch", "--dry-run", "-p1"],
                input=input.patch_diff,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            return TerminalOutput(
                operation="apply_patch",
                success=False,
                error=f"patch validation failed: {e}",
                duration_seconds=round(time.time() - start, 3),
            )
        if check.returncode != 0:
            return TerminalOutput(
                operation="apply_patch",
                success=False,
                error=f"Patch would not apply cleanly: {check.stderr.strip()}",
                duration_seconds=round(time.time() - start, 3),
            )

        # Apply for real.
        try:
            result = subprocess.run(
                ["patch", "-p1"],
                input=input.patch_diff,
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.SubprocessError as e:
            return TerminalOutput(
                operation="apply_patch",
                success=False,
                error=str(e),
                duration_seconds=round(time.time() - start, 3),
            )

        modified = self._parse_modified_files(result.stdout)
        return TerminalOutput(
            operation="apply_patch",
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            files_modified=modified,
            duration_seconds=round(time.time() - start, 3),
        )

    # ------------------------------------------------------------------
    # git_status
    # ------------------------------------------------------------------

    async def _git_status(self, input: TerminalInput) -> TerminalOutput:
        return await self._git(
            input, ["git", "status", "--porcelain"], "git_status"
        )

    # ------------------------------------------------------------------
    # git_diff
    # ------------------------------------------------------------------

    async def _git_diff(self, input: TerminalInput) -> TerminalOutput:
        cmd = ["git", "diff"]
        if input.staged:
            cmd.append("--cached")
        out = await self._git(input, cmd, "git_diff")
        if out.success:
            out.content = out.stdout
        return out

    # ------------------------------------------------------------------
    # Shared git helper
    # ------------------------------------------------------------------

    async def _git(
        self, input: TerminalInput, cmd: list[str], op: str
    ) -> TerminalOutput:
        import time

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=input.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
        except Exception as e:
            return TerminalOutput(
                operation=op,
                success=False,
                error=str(e),
                duration_seconds=round(time.time() - start, 3),
            )
        exit_code = proc.returncode if proc.returncode is not None else -1
        return TerminalOutput(
            operation=op,
            success=exit_code == 0,
            exit_code=exit_code,
            stdout=self._decode(stdout_b, input.max_bytes),
            stderr=self._decode(stderr_b, input.max_bytes),
            duration_seconds=round(time.time() - start, 3),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(repo_path: str, file_path: str) -> Path:
        """Resolve ``file_path`` against ``repo_path`` (absolute wins)."""
        p = Path(file_path)
        if p.is_absolute():
            return p
        return Path(repo_path) / file_path

    @staticmethod
    def _decode(data: bytes, max_bytes: int) -> str:
        if not data:
            return ""
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n...[truncated]"
        return text

    @staticmethod
    def _parse_modified_files(patch_stdout: str) -> list[str]:
        """Extract modified file paths from ``patch`` output."""
        files: list[str] = []
        for line in patch_stdout.splitlines():
            # Lines like: "patching file src/foo.py"
            if line.startswith("patching file "):
                files.append(line[len("patching file ") :].strip())
        return files


__all__ = [
    "TerminalTool",
    "TerminalInput",
    "TerminalOutput",
    "ALLOWED_COMMAND_PREFIXES",
]
