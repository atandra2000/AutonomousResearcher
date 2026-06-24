"""Experiment Runner Tool for Phase 7.

Launches training, evaluation, and validation commands as subprocesses
with safety controls: command allowlist, dry-run default, timeout, and
working directory confinement.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from research_engineer.models.experiment import (
    ExperimentRun,
    ExperimentRunnerInput,
    ExperimentRunnerOutput,
    ExperimentStatus,
    ExperimentType,
    StatusTransition,
)
from research_engineer.tools.base import Tool, ToolError

ALLOWED_COMMAND_PREFIXES = frozenset({
    "python",
    "python3",
    "torchrun",
    "accelerate",
    "pytest",
    "bash",
    "sh",
    "make",
    "uv",
    "pip",
})


def _command_allowed(command: list[str]) -> bool:
    """Return True if the command starts with an allowed prefix."""
    if not command:
        return False
    base = os.path.basename(command[0])
    # Handle paths like /usr/bin/python
    return base in ALLOWED_COMMAND_PREFIXES


class ExperimentRunnerTool(Tool[ExperimentRunnerInput, ExperimentRunnerOutput]):
    """Launch experiments as subprocesses with safety controls."""

    def __init__(self) -> None:
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def validate(self, input: ExperimentRunnerInput) -> bool:
        if not input.command:
            return False
        if not input.working_dir:
            return False
        if not _command_allowed(input.command):
            return False
        return True

    async def execute(self, input: ExperimentRunnerInput) -> ExperimentRunnerOutput:
        try:
            if not _command_allowed(input.command):
                raise ToolError(
                    f"Command not allowed: {input.command[0]}. "
                    f"Allowed prefixes: {sorted(ALLOWED_COMMAND_PREFIXES)}",
                    input,
                    None,
                )

            run = ExperimentRun(
                experiment_id=input.experiment_id,
                command=input.command,
                working_dir=input.working_dir,
                experiment_type=input.experiment_type,
                status=ExperimentStatus.PENDING,
                timeout_seconds=input.timeout_seconds,
            )

            # Dry-run mode: return without executing
            if input.dry_run or input.experiment_type == ExperimentType.DRY_RUN:
                run.status = ExperimentStatus.PENDING
                run.status_history.append(
                    StatusTransition(
                        from_status=ExperimentStatus.PENDING,
                        to_status=ExperimentStatus.PENDING,
                        reason="dry_run",
                    )
                )
                return ExperimentRunnerOutput(
                    run=run,
                    launched=False,
                    message="Dry run: command not executed",
                )

            # Real execution
            run.status = ExperimentStatus.RUNNING
            run.start_time = datetime.now()
            run.status_history.append(
                StatusTransition(
                    from_status=ExperimentStatus.PENDING,
                    to_status=ExperimentStatus.RUNNING,
                    reason="launched",
                )
            )

            env = os.environ.copy()
            env.update(input.env_vars)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *input.command,
                    cwd=input.working_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            except (FileNotFoundError, NotADirectoryError, OSError) as e:
                run.status = ExperimentStatus.CRASHED
                run.end_time = datetime.now()
                run.error_message = str(e)
                run.status_history.append(
                    StatusTransition(
                        from_status=ExperimentStatus.RUNNING,
                        to_status=ExperimentStatus.CRASHED,
                        reason=f"launch_error: {e}",
                    )
                )
                return ExperimentRunnerOutput(
                    run=run,
                    launched=False,
                    message=f"Failed to launch: {e}",
                )

            self._processes[input.experiment_id] = proc
            run.pid = proc.pid

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=input.timeout_seconds,
                )
            except TimeoutError:
                proc.kill()
                run.killed = True
                run.status = ExperimentStatus.TIMEOUT
                run.end_time = datetime.now()
                run.duration_seconds = (
                    run.end_time - run.start_time
                ).total_seconds()
                run.status_history.append(
                    StatusTransition(
                        from_status=ExperimentStatus.RUNNING,
                        to_status=ExperimentStatus.TIMEOUT,
                        reason="timeout",
                    )
                )
                self._processes.pop(input.experiment_id, None)
                return ExperimentRunnerOutput(
                    run=run,
                    launched=True,
                    message="Experiment timed out",
                )

            run.exit_code = proc.returncode
            run.stdout = self._decode(stdout_bytes, input.max_output_bytes)
            run.stderr = self._decode(stderr_bytes, input.max_output_bytes)
            run.end_time = datetime.now()
            run.duration_seconds = (
                run.end_time - run.start_time
            ).total_seconds()

            if proc.returncode == 0:
                run.status = ExperimentStatus.COMPLETED
                reason = "completed"
            else:
                run.status = ExperimentStatus.FAILED
                run.error_message = (
                    f"Process exited with code {proc.returncode}"
                )
                reason = f"exit_code_{proc.returncode}"

            run.status_history.append(
                StatusTransition(
                    from_status=ExperimentStatus.RUNNING,
                    to_status=run.status,
                    reason=reason,
                )
            )
            self._processes.pop(input.experiment_id, None)

            return ExperimentRunnerOutput(
                run=run,
                launched=True,
                message=f"Experiment {run.status.value}",
            )
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Experiment runner failed: {e}", input, e)

    async def cancel(self, experiment_id: str) -> bool:
        """Cancel a running experiment by ID."""
        proc = self._processes.get(experiment_id)
        if proc is None:
            return False
        try:
            proc.kill()
            return True
        except ProcessLookupError:
            return False
        except Exception:
            return False

    def get_process(self, experiment_id: str) -> asyncio.subprocess.Process | None:
        """Get the process handle for an experiment."""
        return self._processes.get(experiment_id)

    @staticmethod
    def _decode(data: bytes, max_bytes: int) -> str:
        """Decode bytes to string, capping size."""
        if not data:
            return ""
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n...[truncated]"
        return text
