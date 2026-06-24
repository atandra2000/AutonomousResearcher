"""Phase 13 - Test agent.

Specialized agent that executes the test suite, parses failures, and
provides structured feedback. Uses :class:`TerminalTool` for safe
command execution with an allowlist.

Interface: ``async execute(ctx: SharedTaskContext) -> dict``
"""

from __future__ import annotations

import re
from typing import Any

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.llm import LLMProvider
from research_engineer.models.delegation import (
    Feedback,
    FeedbackType,
    SharedTaskContext,
)
from research_engineer.tools.terminal import TerminalInput, TerminalTool


class TestAgent:
    """Executes tests and parses failures.

    Reads the test command from ``ctx`` (or config), runs it via
    :class:`TerminalTool`, and writes ``ctx.test_exit_code``,
    ``ctx.test_stdout``, ``ctx.test_stderr``, and
    ``ctx.test_failures``. Returns a :class:`Feedback` dict.
    """

    def __init__(
        self,
        terminal_tool: TerminalTool | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "TestAgent"
        self.terminal = terminal_tool or TerminalTool()
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the test suite and write results to ``ctx``."""
        test_command = kwargs.get(
            "test_command", "uv run pytest"
        )
        timeout = kwargs.get("timeout_seconds", 600)
        out = await self.terminal.execute(
            TerminalInput(
                operation="run_command",
                repo_path=ctx.repo_path,
                command=test_command,
                timeout_seconds=timeout,
                dry_run=False,
            )
        )
        ctx.test_exit_code = out.exit_code
        ctx.test_stdout = out.stdout
        ctx.test_stderr = out.stderr
        failures = self._parse_failures(out.stdout, out.stderr)
        ctx.test_failures = failures
        passed = out.success and out.exit_code == 0
        feedback = Feedback(
            feedback_type=(
                FeedbackType.TEST_PASS if passed else FeedbackType.TEST_FAILURE
            ),
            approved=passed,
            issues=failures,
            summary=(
                f"Tests passed (exit={out.exit_code})"
                if passed
                else f"Tests failed (exit={out.exit_code}): {len(failures)} failure(s)"
            ),
            repair_needed=not passed,
        )
        return {
            "summary": feedback.summary,
            "feedback": feedback.model_dump(),
            "approved": feedback.approved,
            "exit_code": out.exit_code,
        }

    @staticmethod
    def _parse_failures(stdout: str, stderr: str) -> list[str]:
        """Parse test failure messages from pytest output."""
        failures: list[str] = []
        combined = stdout + "\n" + stderr
        # Pytest FAILED lines: "FAILED tests/test_foo.py::test_bar - assert ..."
        for match in re.finditer(
            r"FAILED\s+(\S+)\s*-\s*(.+)", combined
        ):
            failures.append(f"{match.group(1)}: {match.group(2).strip()}")
        # ERROR lines.
        for match in re.finditer(
            r"ERROR\s+(\S+)\s*-\s*(.+)", combined
        ):
            failures.append(f"ERROR {match.group(1)}: {match.group(2).strip()}")
        # AssertionError fallback.
        if not failures:
            for match in re.finditer(
                r"(AssertionError[^\n]*)", combined
            ):
                failures.append(match.group(1).strip()[:200])
        return failures[:20]


__all__ = ["TestAgent"]
