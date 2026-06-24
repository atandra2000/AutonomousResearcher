"""Tests for the TaskAgent (Phase 11)."""

from __future__ import annotations

from typing import Any

import pytest

from research_engineer.llm import LLMProvider, LLMRequest, LLMResponse
from research_engineer.models.task import TaskConfig, TaskStatus
from research_engineer.tools.terminal import TerminalOutput


class _FakeProvider(LLMProvider):
    """No-network LLM provider for deterministic tests."""

    name = "fake"

    def __init__(self, default_model: str = "fake-model") -> None:
        self.default_model = default_model
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            content="1. Locate module.\n2. Implement change.\n3. Run tests.",
            model=self.default_model,
            provider=self.name,
        )


class _FakeTerminalTool:
    """Fake terminal tool that returns canned outputs."""

    async def execute(self, input: Any) -> TerminalOutput:
        op = input.operation
        if op == "git_diff":
            return TerminalOutput(
                operation="git_diff",
                success=True,
                content="diff --git a/foo.py b/foo.py\n+new code",
            )
        if op == "run_command":
            return TerminalOutput(
                operation="run_command",
                success=True,
                exit_code=0,
                stdout="All tests passed",
                stderr="",
            )
        return TerminalOutput(operation=op, success=True)


class _FakeRepoAgent:
    """Fake repository agent returning a minimal summary."""

    async def analyze(self, repo_path: str, output_dir: str = "output", enable_llm: bool = False) -> dict:
        return {
            "repository_name": "test-repo",
            "project_type": "python",
            "architecture_summary": "Simple package",
            "important_files": [],
            "generated_files": [],
            "analysis_time_seconds": 0.1,
            "output_dir": output_dir,
        }


class _FakeCodingAgent:
    """Fake coding agent returning a minimal result."""

    async def implement(self, task_description: str, repo_path: str, paper_input: str | None = None, output_dir: str = "output", **kwargs: Any) -> Any:
        from research_engineer.agents.coding_agent import CodingAgentResult

        return CodingAgentResult(
            implementation_id="impl_test_123",
            request_id="req_test_123",
            paper_id=None,
            repo_path=repo_path,
            task_description=task_description,
            status="completed",
            patches_generated=2,
            tests_generated=3,
            review_status="approved",
            generated_files=[f"{output_dir}/patch.md"],
            implementation_time_seconds=0.5,
            output_dir=output_dir,
        )


class TestTaskAgentConstruction:
    def test_task_agent_resolves_llm_provider(self):
        from research_engineer.agents import TaskAgent

        agent = TaskAgent()
        assert agent.agent_name == "TaskAgent"
        assert hasattr(agent, "llm_provider")

    def test_task_agent_explicit_provider_wins(self):
        from research_engineer.agents import TaskAgent

        custom = _FakeProvider(default_model="custom")
        agent = TaskAgent(llm=custom)
        assert agent.llm_provider is custom

    def test_task_agent_known_in_router(self):
        from research_engineer.llm import get_router

        assert "TaskAgent" in get_router().KNOWN_AGENTS


class TestTaskAgentRun:
    @pytest.mark.asyncio
    async def test_run_completes_all_steps(self, tmp_path):
        from research_engineer.agents import TaskAgent

        agent = TaskAgent(
            repository_agent=_FakeRepoAgent(),  # type: ignore[arg-type]
            coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
            terminal_tool=_FakeTerminalTool(),  # type: ignore[arg-type]
            llm=_FakeProvider(),
        )
        config = TaskConfig(
            goal="Add a utility function",
            repo_path=str(tmp_path),
            run_tests=True,
            stream=False,
            output_dir=str(tmp_path / "out"),
        )
        result = await agent.run("Add a utility function", str(tmp_path), config=config)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.steps) == 5
        assert result.patches_generated == 2
        assert "new code" in result.diff
        assert result.test_exit_code == 0
        assert "All tests passed" in result.test_stdout
        assert result.implementation_id == "impl_test_123"

    @pytest.mark.asyncio
    async def test_run_without_tests_skips_test_step(self, tmp_path):
        from research_engineer.agents import TaskAgent

        agent = TaskAgent(
            repository_agent=_FakeRepoAgent(),  # type: ignore[arg-type]
            coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
            terminal_tool=_FakeTerminalTool(),  # type: ignore[arg-type]
            llm=_FakeProvider(),
        )
        config = TaskConfig(
            goal="Refactor module",
            repo_path=str(tmp_path),
            run_tests=False,
            stream=False,
            output_dir=str(tmp_path / "out"),
        )
        result = await agent.run("Refactor module", str(tmp_path), config=config)

        assert result.status == TaskStatus.COMPLETED
        # 4 steps: analyze, plan, implement, diff (no test)
        assert len(result.steps) == 4
        assert result.test_exit_code is None

    @pytest.mark.asyncio
    async def test_run_repo_failure_fails_gracefully(self, tmp_path):
        from research_engineer.agents import TaskAgent

        class _BadRepoAgent:
            async def analyze(self, *a: Any, **k: Any) -> dict:
                raise RuntimeError("repo scan failed")

        agent = TaskAgent(
            repository_agent=_BadRepoAgent(),  # type: ignore[arg-type]
            coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
            terminal_tool=_FakeTerminalTool(),  # type: ignore[arg-type]
            llm=_FakeProvider(),
        )
        config = TaskConfig(
            goal="x", repo_path=str(tmp_path), stream=False
        )
        result = await agent.run("x", str(tmp_path), config=config)

        assert result.status == TaskStatus.FAILED
        assert len(result.steps) == 1
        assert result.steps[0].status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_run_falls_back_to_rule_based_plan_without_llm(self, tmp_path):
        from research_engineer.agents import TaskAgent

        agent = TaskAgent(
            repository_agent=_FakeRepoAgent(),  # type: ignore[arg-type]
            coding_agent=_FakeCodingAgent(),  # type: ignore[arg-type]
            terminal_tool=_FakeTerminalTool(),  # type: ignore[arg-type]
            llm=None,
        )
        # Force llm_provider to None
        agent.llm_provider = None
        config = TaskConfig(
            goal="do something", repo_path=str(tmp_path), stream=False
        )
        result = await agent.run("do something", str(tmp_path), config=config)

        assert result.status == TaskStatus.COMPLETED
        plan_step = result.steps[1]
        assert "Implementation Plan" in plan_step.summary or "plan" in plan_step.summary.lower()


class TestTaskModels:
    def test_task_config_defaults(self):
        c = TaskConfig(goal="g", repo_path=".")
        assert c.run_tests is False
        assert c.dry_run is True
        assert c.stream is True
        assert c.test_command == "uv run pytest"

    def test_task_result_serialization(self):
        from research_engineer.models.task import TaskResult, TaskStep, TaskStepType

        r = TaskResult(
            task_id="task_abc",
            goal="g",
            repo_path=".",
            status=TaskStatus.COMPLETED,
            steps=[
                TaskStep(
                    step_type=TaskStepType.ANALYZE_REPO,
                    status=TaskStatus.COMPLETED,
                )
            ],
        )
        d = r.model_dump()
        assert d["status"] == "completed"
        assert d["task_id"] == "task_abc"
        assert len(d["steps"]) == 1

    def test_new_task_id_format(self):
        from research_engineer.models.task import new_task_id

        tid = new_task_id()
        assert tid.startswith("task_")
        assert len(tid) > 10