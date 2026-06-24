"""Tests for Phase 14 - Autonomous self-repair framework."""

from __future__ import annotations

from typing import Any

import pytest

from research_engineer.llm import LLMProvider, LLMRequest, LLMResponse
from research_engineer.models.delegation import (
    AgentCapability,
    AgentRole,
    SharedTaskContext,
)
from research_engineer.models.repair import (
    FailureCategory,
    FailureReport,
    FailureSeverity,
    RepairActionType,
    RepairCycle,
    RepairConfig,
    RepairOutcome,
    RepairResult,
    RepairStrategy,
    RepairTerminationReason,
)
from research_engineer.tools.terminal import TerminalOutput


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, default_model: str = "fake-model") -> None:
        self.default_model = default_model
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            content="Root cause: missing import.",
            model=self.default_model,
            provider=self.name,
        )


class _FakeTerminal:
    async def execute(self, input: Any) -> TerminalOutput:
        return TerminalOutput(
            operation=input.operation,
            success=True,
            content="diff --git a/f.py b/f.py\n+fixed code",
            stdout="",
        )


class _StubAgent:
    """Agent that returns a canned result."""

    def __init__(self, name: str = "Stub", result: dict | None = None) -> None:
        self.agent_name = name
        self._result = result or {"summary": "done", "approved": True}

    async def execute(self, ctx, **kwargs):
        return self._result


# ---------------------------------------------------------------------------
# Repair models
# ---------------------------------------------------------------------------


class TestRepairModels:
    def test_failure_category_values(self):
        assert FailureCategory.SYNTAX_ERROR == "syntax_error"
        assert FailureCategory.IMPORT_ERROR == "import_error"
        assert FailureCategory.TEST_FAILURE == "test_failure"

    def test_failure_severity_values(self):
        assert FailureSeverity.CRITICAL == "critical"
        assert FailureSeverity.LOW == "low"

    def test_repair_action_type_values(self):
        assert RepairActionType.FIX_SYNTAX == "fix_syntax"
        assert RepairActionType.REGENERATE == "regenerate"

    def test_repair_outcome_values(self):
        assert RepairOutcome.PENDING == "pending"
        assert RepairOutcome.SUCCESS == "success"
        assert RepairOutcome.STAGNATED == "stagnated"

    def test_termination_reason_values(self):
        assert RepairTerminationReason.SUCCESS == "success"
        assert RepairTerminationReason.BUDGET_EXHAUSTED == "budget_exhausted"
        assert RepairTerminationReason.STAGNATION == "stagnation"

    def test_failure_report(self):
        r = FailureReport(
            report_id="fr_1",
            category=FailureCategory.SYNTAX_ERROR,
            severity=FailureSeverity.HIGH,
            root_cause="Missing colon",
            evidence=["SyntaxError: invalid syntax"],
            affected_files=["src/foo.py"],
        )
        assert r.category == FailureCategory.SYNTAX_ERROR
        assert len(r.evidence) == 1

    def test_repair_strategy(self):
        s = RepairStrategy(
            strategy_id="rs_1",
            action_type=RepairActionType.FIX_SYNTAX,
            description="Fix the syntax",
            priority=9,
            confidence=0.9,
        )
        assert s.priority == 9
        assert s.confidence == 0.9

    def test_repair_config_defaults(self):
        c = RepairConfig()
        assert c.max_iterations == 3
        assert c.stagnation_threshold == 3
        assert c.require_review is True
        assert c.require_tests is True

    def test_repair_cycle(self):
        c = RepairCycle(cycle_number=0)
        assert c.outcome == RepairOutcome.PENDING
        assert c.review_passed is False
        assert c.test_passed is False

    def test_repair_result(self):
        r = RepairResult(total_cycles=2, successful=True)
        assert r.successful is True
        assert r.termination_reason == RepairTerminationReason.BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------


class TestFailureAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_syntax_error(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["SyntaxError: invalid syntax in foo.py"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.SYNTAX_ERROR

    @pytest.mark.asyncio
    async def test_analyze_import_error(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["ImportError: No module named 'foo'"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.IMPORT_ERROR

    @pytest.mark.asyncio
    async def test_analyze_assertion_error(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["AssertionError: expected 5, got 3"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.ASSERTION_ERROR

    @pytest.mark.asyncio
    async def test_analyze_review_rejection(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.review_issues = ["Bare except clause", "Missing docstring"]
        ctx.test_exit_code = None
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.REVIEW_REJECTION

    @pytest.mark.asyncio
    async def test_analyze_extracts_affected_files(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["FAILED tests/test_foo.py::test_bar - assert False"]
        ctx.test_exit_code = 1
        ctx.diff = "+++ b/src/foo.py\n--- a/src/foo.py\n"
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert "tests/test_foo.py" in report.affected_files or "src/foo.py" in report.affected_files

    @pytest.mark.asyncio
    async def test_analyze_extracts_symbols(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["FAILED tests/test_foo.py::test_bar - assert False"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert "test_bar" in report.affected_symbols

    @pytest.mark.asyncio
    async def test_analyze_unknown_failure(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        agent = FailureAnalyzer(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["Something went wrong but no pattern"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.TEST_FAILURE

    @pytest.mark.asyncio
    async def test_analyze_with_llm_augmentation(self, tmp_path):
        from research_engineer.agents import FailureAnalyzer

        provider = _FakeProvider()
        agent = FailureAnalyzer(llm=provider)
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["SyntaxError: invalid syntax"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        report = FailureReport.model_validate(result["failure_report"])
        assert report.category == FailureCategory.SYNTAX_ERROR
        assert len(provider.calls) > 0


# ---------------------------------------------------------------------------
# RepairStrategist
# ---------------------------------------------------------------------------


class TestRepairStrategist:
    @pytest.mark.asyncio
    async def test_generate_syntax_fix_strategy(self, tmp_path):
        from research_engineer.agents import RepairStrategist

        agent = RepairStrategist(llm=None)
        agent.llm_provider = None
        report = FailureReport(
            report_id="fr_1",
            category=FailureCategory.SYNTAX_ERROR,
            affected_files=["src/foo.py"],
        )
        result = await agent.execute(
            SharedTaskContext(goal="g", repo_path=str(tmp_path)),
            failure_report=report.model_dump(),
        )
        strategies = [
            RepairStrategy.model_validate(s)
            for s in result["strategies"]
        ]
        assert len(strategies) >= 1
        assert any(s.action_type == RepairActionType.FIX_SYNTAX for s in strategies)

    @pytest.mark.asyncio
    async def test_generate_import_fix_strategy(self, tmp_path):
        from research_engineer.agents import RepairStrategist

        agent = RepairStrategist(llm=None)
        agent.llm_provider = None
        report = FailureReport(
            report_id="fr_1",
            category=FailureCategory.IMPORT_ERROR,
            affected_files=["src/foo.py"],
        )
        result = await agent.execute(
            SharedTaskContext(goal="g", repo_path=str(tmp_path)),
            failure_report=report.model_dump(),
        )
        strategies = [
            RepairStrategy.model_validate(s)
            for s in result["strategies"]
        ]
        assert any(s.action_type == RepairActionType.FIX_IMPORT for s in strategies)

    @pytest.mark.asyncio
    async def test_strategies_sorted_by_priority(self, tmp_path):
        from research_engineer.agents import RepairStrategist

        agent = RepairStrategist(llm=None)
        agent.llm_provider = None
        report = FailureReport(
            report_id="fr_1",
            category=FailureCategory.SYNTAX_ERROR,
            affected_files=["src/foo.py"],
        )
        result = await agent.execute(
            SharedTaskContext(goal="g", repo_path=str(tmp_path)),
            failure_report=report.model_dump(),
        )
        strategies = [
            RepairStrategy.model_validate(s)
            for s in result["strategies"]
        ]
        priorities = [s.priority for s in strategies]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_generates_from_ctx_without_report(self, tmp_path):
        from research_engineer.agents import RepairStrategist

        agent = RepairStrategist(llm=None)
        agent.llm_provider = None
        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.test_failures = ["FAILED test_foo.py::test_bar"]
        ctx.test_exit_code = 1
        result = await agent.execute(ctx)
        assert len(result["strategies"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_failure_generates_regenerate(self, tmp_path):
        from research_engineer.agents import RepairStrategist

        agent = RepairStrategist(llm=None)
        agent.llm_provider = None
        report = FailureReport(
            report_id="fr_1",
            category=FailureCategory.UNKNOWN,
            affected_files=["src/foo.py"],
        )
        result = await agent.execute(
            SharedTaskContext(goal="g", repo_path=str(tmp_path)),
            failure_report=report.model_dump(),
        )
        strategies = [
            RepairStrategy.model_validate(s)
            for s in result["strategies"]
        ]
        assert any(
            s.action_type in (RepairActionType.REGENERATE, RepairActionType.MANUAL_REVIEW)
            for s in strategies
        )


# ---------------------------------------------------------------------------
# SelfRepairFramework
# ---------------------------------------------------------------------------


class TestSelfRepairFramework:
    @pytest.mark.asyncio
    async def test_repair_succeeds_on_first_cycle(self, tmp_path):
        from research_engineer.agents.delegation import DelegationFramework
        from research_engineer.agents.self_repair import SelfRepairFramework

        fw = DelegationFramework()
        # Reviewer approves.
        fw.register(
            _StubAgent("Reviewer", {"approved": True, "feedback": {"approved": True}}),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        # Tester passes.
        class _PassTester:
            agent_name = "Tester"
            async def execute(self, ctx, **kwargs):
                ctx.test_exit_code = 0
                return {"approved": True, "exit_code": 0}
        fw.register(_PassTester(), AgentRole.TESTER, [AgentCapability.TEST_EXECUTION])
        # Coder (repair).
        fw.register(_StubAgent("Coder"), AgentRole.CODER, [AgentCapability.REPAIR])

        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.diff = "+some code"
        ctx.test_failures = ["FAILED test.py::test_x"]
        ctx.test_exit_code = 1

        sr = SelfRepairFramework(
            delegation=fw,
            terminal=_FakeTerminal(),  # type: ignore[arg-type]
            config=RepairConfig(max_iterations=3, require_tests=True),
        )
        result = await sr.run(ctx, test_command="pytest")
        assert result.successful is True
        assert result.termination_reason == RepairTerminationReason.SUCCESS
        assert result.total_cycles >= 1

    @pytest.mark.asyncio
    async def test_repair_budget_exhausted(self, tmp_path):
        from research_engineer.agents.delegation import DelegationFramework
        from research_engineer.agents.self_repair import SelfRepairFramework

        fw = DelegationFramework()
        # Reviewer always rejects.
        fw.register(
            _StubAgent("Reviewer", {"approved": False, "feedback": {"approved": False}}),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        # Tester always fails.
        class _FailTester:
            agent_name = "Tester"
            async def execute(self, ctx, **kwargs):
                ctx.test_exit_code = 1
                ctx.test_failures = ["FAILED test.py::test_x - assert False"]
                return {"approved": False, "exit_code": 1}
        fw.register(_FailTester(), AgentRole.TESTER, [AgentCapability.TEST_EXECUTION])
        fw.register(_StubAgent("Coder"), AgentRole.CODER, [AgentCapability.REPAIR])

        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.diff = "+bad code"
        ctx.test_failures = ["FAILED test.py::test_x"]
        ctx.test_exit_code = 1

        sr = SelfRepairFramework(
            delegation=fw,
            terminal=_FakeTerminal(),  # type: ignore[arg-type]
            config=RepairConfig(max_iterations=2, require_tests=True),
        )
        result = await sr.run(ctx, test_command="pytest")
        assert result.successful is False
        assert result.termination_reason == RepairTerminationReason.BUDGET_EXHAUSTED
        assert result.total_cycles == 2

    @pytest.mark.asyncio
    async def test_stagnation_detection(self, tmp_path):
        from research_engineer.agents.delegation import DelegationFramework
        from research_engineer.agents.self_repair import SelfRepairFramework

        fw = DelegationFramework()
        fw.register(
            _StubAgent("Reviewer", {"approved": False, "feedback": {"approved": False}}),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        class _FailTester:
            agent_name = "Tester"
            async def execute(self, ctx, **kwargs):
                ctx.test_exit_code = 1
                ctx.test_failures = ["SyntaxError: invalid syntax"]
                return {"approved": False, "exit_code": 1}
        fw.register(_FailTester(), AgentRole.TESTER, [AgentCapability.TEST_EXECUTION])
        fw.register(_StubAgent("Coder"), AgentRole.CODER, [AgentCapability.REPAIR])

        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.diff = "+bad"
        ctx.test_failures = ["SyntaxError: invalid syntax"]
        ctx.test_exit_code = 1

        sr = SelfRepairFramework(
            delegation=fw,
            terminal=_FakeTerminal(),  # type: ignore[arg-type]
            config=RepairConfig(
                max_iterations=10, stagnation_threshold=3, require_tests=True,
            ),
        )
        result = await sr.run(ctx, test_command="pytest")
        assert result.termination_reason == RepairTerminationReason.STAGNATION
        assert result.total_cycles <= 3

    @pytest.mark.asyncio
    async def test_repair_without_tests(self, tmp_path):
        from research_engineer.agents.delegation import DelegationFramework
        from research_engineer.agents.self_repair import SelfRepairFramework

        fw = DelegationFramework()
        fw.register(
            _StubAgent("Reviewer", {"approved": True, "feedback": {"approved": True}}),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        fw.register(_StubAgent("Coder"), AgentRole.CODER, [AgentCapability.REPAIR])

        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.diff = "+code"
        ctx.review_issues = ["some issue"]

        sr = SelfRepairFramework(
            delegation=fw,
            terminal=_FakeTerminal(),  # type: ignore[arg-type]
            config=RepairConfig(max_iterations=2, require_tests=False),
        )
        result = await sr.run(ctx, test_command="pytest")
        assert result.successful is True

    @pytest.mark.asyncio
    async def test_repair_tracks_strategies_tried(self, tmp_path):
        from research_engineer.agents.delegation import DelegationFramework
        from research_engineer.agents.self_repair import SelfRepairFramework

        fw = DelegationFramework()
        fw.register(
            _StubAgent("Reviewer", {"approved": False, "feedback": {"approved": False}}),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        class _FailTester:
            agent_name = "Tester"
            async def execute(self, ctx, **kwargs):
                ctx.test_exit_code = 1
                ctx.test_failures = ["ImportError: no module"]
                return {"approved": False, "exit_code": 1}
        fw.register(_FailTester(), AgentRole.TESTER, [AgentCapability.TEST_EXECUTION])
        fw.register(_StubAgent("Coder"), AgentRole.CODER, [AgentCapability.REPAIR])

        ctx = SharedTaskContext(goal="g", repo_path=str(tmp_path))
        ctx.diff = "+code"
        ctx.test_failures = ["ImportError: no module"]
        ctx.test_exit_code = 1

        sr = SelfRepairFramework(
            delegation=fw,
            terminal=_FakeTerminal(),  # type: ignore[arg-type]
            config=RepairConfig(max_iterations=2, require_tests=True),
        )
        result = await sr.run(ctx, test_command="pytest")
        assert result.strategies_tried > 0
        assert result.unique_failures > 0


# ---------------------------------------------------------------------------
# TaskAgent integration
# ---------------------------------------------------------------------------


class TestTaskAgentSelfRepairIntegration:
    def test_task_agent_accepts_self_repair_components(self):
        from research_engineer.agents import (
            FailureAnalyzer,
            RepairStrategist,
            TaskAgent,
        )

        agent = TaskAgent(
            failure_analyzer=FailureAnalyzer(llm=None),
            repair_strategist=RepairStrategist(llm=None),
        )
        assert agent._failure_analyzer is not None
        assert agent._repair_strategist is not None

    def test_new_agents_in_router(self):
        from research_engineer.llm import get_router

        r = get_router()
        assert "FailureAnalyzer" in r.KNOWN_AGENTS
        assert "RepairStrategist" in r.KNOWN_AGENTS

    @pytest.mark.asyncio
    async def test_delegated_run_with_self_repair(self, tmp_path):
        """Test that delegation mode works with self-repair enabled."""
        from research_engineer.agents import TaskAgent
        from research_engineer.models.task import TaskConfig

        class _FakeRepo:
            async def analyze(self, *a, **k):
                return {"repository_name": "r", "project_type": "py",
                        "generated_files": []}

        class _FakeCoding:
            async def implement(self, *a, **k):
                from research_engineer.agents.coding_agent import CodingAgentResult
                return CodingAgentResult(
                    implementation_id="impl", request_id="req",
                    repo_path=".", task_description="t", status="completed",
                    patches_generated=1, tests_generated=0, review_status="ok",
                    generated_files=[], implementation_time_seconds=0.1,
                    output_dir=".",
                )

        class _FakeTerm:
            async def execute(self, input):
                return TerminalOutput(
                    operation=input.operation, success=True,
                    content="diff", stdout="", exit_code=0,
                )

        agent = TaskAgent(
            repository_agent=_FakeRepo(),  # type: ignore[arg-type]
            coding_agent=_FakeCoding(),  # type: ignore[arg-type]
            terminal_tool=_FakeTerm(),  # type: ignore[arg-type]
            llm=_FakeProvider(),
        )
        config = TaskConfig(
            goal="test", repo_path=str(tmp_path),
            delegate=True, max_repair_iterations=1,
            stream=False, run_tests=True,
            output_dir=str(tmp_path / "out"),
        )
        result = await agent.run("test", str(tmp_path), config=config)
        assert result.delegated is True