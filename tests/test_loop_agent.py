"""Tests for Phase 9 - Research Loop Agent."""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_engineer.agents.research_loop_agent import ResearchLoopAgent
from research_engineer.models.loop import (
    ApprovalGate,
    ApprovalRequest,
    IterationPhase,
    LoopConfig,
    LoopIteration,
    LoopStatus,
    NextAction,
    StoppingCondition,
)
from research_engineer.tools.loop_storage import LoopStorageTool
from research_engineer.tools.report_generator import ReportGeneratorTool
from research_engineer.tools.stopping_condition import (
    StoppingConditionChecker,
)


# --- Helpers ---


def _config(
    max_iterations: int = 3,
    target_metric_name: str | None = "loss",
    target_metric_value: float | None = 0.01,
    dry_run: bool = True,
    **kwargs,
) -> LoopConfig:
    return LoopConfig(
        goal="improve training",
        repo_path="/repo",
        max_iterations=max_iterations,
        target_metric_name=target_metric_name,
        target_metric_value=target_metric_value,
        dry_run=dry_run,
        **kwargs,
    )


def _make_mock_agent(
    return_value=None,
    raise_exc: Exception | None = None,
):
    agent = MagicMock()
    method = AsyncMock()
    if raise_exc:
        method.side_effect = raise_exc
    elif return_value is not None:
        method.return_value = return_value
    return agent, method


# --- Fixtures ---


@pytest.fixture
def mock_memory_agent():
    agent = MagicMock()
    agent.storage = AsyncMock()
    agent.storage.store_relationship = AsyncMock()
    agent.graph = MagicMock()
    agent.graph.add_node = MagicMock()
    agent.graph.add_relationship = MagicMock()
    agent.store_insight = AsyncMock(return_value="mem_insight")
    agent.store_success = AsyncMock(return_value="mem_success")
    agent.store_failure = AsyncMock(return_value="mem_failure")
    agent.store_decision = AsyncMock(return_value="mem_decision")
    agent.get_context = AsyncMock(return_value=[])
    agent.search = AsyncMock(return_value=[])
    return agent


@pytest.fixture
def mock_literature_agent():
    agent = MagicMock()
    agent.discover = AsyncMock()
    agent.search_papers = AsyncMock(return_value=MagicMock(papers=[]))
    return agent


@pytest.fixture
def mock_planner_agent():
    agent = MagicMock()
    result = MagicMock()
    result.plan_id = "plan_test"
    result.paper_id = "2503.12345"
    agent.plan = AsyncMock(return_value=result)
    return agent


@pytest.fixture
def mock_coding_agent():
    agent = MagicMock()
    result = MagicMock()
    result.implementation_id = "impl_test"
    agent.implement = AsyncMock(return_value=result)
    return agent


@pytest.fixture
def mock_experiment_agent():
    agent = MagicMock()
    result = MagicMock()
    result.experiment_id = "exp_test"
    result.metrics = MagicMock(summary_metrics={"loss": 0.5})
    record = MagicMock()
    record.experiment_id = "exp_test"
    record.metrics = {"loss": 0.5}
    record.metric_series = []
    record.status = "completed"
    record.experiment_type = "training"
    record.command = ["python", "train.py"]
    record.repo_path = "/repo"
    record.start_time = datetime.now()
    record.duration_seconds = 10.0
    record.failure_mode = None
    record.failure_severity = "none"
    record.exit_code = 0
    record.artifacts = []
    record.paper_id = "2503.12345"
    record.plan_id = "plan_test"
    record.patch_id = None
    record.implementation_id = "impl_test"
    record.end_time = datetime.now()
    record.root_cause = None
    record.lessons_learned = []
    record.output_dir = None
    record.memory_id = None
    record.tags = []
    record.notes = ""
    record.created_at = datetime.now()
    record.updated_at = None
    agent.run = AsyncMock(return_value=result)
    agent.get_experiment = AsyncMock(return_value=record)
    return agent


@pytest.fixture
def mock_evaluation_agent():
    agent = MagicMock()
    result = MagicMock()
    result.evaluation_id = "eval_test"
    result.is_positive = MagicMock(return_value=True)
    result.record = MagicMock()
    result.record.summary_metrics = {"loss": 0.5}
    agent.analyze = AsyncMock(return_value=result)
    return agent


@pytest.fixture
def loop_agent(
    mock_memory_agent,
    mock_literature_agent,
    mock_planner_agent,
    mock_coding_agent,
    mock_experiment_agent,
    mock_evaluation_agent,
    tmp_path,
):
    return ResearchLoopAgent(
        memory_agent=mock_memory_agent,
        literature_agent=mock_literature_agent,
        planner_agent=mock_planner_agent,
        coding_agent=mock_coding_agent,
        experiment_agent=mock_experiment_agent,
        evaluation_agent=mock_evaluation_agent,
        storage_tool=LoopStorageTool(db_path=str(tmp_path / "test.db")),
        stopping_checker=StoppingConditionChecker(),
        report_generator=ReportGeneratorTool(),
        config=_config(max_iterations=3),
    )


# --- Tests ---


class TestResearchLoopAgentInit:
    def test_defaults(self):
        agent = ResearchLoopAgent()
        assert agent.config is not None
        assert agent.storage is not None
        assert agent.stopping_checker is not None
        assert agent.report_gen is not None

    def test_with_deps(self, loop_agent):
        assert loop_agent.memory is not None
        assert loop_agent.literature is not None
        assert loop_agent.planner is not None


class TestRunSingleIteration:
    @pytest.mark.asyncio
    async def test_one_iteration_stops_max(
        self, loop_agent, mock_literature_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1,
            target_metric_value=0.001,
            output_dir=str(tmp_path / "loops"),
        )
        result = await loop_agent.run(
            "test goal", "/repo", config=cfg
        )
        assert result.loop_id.startswith("loop_")
        assert result.iteration_count == 1
        assert result.status == LoopStatus.STOPPED
        assert (
            result.stopping_condition
            == StoppingCondition.MAX_ITERATIONS_REACHED
        )

    @pytest.mark.asyncio
    async def test_loop_result_has_generated_files(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        result = await loop_agent.run(
            "test goal", "/repo", config=cfg
        )
        if result.generated_files:
            for f in result.generated_files:
                assert Path(f).exists()


class TestRunMultiIteration:
    @pytest.mark.asyncio
    async def test_target_achieved(
        self,
        loop_agent,
        mock_experiment_agent,
        mock_evaluation_agent,
        tmp_path,
    ):
        exp_result = MagicMock()
        exp_result.experiment_id = "exp_target"
        exp_result.metrics = MagicMock(
            summary_metrics={"loss": 0.001}
        )
        mock_experiment_agent.run.return_value = exp_result
        record = MagicMock()
        record.metrics = {"loss": 0.001}
        record.metric_series = []
        record.status = "completed"
        record.experiment_type = "training"
        record.command = ["python", "train.py"]
        record.repo_path = "/repo"
        record.start_time = datetime.now()
        record.duration_seconds = 10.0
        record.failure_mode = None
        record.failure_severity = "none"
        record.exit_code = 0
        record.artifacts = []
        record.paper_id = None
        record.plan_id = None
        record.patch_id = None
        record.implementation_id = None
        record.end_time = datetime.now()
        record.root_cause = None
        record.lessons_learned = []
        record.output_dir = None
        record.memory_id = None
        record.tags = []
        record.notes = ""
        record.created_at = datetime.now()
        record.updated_at = None
        mock_experiment_agent.get_experiment.return_value = record
        eval_result = MagicMock()
        eval_result.evaluation_id = "eval_target"
        eval_result.is_positive = MagicMock(return_value=True)
        eval_result.record = MagicMock()
        eval_result.record.summary_metrics = {"loss": 0.001}
        mock_evaluation_agent.analyze.return_value = eval_result

        cfg = _config(
            max_iterations=5,
            target_metric_value=0.01,
            output_dir=str(tmp_path / "loops"),
        )
        result = await loop_agent.run(
            "achieve target", "/repo", config=cfg
        )
        assert result.status == LoopStatus.STOPPED
        assert (
            result.stopping_condition
            == StoppingCondition.TARGET_ACHIEVED
        )


class TestNoImprovementStop:
    @pytest.mark.asyncio
    async def test_no_improvement(
        self,
        loop_agent,
        mock_experiment_agent,
        mock_evaluation_agent,
        tmp_path,
    ):
        exp_result = MagicMock()
        exp_result.experiment_id = "exp_noimp"
        exp_result.metrics = MagicMock(
            summary_metrics={"loss": 0.5}
        )
        mock_experiment_agent.run.return_value = exp_result
        record = MagicMock()
        record.metrics = {"loss": 0.5}
        record.metric_series = []
        record.status = "completed"
        record.experiment_type = "training"
        record.command = ["python", "train.py"]
        record.repo_path = "/repo"
        record.start_time = datetime.now()
        record.duration_seconds = 10.0
        record.failure_mode = None
        record.failure_severity = "none"
        record.exit_code = 0
        record.artifacts = []
        record.paper_id = None
        record.plan_id = None
        record.patch_id = None
        record.implementation_id = None
        record.end_time = datetime.now()
        record.root_cause = None
        record.lessons_learned = []
        record.output_dir = None
        record.memory_id = None
        record.tags = []
        record.notes = ""
        record.created_at = datetime.now()
        record.updated_at = None
        mock_experiment_agent.get_experiment.return_value = record
        eval_result = MagicMock()
        eval_result.evaluation_id = "eval_noimp"
        eval_result.is_positive = MagicMock(return_value=True)
        eval_result.record = MagicMock()
        eval_result.record.summary_metrics = {"loss": 0.5}
        mock_evaluation_agent.analyze.return_value = eval_result

        cfg = _config(
            max_iterations=10,
            target_metric_value=0.01,
            stagnation_window=3,
            output_dir=str(tmp_path / "loops"),
        )
        result = await loop_agent.run(
            "no improvement", "/repo", config=cfg
        )
        assert result.status == LoopStatus.STOPPED
        assert (
            result.stopping_condition
            == StoppingCondition.NO_IMPROVEMENT
        )


class TestBudgetExceeded:
    @pytest.mark.asyncio
    async def test_budget_hours(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=100,
            target_metric_value=0.001,
            budget_hours=0.001,
            output_dir=str(tmp_path / "loops"),
        )
        result = await loop_agent.run(
            "budget test", "/repo", config=cfg
        )
        assert result.status == LoopStatus.STOPPED
        assert (
            result.stopping_condition
            == StoppingCondition.BUDGET_EXCEEDED
        )


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_iteration_error_continues(
        self,
        mock_memory_agent,
        mock_literature_agent,
        mock_planner_agent,
        mock_coding_agent,
        mock_evaluation_agent,
        tmp_path,
    ):
        exp_agent = MagicMock()
        exp_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        exp_agent.get_experiment = AsyncMock(return_value=None)
        agent = ResearchLoopAgent(
            memory_agent=mock_memory_agent,
            literature_agent=mock_literature_agent,
            planner_agent=mock_planner_agent,
            coding_agent=mock_coding_agent,
            experiment_agent=exp_agent,
            evaluation_agent=mock_evaluation_agent,
            storage_tool=LoopStorageTool(
                db_path=str(tmp_path / "test.db")
            ),
            stopping_checker=StoppingConditionChecker(),
            report_generator=ReportGeneratorTool(),
            config=_config(max_iterations=2),
        )
        result = await agent.run(
            "error test", "/repo", config=_config(max_iterations=1)
        )
        assert result.iteration_count == 1
        assert result.status == LoopStatus.STOPPED


class TestMemoryIntegration:
    @pytest.mark.asyncio
    async def test_memory_called(
        self, loop_agent, mock_memory_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        await loop_agent.run("memory test", "/repo", config=cfg)
        assert mock_memory_agent.get_context.called
        assert mock_memory_agent.store_success.called
        assert mock_memory_agent.graph.add_node.called
        assert mock_memory_agent.graph.add_relationship.called


class TestDeriveNextCommand:
    """Unit tests for each decision branch of _derive_next_command."""

    def _iteration(
        self,
        number: int = 1,
        primary_metric_value: float | None = 0.5,
        best_metric_value: float | None = 0.5,
        improvement: float | None = None,
        primary_metric_name: str | None = "loss",
    ) -> LoopIteration:
        return LoopIteration(
            loop_id="loop_1",
            iteration_number=number,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            best_metric_value=best_metric_value,
            improvement=improvement,
        )

    def test_no_target_metric(self):
        cfg = _config(target_metric_name=None)
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(), cfg, []
        )
        assert cmd.action == NextAction.NONE
        assert cmd.iteration_number == 1

    def test_no_primary_metric_value(self):
        cfg = _config()
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(primary_metric_value=None), cfg, []
        )
        assert cmd.action == NextAction.NONE
        assert cmd.target_metric == "loss"

    def test_improved_continues(self):
        cfg = _config()
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(
                primary_metric_value=0.3,
                best_metric_value=0.3,
                improvement=0.2,
            ),
            cfg,
            [],
        )
        assert cmd.action == NextAction.CONTINUE
        assert "improved" in cmd.rationale
        assert cmd.target_metric == "loss"

    def test_regressed_corrects(self):
        cfg = _config()
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(
                primary_metric_value=0.8,
                best_metric_value=0.5,
                improvement=None,
            ),
            cfg,
            [],
        )
        assert cmd.action == NextAction.CORRECT
        assert "regressed" in cmd.rationale
        assert any("hyperparameters" in c for c in cmd.suggested_changes)

    def test_regressed_higher_is_better(self):
        cfg = _config(higher_is_better=True)
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(
                primary_metric_value=0.4,
                best_metric_value=0.9,
                improvement=None,
            ),
            cfg,
            [],
        )
        assert cmd.action == NextAction.CORRECT

    def test_no_improvement_over_window_rediscovers(self):
        cfg = _config(stagnation_window=3)
        history = [
            self._iteration(number=1, improvement=None),
            self._iteration(number=2, improvement=None),
            self._iteration(number=3, improvement=None),
        ]
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(number=4, improvement=None), cfg, history
        )
        assert cmd.action == NextAction.REDISCOVER
        assert "re-discovery" in cmd.rationale

    def test_stagnant_within_window_continues(self):
        cfg = _config(stagnation_window=3)
        history = [
            self._iteration(number=1, improvement=None),
        ]
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(number=2, improvement=None), cfg, history
        )
        assert cmd.action == NextAction.CONTINUE
        assert "cautiously" in cmd.rationale

    def test_improvement_breaks_stagnation(self):
        cfg = _config(stagnation_window=3)
        history = [
            self._iteration(number=1, improvement=None),
            self._iteration(number=2, improvement=None),
            self._iteration(number=3, improvement=0.1),
        ]
        cmd = ResearchLoopAgent._derive_next_command(
            self._iteration(number=4, improvement=None), cfg, history
        )
        assert cmd.action == NextAction.CONTINUE


class TestQueryMethods:
    @pytest.mark.asyncio
    async def test_list_loops(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        await loop_agent.run("query test", "/repo", config=cfg)
        out = await loop_agent.list_loops()
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_get_loop(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        result = await loop_agent.run("get test", "/repo", config=cfg)
        record = await loop_agent.get_loop(result.loop_id)
        assert record is not None
        assert record.goal == "get test"

    @pytest.mark.asyncio
    async def test_get_iterations(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        result = await loop_agent.run(
            "iter test", "/repo", config=cfg
        )
        out = await loop_agent.get_iterations(result.loop_id)
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_search_loops(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        await loop_agent.run(
            "searchable goal", "/repo", config=cfg
        )
        out = await loop_agent.search_loops("searchable")
        assert out.total >= 1


class TestReportGeneration:
    @pytest.mark.asyncio
    async def test_generate_report(
        self, loop_agent, tmp_path
    ):
        cfg = _config(
            max_iterations=1, output_dir=str(tmp_path / "loops")
        )
        result = await loop_agent.run(
            "report test", "/repo", config=cfg
        )
        report = await loop_agent.generate_report(result.loop_id)
        assert Path(report.report_path).exists()
        assert Path(report.json_path).exists()


class TestGracefulNoDeps:
    @pytest.mark.asyncio
    async def test_no_memory(self, tmp_path):
        agent = ResearchLoopAgent(
            storage_tool=LoopStorageTool(
                db_path=str(tmp_path / "test.db")
            ),
            stopping_checker=StoppingConditionChecker(),
            report_generator=ReportGeneratorTool(),
            config=_config(max_iterations=1),
        )
        result = await agent.run(
            "no deps", "/repo", config=_config(max_iterations=1)
        )
        assert result.status == LoopStatus.STOPPED
        assert result.memory_ids == []