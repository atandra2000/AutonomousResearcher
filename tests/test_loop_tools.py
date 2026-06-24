"""Tests for Phase 9 - Autonomous Research Loop tools."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from research_engineer.models.loop import (
    IterationQueryInput,
    IterationStorageInput,
    LoopConfig,
    LoopIteration,
    LoopQueryInput,
    LoopRecord,
    LoopState,
    LoopStatus,
    LoopStorageInput,
    ReportInput,
    StoppingCheckInput,
    StoppingCondition,
)
from research_engineer.tools.loop_storage import LoopStorageTool
from research_engineer.tools.report_generator import ReportGeneratorTool
from research_engineer.tools.stopping_condition import (
    StoppingConditionChecker,
)


# --- Helpers ---


def _config(
    max_iterations: int = 5,
    target_metric_name: str | None = "loss",
    target_metric_value: float | None = 0.1,
    **kwargs,
) -> LoopConfig:
    return LoopConfig(
        goal="test goal",
        repo_path="/repo",
        max_iterations=max_iterations,
        target_metric_name=target_metric_name,
        target_metric_value=target_metric_value,
        **kwargs,
    )


def _iteration(
    n: int = 1,
    loop_id: str = "loop_test",
    metric_value: float | None = 0.5,
    status: LoopStatus = LoopStatus.EVALUATED,
) -> LoopIteration:
    return LoopIteration(
        loop_id=loop_id,
        iteration_number=n,
        primary_metric_name="loss",
        primary_metric_value=metric_value,
        status=status,
        timestamp=datetime.now(),
    )


def _loop_record(
    loop_id: str = "loop_test",
    goal: str = "test goal",
    status: LoopStatus = LoopStatus.STOPPED,
) -> LoopRecord:
    return LoopRecord(
        loop_id=loop_id,
        goal=goal,
        config_json='{"goal": "test"}',
        status=status,
        iteration_count=3,
        best_metric_value=0.1,
        primary_metric_name="loss",
        stopping_condition=StoppingCondition.TARGET_ACHIEVED,
        created_at=datetime.now(),
    )


# --- LoopStorageTool ---


class TestLoopStorageTool:
    @pytest.fixture
    def storage(self, tmp_path):
        return LoopStorageTool(db_path=str(tmp_path / "test.db"))

    @pytest.mark.asyncio
    async def test_store_and_get_loop(self, storage):
        rec = _loop_record()
        out = await storage.execute(LoopStorageInput(loop=rec))
        assert out.success is True
        fetched = await storage.get_loop_by_id(rec.loop_id)
        assert fetched is not None
        assert fetched.goal == "test goal"
        assert fetched.status == LoopStatus.STOPPED

    @pytest.mark.asyncio
    async def test_get_loop_missing(self, storage):
        assert await storage.get_loop_by_id("missing") is None

    @pytest.mark.asyncio
    async def test_query_loops_by_status(self, storage):
        rec1 = _loop_record(loop_id="l1", status=LoopStatus.STOPPED)
        rec2 = _loop_record(loop_id="l2", status=LoopStatus.RUNNING)
        await storage.execute(LoopStorageInput(loop=rec1))
        await storage.execute(LoopStorageInput(loop=rec2))
        out = await storage.execute(
            LoopQueryInput(status=LoopStatus.STOPPED)
        )
        assert out.total == 1
        assert out.loops[0].loop_id == "l1"

    @pytest.mark.asyncio
    async def test_query_loops_search_text(self, storage):
        rec = _loop_record(goal="improve training stability")
        await storage.execute(LoopStorageInput(loop=rec))
        out = await storage.execute(LoopQueryInput(search_text="training"))
        assert out.total == 1

    @pytest.mark.asyncio
    async def test_store_and_get_iteration(self, storage):
        it = _iteration()
        out = await storage.execute(IterationStorageInput(iteration=it))
        assert out.success is True
        fetched = await storage.get_iteration_by_id(it.iteration_id)
        assert fetched is not None
        assert fetched.iteration_number == 1

    @pytest.mark.asyncio
    async def test_get_iteration_missing(self, storage):
        assert await storage.get_iteration_by_id("missing") is None

    @pytest.mark.asyncio
    async def test_query_iterations_by_loop(self, storage):
        for n in range(3):
            it = _iteration(n=n + 1, loop_id="loop_x")
            await storage.execute(IterationStorageInput(iteration=it))
        out = await storage.execute(
            IterationQueryInput(loop_id="loop_x")
        )
        assert out.total == 3

    @pytest.mark.asyncio
    async def test_query_iterations_pagination(self, storage):
        for n in range(5):
            it = _iteration(n=n + 1, loop_id="loop_p")
            await storage.execute(IterationStorageInput(iteration=it))
        out = await storage.execute(
            IterationQueryInput(loop_id="loop_p", limit=2, offset=0)
        )
        assert len(out.iterations) == 2
        assert out.total == 5


# --- StoppingConditionChecker ---


class TestStoppingConditionChecker:
    @pytest.fixture
    def checker(self):
        return StoppingConditionChecker()

    @pytest.mark.asyncio
    async def test_target_achieved_higher(self, checker):
        state = LoopState(
            loop_id="l", goal="g", best_metric_value=0.96
        )
        cfg = _config(
            target_metric_name="accuracy",
            target_metric_value=0.95,
            higher_is_better=True,
        )
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.TARGET_ACHIEVED

    @pytest.mark.asyncio
    async def test_target_achieved_lower(self, checker):
        state = LoopState(
            loop_id="l", goal="g", best_metric_value=0.05
        )
        cfg = _config(target_metric_value=0.1, higher_is_better=False)
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.TARGET_ACHIEVED

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, checker):
        state = LoopState(
            loop_id="l", goal="g", current_iteration=5
        )
        cfg = _config(max_iterations=5, target_metric_value=None)
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.MAX_ITERATIONS_REACHED

    @pytest.mark.asyncio
    async def test_budget_hours_exceeded(self, checker):
        state = LoopState(
            loop_id="l", goal="g", cumulative_cost_hours=10.0
        )
        cfg = _config(
            max_iterations=100,
            target_metric_value=None,
            budget_hours=5.0,
        )
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_budget_cost_exceeded(self, checker):
        state = LoopState(
            loop_id="l", goal="g", cumulative_cost_usd=100.0
        )
        cfg = _config(
            max_iterations=100,
            target_metric_value=None,
            budget_cost=50.0,
        )
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_no_improvement(self, checker):
        state = LoopState(
            loop_id="l", goal="g", best_metric_value=0.5
        )
        cfg = _config(
            max_iterations=100,
            target_metric_value=0.01,
            stagnation_window=3,
        )
        history = [
            _iteration(n=1, metric_value=0.5),
            _iteration(n=2, metric_value=0.5),
            _iteration(n=3, metric_value=0.5),
            _iteration(n=4, metric_value=0.5),
            _iteration(n=5, metric_value=0.5),
            _iteration(n=6, metric_value=0.5),
        ]
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=history)
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.NO_IMPROVEMENT

    @pytest.mark.asyncio
    async def test_no_stop(self, checker):
        state = LoopState(
            loop_id="l", goal="g", current_iteration=2
        )
        cfg = _config(max_iterations=10, target_metric_value=0.01)
        out = await checker.execute(
            StoppingCheckInput(state=state, config=cfg, history=[])
        )
        assert out.should_stop is False
        assert out.condition is None


# --- ReportGeneratorTool ---


class TestReportGeneratorTool:
    @pytest.fixture
    def generator(self):
        return ReportGeneratorTool()

    @pytest.mark.asyncio
    async def test_generates_files(self, generator, tmp_path):
        rec = _loop_record()
        cfg = _config()
        iterations = [_iteration(n=1), _iteration(n=2), _iteration(n=3)]
        out = await generator.execute(
            ReportInput(
                loop=rec,
                iterations=iterations,
                config=cfg,
                output_dir=str(tmp_path / "loops"),
            )
        )
        assert Path(out.report_path).exists()
        assert Path(out.json_path).exists()
        assert out.summary != ""

    @pytest.mark.asyncio
    async def test_report_contains_sections(self, generator, tmp_path):
        rec = _loop_record()
        cfg = _config()
        iterations = [_iteration(n=1)]
        out = await generator.execute(
            ReportInput(
                loop=rec,
                iterations=iterations,
                config=cfg,
                output_dir=str(tmp_path / "loops"),
            )
        )
        md = Path(out.report_path).read_text(encoding="utf-8")
        assert "# Autonomous Research Report" in md
        assert "## Executive Summary" in md
        assert "## Iteration History" in md
        assert "## Conclusions" in md

    @pytest.mark.asyncio
    async def test_json_valid(self, generator, tmp_path):
        rec = _loop_record()
        cfg = _config()
        out = await generator.execute(
            ReportInput(
                loop=rec,
                iterations=[],
                config=cfg,
                output_dir=str(tmp_path / "loops"),
            )
        )
        data = json.loads(Path(out.json_path).read_text())
        assert "loop" in data
        assert "iterations" in data