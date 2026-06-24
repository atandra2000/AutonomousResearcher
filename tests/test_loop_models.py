"""Tests for Phase 9 - Autonomous Research Loop models."""

import json
from datetime import datetime

import pytest

from research_engineer.models.loop import (
    ApprovalGate,
    ApprovalRequest,
    IterationPhase,
    IterationQueryInput,
    IterationQueryOutput,
    IterationStorageInput,
    IterationStorageOutput,
    LoopConfig,
    LoopIteration,
    LoopQueryInput,
    LoopQueryOutput,
    LoopRecord,
    LoopResult,
    LoopState,
    LoopStatus,
    LoopStorageInput,
    LoopStorageOutput,
    ReportInput,
    ReportOutput,
    StoppingCheckInput,
    StoppingCheckOutput,
    StoppingCondition,
)


class TestLoopStatus:
    def test_values(self):
        assert LoopStatus.CREATED == "created"
        assert LoopStatus.RUNNING == "running"
        assert LoopStatus.STOPPED == "stopped"
        assert LoopStatus.FAILED == "failed"
        assert LoopStatus.ITERATING == "iterating"
        assert LoopStatus.AWAITING_APPROVAL == "awaiting_approval"
        assert LoopStatus.EVALUATED == "evaluated"


class TestIterationPhase:
    def test_values(self):
        assert IterationPhase.LITERATURE == "literature"
        assert IterationPhase.PLANNING == "planning"
        assert IterationPhase.IMPLEMENTATION == "implementation"
        assert IterationPhase.EXPERIMENT == "experiment"
        assert IterationPhase.EVALUATION == "evaluation"
        assert IterationPhase.DECISION == "decision"


class TestStoppingCondition:
    def test_values(self):
        assert StoppingCondition.TARGET_ACHIEVED == "target_achieved"
        assert (
            StoppingCondition.MAX_ITERATIONS_REACHED
            == "max_iterations_reached"
        )
        assert StoppingCondition.BUDGET_EXCEEDED == "budget_exceeded"
        assert StoppingCondition.NO_IMPROVEMENT == "no_improvement"


class TestApprovalGate:
    def test_values(self):
        assert ApprovalGate.PLAN == "plan"
        assert ApprovalGate.IMPLEMENTATION == "implementation"
        assert ApprovalGate.NEXT_ITERATION == "next_iteration"


class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig(goal="test goal", repo_path="/repo")
        assert cfg.goal == "test goal"
        assert cfg.repo_path == "/repo"
        assert cfg.max_iterations == 5
        assert cfg.higher_is_better is False
        assert cfg.approval_mode is False
        assert cfg.dry_run is True
        assert cfg.skip_literature_after_first is True
        assert cfg.stagnation_window == 3
        assert cfg.improvement_threshold == 1e-4
        assert cfg.output_dir == "output/loops"

    def test_validation_max_iterations(self):
        with pytest.raises(Exception):
            LoopConfig(goal="g", repo_path="/r", max_iterations=0)

    def test_serialization(self):
        cfg = LoopConfig(goal="test", repo_path="/r", max_iterations=3)
        json_str = cfg.model_dump_json()
        restored = LoopConfig.model_validate_json(json_str)
        assert restored.goal == "test"
        assert restored.max_iterations == 3


class TestApprovalRequest:
    def test_defaults(self):
        req = ApprovalRequest(
            loop_id="loop_1",
            iteration_number=1,
            gate=ApprovalGate.PLAN,
        )
        assert req.loop_id == "loop_1"
        assert req.iteration_number == 1
        assert req.gate == ApprovalGate.PLAN
        assert "approve" in req.options


class TestLoopState:
    def test_defaults(self):
        state = LoopState(loop_id="loop_1", goal="test")
        assert state.loop_id == "loop_1"
        assert state.status == LoopStatus.CREATED
        assert state.current_iteration == 0
        assert state.iterations == []
        assert state.best_metric_value is None
        assert state.cumulative_cost_hours == 0.0
        assert state.pending_approval is None


class TestLoopIteration:
    def test_defaults(self):
        it = LoopIteration(
            loop_id="loop_1",
            iteration_number=1,
        )
        assert it.loop_id == "loop_1"
        assert it.iteration_number == 1
        assert it.phase == IterationPhase.LITERATURE
        assert it.metrics == {}
        assert it.memory_ids == []
        assert it.status == LoopStatus.CREATED

    def test_is_success(self):
        it = LoopIteration(loop_id="l", iteration_number=1)
        assert it.is_success() is True
        it.error = "failed"
        assert it.is_success() is False
        it.error = None
        it.status = LoopStatus.FAILED
        assert it.is_success() is False

    def test_serialization_roundtrip(self):
        it = LoopIteration(
            loop_id="loop_1",
            iteration_number=2,
            phase=IterationPhase.EVALUATION,
            paper_id="2503.12345",
            metrics={"loss": 0.5},
            primary_metric_name="loss",
            primary_metric_value=0.5,
        )
        json_str = it.model_dump_json()
        restored = LoopIteration.model_validate_json(json_str)
        assert restored.loop_id == "loop_1"
        assert restored.iteration_number == 2
        assert restored.phase == IterationPhase.EVALUATION
        assert restored.paper_id == "2503.12345"
        assert restored.metrics == {"loss": 0.5}
        assert restored.primary_metric_value == 0.5


class TestStorageModels:
    def test_iteration_storage_io(self):
        it = LoopIteration(loop_id="l", iteration_number=1)
        inp = IterationStorageInput(iteration=it)
        out = IterationStorageOutput(
            iteration_id=it.iteration_id, success=True
        )
        assert inp.iteration.iteration_id == it.iteration_id
        assert out.success is True

    def test_iteration_query_io(self):
        inp = IterationQueryInput(loop_id="loop_1", limit=10)
        out = IterationQueryOutput(total=5)
        assert inp.loop_id == "loop_1"
        assert out.total == 5

    def test_loop_storage_io(self):
        rec = LoopRecord(
            loop_id="loop_1", goal="test", config_json="{}"
        )
        inp = LoopStorageInput(loop=rec)
        out = LoopStorageOutput(loop_id="loop_1", success=True)
        assert inp.loop.loop_id == "loop_1"
        assert out.success is True

    def test_loop_query_io(self):
        inp = LoopQueryInput(status=LoopStatus.STOPPED)
        out = LoopQueryOutput(total=3)
        assert inp.status == LoopStatus.STOPPED
        assert out.total == 3


class TestLoopRecord:
    def test_defaults(self):
        rec = LoopRecord(goal="test", config_json="{}")
        assert rec.goal == "test"
        assert rec.status == LoopStatus.CREATED
        assert rec.iteration_count == 0
        assert rec.memory_ids == []

    def test_serialization(self):
        rec = LoopRecord(
            goal="test",
            config_json='{"goal": "test"}',
            status=LoopStatus.STOPPED,
            iteration_count=3,
            best_metric_value=0.5,
            stopping_condition=StoppingCondition.TARGET_ACHIEVED,
        )
        json_str = rec.model_dump_json()
        restored = LoopRecord.model_validate_json(json_str)
        assert restored.status == LoopStatus.STOPPED
        assert (
            restored.stopping_condition
            == StoppingCondition.TARGET_ACHIEVED
        )


class TestStoppingCheckModels:
    def test_input(self):
        state = LoopState(loop_id="l", goal="g")
        cfg = LoopConfig(goal="g", repo_path="/r")
        inp = StoppingCheckInput(state=state, config=cfg, history=[])
        assert inp.state.loop_id == "l"

    def test_output(self):
        out = StoppingCheckOutput(
            should_stop=True,
            condition=StoppingCondition.TARGET_ACHIEVED,
            reason="done",
        )
        assert out.should_stop is True
        assert out.condition == StoppingCondition.TARGET_ACHIEVED


class TestReportModels:
    def test_input(self):
        rec = LoopRecord(goal="g", config_json="{}")
        cfg = LoopConfig(goal="g", repo_path="/r")
        inp = ReportInput(
            loop=rec, iterations=[], config=cfg, output_dir="/out"
        )
        assert inp.output_dir == "/out"

    def test_output(self):
        out = ReportOutput(
            report_path="/out/report.md",
            json_path="/out/report.json",
            summary="test",
        )
        assert out.report_path == "/out/report.md"


class TestLoopResult:
    def test_defaults(self):
        result = LoopResult(
            loop_id="loop_1", goal="test", status=LoopStatus.STOPPED
        )
        assert result.loop_id == "loop_1"
        assert result.iterations == []
        assert result.iteration_count == 0

    def test_is_complete(self):
        result = LoopResult(
            loop_id="l", goal="g", status=LoopStatus.STOPPED
        )
        assert result.is_complete() is True
        result.status = LoopStatus.RUNNING
        assert result.is_complete() is False

    def test_is_success(self):
        result = LoopResult(
            loop_id="l",
            goal="g",
            status=LoopStatus.STOPPED,
            iteration_count=1,
        )
        assert result.is_success() is True
        result.stopping_condition = StoppingCondition.TARGET_ACHIEVED
        assert result.is_success() is True
        result.stopping_condition = None
        result.status = LoopStatus.FAILED
        assert result.is_success() is False