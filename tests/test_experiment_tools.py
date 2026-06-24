"""Tests for Phase 7 experiment tools."""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_engineer.models.experiment import (
    AnomalyIndicator,
    ArtifactCollectorInput,
    ArtifactCollectorInput as _ACI,
    ArtifactPattern,
    ArtifactType,
    ExperimentArtifact,
    ExperimentQueryInput,
    ExperimentRecord,
    ExperimentRunnerInput,
    ExperimentRun,
    ExperimentStatus,
    ExperimentStorageInput,
    ExperimentType,
    FailureDetectorInput,
    FailureSeverity,
    MetricCollectorInput,
    MetricPattern,
    MetricReading,
    MetricType,
    MonitoringInput,
)
from research_engineer.tools.artifact_collector import ArtifactCollectorTool
from research_engineer.tools.experiment_runner import (
    ALLOWED_COMMAND_PREFIXES,
    ExperimentRunnerTool,
    _command_allowed,
)
from research_engineer.tools.experiment_storage import ExperimentStorageTool
from research_engineer.tools.failure_detector import FailureDetectorTool
from research_engineer.tools.metric_collector import MetricCollectorTool
from research_engineer.tools.monitoring import MonitoringTool


# ---------------------------------------------------------------------------
# ExperimentRunnerTool
# ---------------------------------------------------------------------------


class TestExperimentRunnerTool:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=["python", "-c", "print(1)"],
            working_dir=".",
            experiment_id="dry1",
            dry_run=True,
        )
        out = await tool.execute(inp)
        assert out.launched is False
        assert out.run.status == ExperimentStatus.PENDING

    @pytest.mark.asyncio
    async def test_real_execution_success(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=[sys.executable, "-c", "print('hello')"],
            working_dir=".",
            experiment_id="real1",
            dry_run=False,
            timeout_seconds=10,
        )
        out = await tool.execute(inp)
        assert out.launched is True
        assert out.run.status == ExperimentStatus.COMPLETED
        assert out.run.exit_code == 0
        assert "hello" in out.run.stdout

    @pytest.mark.asyncio
    async def test_nonzero_exit(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=[sys.executable, "-c", "import sys; sys.exit(1)"],
            working_dir=".",
            experiment_id="fail1",
            dry_run=False,
            timeout_seconds=10,
        )
        out = await tool.execute(inp)
        assert out.run.status == ExperimentStatus.FAILED
        assert out.run.exit_code == 1

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=[sys.executable, "-c", "import time; time.sleep(5)"],
            working_dir=".",
            experiment_id="timeout1",
            dry_run=False,
            timeout_seconds=1,
        )
        out = await tool.execute(inp)
        assert out.run.status == ExperimentStatus.TIMEOUT
        assert out.run.killed is True

    @pytest.mark.asyncio
    async def test_disallowed_command_rejected(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=["rm", "-rf", "/"],
            working_dir=".",
            experiment_id="bad1",
            dry_run=False,
        )
        from research_engineer.tools.base import ToolError

        with pytest.raises(ToolError):
            await tool.execute(inp)

    def test_command_allowed(self):
        assert _command_allowed(["python", "script.py"]) is True
        assert _command_allowed(["python3", "-c", "print(1)"]) is True
        assert _command_allowed(["torchrun", "train.py"]) is True
        assert _command_allowed(["rm", "-rf"]) is False
        assert _command_allowed([]) is False

    @pytest.mark.asyncio
    async def test_status_history_recorded(self):
        tool = ExperimentRunnerTool()
        inp = ExperimentRunnerInput(
            command=[sys.executable, "-c", "print(1)"],
            working_dir=".",
            experiment_id="hist1",
            dry_run=False,
            timeout_seconds=10,
        )
        out = await tool.execute(inp)
        assert len(out.run.status_history) >= 2
        assert out.run.status_history[0].from_status == ExperimentStatus.PENDING


# ---------------------------------------------------------------------------
# MonitoringTool
# ---------------------------------------------------------------------------


class TestMonitoringTool:
    @pytest.mark.asyncio
    async def test_basic_analysis(self):
        tool = MonitoringTool()
        inp = MonitoringInput(
            experiment_id="exp1",
            stdout="loss=0.5\naccuracy=0.85\nstep=100",
            stderr="",
        )
        out = await tool.execute(inp)
        assert out.experiment_id == "exp1"
        assert out.total_stdout_lines == 3
        assert len(out.metrics_detected) >= 2

    @pytest.mark.asyncio
    async def test_metric_detection(self):
        tool = MonitoringTool()
        inp = MonitoringInput(
            experiment_id="exp1",
            stdout="epoch=1 loss=1.5\nepoch=2 loss=1.0\nepoch=3 loss=0.5",
        )
        out = await tool.execute(inp)
        loss_metrics = [m for m in out.metrics_detected if m.name == "loss"]
        assert len(loss_metrics) == 3

    @pytest.mark.asyncio
    async def test_anomaly_detection_nan(self):
        tool = MonitoringTool()
        inp = MonitoringInput(
            experiment_id="exp1",
            stdout="loss=nan",
        )
        out = await tool.execute(inp)
        assert len(out.anomalies) >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_scan(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"checkpoint")
        tool = MonitoringTool()
        inp = MonitoringInput(
            experiment_id="exp1",
            stdout="",
            working_dir=str(tmp_path),
        )
        out = await tool.execute(inp)
        assert len(out.checkpoints_found) >= 1

    @pytest.mark.asyncio
    async def test_empty_input(self):
        tool = MonitoringTool()
        inp = MonitoringInput(experiment_id="exp1")
        out = await tool.execute(inp)
        assert out.total_stdout_lines == 0
        assert out.metrics_detected == []


# ---------------------------------------------------------------------------
# MetricCollectorTool
# ---------------------------------------------------------------------------


class TestMetricCollectorTool:
    @pytest.mark.asyncio
    async def test_parse_loss_from_stdout(self):
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="epoch=1 loss=1.5\nepoch=2 loss=1.0\nepoch=3 loss=0.5",
        )
        out = await tool.execute(inp)
        loss_readings = [m for m in out.metrics if m.name == "loss"]
        assert len(loss_readings) == 3
        assert "loss" in out.summary_metrics

    @pytest.mark.asyncio
    async def test_parse_accuracy_max(self):
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="accuracy=0.8\naccuracy=0.9\naccuracy=0.85",
        )
        out = await tool.execute(inp)
        assert out.summary_metrics["accuracy"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_parse_json_file(self, tmp_path):
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(
            json.dumps([{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.3}])
        )
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="",
            output_dir=str(tmp_path),
        )
        out = await tool.execute(inp)
        loss_readings = [m for m in out.metrics if m.name == "loss"]
        assert len(loss_readings) == 3

    @pytest.mark.asyncio
    async def test_parse_csv_file(self, tmp_path):
        csv_file = tmp_path / "metrics.csv"
        csv_file.write_text("step,loss,accuracy\n1,1.0,0.7\n2,0.5,0.85\n")
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="",
            output_dir=str(tmp_path),
        )
        out = await tool.execute(inp)
        loss_readings = [m for m in out.metrics if m.name == "loss"]
        assert len(loss_readings) == 2

    @pytest.mark.asyncio
    async def test_empty_input(self):
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(experiment_id="exp1")
        out = await tool.execute(inp)
        assert out.metrics == []
        assert out.summary_metrics == {}

    @pytest.mark.asyncio
    async def test_custom_patterns(self):
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="custom_metric=42.5",
            metric_patterns=[
                MetricPattern(
                    name="custom_metric",
                    regex=r"custom_metric=([0-9.]+)",
                    aggregate="last",
                )
            ],
        )
        out = await tool.execute(inp)
        assert any(m.name == "custom_metric" for m in out.metrics)

    @pytest.mark.asyncio
    async def test_metric_series(self):
        tool = MetricCollectorTool()
        inp = MetricCollectorInput(
            experiment_id="exp1",
            stdout="loss=1.0\nloss=0.5\nloss=0.3",
        )
        out = await tool.execute(inp)
        loss_series = [s for s in out.metric_series if s.name == "loss"]
        assert len(loss_series) == 1
        assert loss_series[0].final_value == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# ArtifactCollectorTool
# ---------------------------------------------------------------------------


class TestArtifactCollectorTool:
    @pytest.mark.asyncio
    async def test_collect_checkpoints(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"checkpoint1")
        (tmp_path / "ckpt").with_suffix(".pt").write_bytes(b"ckpt2")
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            copy_artifacts=False,
        )
        out = await tool.execute(inp)
        ckpts = [a for a in out.artifacts if a.artifact_type == ArtifactType.CHECKPOINT]
        assert len(ckpts) >= 1

    @pytest.mark.asyncio
    async def test_collect_logs(self, tmp_path):
        (tmp_path / "run.log").write_text("log content")
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            copy_artifacts=False,
        )
        out = await tool.execute(inp)
        logs = [a for a in out.artifacts if a.artifact_type == ArtifactType.LOG]
        assert len(logs) >= 1

    @pytest.mark.asyncio
    async def test_size_cap_skips_large(self, tmp_path):
        big_file = tmp_path / "big.pt"
        big_file.write_bytes(b"x" * (2 * 1024 * 1024))
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            max_artifact_size_mb=1.0,
            copy_artifacts=False,
        )
        out = await tool.execute(inp)
        assert len(out.artifacts) == 0
        assert len(out.collection_errors) >= 1

    @pytest.mark.asyncio
    async def test_copy_with_checksum(self, tmp_path):
        (tmp_path / "model.pt").write_bytes(b"checkpoint")
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            copy_artifacts=True,
        )
        out = await tool.execute(inp)
        ckpts = [a for a in out.artifacts if a.artifact_type == ArtifactType.CHECKPOINT]
        if ckpts:
            assert ckpts[0].stored_path is not None
            assert ckpts[0].checksum is not None

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_path):
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            copy_artifacts=False,
        )
        out = await tool.execute(inp)
        assert out.artifacts == []

    @pytest.mark.asyncio
    async def test_max_files_limit(self, tmp_path):
        for i in range(15):
            (tmp_path / f"model_{i}.pt").write_bytes(b"x")
        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp1",
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path),
            copy_artifacts=False,
            artifact_patterns=[
                ArtifactPattern(
                    name="ckpts",
                    glob_pattern="*.pt",
                    artifact_type=ArtifactType.CHECKPOINT,
                    max_files=5,
                )
            ],
        )
        out = await tool.execute(inp)
        assert len(out.artifacts) == 5


# ---------------------------------------------------------------------------
# FailureDetectorTool
# ---------------------------------------------------------------------------


class TestFailureDetectorTool:
    def _make_run(
        self,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        exit_code: int | None = 0,
        stderr: str = "",
        stdout: str = "",
    ) -> ExperimentRun:
        return ExperimentRun(
            experiment_id="exp1",
            command=["python", "train.py"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
            status=status,
            exit_code=exit_code,
            stderr=stderr,
            stdout=stdout,
        )

    @pytest.mark.asyncio
    async def test_success_no_failure(self):
        tool = FailureDetectorTool()
        run = self._make_run()
        inp = FailureDetectorInput(
            run=run,
            metrics=[MetricReading(name="loss", value=0.5)],
        )
        out = await tool.execute(inp)
        assert out.detected_failure is False
        assert out.severity == FailureSeverity.NONE

    @pytest.mark.asyncio
    async def test_oom_detection(self):
        tool = FailureDetectorTool()
        run = self._make_run(
            status=ExperimentStatus.FAILED,
            exit_code=1,
            stderr="RuntimeError: CUDA out of memory",
        )
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert out.detected_failure is True
        assert out.failure_mode == "memory_overflow"
        assert out.severity == FailureSeverity.HIGH
        assert len(out.recommendations) > 0

    @pytest.mark.asyncio
    async def test_crash_detection(self):
        tool = FailureDetectorTool()
        run = self._make_run(
            status=ExperimentStatus.FAILED,
            exit_code=1,
            stderr="RuntimeError: something went wrong",
        )
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert out.detected_failure is True
        assert out.failure_mode == "crash"

    @pytest.mark.asyncio
    async def test_nan_loss_detection(self):
        tool = FailureDetectorTool()
        run = self._make_run()
        inp = FailureDetectorInput(
            run=run,
            metrics=[MetricReading(name="loss", value=float("nan"))],
        )
        out = await tool.execute(inp)
        assert out.detected_failure is True
        assert any(a.indicator == "loss_nan" for a in out.anomaly_indicators)

    @pytest.mark.asyncio
    async def test_divergence_detection(self):
        tool = FailureDetectorTool()
        run = self._make_run()
        metrics = [
            MetricReading(name="loss", value=1.0),
            MetricReading(name="loss", value=1.1),
            MetricReading(name="loss", value=1.5),
            MetricReading(name="loss", value=2.0),
        ]
        inp = FailureDetectorInput(run=run, metrics=metrics)
        out = await tool.execute(inp)
        anomalies = [a for a in out.anomaly_indicators if a.indicator == "loss_divergence"]
        assert len(anomalies) == 1

    @pytest.mark.asyncio
    async def test_missing_metrics(self):
        tool = FailureDetectorTool()
        run = self._make_run()
        inp = FailureDetectorInput(
            run=run,
            metrics=[],
            expected_metrics=["loss", "accuracy"],
        )
        out = await tool.execute(inp)
        assert out.detected_failure is True
        assert out.failure_mode == "poor_performance"

    @pytest.mark.asyncio
    async def test_api_incompatibility(self):
        tool = FailureDetectorTool()
        run = self._make_run(
            status=ExperimentStatus.FAILED,
            exit_code=1,
            stderr="KeyError: 'model_config'",
        )
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert out.failure_mode == "api_incompatibility"

    @pytest.mark.asyncio
    async def test_recommendations_generated(self):
        tool = FailureDetectorTool()
        run = self._make_run(
            status=ExperimentStatus.FAILED,
            exit_code=1,
            stderr="CUDA out of memory",
        )
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert len(out.recommendations) > 0
        assert len(out.lessons_learned) > 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        tool = FailureDetectorTool()
        run = self._make_run(status=ExperimentStatus.TIMEOUT, exit_code=None)
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert out.detected_failure is True

    @pytest.mark.asyncio
    async def test_crashed_status(self):
        tool = FailureDetectorTool()
        run = self._make_run(
            status=ExperimentStatus.CRASHED, exit_code=None
        )
        inp = FailureDetectorInput(run=run, metrics=[])
        out = await tool.execute(inp)
        assert out.detected_failure is True
        assert out.failure_mode == "crash"


# ---------------------------------------------------------------------------
# ExperimentStorageTool
# ---------------------------------------------------------------------------


class TestExperimentStorageTool:
    def _make_record(self, experiment_id: str = "exp1") -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=experiment_id,
            paper_id="2503.12345",
            plan_id="plan1",
            patch_id=None,
            implementation_id=None,
            repo_path="./my_repo",
            command=["python", "train.py"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T01:00:00",
            duration_seconds=3600.0,
            exit_code=0,
            metrics={"loss": 0.5, "accuracy": 0.9},
            tags=["my_repo", "experiment"],
        )

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        record = self._make_record()
        out = await tool.execute(
            ExperimentStorageInput(experiment=record)
        )
        assert out.success is True
        retrieved = await tool.get_by_id("exp1")
        assert retrieved is not None
        assert retrieved.experiment_id == "exp1"
        assert retrieved.metrics["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_query_by_paper(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        for i in range(3):
            record = self._make_record(f"exp{i}")
            record.paper_id = "2503.12345"
            await tool.execute(ExperimentStorageInput(experiment=record))
        out = await tool.execute(
            ExperimentQueryInput(paper_id="2503.12345")
        )
        assert out.total == 3
        assert len(out.experiments) == 3

    @pytest.mark.asyncio
    async def test_query_by_status(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        rec1 = self._make_record("exp1")
        rec1.status = ExperimentStatus.COMPLETED
        rec2 = self._make_record("exp2")
        rec2.status = ExperimentStatus.FAILED
        await tool.execute(ExperimentStorageInput(experiment=rec1))
        await tool.execute(ExperimentStorageInput(experiment=rec2))
        out = await tool.execute(
            ExperimentQueryInput(status=ExperimentStatus.COMPLETED)
        )
        assert out.total == 1
        assert out.experiments[0].experiment_id == "exp1"

    @pytest.mark.asyncio
    async def test_query_by_type(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        rec1 = self._make_record("exp1")
        rec1.experiment_type = ExperimentType.TRAINING
        rec2 = self._make_record("exp2")
        rec2.experiment_type = ExperimentType.EVALUATION
        await tool.execute(ExperimentStorageInput(experiment=rec1))
        await tool.execute(ExperimentStorageInput(experiment=rec2))
        out = await tool.execute(
            ExperimentQueryInput(experiment_type=ExperimentType.TRAINING)
        )
        assert out.total == 1

    @pytest.mark.asyncio
    async def test_search_text(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        rec = self._make_record("exp1")
        rec.notes = "attention experiment with OOM"
        await tool.execute(ExperimentStorageInput(experiment=rec))
        out = await tool.execute(
            ExperimentQueryInput(search_text="attention")
        )
        assert out.total == 1

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        for i in range(5):
            record = self._make_record(f"exp{i}")
            await tool.execute(ExperimentStorageInput(experiment=record))
        out = await tool.execute(
            ExperimentQueryInput(limit=2, offset=0)
        )
        assert len(out.experiments) == 2
        assert out.total == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tmp_path):
        tool = ExperimentStorageTool(db_path=str(tmp_path / "test.db"))
        result = await tool.get_by_id("nonexistent")
        assert result is None