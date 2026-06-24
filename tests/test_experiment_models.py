"""Tests for Phase 7 experiment models."""

import math

import pytest

from research_engineer.models.experiment import (
    AnomalyIndicator,
    ArtifactCollectorInput,
    ArtifactCollectorOutput,
    ArtifactPattern,
    ArtifactType,
    ExperimentArtifact,
    ExperimentConfig,
    ExperimentQueryInput,
    ExperimentQueryOutput,
    ExperimentRecord,
    ExperimentResult,
    ExperimentRun,
    ExperimentRunnerInput,
    ExperimentRunnerOutput,
    ExperimentStatus,
    ExperimentStorageInput,
    ExperimentStorageOutput,
    ExperimentType,
    FailureDetectorInput,
    FailureDetectorOutput,
    FailureSeverity,
    MetricCollectorInput,
    MetricCollectorOutput,
    MetricPattern,
    MetricReading,
    MetricSeries,
    MetricType,
    MonitoringInput,
    MonitoringOutput,
    StatusTransition,
)


class TestEnums:
    def test_experiment_type_values(self):
        assert ExperimentType.TRAINING.value == "training"
        assert ExperimentType.EVALUATION.value == "evaluation"
        assert ExperimentType.VALIDATION.value == "validation"
        assert ExperimentType.DRY_RUN.value == "dry_run"

    def test_experiment_status_values(self):
        assert ExperimentStatus.PENDING.value == "pending"
        assert ExperimentStatus.RUNNING.value == "running"
        assert ExperimentStatus.COMPLETED.value == "completed"
        assert ExperimentStatus.FAILED.value == "failed"
        assert ExperimentStatus.TIMEOUT.value == "timeout"
        assert ExperimentStatus.CANCELLED.value == "cancelled"
        assert ExperimentStatus.CRASHED.value == "crashed"

    def test_metric_type_values(self):
        assert MetricType.SCALAR.value == "scalar"
        assert MetricType.LOSS.value == "loss"
        assert MetricType.ACCURACY.value == "accuracy"

    def test_artifact_type_values(self):
        assert ArtifactType.CHECKPOINT.value == "checkpoint"
        assert ArtifactType.LOG.value == "log"
        assert ArtifactType.CONFIG.value == "config"

    def test_failure_severity_values(self):
        assert FailureSeverity.NONE.value == "none"
        assert FailureSeverity.LOW.value == "low"
        assert FailureSeverity.HIGH.value == "high"
        assert FailureSeverity.CRITICAL.value == "critical"


class TestExperimentRun:
    def test_defaults(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python", "train.py"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
        )
        assert run.status == ExperimentStatus.PENDING
        assert run.pid is None
        assert run.exit_code is None
        assert run.stdout == ""
        assert run.status_history == []
        assert run.duration_seconds == 0.0
        assert run.killed is False

    def test_is_success(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
        )
        assert run.is_success() is True

    def test_is_not_success(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.FAILED,
        )
        assert run.is_success() is False

    def test_is_terminal(self):
        for status in [
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.TIMEOUT,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.CRASHED,
        ]:
            run = ExperimentRun(
                experiment_id="exp1",
                command=["python"],
                working_dir=".",
                experiment_type=ExperimentType.TRAINING,
                status=status,
            )
            assert run.is_terminal() is True

    def test_not_terminal(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.RUNNING,
        )
        assert run.is_terminal() is False


class TestStatusTransition:
    def test_basic(self):
        t = StatusTransition(
            from_status=ExperimentStatus.PENDING,
            to_status=ExperimentStatus.RUNNING,
        )
        assert t.from_status == ExperimentStatus.PENDING
        assert t.to_status == ExperimentStatus.RUNNING
        assert t.reason == ""


class TestRunnerModels:
    def test_runner_input_defaults(self):
        inp = ExperimentRunnerInput(
            command=["python", "train.py"],
            working_dir=".",
            experiment_id="exp1",
        )
        assert inp.dry_run is True
        assert inp.timeout_seconds == 3600
        assert inp.capture_output is True
        assert inp.env_vars == {}

    def test_runner_output(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
        )
        out = ExperimentRunnerOutput(run=run, launched=False, message="dry")
        assert out.launched is False


class TestMonitoringModels:
    def test_monitoring_input_defaults(self):
        inp = MonitoringInput(experiment_id="exp1")
        assert inp.stdout == ""
        assert inp.log_tail_lines == 100
        assert inp.status == ExperimentStatus.COMPLETED

    def test_monitoring_output(self):
        out = MonitoringOutput(
            experiment_id="exp1",
            final_status=ExperimentStatus.COMPLETED,
        )
        assert out.total_stdout_lines == 0
        assert out.metrics_detected == []
        assert out.anomalies == []


class TestMetricModels:
    def test_metric_pattern(self):
        p = MetricPattern(name="loss", regex=r"loss=(\d+)")
        assert p.aggregate == "last"
        assert p.metric_type == MetricType.SCALAR

    def test_metric_reading(self):
        r = MetricReading(name="loss", value=0.5)
        assert r.value == 0.5
        assert r.source == "stdout"
        assert r.step is None

    def test_metric_series(self):
        s = MetricSeries(
            name="loss",
            values=[1.0, 0.5, 0.3],
            steps=[1, 2, 3],
            best_value=0.3,
            best_step=3,
            final_value=0.3,
        )
        assert len(s.values) == 3
        assert s.best_value == 0.3

    def test_metric_collector_output(self):
        out = MetricCollectorOutput()
        assert out.metrics == []
        assert out.summary_metrics == {}


class TestArtifactModels:
    def test_artifact_pattern(self):
        p = ArtifactPattern(
            name="ckpts",
            glob_pattern="**/*.pt",
            artifact_type=ArtifactType.CHECKPOINT,
        )
        assert p.max_files == 10

    def test_experiment_artifact(self):
        a = ExperimentArtifact(
            name="model.pt",
            original_path="/path/to/model.pt",
            artifact_type=ArtifactType.CHECKPOINT,
        )
        assert a.size_bytes == 0
        assert a.checksum is None
        assert a.artifact_id  # auto-generated

    def test_artifact_collector_output(self):
        out = ArtifactCollectorOutput(output_dir="/tmp/out")
        assert out.artifacts == []
        assert out.total_size_mb == 0.0


class TestFailureModels:
    def test_anomaly_indicator(self):
        a = AnomalyIndicator(
            indicator="loss_nan",
            description="Loss is NaN",
            confidence=0.9,
        )
        assert a.confidence == 0.9

    def test_failure_detector_output(self):
        out = FailureDetectorOutput()
        assert out.detected_failure is False
        assert out.severity == FailureSeverity.NONE
        assert out.recommendations == []


class TestStorageModels:
    def test_experiment_record_defaults(self):
        rec = ExperimentRecord(
            experiment_id="exp1",
            repo_path=".",
            command=["python"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
            start_time="2026-01-01T00:00:00",
        )
        assert rec.metrics == {}
        assert rec.tags == []
        assert rec.failure_mode is None

    def test_experiment_record_is_success(self):
        rec = ExperimentRecord(
            experiment_id="exp1",
            repo_path=".",
            command=["python"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
            start_time="2026-01-01T00:00:00",
        )
        assert rec.is_success() is True

    def test_storage_input_output(self):
        rec = ExperimentRecord(
            experiment_id="exp1",
            repo_path=".",
            command=["python"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
            start_time="2026-01-01T00:00:00",
        )
        inp = ExperimentStorageInput(experiment=rec)
        assert inp.operation == "store"
        out = ExperimentStorageOutput(
            experiment_id="exp1", success=True, message="ok"
        )
        assert out.success is True

    def test_query_input_output(self):
        inp = ExperimentQueryInput(paper_id="2503.12345")
        assert inp.limit == 100
        out = ExperimentQueryOutput(experiments=[], total=0)
        assert out.total == 0


class TestExperimentResult:
    def test_defaults(self):
        r = ExperimentResult(
            experiment_id="exp1",
            repo_path=".",
        )
        assert r.paper_id is None
        assert r.memory_ids == []
        assert r.processing_time_seconds == 0.0

    def test_is_success_with_run(self):
        run = ExperimentRun(
            experiment_id="exp1",
            command=["python"],
            working_dir=".",
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
        )
        r = ExperimentResult(
            experiment_id="exp1", repo_path=".", run=run
        )
        assert r.is_success() is True

    def test_is_success_with_record(self):
        rec = ExperimentRecord(
            experiment_id="exp1",
            repo_path=".",
            command=["python"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.FAILED,
            start_time="2026-01-01T00:00:00",
        )
        r = ExperimentResult(
            experiment_id="exp1", repo_path=".", record=rec
        )
        assert r.is_success() is False

    def test_serialization(self):
        r = ExperimentResult(experiment_id="exp1", repo_path=".")
        data = r.model_dump()
        assert data["experiment_id"] == "exp1"
        r2 = ExperimentResult(**data)
        assert r2.experiment_id == "exp1"


class TestExperimentConfig:
    def test_defaults(self):
        c = ExperimentConfig()
        assert c.default_timeout_seconds == 3600
        assert c.dry_run_default is True
        assert c.store_results is True
        assert c.update_graph is True
        assert c.output_dir == "output/experiments"