"""Tests for Phase 7 experiment CLI commands."""

import sys

import pytest
from typer.testing import CliRunner

from research_engineer.cli import app, experiment_app


runner = CliRunner()


class TestExperimentRun:
    def test_dry_run(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "experiment", "run",
                "--command", "python -c print(1)",
                "--repo", str(tmp_path),
                "--dry-run",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code == 0
        assert "Experiment" in result.output

    def test_dry_run_json(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "experiment", "run",
                "--command", "python -c print(1)",
                "--repo", str(tmp_path),
                "--dry-run",
                "--output-dir", str(tmp_path / "out"),
                "--format", "json",
            ],
        )
        assert result.exit_code == 0
        assert "experiment_id" in result.output

    def test_invalid_type(self, tmp_path):
        result = runner.invoke(
            app,
            [
                "experiment", "run",
                "--command", "python -c print(1)",
                "--repo", str(tmp_path),
                "--type", "invalid_type",
            ],
        )
        assert "Invalid type" in result.output


class TestExperimentList:
    def test_list_empty(self):
        result = runner.invoke(app, ["experiment", "list"])
        assert result.exit_code == 0

    def test_list_json(self):
        result = runner.invoke(
            app, ["experiment", "list", "--format", "json"]
        )
        assert result.exit_code == 0


class TestExperimentGet:
    def test_get_nonexistent(self):
        result = runner.invoke(
            app, ["experiment", "get", "nonexistent_id"]
        )
        assert "not found" in result.output


class TestExperimentSearch:
    def test_search(self):
        result = runner.invoke(
            app, ["experiment", "search", "attention"]
        )
        assert result.exit_code == 0


class TestExperimentCancel:
    def test_cancel_nonexistent(self):
        result = runner.invoke(
            app, ["experiment", "cancel", "nonexistent_id"]
        )
        assert result.exit_code == 0


class TestExperimentHistory:
    def test_history(self):
        result = runner.invoke(
            app, ["experiment", "history", "2503.12345"]
        )
        assert result.exit_code == 0


class TestExperimentMonitor:
    def test_monitor_nonexistent(self):
        result = runner.invoke(
            app, ["experiment", "monitor", "nonexistent_id"]
        )
        assert "not found" in result.output