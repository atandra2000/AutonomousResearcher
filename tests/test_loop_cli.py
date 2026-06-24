"""Tests for Phase 9 - Autonomous Research Loop CLI commands."""

from typer.testing import CliRunner

from research_engineer.cli import app

runner = CliRunner()


class TestLoopRun:
    def test_run_help(self):
        result = runner.invoke(app, ["loop", "run", "--help"])
        assert result.exit_code == 0
        assert "research goal" in result.output.lower()


class TestLoopList:
    def test_list_empty(self):
        result = runner.invoke(app, ["loop", "list"])
        assert result.exit_code == 0

    def test_list_json(self):
        result = runner.invoke(
            app, ["loop", "list", "--format", "json"]
        )
        assert result.exit_code == 0


class TestLoopGet:
    def test_get_nonexistent(self):
        result = runner.invoke(
            app, ["loop", "get", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output


class TestLoopIterations:
    def test_iterations_nonexistent(self):
        result = runner.invoke(
            app, ["loop", "iterations", "nonexistent_id"]
        )
        assert result.exit_code == 0


class TestLoopIteration:
    def test_iteration_nonexistent(self):
        result = runner.invoke(
            app, ["loop", "iteration", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output


class TestLoopSearch:
    def test_search(self):
        result = runner.invoke(app, ["loop", "search", "test"])
        assert result.exit_code == 0


class TestLoopReport:
    def test_report_nonexistent(self):
        result = runner.invoke(
            app, ["loop", "report", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "error" in result.output.lower()