"""Tests for Phase 8 evaluation CLI commands."""


from typer.testing import CliRunner

from research_engineer.cli import app

runner = CliRunner()


class TestEvaluateRun:
    def test_run_nonexistent(self):
        result = runner.invoke(
            app, ["evaluate", "run", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output


class TestEvaluateCompare:
    def test_compare_too_few(self):
        result = runner.invoke(
            app, ["evaluate", "compare", "only_one"]
        )
        assert result.exit_code == 0
        assert "at least 2" in result.output


class TestEvaluateAnalyze:
    def test_analyze_no_ids(self):
        result = runner.invoke(app, ["evaluate", "analyze", ""])
        assert result.exit_code == 0


class TestEvaluateDynamics:
    def test_dynamics_nonexistent(self):
        result = runner.invoke(
            app, ["evaluate", "dynamics", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output


class TestEvaluateSignificance:
    def test_significance_too_few(self):
        result = runner.invoke(
            app,
            [
                "evaluate", "significance", "only_one", "--metric", "loss",
            ],
        )
        assert result.exit_code == 0
        assert "at least 2" in result.output


class TestEvaluateNext:
    def test_next_empty(self):
        result = runner.invoke(app, ["evaluate", "next", "nonexistent"])
        assert result.exit_code == 0


class TestEvaluateList:
    def test_list_empty(self):
        result = runner.invoke(app, ["evaluate", "list"])
        assert result.exit_code == 0

    def test_list_json(self):
        result = runner.invoke(
            app, ["evaluate", "list", "--format", "json"]
        )
        assert result.exit_code == 0


class TestEvaluateGet:
    def test_get_nonexistent(self):
        result = runner.invoke(
            app, ["evaluate", "get", "nonexistent_id"]
        )
        assert result.exit_code == 0
        assert "not found" in result.output


class TestEvaluateSearch:
    def test_search(self):
        result = runner.invoke(
            app, ["evaluate", "search", "overfitting"]
        )
        assert result.exit_code == 0
