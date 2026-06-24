"""CLI tests for the `task` command (Phase 11)."""

from __future__ import annotations

from typer.testing import CliRunner

from research_engineer.cli import app

runner = CliRunner()


class TestTaskCommand:
    """Test the `task` CLI command."""

    def test_task_help(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "coding goal" in result.output.lower() or "goal" in result.output.lower()

    def test_task_missing_goal(self):
        result = runner.invoke(app, ["task"])
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "error" in result.output.lower()

    def test_task_nonexistent_repo(self):
        """Running against a nonexistent repo should fail gracefully."""
        result = runner.invoke(
            app,
            [
                "task",
                "do something",
                "--repo",
                "/nonexistent/path/xyz",
                "--no-stream",
            ],
        )
        # Should exit 0 or 1 (graceful error), not crash with traceback
        assert result.exit_code in (0, 1)

    def test_task_dry_run_flag_documented(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "dry_run" in result.output.lower()

    def test_task_run_tests_flag_documented(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "run-tests" in result.output.lower() or "run_tests" in result.output.lower()

    def test_task_stream_flag_documented(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "stream" in result.output.lower()

    def test_task_paper_flag_documented(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "paper" in result.output.lower()

    def test_task_output_format_json_documented(self):
        result = runner.invoke(app, ["task", "--help"])
        assert result.exit_code == 0
        assert "format" in result.output.lower()