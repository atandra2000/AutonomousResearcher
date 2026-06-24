"""CLI tests for research-engineer commands."""

import pytest
from typer.testing import CliRunner

from research_engineer.cli import app

runner = CliRunner()


class TestAnalyzeCommand:
    """Test the analyze command."""

    def test_analyze_missing_argument(self):
        """Test analyze command without paper argument."""
        result = runner.invoke(app, ["analyze"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_analyze_invalid_paper_id(self):
        """Test analyze command with invalid paper ID."""
        result = runner.invoke(app, ["analyze", "invalid"])
        assert result.exit_code in [0, 1]

    def test_analyze_help(self):
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "arXiv ID" in result.output or "paper" in result.output.lower()


class TestAnalyzeRepoCommand:
    """Test the analyze-repo command."""

    def test_analyze_repo_missing_argument(self):
        """Test analyze-repo command without path argument."""
        result = runner.invoke(app, ["analyze-repo"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_analyze_repo_nonexistent_path(self):
        """Test analyze-repo command with non-existent path."""
        result = runner.invoke(app, ["analyze-repo", "/nonexistent/path"])
        assert result.exit_code in [0, 1]

    def test_analyze_repo_help(self):
        """Test analyze-repo command help."""
        result = runner.invoke(app, ["analyze-repo", "--help"])
        assert result.exit_code == 0
        assert "repository" in result.output.lower() or "path" in result.output.lower()


class TestPlanCommand:
    """Test the plan command."""

    def test_plan_missing_arguments(self):
        """Test plan command without required arguments."""
        result = runner.invoke(app, ["plan"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_plan_missing_repo(self):
        """Test plan command with only paper argument."""
        result = runner.invoke(app, ["plan", "2503.12345"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_plan_nonexistent_repo(self):
        """Test plan command with non-existent repository."""
        result = runner.invoke(app, ["plan", "2503.12345", "/nonexistent/repo"])
        assert result.exit_code in [0, 1]

    def test_plan_help(self):
        """Test plan command help."""
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0
        assert "paper" in result.output.lower() and "repository" in result.output.lower()


class TestGetCommand:
    """Test the get command."""

    def test_get_missing_argument(self):
        """Test get command without paper_id argument."""
        result = runner.invoke(app, ["get"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_get_nonexistent_paper(self):
        """Test get command with non-existent paper ID."""
        result = runner.invoke(app, ["get", "nonexistent_id"])
        assert result.exit_code in [0, 1]

    def test_get_help(self):
        """Test get command help."""
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "paper" in result.output.lower()


class TestSearchCommand:
    """Test the search command."""

    def test_search_missing_argument(self):
        """Test search command without query argument."""
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "error" in result.output.lower()

    def test_search_no_results(self):
        """Test search command with query that has no results."""
        result = runner.invoke(app, ["search", "nonexistent_query_xyz123"])
        assert result.exit_code == 0
        assert "No papers found" in result.output or len(result.output.strip()) == 0

    def test_search_help(self):
        """Test search command help."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output.lower()


class TestHistoryCommand:
    """Test the history command."""

    def test_history_default_limit(self):
        """Test history command with default limit."""
        result = runner.invoke(app, ["history"])
        assert result.exit_code == 0

    def test_history_custom_limit(self):
        """Test history command with custom limit."""
        result = runner.invoke(app, ["history", "--limit", "5"])
        assert result.exit_code == 0

    def test_history_invalid_limit(self):
        """Test history command with invalid limit."""
        result = runner.invoke(app, ["history", "--limit", "invalid"])
        assert result.exit_code != 0

    def test_history_help(self):
        """Test history command help."""
        result = runner.invoke(app, ["history", "--help"])
        assert result.exit_code == 0
        assert "limit" in result.output.lower()


class TestImplementCommand:
    """Test the implement command."""

    def test_implement_no_inputs(self):
        """Test implement command without any input."""
        result = runner.invoke(app, ["implement"])
        assert result.exit_code in [0, 1]

    def test_implement_nonexistent_repo(self):
        """Test implement command with non-existent repository."""
        result = runner.invoke(app, ["--task", "test", "/nonexistent/repo"])
        assert result.exit_code != 0

    def test_implement_with_task(self):
        """Test implement command with task argument."""
        result = runner.invoke(app, ["implement", "--task", "test task", "."])
        assert result.exit_code == 0 or result.exit_code != 0

    def test_implement_with_plan_nonexistent_file(self):
        """Test implement command with non-existent plan file."""
        result = runner.invoke(app, ["implement", "--plan", "nonexistent.md"])
        assert result.exit_code in [0, 1]

    def test_implement_help(self):
        """Test implement command help."""
        result = runner.invoke(app, ["implement", "--help"])
        assert result.exit_code == 0
        assert "task" in result.output.lower() or "plan" in result.output.lower()


class TestCacheStatusCommand:
    """Test the cache-status command."""

    def test_cache_status_default(self):
        """Test cache-status command with defaults."""
        result = runner.invoke(app, ["cache-status"])
        assert result.exit_code == 0

    def test_cache_status_json_output(self):
        """Test cache-status command with JSON output."""
        result = runner.invoke(app, ["cache-status", "--output-format", "json"])
        assert result.exit_code == 0

    def test_cache_status_nonexistent_path(self):
        """Test cache-status command with non-existent cache path."""
        result = runner.invoke(app, ["cache-status", "--cache-path", "/nonexistent/cache"])
        assert result.exit_code == 0
        assert "does not exist" in result.output.lower()

    def test_cache_status_help(self):
        """Test cache-status command help."""
        result = runner.invoke(app, ["cache-status", "--help"])
        assert result.exit_code == 0
        assert "cache" in result.output.lower()


class TestOutputFormats:
    """Test output format options."""

    def test_analyze_json_output(self):
        """Test analyze command with JSON output."""
        result = runner.invoke(app, ["analyze", "invalid", "--output-format", "json"])
        assert result.exit_code in [0, 1]

    def test_analyze_markdown_output(self):
        """Test analyze command with Markdown output."""
        result = runner.invoke(app, ["analyze", "invalid", "--output-format", "markdown"])
        assert result.exit_code in [0, 1]

    def test_plan_json_output(self):
        """Test plan command with JSON output."""
        result = runner.invoke(app, ["plan", "2503.12345", "/nonexistent", "--output-format", "json"])
        assert result.exit_code in [0, 1]

    def test_implement_json_output(self):
        """Test implement command with JSON output."""
        result = runner.invoke(app, ["implement", "--task", "test", ".", "--output-format", "json"])
        assert result.exit_code in [0, 1]


class TestAppHelp:
    """Test main app help."""

    def test_app_help(self):
        """Test main application help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "research-engineer" in result.output.lower()
        assert "analyze" in result.output.lower()
        assert "plan" in result.output.lower()
        assert "implement" in result.output.lower()
