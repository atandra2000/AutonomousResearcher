"""Tests for memory CLI commands."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_engineer.cli import app


runner = CliRunner()


class TestMemorySearchCommand:
    """Test memory search subcommand."""

    def test_search_runs(self, tmp_path, monkeypatch):
        """Search should run without crashing (empty result OK)."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "search", "attention"])
        assert result.exit_code == 0

    def test_search_json_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "search", "attention", "--format", "json"])
        assert result.exit_code == 0

    def test_search_invalid_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "search", "test", "--type", "invalid_type"])
        assert result.exit_code == 1


class TestMemoryListCommand:
    """Test memory list subcommand."""

    def test_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "list"])
        assert result.exit_code == 0

    def test_list_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "list", "--format", "json"])
        assert result.exit_code == 0

    def test_list_with_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "list", "--type", "paper"])
        assert result.exit_code == 0

    def test_list_invalid_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "list", "--type", "bogus"])
        assert result.exit_code == 1

    def test_list_include_archived(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "list", "--include-archived"])
        assert result.exit_code == 0


class TestMemoryStatsCommand:
    """Test memory stats subcommand."""

    def test_stats_console(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "stats"])
        assert result.exit_code == 0

    def test_stats_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "stats", "--format", "json"])
        assert result.exit_code == 0


class TestMemoryRelatedCommand:
    """Test memory related subcommand."""

    def test_related_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "related", "nonexistent_id"])
        assert result.exit_code == 0

    def test_related_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "related", "nonexistent", "--format", "json"])
        assert result.exit_code == 0


class TestMemoryGraphCommand:
    """Test memory graph subcommand."""

    def test_graph_stats(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "graph", "--op", "stats"])
        assert result.exit_code == 0

    def test_graph_stats_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "graph", "--op", "stats", "--format", "json"])
        assert result.exit_code == 0

    def test_graph_neighbors_no_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "graph", "--op", "neighbors"])
        assert result.exit_code == 1


class TestMemoryExportImport:
    """Test export and import round-trip."""

    def test_export_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_file = str(tmp_path / "export.json")
        result = runner.invoke(app, ["memory", "export", "--output", out_file])
        assert result.exit_code == 0, f"Output: {result.output}"
        assert Path(out_file).exists()
        data = json.loads(Path(out_file).read_text())
        assert "memories" in data
        assert "relationships" in data

    def test_import_nonexistent_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "import", str(tmp_path / "nonexistent.json")])
        assert result.exit_code == 1

    def test_import_empty_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"memories": [], "relationships": []}))
        result = runner.invoke(app, ["memory", "import", str(f)])
        assert result.exit_code == 0

    def test_export_import_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        export_file = str(tmp_path / "export.json")
        # Export empty
        runner.invoke(app, ["memory", "export", "--output", export_file])
        # Import it back
        result = runner.invoke(app, ["memory", "import", export_file])
        assert result.exit_code == 0


class TestMemoryArchiveCommand:
    """Test memory archive subcommand."""

    def test_archive_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "archive", "--older-than", "90", "--dry-run"])
        assert result.exit_code == 0

    def test_archive_with_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["memory", "archive", "--older-than", "30", "--type", "paper", "--dry-run"]
        )
        assert result.exit_code == 0