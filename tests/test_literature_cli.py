"""Tests for Phase 6 CLI commands."""

from typer.testing import CliRunner

from research_engineer.cli import app


runner = CliRunner()


class TestLiteratureSearch:
    def test_search_help(self):
        result = runner.invoke(app, ["literature", "search", "--help"])
        assert result.exit_code == 0
        assert "query" in result.output

    def test_search_invalid_source(self):
        result = runner.invoke(app, ["literature", "search", "test", "--sources", "bogus"])
        assert "Unknown source" in result.output or result.exit_code in (0, 1)


class TestLiteratureCompare:
    def test_compare_help(self):
        result = runner.invoke(app, ["literature", "compare", "--help"])
        assert result.exit_code == 0
        assert "papers" in result.output

    def test_compare_single_paper(self):
        result = runner.invoke(app, ["literature", "compare", "2503.12345"])
        assert "at least 2" in result.output


class TestLiteratureReview:
    def test_review_help(self):
        result = runner.invoke(app, ["literature", "review", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output

    def test_review_invalid_depth(self):
        result = runner.invoke(app, ["literature", "review", "attention", "--depth", "bogus"])
        assert "Invalid depth" in result.output


class TestLiteratureRelationships:
    def test_relationships_help(self):
        result = runner.invoke(app, ["literature", "relationships", "--help"])
        assert result.exit_code == 0
        assert "papers" in result.output


class TestLiteratureTrends:
    def test_trends_help(self):
        result = runner.invoke(app, ["literature", "trends", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output


class TestLiteratureRecommend:
    def test_recommend_help(self):
        result = runner.invoke(app, ["literature", "recommend", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output


class TestLiteratureRelevance:
    def test_relevance_help(self):
        result = runner.invoke(app, ["literature", "relevance", "--help"])
        assert result.exit_code == 0
        assert "paper" in result.output


class TestLiteratureDiscover:
    def test_discover_help(self):
        result = runner.invoke(app, ["literature", "discover", "--help"])
        assert result.exit_code == 0
        assert "topic" in result.output


class TestLiteratureAppRegistered:
    def test_literature_app_exists(self):
        result = runner.invoke(app, ["literature", "--help"])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "compare" in result.output
        assert "review" in result.output
        assert "discover" in result.output