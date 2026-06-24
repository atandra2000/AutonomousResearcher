"""Tests for research engine models."""


import pytest
from pydantic import ValidationError

from research_engineer.models import (
    Author,
    ComplexityMetrics,
    EngineeringReport,
    FileRequirement,
    Paper,
    ResearchSummary,
)


class TestAuthor:
    """Tests for Author model."""

    def test_author_creation(self):
        """Test author creation with required fields."""
        author = Author(name="John Doe")
        assert author.name == "John Doe"
        assert author.affiliation is None
        assert author.email is None

    def test_author_with_optional_fields(self):
        """Test author creation with optional fields."""
        author = Author(
            name="John Doe",
            affiliation="MIT",
            email="joe@example.com",
            orcid="0000-0000-0000-0000"
        )
        assert author.affiliation == "MIT"
        assert author.email == "joe@example.com"
        assert author.orcid == "0000-0000-0000-0000"

    def test_author_validation_missing_name(self):
        """Test that author requires name."""
        with pytest.raises(ValidationError):
            Author()


class TestPaper:
    """Tests for Paper model."""

    def test_paper_creation(self):
        """Test paper creation with required fields."""
        author = Author(name="John Doe")
        paper = Paper(
            paper_id="2503.12345",
            title="Test Paper",
            authors=[author],
            abstract="Test abstract",
            url="https://arxiv.org/abs/2503.12345",
            content_type="arxiv"
        )
        assert paper.paper_id == "2503.12345"
        assert paper.title == "Test Paper"

    def test_paper_validation_invalid_id(self):
        """Test that paper validates paper_id format."""
        author = Author(name="John Doe")
        with pytest.raises(ValidationError):
            Paper(
                paper_id="invalid",
                title="Test",
                authors=[author],
                abstract="Test",
                url="https://arxiv.org/abs/invalid",
                content_type="arxiv"
            )

    def test_paper_default_content_type(self):
        """Test default content type."""
        author = Author(name="John Doe")
        paper = Paper(
            paper_id="2503.12345",
            title="Test",
            authors=[author],
            abstract="Test",
            url="https://arxiv.org/abs/2503.12345"
        )
        assert paper.content_type == "pdf"


class TestComplexityMetrics:
    """Tests for ComplexityMetrics model."""

    def test_default_metrics(self):
        """Test default complexity metrics."""
        metrics = ComplexityMetrics()
        assert metrics.code_complexity == "Medium"
        assert metrics.training_time == "1-2 days on 2x A100"

    def test_custom_metrics(self):
        """Test custom complexity metrics."""
        metrics = ComplexityMetrics(
            code_complexity="High",
            data_requirements="High",
            compute_requirements="High"
        )
        assert metrics.code_complexity == "High"
        assert metrics.data_requirements == "High"
        assert metrics.compute_requirements == "High"


class TestFileRequirement:
    """Tests for FileRequirement model."""

    def test_file_requirement(self):
        """Test file requirement creation."""
        req = FileRequirement(
            filename="model.py",
            path="model/",
            purpose="Model architecture",
            complexity="Medium",
            estimated_lines=100
        )
        assert req.filename == "model.py"
        assert req.purpose == "Model architecture"

    def test_file_requirement_defaults(self):
        """Test file requirement with defaults."""
        req = FileRequirement(
            filename="test.py",
            path="",
            purpose="Tests"
        )
        assert req.complexity == "Medium"
        assert req.estimated_lines == 100


class TestResearchSummary:
    """Tests for ResearchSummary model."""

    def test_summary_creation(self):
        """Test research summary creation."""
        summary = ResearchSummary(
            paper_id="2503.12345",
            executive_summary="Summary.",
            problem_statement="Problem.",
            core_contributions=["Contribution 1"],
            model_architecture="Transformer.",
            training_methodology="Training.",
            dataset_information="Dataset.",
            evaluation_methodology="Evaluation.",
            key_results=["Result 1"],
            limitations=["Limitation 1"],
            reproduction_challenges=["Challenge 1"]
        )
        assert summary.paper_id == "2503.12345"
        assert len(summary.core_contributions) == 1
        assert len(summary.key_results) == 1
        assert len(summary.limitations) == 1

    def test_summary_validation_min_items(self):
        """Test that summary requires min items for lists."""
        with pytest.raises(ValidationError):
            ResearchSummary(
                paper_id="2503.12345",
                executive_summary="Summary.",
                problem_statement="Problem.",
                core_contributions=[],  # Empty list
                model_architecture="Transformer.",
                training_methodology="Training.",
                dataset_information="Dataset.",
                evaluation_methodology="Evaluation.",
                key_results=["Result"],
                limitations=["Limitation"],
                reproduction_challenges=["Challenge"]
            )


class TestEngineeringReport:
    """Tests for EngineeringReport model."""

    def test_report_creation(self):
        """Test engineering report creation."""
        metrics = ComplexityMetrics()
        file_req = FileRequirement(
            filename="model.py",
            path="src/",
            purpose="Model architecture"
        )
        report = EngineeringReport(
            paper_id="2503.12345",
            complexity_analysis=metrics,
            step_by_step_implementation="Steps.",
            files_required=[file_req],
            development_effort="3-5 days",
            dependencies=["torch"],
            pytorch_modules=["torch.nn"]
        )
        assert report.paper_id == "2503.12345"
        assert "torch" in report.dependencies
        assert "torch.nn" in report.pytorch_modules
        assert len(report.files_required) == 1
