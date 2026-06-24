"""Integration tests for research engine."""

from datetime import datetime

import pytest

from research_engineer.agents import ResearchAgent
from research_engineer.models import (
    Author,
    ComplexityMetrics,
    EngineeringReport,
    FileRequirement,
    Paper,
    ResearchSummary,
)
from research_engineer.tools import StorageInput


def _make_paper(paper_id="2503.12345"):
    return Paper(
        paper_id=paper_id,
        title="Test Paper",
        authors=[Author(name="Test Author")],
        abstract="Test abstract.",
        url=f"https://arxiv.org/abs/{paper_id}",
        content_type="arxiv",
        published=datetime.now(),
    )


def _make_summary(paper_id="2503.12345"):
    return ResearchSummary(
        paper_id=paper_id,
        executive_summary="Summary.",
        problem_statement="Problem.",
        core_contributions=["Contribution 1"],
        model_architecture="Transformer.",
        training_methodology="Training.",
        dataset_information="Dataset.",
        evaluation_methodology="Evaluation.",
        key_results=["Result 1"],
        limitations=["Limitation 1"],
        reproduction_challenges=["Challenge 1"],
    )


def _make_plan(paper_id="2503.12345"):
    return EngineeringReport(
        paper_id=paper_id,
        complexity_analysis=ComplexityMetrics(),
        step_by_step_implementation="Steps.",
        files_required=[
            FileRequirement(
                filename="model.py",
                path="src/",
                purpose="Model architecture",
            )
        ],
        development_effort="3-5 days",
        dependencies=["torch"],
        pytorch_modules=["torch.nn"],
    )


@pytest.mark.asyncio
async def test_full_pipeline_integration():
    """Test full pipeline from start to finish."""
    agent = ResearchAgent()

    mock_paper = _make_paper()
    mock_summary = _make_summary()
    mock_plan = _make_plan()

    storage_input = StorageInput(
        paper=mock_paper,
        summary=mock_summary,
        plan=mock_plan,
    )

    storage_output = await agent.storage.execute(storage_input)
    assert storage_output.record_id is not None
    assert storage_output.paper_id == "2503.12345"


@pytest.mark.asyncio
async def test_storage_roundtrip():
    """Test storing and retrieving papers."""
    agent = ResearchAgent()

    paper = _make_paper("2503.54321")
    summary = _make_summary("2503.54321")
    plan = _make_plan("2503.54321")

    storage_input = StorageInput(paper=paper, summary=summary, plan=plan)
    await agent.storage.execute(storage_input)

    retrieved = await agent.storage.get_paper("2503.54321")
    assert retrieved is not None
    assert retrieved["paper_id"] == "2503.54321"

    await agent.storage.delete_paper("2503.54321")


@pytest.mark.asyncio
async def test_list_papers():
    """Test listing papers from storage."""
    agent = ResearchAgent()
    papers = await agent.storage.list_papers(limit=5)
    assert isinstance(papers, list)


@pytest.mark.asyncio
async def test_search_papers():
    """Test searching papers in storage."""
    agent = ResearchAgent()
    papers = await agent.storage.search_papers("nonexistent")
    assert isinstance(papers, list)
