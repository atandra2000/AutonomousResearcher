"""Tests for research agent."""

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_research_agent_import():
    """Test that ResearchAgent can be imported."""
    from research_engineer.agents import ResearchAgent
    assert ResearchAgent is not None


@pytest.mark.asyncio
async def test_agent_creation():
    """Test agent creation with defaults."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()
    assert agent is not None


@pytest.mark.asyncio
async def test_agent_creation_with_custom_tools():
    """Test agent creation with custom tools."""
    from research_engineer.agents import ResearchAgent
    from research_engineer.tools import ArxivTool, PaperParserTool, PDFTool, StorageTool

    agent = ResearchAgent(
        arxiv_tool=ArxivTool(),
        pdf_tool=PDFTool(),
        parser_tool=PaperParserTool(),
        storage_tool=StorageTool()
    )

    assert agent.arxiv is not None
    assert agent.pdf is not None
    assert agent.parser is not None
    assert agent.storage is not None


@pytest.mark.asyncio
async def test_detect_input_type_arxiv_id():
    """Test input type detection for arXiv ID."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()
    input_type = agent._detect_input_type("2503.12345")
    assert input_type == "arxiv_id"


@pytest.mark.asyncio
async def test_detect_input_type_arxiv_url():
    """Test input type detection for arXiv URL."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()
    input_type = agent._detect_input_type("https://arxiv.org/abs/2503.12345")
    assert input_type == "arxiv_url"

    input_type = agent._detect_input_type("https://arxiv.org/pdf/2503.12345.pdf")
    assert input_type == "arxiv_url"


@pytest.mark.asyncio
async def test_detect_input_type_pdf_path():
    """Test input type detection for PDF path."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()
    input_type = agent._detect_input_type("/path/to/paper.pdf")
    assert input_type == "pdf_file"

    input_type = agent._detect_input_type("paper.pdf")
    assert input_type == "pdf_file"


@pytest.mark.asyncio
async def test_summary_extraction():
    """Test summary extraction methods."""
    from research_engineer.agents import ResearchAgent
    from research_engineer.models import Paper

    agent = ResearchAgent()

    # Create mock parsed content
    mock_parsed = MagicMock()
    mock_parsed.sections = {
        "abstract": "This is a test abstract.",
        "introduction": "This is the introduction.",
        "methods": "This describes the methods.",
        "results": "This reports the results."
    }

    # Create mock paper
    mock_paper = MagicMock(spec=Paper)
    mock_paper.paper_id = "2503.12345"
    mock_paper.abstract = "Default abstract."
    mock_paper.raw_content = "Full paper content."

    # Test extraction methods
    executive = agent._extract_executive_summary(mock_parsed.sections, mock_paper)
    assert len(executive) > 0

    problem = agent._extract_problem_statement(mock_parsed.sections)
    assert len(problem) > 0


@pytest.mark.asyncio
async def test_plan_generation():
    """Test plan generation methods."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()

    # Create mock parsed content
    mock_parsed = MagicMock()
    mock_parsed.sections = {
        "methods": "Transformers and attention mechanisms.",
        "results": "Achieved 90% accuracy."
    }

    complexity = agent._assess_complexity(mock_parsed)
    assert complexity in ["Low", "Medium", "High"]


@pytest.mark.asyncio
async def test_complexity_assessment():
    """Test complexity assessment."""
    from research_engineer.agents import ResearchAgent

    agent = ResearchAgent()

    # Test with Transformer mention
    mock_parsed = MagicMock()
    mock_parsed.sections = {"methods": "Transformer model"}
    complexity = agent._assess_complexity(mock_parsed)
    assert complexity == "High"

    # Test with CNN mention
    mock_parsed = MagicMock()
    mock_parsed.sections = {"methods": "CNN architecture"}
    complexity = agent._assess_complexity(mock_parsed)
    assert complexity == "Medium"
