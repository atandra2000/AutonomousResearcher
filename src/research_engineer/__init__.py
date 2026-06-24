"""Autonomous ML Research Engine - Analyze papers and generate implementation plans."""

__version__ = "0.8.0"
__author__ = "Autonomous ML Research Engineer Team"

from .agents import (
    CodingAgent,
    EvaluationAgent,
    ExperimentAgent,
    ExperimentPlannerAgent,
    LiteratureAgent,
    MemoryAgent,
    RepositoryAgent,
    ResearchAgent,
)
from .models import (
    EngineeringReport,
    Paper,
    ResearchSummary,
)
from .tools import (
    ArxivTool,
    PaperParserTool,
    PDFTool,
    StorageTool,
)

__all__ = [
    # Core Agents (Phase 1-8)
    "ResearchAgent",
    "RepositoryAgent",
    "ExperimentPlannerAgent",
    "CodingAgent",
    "MemoryAgent",
    "LiteratureAgent",
    "ExperimentAgent",
    "EvaluationAgent",
    # Tools
    "ArxivTool",
    "PDFTool",
    "PaperParserTool",
    "StorageTool",
    # Models
    "Paper",
    "ResearchSummary",
    "EngineeringReport",
]
