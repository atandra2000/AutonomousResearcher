"""Tests for research engine tools."""


import pytest


@pytest.mark.asyncio
async def test_arxiv_tool_import():
    """Test that ArxivTool can be imported."""
    from research_engineer.tools import ArxivTool
    assert ArxivTool is not None


@pytest.mark.asyncio
async def test_pdf_tool_import():
    """Test that PDFTool can be imported."""
    from research_engineer.tools import PDFTool
    assert PDFTool is not None


@pytest.mark.asyncio
async def test_parser_tool_import():
    """Test that PaperParserTool can be imported."""
    from research_engineer.tools import PaperParserTool
    assert PaperParserTool is not None


@pytest.mark.asyncio
async def test_storage_tool_import():
    """Test that StorageTool can be imported."""
    from research_engineer.tools import StorageTool
    assert StorageTool is not None


@pytest.mark.asyncio
async def test_tool_base():
    """Test base Tool interface."""
    from research_engineer.tools.base import Tool

    # Check that Tool is abstract
    assert hasattr(Tool, 'execute')


@pytest.mark.asyncio
async def test_tool_error():
    """Test ToolError exception."""
    from research_engineer.tools.base import ToolError

    error = ToolError("Test error")
    assert str(error) == "Test error"

    cause = Exception("Cause")
    error = ToolError("Test error", cause=cause)
    assert error.cause == cause
