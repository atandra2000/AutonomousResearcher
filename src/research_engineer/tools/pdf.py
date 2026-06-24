"""PDF tool for parsing local PDF files."""

import re
from datetime import datetime
from pathlib import Path

import fitz as pymupdf
import httpx
from pydantic import BaseModel, Field

from research_engineer.models.paper import Author, Paper
from research_engineer.tools.base import Tool, ToolError


class PDFInput(BaseModel):
    """Input for PDF tool."""

    file_path: str | None = Field(
        default=None,
        description="Path to PDF file"
    )
    file_bytes: bytes | None = Field(
        default=None,
        description="PDF file bytes"
    )
    file_url: str | None = Field(
        default=None,
        description="URL to download PDF"
    )


class FigureInfo(BaseModel):
    """Information about a figure in the PDF."""

    page: int = Field(..., description="Page number")
    bbox: list = Field(..., description="Bounding box [x0, y0, x1, y1]")
    width: int = Field(..., description="Figure width")
    height: int = Field(..., description="Figure height")
    type: str = Field(default="image", description="Figure type")


class TableInfo(BaseModel):
    """Information about a table in the PDF."""

    page: int = Field(..., description="Page number")
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    bbox: list = Field(..., description="Bounding box")


class PDFOutput(BaseModel):
    """Output from PDF tool."""

    paper: Paper = Field(..., description="Parsed paper")
    pages: int = Field(..., description="Number of pages")
    figures: list = Field(default_factory=list, description="List of figures")
    tables: list = Field(default_factory=list, description="List of tables")
    raw_text: str = Field(..., description="Extracted text from all pages")
    text_per_page: dict = Field(default_factory=dict, description="Text per page")


class PDFTool(Tool[PDFInput, PDFOutput]):
    """Parse PDF files and extract paper metadata."""

    def __init__(self):
        self.supported_extensions = {".pdf"}

    async def validate(self, input: PDFInput) -> bool:
        """Validate PDF input."""
        if input.file_path:
            return Path(input.file_path).exists()
        elif input.file_bytes:
            return len(input.file_bytes) > 0
        elif input.file_url:
            return bool(re.match(r"^https?://.*\.pdf", input.file_url))
        return False

    async def _extract_text(self, doc: pymupdf.Document) -> str:
        """Extract text from PDF document."""
        raw_text = ""
        for page_num, page in enumerate(doc):
            raw_text += f"\n\n--- PAGE {page_num + 1} ---\n"
            raw_text += page.get_text()
        return raw_text

    async def _find_figures(self, doc: pymupdf.Document) -> list:
        """Find figures in PDF."""
        figures = []
        for page_num, page in enumerate(doc):
            # Find image references
            images = page.get_images(full=True)
            for img in images:
                figures.append({
                    "page": page_num,
                    "type": "image",
                    "xref": img[0]
                })

            # Try to find figure captions
            text = page.get_text()
            figure_matches = re.finditer(
                r"(?i)(figure|fig\.?)\s*(\d+)",
                text
            )
            for match in figure_matches:
                figures.append({
                    "page": page_num,
                    "type": "figure_caption",
                    "number": match.group(2)
                })

        return figures

    async def _find_tables(self, doc: pymupdf.Document) -> list:
        """Find tables in PDF."""
        tables = []
        for page_num, page in enumerate(doc):
            # Find tables using pymupdf's table finder
            tables_on_page = page.find_tables()
            for table in tables_on_page:
                tables.append({
                    "page": page_num,
                    "rows": len(table.rows),
                    "columns": len(table.cols)
                })

        return tables

    async def execute(self, input: PDFInput) -> PDFOutput:
        """Parse PDF and extract content."""
        try:
            # Load document
            doc: pymupdf.Document
            if input.file_path:
                doc = pymupdf.open(input.file_path)
            elif input.file_bytes:
                doc = pymupdf.open(stream=input.file_bytes, filetype="pdf")
            elif input.file_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(input.file_url)
                    doc = pymupdf.open(stream=response.content, filetype="pdf")

            # Extract pages
            pages = len(doc)

            # Extract text
            raw_text = await self._extract_text(doc)

            # Extract figures and tables
            figures = await self._find_figures(doc)
            tables = await self._find_tables(doc)

            # Extract metadata
            metadata = doc.metadata
            title = metadata.get("title", "Unknown")

            # Parse authors (simple heuristic)
            authors = []
            author_text = metadata.get("author", "")
            if author_text:
                for name in re.split(r"\s+and\s+|,\s+", author_text):
                    authors.append(Author(name=name.strip()))

            # Generate paper ID
            import hashlib
            paper_id = f"local_{hashlib.md5(raw_text[:200].encode()).hexdigest()[:8]}"

            # Build Paper model
            paper = Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=raw_text[:500] if raw_text else "",
                url=input.file_path or input.file_url or "manual",
                published=datetime.now(),
                raw_content=raw_text,
                content_type="pdf"
            )

            # Build metadata per page
            text_per_page = {}
            for page_num, page in enumerate(doc):
                text_per_page[str(page_num + 1)] = page.get_text()

            return PDFOutput(
                paper=paper,
                pages=pages,
                figures=figures,
                tables=tables,
                raw_text=raw_text,
                text_per_page=text_per_page
            )

        except pymupdf.FileDataError as e:
            raise ToolError(f"Invalid PDF file: {e}", input, e)
        except Exception as e:
            raise ToolError(f"Failed to parse PDF: {e}", input, e)
        finally:
            if 'doc' in locals():
                doc.close()

    async def close(self):
        """Close resources."""
        pass
