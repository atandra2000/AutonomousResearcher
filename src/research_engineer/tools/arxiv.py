"""arXiv tool for paper metadata fetching."""

import re

import arxiv
import httpx
from pydantic import BaseModel, Field

from research_engineer.models.paper import Author, Paper
from research_engineer.tools.base import Tool, ToolError


class ArxivInput(BaseModel):
    """Input for arXiv tool."""

    arxiv_id: str | None = Field(
        default=None,
        pattern=r"^\d{4}\.\d{5}(v\d+)?$",
        description="arXiv paper ID (e.g., 2503.12345 or 2503.12345v1)"
    )
    arxiv_url: str | None = Field(
        default=None,
        description="arXiv URL (e.g., https://arxiv.org/abs/2503.12345)"
    )


class ArxivOutput(BaseModel):
    """Output from arXiv tool."""

    paper: Paper = Field(..., description="Paper metadata")
    pdf_url: str | None = Field(None, description="PDF download URL")
    references: list = Field(default_factory=list, description="Reference papers")
    citations: list = Field(default_factory=list, description="Citation papers")
    arxiv_id: str = Field(..., description="Extracted arXiv ID")
    version: int | None = Field(None, description="arXiv version")


class ArxivTool(Tool[ArxivInput, ArxivOutput]):
    """Fetch paper metadata from arXiv API."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=30.0)

    async def _extract_arxiv_id(self, arxiv_url: str) -> str:
        """Extract arXiv ID from URL."""
        patterns = [
            r"arxiv\.org/abs/(\d{4}\.\d{5}(?:v\d+)?)",
            r"arxiv\.org/pdf/(\d{4}\.\d{5}(?:v\d+)?)\.pdf",
            r"arXiv:(\d{4}\.\d{5}(?:v\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, arxiv_url, re.IGNORECASE)
            if match:
                return match.group(1)
        raise ValueError(f"Invalid arXiv URL: {arxiv_url}")

    async def validate(self, input: ArxivInput) -> bool:
        """Validate arXiv input."""
        if input.arxiv_id:
            return bool(re.match(r"^\d{4}\.\d{5}(v\d+)?$", input.arxiv_id))
        elif input.arxiv_url:
            try:
                await self._extract_arxiv_id(input.arxiv_url)
                return True
            except ValueError:
                return False
        return False

    async def execute(self, input: ArxivInput) -> ArxivOutput:
        """Fetch paper from arXiv."""
        try:
            # Extract ID if URL provided
            if input.arxiv_url and not input.arxiv_id:
                arxiv_id = await self._extract_arxiv_id(input.arxiv_url)
            else:
                arxiv_id = input.arxiv_id

            # Query arXiv API with Client for newer library versions
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
            results = list(client.results(search))

            if not results:
                raise ToolError(f"Paper not found: {arxiv_id}", input)

            result = results[0]

            # Parse authors
            authors = []
            for author_name in result.authors:
                authors.append(Author(name=str(author_name)))

            # Get version from entry_id (format: http://arxiv.org/abs/2503.12345v3)
            version = None
            if 'v' in result.entry_id:
                version = int(result.entry_id.split('v')[-1])

            # Build Paper model
            paper = Paper(
                paper_id=arxiv_id,
                arxiv_id=arxiv_id,
                arxiv_version=version,
                title=result.title,
                authors=authors,
                abstract=result.summary,
                url=result.entry_id,
                published=result.published,
                updated=result.updated,
                categories=list(result.categories),
                primary_category=result.primary_category,
                comments=result.comment,
                journal_ref=result.journal_ref,
                doi=result.doi,
                content_type="arxiv"
            )

            return ArxivOutput(
                paper=paper,
                pdf_url=result.pdf_url,
                references=[],
                citations=[],
                arxiv_id=arxiv_id,
                version=version
            )

        except arxiv.HTTPError as e:
            raise ToolError(f"arXiv API error: {e}", input, e)
        except Exception as e:
            raise ToolError(f"Failed to fetch arXiv paper: {e}", input, e)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
