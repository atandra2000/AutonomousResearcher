"""Paper domain models."""

from datetime import datetime

from pydantic import BaseModel, Field


class Author(BaseModel):
    """Represents an author of a paper."""

    name: str = Field(..., description="Author's full name")
    affiliation: str | None = Field(None, description="Author's institution")
    email: str | None = Field(None, description="Author's email")
    orcid: str | None = Field(None, description="Author's ORCID")
    roles: list[str] | None = Field(None, description="Author's roles in the paper")


class Paper(BaseModel):
    """Represents an ML research paper."""

    paper_id: str = Field(
        ...,
        pattern=r"^\d{4}\.\d{5}$",
        description="arXiv paper ID (e.g., 2503.12345)"
    )
    arxiv_id: str | None = Field(None, description="arXiv ID for references")
    arxiv_version: int | None = Field(None, description="arXiv version number")
    title: str = Field(..., description="Paper title")
    authors: list[Author] = Field(..., description="List of authors")
    abstract: str = Field(..., description="Paper abstract")
    url: str = Field(..., description="Paper URL")
    published: datetime = Field(default_factory=datetime.now, description="Publication date")
    updated: datetime | None = Field(None, description="Last updated date")
    categories: list[str] = Field(default_factory=list, description="arXiv categories")
    primary_category: str | None = Field(None, description="Primary arXiv category")
    comments: str | None = Field(None, description="Author comments")
    journal_ref: str | None = Field(None, description="Journal reference")
    doi: str | None = Field(None, description="DOI")
    acm_class: str | None = Field(None, description="ACM classification")
    msc_class: str | None = Field(None, description="Mathematics Subject Classification")
    report_num: str | None = Field(None, description="Report number")
    identifiers: list[str] = Field(default_factory=list, description="Other identifiers")
    raw_content: str | None = Field(None, description="Full paper text from PDF parsing")
    content_type: str = Field(default="pdf", description="Content type: 'pdf' or 'arxiv'")
