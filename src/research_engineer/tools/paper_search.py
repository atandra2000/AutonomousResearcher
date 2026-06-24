"""Paper Search Tool for Phase 6 - Literature Intelligence.

Searches for papers across multiple sources: local storage (previously
analyzed papers), arXiv API, and Semantic Scholar API. Deduplicates and
ranks results by relevance to the query.
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import httpx

from research_engineer.models.literature import (
    PaperSearchInput,
    PaperSearchOutput,
    SearchResult,
    SearchSource,
)
from research_engineer.tools.base import Tool, ToolError

if TYPE_CHECKING:
    from research_engineer.tools.storage import StorageTool


class PaperSearchTool(Tool[PaperSearchInput, PaperSearchOutput]):
    """Search for papers from multiple sources.

    Sources:
    - LOCAL: Query the SQLite storage for previously analyzed papers
    - ARXIV: Use the arXiv API via the `arxiv` library
    - SEMANTIC_SCHOLAR: Query the Semantic Scholar API via httpx
    """

    S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(
        self,
        storage_tool: StorageTool | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.storage = storage_tool
        self._http = http_client or httpx.AsyncClient(timeout=30.0)

    async def validate(self, input: PaperSearchInput) -> bool:
        return bool(input.query and input.sources)

    async def execute(self, input: PaperSearchInput) -> PaperSearchOutput:
        start = time.time()
        try:
            all_results: list[SearchResult] = []
            sources_searched: list[str] = []

            if SearchSource.LOCAL in input.sources:
                local = await self._search_local(input)
                all_results.extend(local)
                sources_searched.append(SearchSource.LOCAL.value)

            if SearchSource.ARXIV in input.sources:
                arxiv_res = await self._search_arxiv(input)
                all_results.extend(arxiv_res)
                sources_searched.append(SearchSource.ARXIV.value)

            if SearchSource.SEMANTIC_SCHOLAR in input.sources:
                s2_res = await self._search_semantic_scholar(input)
                all_results.extend(s2_res)
                sources_searched.append(SearchSource.SEMANTIC_SCHOLAR.value)

            deduped = self._deduplicate(all_results)
            ranked = self._rank_results(deduped, input.query, input.sort)

            if input.min_citation_count is not None:
                ranked = [
                    r for r in ranked if r.citation_count >= input.min_citation_count
                ]

            elapsed = time.time() - start
            return PaperSearchOutput(
                papers=ranked[: input.max_results * len(input.sources)],
                total_found=len(ranked),
                sources_searched=sources_searched,
                search_time_seconds=round(elapsed, 3),
            )
        except Exception as e:
            raise ToolError(f"Paper search failed: {e}", input, e)

    async def _search_local(self, input: PaperSearchInput) -> list[SearchResult]:
        """Search locally stored papers via StorageTool."""
        if self.storage is None:
            return []
        try:
            papers = await self.storage.search_papers(input.query)  # type: ignore[union-attr]
            results: list[SearchResult] = []
            for p in papers:
                results.append(
                    SearchResult(
                        paper_id=p.get("paper_id", ""),
                        title=p.get("title", ""),
                        source=SearchSource.LOCAL,
                    )
                )
            return results
        except Exception:
            return []

    async def _search_arxiv(self, input: PaperSearchInput) -> list[SearchResult]:
        """Search arXiv via the arxiv library."""
        try:
            import arxiv

            client = arxiv.Client()
            search = arxiv.Search(
                query=input.query,
                max_results=input.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results: list[SearchResult] = []
            for entry in client.results(search):
                arxiv_id = self._extract_arxiv_id(entry.entry_id)
                year = None
                if entry.published:
                    year = entry.published.year
                results.append(
                    SearchResult(
                        paper_id=arxiv_id,
                        title=entry.title,
                        authors=[str(a) for a in entry.authors],
                        abstract=entry.summary or "",
                        year=year,
                        citation_count=0,
                        source=SearchSource.ARXIV,
                        url=entry.entry_id,
                        doi=entry.doi,
                        relevance_score=0.0,
                    )
                )
            return results
        except Exception:
            return []

    async def _search_semantic_scholar(self, input: PaperSearchInput) -> list[SearchResult]:
        """Search Semantic Scholar API."""
        try:
            params: dict[str, str | int] = {
                "query": input.query,
                "limit": min(input.max_results, 50),
                "fields": "title,authors,abstract,year,citationCount,externalIds,url",
            }
            if input.year_range:
                params["year"] = input.year_range
            if input.fields_of_study:
                params["fieldsOfStudy"] = ",".join(input.fields_of_study)

            resp = await self._http.get(self.S2_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            results: list[SearchResult] = []
            for paper in data.get("data", []):
                arxiv_id = None
                ext_ids = paper.get("externalIds") or {}
                if ext_ids and ext_ids.get("ArXiv"):
                    arxiv_id = ext_ids["ArXiv"]
                paper_id = arxiv_id or ext_ids.get("DOI") or paper.get("paperId", "")
                authors = [
                    a.get("name", "")
                    for a in (paper.get("authors") or [])
                ]
                results.append(
                    SearchResult(
                        paper_id=paper_id,
                        title=paper.get("title", ""),
                        authors=authors,
                        abstract=paper.get("abstract") or "",
                        year=paper.get("year"),
                        citation_count=paper.get("citationCount", 0),
                        source=SearchSource.SEMANTIC_SCHOLAR,
                        url=paper.get("url"),
                        doi=ext_ids.get("DOI"),
                        relevance_score=0.0,
                    )
                )
            return results
        except Exception:
            return []

    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract arXiv ID from entry URL."""
        match = re.search(r"(\d{4}\.\d{5}(?:v\d+)?)", entry_id)
        return match.group(1) if match else entry_id

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Deduplicate by paper_id, keeping highest citation count."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            if not r.paper_id:
                continue
            if r.paper_id not in seen:
                seen[r.paper_id] = r
            else:
                if r.citation_count > seen[r.paper_id].citation_count:
                    seen[r.paper_id] = r
        return list(seen.values())

    def _rank_results(
        self, results: list[SearchResult], query: str, sort: str
    ) -> list[SearchResult]:
        """Rank results by relevance, citation count, or year."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        for r in results:
            r.relevance_score = self._compute_relevance(r, query_lower, query_tokens)

        if sort == "citationCount":
            results.sort(key=lambda x: x.citation_count, reverse=True)
        elif sort == "year":
            results.sort(key=lambda x: x.year or 0, reverse=True)
        else:
            results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    def _compute_relevance(
        self, result: SearchResult, query_lower: str, query_tokens: set[str]
    ) -> float:
        """Compute relevance score based on keyword overlap."""
        title_lower = result.title.lower()
        abstract_lower = result.abstract.lower()
        combined = f"{title_lower} {abstract_lower}"
        combined_tokens = set(combined.split())

        if not query_tokens or not combined_tokens:
            return 0.0

        overlap = len(query_tokens & combined_tokens) / len(query_tokens | combined_tokens)
        title_ratio = SequenceMatcher(None, query_lower, title_lower).ratio()
        score = 0.5 * overlap + 0.5 * title_ratio
        return round(min(1.0, score), 3)

    async def close(self) -> None:
        await self._http.aclose()
