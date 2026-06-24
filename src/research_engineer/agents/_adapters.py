"""Phase 13 - Agent adapters for the delegation framework.

Wraps existing Phase 1-4 agents (RepositoryAgent, ResearchAgent,
CodingAgent) so they conform to the delegation framework's
``async execute(ctx: SharedTaskContext) -> dict`` interface. This avoids
modifying the original agents' signatures while enabling them to
participate in the multi-agent pipeline.
"""

from __future__ import annotations

from typing import Any

from research_engineer.agents.coding_agent import CodingAgent
from research_engineer.agents.repository_agent import RepositoryAgent
from research_engineer.agents.research_agent import ResearchAgent
from research_engineer.models.delegation import SharedTaskContext


class RepositoryAgentAdapter:
    """Adapts :class:`RepositoryAgent` for the delegation framework.

    Reads ``ctx.repo_path``, runs repository analysis, and writes
    ``ctx.repo_summary``.
    """

    def __init__(self, agent: RepositoryAgent | None = None) -> None:
        self.agent_name = "RepositoryAgentAdapter"
        self._agent = agent or RepositoryAgent(llm_enabled=False)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        result = await self._agent.analyze(
            ctx.repo_path,
            output_dir=ctx.output_dir,
            enable_llm=False,
        )
        ctx.repo_summary = result
        ctx.add_files(result.get("generated_files", []))
        return {
            "summary": (
                f"{result.get('repository_name', 'repo')} "
                f"({result.get('project_type', 'unknown')})"
            ),
            "repo_summary": result,
        }


class ResearchAgentAdapter:
    """Adapts :class:`ResearchAgent` for the delegation framework.

    Optionally analyzes a paper (if ``ctx.paper_input`` is set) and
    writes research context. If no paper is provided, performs a
    lightweight LLM-based research summary of the goal.
    """

    def __init__(self, agent: ResearchAgent | None = None) -> None:
        self.agent_name = "ResearchAgentAdapter"
        self._agent = agent or ResearchAgent()

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        if ctx.paper_input:
            try:
                result = await self._agent.analyze(
                    ctx.paper_input, output_dir=ctx.output_dir
                )
                summary = result.get("summary", {})
                ctx.research_context = (
                    f"Paper: {result.get('title', 'N/A')}\n"
                    f"Summary: {summary.get('executive_summary', '')}"
                )
                ctx.add_files(result.get("output_files", []))
                return {
                    "summary": f"Analyzed paper: {result.get('title', 'N/A')[:80]}",
                    "research_context": ctx.research_context,
                }
            except Exception as e:
                return {"summary": f"Research failed: {e}", "error": str(e)}
        # No paper: lightweight goal analysis.
        ctx.research_context = (
            f"Goal: {ctx.goal}\nNo paper provided; "
            "proceeding with repository-grounded approach."
        )
        return {
            "summary": "No paper to research; using goal as context.",
            "research_context": ctx.research_context,
        }


class CodingAgentAdapter:
    """Adapts :class:`CodingAgent` for the delegation framework.

    Reads ``ctx.goal``, ``ctx.repo_path``, and
    ``ctx.implementation_plan``, generates patches, and writes
    ``ctx.implementation_id``, ``ctx.patches_generated``, and
    ``ctx.diff`` (via TerminalTool git_diff after patching).
    """

    def __init__(
        self,
        agent: CodingAgent | None = None,
    ) -> None:
        self.agent_name = "CodingAgentAdapter"
        self._agent = agent or CodingAgent()

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        result = await self._agent.implement(
            task_description=ctx.goal,
            repo_path=ctx.repo_path,
            paper_input=ctx.paper_input,
            output_dir=ctx.output_dir,
        )
        ctx.implementation_id = result.implementation_id
        ctx.patches_generated = result.patches_generated
        ctx.add_files(result.generated_files)
        return {
            "summary": (
                f"Generated {result.patches_generated} patch(es); "
                f"review={result.review_status}"
            ),
            "implementation_id": result.implementation_id,
            "patches_generated": result.patches_generated,
        }


__all__ = [
    "RepositoryAgentAdapter",
    "ResearchAgentAdapter",
    "CodingAgentAdapter",
]
