"""Phase 15 - Research orchestrator agent.

The top-level coordinator for autonomous research workflows. Wraps the
:class:`ResearchWorkflowFramework` and integrates with the existing
agent ecosystem (LiteratureAgent, RepositoryMemory, TerminalTool,
ModelRouter).

This agent is the entry point for the ``research-engineer research``
CLI command. It builds the workflow framework with the appropriate
agents, runs the full pipeline, and returns a :class:`ResearchResult`.
"""

from __future__ import annotations

from typing import Any

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.agents.research_workflow import (
    ResearchConfig,
    ResearchWorkflowFramework,
)
from research_engineer.llm import LLMProvider
from research_engineer.models.research import ResearchResult


class ResearchOrchestrator:
    """Coordinates the full autonomous research workflow.

    Parameters
    ----------
    literature_agent:
        Optional existing :class:`LiteratureAgent` for paper discovery.
    terminal_tool:
        :class:`TerminalTool` for experiment execution.
    repository_memory:
        Optional :class:`RepositoryMemory` (Phase 12) for context.
    llm:
        Optional explicit LLM provider (overrides router).
    """

    def __init__(
        self,
        *,
        literature_agent: Any | None = None,
        terminal_tool: Any | None = None,
        repository_memory: Any | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "ResearchOrchestrator"
        self.literature_agent = literature_agent
        self.terminal_tool = terminal_tool
        self.repository_memory = repository_memory
        self.llm_provider = resolve_llm(self.agent_name, llm)
        self._framework: ResearchWorkflowFramework | None = None

    def _get_framework(
        self, config: ResearchConfig | None = None
    ) -> ResearchWorkflowFramework:
        """Build or return the workflow framework."""
        if self._framework is None:
            self._framework = ResearchWorkflowFramework(
                literature_agent=self.literature_agent,
                terminal_tool=self.terminal_tool,
                config=config,
            )
        return self._framework

    async def run(
        self,
        research_goal: str,
        repo_path: str = ".",
        config: ResearchConfig | None = None,
        stream_sink: Any | None = None,
    ) -> ResearchResult:
        """Run the full autonomous research workflow.

        Args:
            research_goal: The research objective to investigate.
            repo_path: Repository for experiment execution.
            config: Optional :class:`ResearchConfig` override.
            stream_sink: Optional LLM streaming sink.

        Returns:
            :class:`ResearchResult` with all stage records and the
            final report.
        """
        framework = self._get_framework(config)
        return await framework.run(
            research_goal=research_goal,
            repo_path=repo_path,
            config=config,
            stream_sink=stream_sink,
        )


__all__ = ["ResearchOrchestrator"]
