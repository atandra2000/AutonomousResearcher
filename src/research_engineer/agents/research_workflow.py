"""Phase 15 - Research workflow stage authority.

The seven-stage research pipeline — literature discovery → knowledge
synthesis → hypothesis generation → experiment planning → experiment
execution → result analysis → report generation — is orchestrated
exclusively by the LangGraph engine (:mod:`research_engineer.graphs`).
This module is the stage-executor authority the graph delegates to:
it owns the stage agents and the shared structured context
(:class:`SharedResearchContext`), and executes one stage at a time.

Design principles:
- **Structured artifacts**: All inter-stage communication flows through
  :class:`SharedResearchContext`; stages never call each other directly.
- **Configurable pipeline**: Stages can be skipped via
  :class:`ResearchConfig` (honored by the graph).
- **Full traceability**: Each stage produces a
  :class:`ResearchStageRecord` with timing, status, and output.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from research_engineer.agents.research_stages import (
    ExperimentExecutorAgent,
    HypothesisGeneratorAgent,
    KnowledgeSynthesisAgent,
    LiteratureDiscoveryAgent,
    ReportGeneratorAgent,
    ResearchExperimentPlannerAgent,
    ResultAnalyzerAgent,
)
from research_engineer.models.research import (
    ResearchStageRecord,
    ResearchStageStatus,
    ResearchStageType,
    SharedResearchContext,
)


class ResearchConfig:
    """Configuration for the research workflow.

    Controls which stages run, paper/hypothesis limits, and experiment
    execution settings. Orchestration always goes through the LangGraph
    engine; ``thread_id`` names the checkpoint thread.
    """

    def __init__(
        self,
        *,
        max_papers: int = 20,
        max_hypotheses: int = 5,
        dry_run_experiments: bool = True,
        experiment_timeout: int = 3600,
        skip_stages: list[ResearchStageType] | None = None,
        stream: bool = True,
        output_dir: str = "output/research",
        llm_enabled: bool = True,
        thread_id: str | None = None,
    ) -> None:
        self.max_papers = max_papers
        self.max_hypotheses = max_hypotheses
        self.dry_run_experiments = dry_run_experiments
        self.experiment_timeout = experiment_timeout
        self.skip_stages = skip_stages or []
        self.stream = stream
        self.output_dir = output_dir
        self.llm_enabled = llm_enabled
        self.thread_id = thread_id


class ResearchWorkflowFramework:
    """Stage executor for the seven-stage research pipeline.

    The LangGraph engine (``research_engineer.graphs.ResearchGraph``)
    drives orchestration; this class owns the stage agents and executes
    one stage at a time against the shared research context.

    Parameters
    ----------
    literature_agent:
        Optional existing :class:`LiteratureAgent` for paper discovery.
    terminal_tool:
        :class:`TerminalTool` for experiment execution.
    config:
        :class:`ResearchConfig` controlling the workflow.
    stage_agents:
        Optional dict mapping :class:`ResearchStageType` to a custom
        stage agent. If not provided, default agents are created.
    """

    def __init__(
        self,
        *,
        literature_agent: Any | None = None,
        terminal_tool: Any | None = None,
        config: ResearchConfig | None = None,
        stage_agents: dict[ResearchStageType, Any] | None = None,
    ) -> None:
        self.config = config or ResearchConfig()
        self._stage_agents: dict[ResearchStageType, Any] = {}
        self._terminal = terminal_tool
        self._literature_agent = literature_agent
        if stage_agents:
            self._stage_agents.update(stage_agents)
        self._init_default_agents()

    def _init_default_agents(self) -> None:
        """Initialize default stage agents if not overridden."""
        defaults: dict[ResearchStageType, Any] = {
            ResearchStageType.LITERATURE_DISCOVERY: LiteratureDiscoveryAgent(
                literature_agent=self._literature_agent
            ),
            ResearchStageType.KNOWLEDGE_SYNTHESIS: KnowledgeSynthesisAgent(),
            ResearchStageType.HYPOTHESIS_GENERATION: HypothesisGeneratorAgent(),
            ResearchStageType.EXPERIMENT_PLANNING: ResearchExperimentPlannerAgent(),
            ResearchStageType.EXPERIMENT_EXECUTION: ExperimentExecutorAgent(
                terminal_tool=self._terminal
            ),
            ResearchStageType.RESULT_ANALYSIS: ResultAnalyzerAgent(),
            ResearchStageType.REPORT_GENERATION: ReportGeneratorAgent(),
        }
        for stage_type, agent in defaults.items():
            if stage_type not in self._stage_agents:
                # When LLM is disabled, force rule-based mode so the
                # workflow never blocks on an unreachable provider.
                if not self.config.llm_enabled:
                    agent.llm_provider = None
                self._stage_agents[stage_type] = agent

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        stage_type: ResearchStageType,
        ctx: SharedResearchContext,
        stream_sink: Any | None,
    ) -> ResearchStageRecord:
        """Execute a single research stage."""
        agent = self._stage_agents.get(stage_type)
        stage_id = f"stage_{uuid4().hex[:8]}"
        if agent is None:
            return ResearchStageRecord(
                stage_id=stage_id,
                stage_type=stage_type,
                status=ResearchStageStatus.SKIPPED,
                summary=f"No agent registered for {stage_type.value}",
            )
        record = ResearchStageRecord(
            stage_id=stage_id,
            stage_type=stage_type,
            status=ResearchStageStatus.RUNNING,
        )
        t0 = time.time()
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink

        # B3: For report generation, first build a draft, evaluate it,
        # then re-generate with the evaluation for reasoned conclusions.
        if stage_type == ResearchStageType.REPORT_GENERATION:
            kwargs.update(await self._evaluate_draft(ctx, agent))

        try:
            result = await agent.execute(ctx, **kwargs)
            record.status = ResearchStageStatus.COMPLETED
            record.finished_at = datetime.now()
            record.duration_seconds = round(time.time() - t0, 3)
            if isinstance(result, dict):
                record.output = result
                record.summary = str(result.get("summary", ""))[:200]
            else:
                record.summary = str(result)[:200]
                record.output = {"result": str(result)}
        except Exception as e:
            record.status = ResearchStageStatus.FAILED
            record.finished_at = datetime.now()
            record.duration_seconds = round(time.time() - t0, 3)
            record.error = str(e)
        # D2 observability: emit a structured stage event (best-effort).
        try:
            from research_engineer.observability import get_event_bus

            get_event_bus().emit_stage(
                stage_id=stage_id,
                stage_type=stage_type.value,
                status=record.status.value,
                duration_seconds=record.duration_seconds or 0.0,
                workflow_id=getattr(ctx, "workflow_id", None),
                error=record.error,
            )
        except Exception:
            pass
        return record

    async def _evaluate_draft(
        self,
        ctx: SharedResearchContext,
        report_agent: Any,
    ) -> dict[str, Any]:
        """Evaluate a draft report before regeneration (B3).

        Returns extra kwargs to pass to the report generation stage
        when an evaluation is available; empty dict otherwise.
        """
        try:
            from research_engineer.agents.evaluation_agent import EvaluationAgent

            evaluation_agent = EvaluationAgent()
            # Build the draft report.
            draft = report_agent._build_report(ctx)
            hypotheses = [h.statement for h in ctx.hypotheses] if ctx.hypotheses else []
            analyses_data = (
                [a.model_dump() for a in ctx.analyses] if ctx.analyses else []
            )
            ev = await evaluation_agent.evaluate_research_output(
                report_markdown=draft,
                hypotheses=hypotheses,
                analyses=analyses_data,
                experiment_count=len(ctx.experiment_outcomes),
                paper_count=len(ctx.papers),
                research_goal=ctx.research_goal,
            )
            ctx.output_evaluation = ev.model_dump()
            return {"evaluation": ev.model_dump()}
        except Exception:
            return {}


__all__ = ["ResearchWorkflowFramework", "ResearchConfig"]
