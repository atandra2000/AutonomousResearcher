"""Phase 15 - Research workflow framework.

A generic multi-stage research orchestrator that transforms a research
goal into a complete research workflow: literature review → knowledge
synthesis → hypothesis generation → experiment planning → experiment
execution → result analysis → report generation.

The framework is built on the same principles as the
:class:`DelegationFramework` (Phase 13): stages communicate through a
shared structured context (:class:`SharedResearchContext`), not prompt
chaining. Each stage reads what it needs and writes its outputs back,
enabling full traceability from research goal to final conclusions.

Design principles:
- **Structured artifacts**: All inter-stage communication flows through
  :class:`SharedResearchContext`; stages never call each other directly.
- **Configurable pipeline**: Stages can be skipped, reordered, or
  replaced via :class:`ResearchConfig`.
- **Full traceability**: Each stage produces a
  :class:`ResearchStageRecord` with timing, status, and output.
- **Backward compatible**: The framework is additive; existing agents
  and workflows are unchanged.
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
    ResearchResult,
    ResearchStageRecord,
    ResearchStageStatus,
    ResearchStageType,
    ResearchWorkflowStatus,
    SharedResearchContext,
)


class ResearchConfig:
    """Configuration for the research workflow.

    Controls which stages run, paper/hypothesis limits, and experiment
    execution settings.
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
    ) -> None:
        self.max_papers = max_papers
        self.max_hypotheses = max_hypotheses
        self.dry_run_experiments = dry_run_experiments
        self.experiment_timeout = experiment_timeout
        self.skip_stages = skip_stages or []
        self.stream = stream
        self.output_dir = output_dir


class ResearchWorkflowFramework:
    """Generic multi-stage research workflow orchestrator.

    Runs the full research pipeline: literature discovery → knowledge
    synthesis → hypothesis generation → experiment planning → experiment
    execution → result analysis → report generation.

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
                self._stage_agents[stage_type] = agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            config: Optional config override.
            stream_sink: Optional LLM streaming sink.

        Returns:
            :class:`ResearchResult` with all stage records and the
            final report.
        """
        start = time.time()
        cfg = config or self.config
        ctx = SharedResearchContext(
            research_goal=research_goal,
            repo_path=repo_path,
            output_dir=cfg.output_dir,
            max_papers=cfg.max_papers,
            max_hypotheses=cfg.max_hypotheses,
            dry_run_experiments=cfg.dry_run_experiments,
            experiment_timeout=cfg.experiment_timeout,
            stream=cfg.stream,
            skip_stages=cfg.skip_stages,
        )
        stages: list[ResearchStageRecord] = []
        status = ResearchWorkflowStatus.RUNNING
        top_error: str | None = None

        pipeline: list[ResearchStageType] = [
            ResearchStageType.LITERATURE_DISCOVERY,
            ResearchStageType.KNOWLEDGE_SYNTHESIS,
            ResearchStageType.HYPOTHESIS_GENERATION,
            ResearchStageType.EXPERIMENT_PLANNING,
            ResearchStageType.EXPERIMENT_EXECUTION,
            ResearchStageType.RESULT_ANALYSIS,
            ResearchStageType.REPORT_GENERATION,
        ]

        try:
            for stage_type in pipeline:
                if stage_type in ctx.skip_stages:
                    stages.append(
                        ResearchStageRecord(
                            stage_id=f"stage_{uuid4().hex[:8]}",
                            stage_type=stage_type,
                            status=ResearchStageStatus.SKIPPED,
                            summary="Skipped per config.",
                        )
                    )
                    continue
                stage_record = await self._run_stage(
                    stage_type, ctx, stream_sink
                )
                stages.append(stage_record)
                if stage_record.status == ResearchStageStatus.FAILED:
                    status = ResearchWorkflowStatus.PARTIAL
                    top_error = stage_record.error or "Stage failed"
                    break
            if status == ResearchWorkflowStatus.RUNNING:
                status = ResearchWorkflowStatus.COMPLETED
        except Exception as e:
            status = ResearchWorkflowStatus.FAILED
            top_error = str(e)

        return ResearchResult(
            workflow_id=ctx.workflow_id,
            research_goal=research_goal,
            status=status,
            stages=stages,
            papers_found=len(ctx.papers),
            hypotheses_generated=len(ctx.hypotheses),
            experiments_run=len(ctx.experiment_outcomes),
            final_report=ctx.final_report,
            report_path=ctx.report_path,
            generated_files=[ctx.report_path] if ctx.report_path else [],
            processing_time_seconds=round(time.time() - start, 2),
            timestamp=datetime.now(),
            error=top_error,
        )

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
        return record


__all__ = ["ResearchWorkflowFramework", "ResearchConfig"]
