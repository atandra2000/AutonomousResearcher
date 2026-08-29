"""LangGraph execution for the existing seven-stage research workflow."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from research_engineer.agents.research_workflow import (
    ResearchConfig,
    ResearchWorkflowFramework,
)
from research_engineer.models.research import (
    ResearchResult,
    ResearchStageRecord,
    ResearchStageStatus,
    ResearchStageType,
    ResearchWorkflowStatus,
    SharedResearchContext,
)

_PIPELINE = list(ResearchStageType)


class ResearchGraphState(TypedDict, total=False):
    """Serializable state shared by the LangGraph research nodes."""

    research_goal: str
    repo_path: str
    options: dict[str, Any]
    context: SharedResearchContext
    stages: list[ResearchStageRecord]
    status: ResearchWorkflowStatus
    error: str | None
    started_at: float


class ResearchGraph:
    """Run :class:`ResearchWorkflowFramework` stages as durable graph nodes.

    The graph owns ordering and persistence while the existing framework
    remains the authority for stage execution, report drafting, and events.
    Pass a LangGraph checkpointer (for example ``PostgresSaver`` in the
    deployed service) to enable per-node snapshots.
    """

    def __init__(
        self,
        workflow: ResearchWorkflowFramework,
        *,
        checkpointer: Any | None = None,
    ) -> None:
        self._workflow = workflow
        self._checkpointer = checkpointer
        self._graph = self._compile()

    async def run(
        self,
        research_goal: str,
        repo_path: str = ".",
        config: ResearchConfig | None = None,
        stream_sink: Any | None = None,
        thread_id: str | None = None,
    ) -> ResearchResult:
        """Run the workflow through LangGraph and return the legacy result."""
        cfg = config or self._workflow.config
        state = await self._graph.ainvoke(
            {
                "research_goal": research_goal,
                "repo_path": repo_path,
                "options": _options(cfg),
                "stages": [],
                "status": ResearchWorkflowStatus.RUNNING,
                "error": None,
                "started_at": time.time(),
            },
            {
                "configurable": {
                    "thread_id": thread_id or f"research_{uuid4().hex}",
                    "stream_sink": stream_sink,
                }
            },
        )
        return _result_from_state(state)

    def _compile(self) -> Any:
        graph = StateGraph(ResearchGraphState)
        graph.add_node("initialize", self._initialize)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", _PIPELINE[0].value)
        for index, stage_type in enumerate(_PIPELINE):
            graph.add_node(stage_type.value, self._stage_node(stage_type))
            graph.add_conditional_edges(
                stage_type.value,
                self._next_node(stage_type, index),
            )
        return graph.compile(checkpointer=self._checkpointer)

    @staticmethod
    def _initialize(state: ResearchGraphState) -> dict[str, Any]:
        options = state["options"]
        context = SharedResearchContext(
            research_goal=state["research_goal"],
            repo_path=state["repo_path"],
            output_dir=str(options["output_dir"]),
            max_papers=int(options["max_papers"]),
            max_hypotheses=int(options["max_hypotheses"]),
            dry_run_experiments=bool(options["dry_run_experiments"]),
            experiment_timeout=int(options["experiment_timeout"]),
            stream=bool(options["stream"]),
            skip_stages=[ResearchStageType(value) for value in options["skip_stages"]],
        )
        return {"context": context}

    def _stage_node(self, stage_type: ResearchStageType) -> Any:
        async def node(
            state: ResearchGraphState,
            config: RunnableConfig,
        ) -> dict[str, Any]:
            context = state["context"]
            if stage_type in context.skip_stages:
                record = ResearchStageRecord(
                    stage_id=f"stage_{uuid4().hex[:8]}",
                    stage_type=stage_type,
                    status=ResearchStageStatus.SKIPPED,
                    summary="Skipped per config.",
                )
            else:
                configurable = config.get("configurable", {})
                stream_sink = configurable.get("stream_sink")
                record = await self._workflow._run_stage(
                    stage_type,
                    context,
                    stream_sink,
                )
            stages = [*state["stages"], record]
            if record.status == ResearchStageStatus.FAILED:
                return {
                    "context": context,
                    "stages": stages,
                    "status": ResearchWorkflowStatus.PARTIAL,
                    "error": record.error or "Stage failed",
                }
            return {"context": context, "stages": stages}

        return node

    @staticmethod
    def _next_node(stage_type: ResearchStageType, index: int) -> Any:
        def route(state: ResearchGraphState) -> str:
            if state.get("status") == ResearchWorkflowStatus.PARTIAL:
                return END
            if index == len(_PIPELINE) - 1:
                return END
            return _PIPELINE[index + 1].value

        return route


def _options(config: ResearchConfig) -> dict[str, Any]:
    """Extract only JSON-serializable per-run configuration."""
    return {
        "max_papers": config.max_papers,
        "max_hypotheses": config.max_hypotheses,
        "dry_run_experiments": config.dry_run_experiments,
        "experiment_timeout": config.experiment_timeout,
        "skip_stages": [stage.value for stage in config.skip_stages],
        "stream": config.stream,
        "output_dir": config.output_dir,
    }


def _result_from_state(state: ResearchGraphState) -> ResearchResult:
    """Translate final graph state back into the existing public result."""
    context = state["context"]
    status = state.get("status", ResearchWorkflowStatus.FAILED)
    if status == ResearchWorkflowStatus.RUNNING:
        status = ResearchWorkflowStatus.COMPLETED
    return ResearchResult(
        workflow_id=context.workflow_id,
        research_goal=state["research_goal"],
        status=status,
        stages=state["stages"],
        papers_found=len(context.papers),
        hypotheses_generated=len(context.hypotheses),
        experiments_run=len(context.experiment_outcomes),
        final_report=context.final_report,
        report_path=context.report_path,
        generated_files=[context.report_path] if context.report_path else [],
        processing_time_seconds=round(time.time() - state["started_at"], 2),
        timestamp=datetime.now(),
        error=state.get("error"),
    )


__all__ = ["ResearchGraph", "ResearchGraphState"]
