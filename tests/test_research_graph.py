"""Contract tests for the LangGraph research-workflow adapter."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from research_engineer.agents.research_workflow import (
    ResearchConfig,
    ResearchWorkflowFramework,
)
from research_engineer.graphs import ResearchGraph
from research_engineer.models.research import (
    ResearchStageStatus,
    ResearchStageType,
    ResearchWorkflowStatus,
)


class _Stage:
    def __init__(self, stage_type: ResearchStageType, *, fail: bool = False) -> None:
        self.stage_type = stage_type
        self.fail = fail

    async def execute(self, ctx, **kwargs):
        if self.fail:
            raise RuntimeError(f"{self.stage_type.value} failed")
        return {"summary": self.stage_type.value, "has_stream_sink": bool(kwargs)}


def _framework(*, fail_stage: ResearchStageType | None = None) -> ResearchWorkflowFramework:
    stage_agents = {
        stage: _Stage(stage, fail=stage == fail_stage)
        for stage in ResearchStageType
    }
    return ResearchWorkflowFramework(
        config=ResearchConfig(llm_enabled=False),
        stage_agents=stage_agents,
    )


@pytest.mark.asyncio
async def test_research_graph_preserves_the_seven_stage_result_contract(tmp_path) -> None:
    graph = ResearchGraph(_framework(), checkpointer=InMemorySaver())

    result = await graph.run(
        "Evaluate a safe migration path.",
        repo_path=str(tmp_path),
        config=ResearchConfig(output_dir=str(tmp_path / "out"), llm_enabled=False),
        thread_id="research_graph_contract",
    )

    assert result.status == ResearchWorkflowStatus.COMPLETED
    assert [record.stage_type for record in result.stages] == list(ResearchStageType)
    assert all(record.status == ResearchStageStatus.COMPLETED for record in result.stages)


@pytest.mark.asyncio
async def test_research_graph_stops_after_a_failed_stage(tmp_path) -> None:
    graph = ResearchGraph(
        _framework(fail_stage=ResearchStageType.HYPOTHESIS_GENERATION)
    )

    result = await graph.run("Find a failure boundary.", repo_path=str(tmp_path))

    assert result.status == ResearchWorkflowStatus.PARTIAL
    assert len(result.stages) == 3
    assert result.stages[-1].status == ResearchStageStatus.FAILED
    assert result.error == "hypothesis_generation failed"


@pytest.mark.asyncio
async def test_research_graph_keeps_skipped_stages_in_the_audit_trail(tmp_path) -> None:
    graph = ResearchGraph(_framework())
    config = ResearchConfig(
        llm_enabled=False,
        skip_stages=[ResearchStageType.EXPERIMENT_EXECUTION],
    )

    result = await graph.run("Skip unsafe execution.", repo_path=str(tmp_path), config=config)

    skipped = [record for record in result.stages if record.status == ResearchStageStatus.SKIPPED]
    assert len(skipped) == 1
    assert skipped[0].stage_type == ResearchStageType.EXPERIMENT_EXECUTION
