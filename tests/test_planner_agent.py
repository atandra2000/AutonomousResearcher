"""Tests for ExperimentPlannerAgent."""

import pytest

from research_engineer.agents.experiment_planner_agent import ExperimentPlannerAgent


@pytest.mark.asyncio
async def test_planner_agent_import():
    from research_engineer.agents import ExperimentPlannerAgent
    assert ExperimentPlannerAgent is not None


@pytest.mark.asyncio
async def test_planner_agent_creation():
    agent = ExperimentPlannerAgent()
    assert agent is not None
    assert agent.compatibility is not None
    assert agent.implementation is not None
    assert agent.impact is not None
    assert agent.experiment is not None
    assert agent.validation is not None
    assert agent.risk is not None
    assert agent.compute is not None
    assert agent.prediction is not None
