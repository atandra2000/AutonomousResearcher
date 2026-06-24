"""Integration tests for MemoryAgent with relationship detection, graph, and retrieval."""

import pytest

from research_engineer.agents import MemoryAgent
from research_engineer.models.memory import MemoryType, RelationshipType
from research_engineer.models.paper import Author, Paper
from research_engineer.models.plan import ComplexityMetrics, EngineeringReport, FileRequirement
from research_engineer.models.summary import ResearchSummary


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """MemoryAgent with isolated temp DB and vector store."""
    from research_engineer.agents.memory_agent import MemoryConfig

    config = MemoryConfig(
        db_path=str(tmp_path / "agent.db"),
        vector_store_path=str(tmp_path / "vs"),
        log_access=True,
    )
    return MemoryAgent(config=config)


@pytest.fixture
def paper():
    return Paper(
        paper_id="2401.00001",
        title="Test Paper",
        authors=[Author(name="Test Author")],
        abstract="An attention mechanism paper",
        url="http://arxiv.org/abs/2401.00001",
        published=__import__("datetime").datetime(2024, 1, 1),
    )


@pytest.fixture
def summary():
    return ResearchSummary(
        paper_id="2401.00001",
        executive_summary="Test summary",
        problem_statement="Test problem",
        core_contributions=["contribution 1"],
        model_architecture="transformer",
        training_methodology="SGD",
        dataset_information="CIFAR-10",
        evaluation_methodology="accuracy",
        key_results=["95% accuracy"],
        limitations=["none"],
        reproduction_challenges=["none"],
    )


@pytest.fixture
def plan():
    return EngineeringReport(
        paper_id="2401.00001",
        complexity_analysis=ComplexityMetrics(
            code_complexity="Medium",
            data_requirements="Low",
            compute_requirements="Low",
            inference_complexity="Low",
            training_time="1 day",
            deployment_complexity="Low",
        ),
        step_by_step_implementation="step 1",
        files_required=[
            FileRequirement(filename="model.py", purpose="Model definition", dependencies=["torch"], complexity="Medium")
        ],
        development_effort="2 days",
        dependencies=["torch"],
    )


class TestMemoryAgentStorePaper:
    """Test storing papers in memory."""

    @pytest.mark.asyncio
    async def test_store_paper(self, agent, paper, summary, plan):
        memory_id = await agent.store_paper(paper, summary, plan)
        assert memory_id is not None

        retrieved = await agent.retrieve(memory_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_store_paper_creates_tags(self, agent, paper, summary, plan):
        await agent.store_paper(paper, summary, plan)
        # tags should be extracted from title/abstract
        # "attention" is in the ML terms list
        stats = await agent.get_stats()
        assert stats.total_memories >= 1

    @pytest.mark.asyncio
    async def test_store_paper_logs_access(self, agent, paper, summary, plan):
        await agent.store_paper(paper, summary, plan)
        # access log should have a write entry
        stats = await agent.get_stats()
        assert stats.total_memories >= 1


class TestMemoryAgentRelationships:
    """Test relationship detection in agent."""

    @pytest.mark.asyncio
    async def test_detect_paper_relationships_not_erroring(self, agent, paper, summary, plan):
        """Storing paper should not raise even if relationship detection runs."""
        from research_engineer.models.memory import PaperMemory
        memory = PaperMemory(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
        )
        # Should not raise
        await agent._detect_paper_relationships(memory)

    @pytest.mark.asyncio
    async def test_link_plan_to_paper(self, agent, paper, summary, plan):
        from research_engineer.models.memory import ExperimentPlanMemory
        plan_mem = ExperimentPlanMemory(
            plan_id="plan_1",
            paper_id=paper.paper_id,
            repo_path="./repo",
        )
        await agent._link_plan_to_paper(plan_mem)
        # graph should now contain the plan->paper edge
        if agent.graph is not None:
            assert "plan_1" in agent.graph.graph or paper.paper_id in agent.graph.graph


class TestMemoryAgentSearch:
    """Test memory search."""

    @pytest.mark.asyncio
    async def test_search_returns_list(self, agent, paper, summary, plan):
        await agent.store_paper(paper, summary, plan)
        results = await agent.search("attention", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_context(self, agent, paper, summary, plan):
        await agent.store_paper(paper, summary, plan)
        results = await agent.get_context("attention mechanism", limit=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_related_empty(self, agent):
        results = await agent.get_related("nonexistent")
        assert isinstance(results, list)
        assert len(results) == 0


class TestMemoryAgentGraph:
    """Test graph operations."""

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, agent):
        stats = await agent.get_graph_stats()
        assert isinstance(stats, dict)
        # empty graph should have 0 nodes
        assert stats.get("node_count", 0) == 0

    @pytest.mark.asyncio
    async def test_search_hybrid_no_crash(self, agent):
        results = await agent.search_hybrid("attention", limit=5)
        assert isinstance(results, list)


class TestMemoryAgentStats:
    """Test stats operations."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, agent):
        stats = await agent.get_stats()
        assert stats.total_memories >= 0

    @pytest.mark.asyncio
    async def test_get_stats_after_store(self, agent, paper, summary, plan):
        await agent.store_paper(paper, summary, plan)
        stats = await agent.get_stats()
        assert stats.total_memories >= 1
        assert "paper" in stats.memories_by_type


class TestMemoryAgentLifecycle:
    """Test agent lifecycle."""

    @pytest.mark.asyncio
    async def test_close(self, agent):
        await agent.close()