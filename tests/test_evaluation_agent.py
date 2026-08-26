"""Tests for EvaluationAgent (Phase 8)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from research_engineer.agents.evaluation_agent import EvaluationAgent
from research_engineer.models.evaluation import (
    EvaluationConfig,
    ResearchOutputEvaluationOutput,
)
from research_engineer.models.experiment import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentType,
    MetricSeries,
)
from research_engineer.tools.evaluation_storage import EvaluationStorageTool


def _record(
    exp_id: str = "exp1",
    metrics: dict[str, float] | None = None,
    series: list[MetricSeries] | None = None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=exp_id,
        repo_path="./repo",
        command=["python", "train.py"],
        experiment_type=ExperimentType.TRAINING,
        status=status,
        start_time="2026-01-01T00:00:00",
        metrics=metrics or {"loss": 1.0},
        metric_series=series or [],
    )


def _series(name: str, values: list[float]) -> MetricSeries:
    return MetricSeries(
        name=name,
        values=values,
        steps=list(range(len(values))),
        best_value=min(values) if "loss" in name.lower() else max(values),
        best_step=values.index(min(values))
        if "loss" in name.lower()
        else values.index(max(values)),
        final_value=values[-1],
    )


@pytest.fixture
def mock_memory_agent():
    agent = MagicMock()
    agent.storage = AsyncMock()
    agent.storage.store_relationship = AsyncMock()
    agent.graph = MagicMock()
    agent.graph.add_node = MagicMock()
    agent.graph.add_relationship = MagicMock()
    agent.store_insight = AsyncMock(return_value="mem_insight")
    agent.store_success = AsyncMock(return_value="mem_success")
    agent.store_failure = AsyncMock(return_value="mem_failure")
    return agent


@pytest.fixture
def mock_literature_agent():
    agent = MagicMock()
    result = MagicMock()
    result.papers = []
    agent.search_papers = AsyncMock(return_value=result)
    return agent


@pytest.fixture
def evaluation_agent(mock_memory_agent, mock_literature_agent, tmp_path):
    config = EvaluationConfig(
        output_dir=str(tmp_path / "evaluations"),
        store_conclusions=True,
        update_graph=True,
        recommend_papers=True,
    )
    agent = EvaluationAgent(
        memory_agent=mock_memory_agent,
        literature_agent=mock_literature_agent,
        storage_tool=EvaluationStorageTool(db_path=str(tmp_path / "evaluations.db")),
        config=config,
    )
    agent.llm_provider = None  # Disable LLM to avoid real API calls in tests.
    return agent


@pytest.fixture
def evaluation_agent_no_deps(tmp_path):
    config = EvaluationConfig(
        output_dir=str(tmp_path / "evaluations"),
    )
    agent = EvaluationAgent(
        storage_tool=EvaluationStorageTool(db_path=str(tmp_path / "evaluations.db")),
        config=config,
    )
    agent.llm_provider = None  # Disable LLM to avoid real API calls in tests.
    return agent


class TestEvaluationAgentInit:
    def test_defaults(self):
        agent = EvaluationAgent()
        assert agent.memory is None
        assert agent.literature is None
        assert agent.comparison is not None
        assert agent.dynamics is not None
        assert agent.significance is not None
        assert agent.next_tool is not None
        assert agent.storage is not None

    def test_config(self):
        agent = EvaluationAgent()
        assert agent.config.store_conclusions is True


class TestAnalyzeWorkflow:
    @pytest.mark.asyncio
    async def test_analyze_two_experiments(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        result = await evaluation_agent.analyze(
            experiments=[a, b],
            primary_metric="loss",
            higher_is_better=False,
        )
        assert result.evaluation_id.startswith("eval_")
        assert result.comparison is not None
        assert result.comparison.experiments_compared == 2
        assert len(result.dynamics) == 2
        assert result.significance is not None
        assert result.next_experiments is not None
        assert result.record is not None
        assert result.processing_time_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_analyze_single_experiment(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.0}, [_series("loss", [1.5, 1.3, 1.1, 1.0])])
        result = await evaluation_agent.analyze(experiments=[a])
        assert result.comparison is None
        assert len(result.dynamics) == 1
        assert result.significance is None
        assert result.next_experiments is not None

    @pytest.mark.asyncio
    async def test_evaluate_single(self, evaluation_agent, tmp_path):
        a = _record("a", {"loss": 1.0}, [_series("loss", [1.5, 1.3, 1.1, 1.0])])
        result = await evaluation_agent.evaluate_single(a)
        assert len(result.dynamics) == 1
        assert result.comparison is None

    @pytest.mark.asyncio
    async def test_stores_in_memory(
        self, evaluation_agent, mock_memory_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        await evaluation_agent.analyze(
            experiments=[a, b], primary_metric="loss"
        )
        assert mock_memory_agent.store_insight.called

    @pytest.mark.asyncio
    async def test_graph_updated(
        self, evaluation_agent, mock_memory_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        await evaluation_agent.analyze(
            experiments=[a, b], primary_metric="loss"
        )
        assert mock_memory_agent.graph.add_node.called
        assert mock_memory_agent.graph.add_relationship.called

    @pytest.mark.asyncio
    async def test_output_files_generated(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        result = await evaluation_agent.analyze(
            experiments=[a, b],
            primary_metric="loss",
            output_dir=str(tmp_path / "out"),
        )
        assert len(result.generated_files) > 0
        from pathlib import Path

        for f in result.generated_files:
            assert Path(f).exists()

    @pytest.mark.asyncio
    async def test_with_paper_id(
        self, evaluation_agent, mock_memory_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        result = await evaluation_agent.analyze(
            experiments=[a, b],
            primary_metric="loss",
            paper_id="2503.12345",
            repo_path="./repo",
        )
        assert result.paper_id == "2503.12345"
        assert result.repo_path == "./repo"

    @pytest.mark.asyncio
    async def test_graceful_no_memory(
        self, evaluation_agent_no_deps, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        result = await evaluation_agent_no_deps.analyze(
            experiments=[a, b], primary_metric="loss"
        )
        assert result.comparison is not None
        assert result.memory_ids == []

    @pytest.mark.asyncio
    async def test_graceful_no_literature(
        self, evaluation_agent_no_deps, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        result = await evaluation_agent_no_deps.analyze(experiments=[a])
        assert result.next_experiments is not None
        assert result.next_experiments.paper_suggestions == []


class TestIndividualOperations:
    @pytest.mark.asyncio
    async def test_compare(self, evaluation_agent):
        a = _record("a", {"loss": 1.5})
        b = _record("b", {"loss": 1.0})
        out = await evaluation_agent.compare(
            [a, b], primary_metric="loss", higher_is_better=False
        )
        assert out.winner_experiment_id == "b"

    @pytest.mark.asyncio
    async def test_dynamics_analysis(self, evaluation_agent):
        a = _record("a", {"loss": 1.0}, [_series("loss", [1.5, 1.3, 1.1, 1.0])])
        out = await evaluation_agent.dynamics_analysis(a)
        assert out.experiment_id == "a"

    @pytest.mark.asyncio
    async def test_significance_test(self, evaluation_agent):
        a = _record("a", series=[_series("loss", [1.0, 1.1, 0.9, 1.0, 1.05])])
        b = _record("b", series=[_series("loss", [5.0, 5.1, 4.9, 5.0, 5.05])])
        out = await evaluation_agent.significance_test(
            [a, b], metric="loss"
        )
        assert out.pairwise_count == 1

    @pytest.mark.asyncio
    async def test_next_experiments(self, evaluation_agent):
        a = _record("a", {"loss": 1.0}, [_series("loss", [1.5, 1.3, 1.1, 1.0])])
        out = await evaluation_agent.next_experiments([a])
        assert len(out.experiment_recommendations) >= 1


class TestQueryMethods:
    @pytest.mark.asyncio
    async def test_list_evaluations(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        await evaluation_agent.analyze(
            experiments=[a, b],
            primary_metric="loss",
            paper_id="2503.12345",
        )
        out = await evaluation_agent.list_evaluations(paper_id="2503.12345")
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_get_evaluation(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        result = await evaluation_agent.analyze(
            experiments=[a, b], primary_metric="loss"
        )
        got = await evaluation_agent.get_evaluation(result.evaluation_id)
        assert got is not None
        assert got.evaluation_id == result.evaluation_id

    @pytest.mark.asyncio
    async def test_search_evaluations(
        self, evaluation_agent, tmp_path
    ):
        a = _record("a", {"loss": 1.5}, [_series("loss", [1.5, 1.4, 1.3])])
        b = _record("b", {"loss": 1.0}, [_series("loss", [1.2, 1.1, 1.0])])
        await evaluation_agent.analyze(
            experiments=[a, b], primary_metric="loss"
        )
        out = await evaluation_agent.search_evaluations("loss")
        assert out.total >= 1


class TestPaperSuggestions:
    @pytest.mark.asyncio
    async def test_paper_suggestions_populated(
        self, mock_memory_agent, tmp_path
    ):
        from research_engineer.models.literature import SearchResult, SearchSource

        lit = MagicMock()
        search_out = MagicMock()
        search_out.papers = [
            SearchResult(
                paper_id="p1",
                title="Regularization for overfitting",
                source=SearchSource.ARXIV,
            )
        ]
        lit.search_papers = AsyncMock(return_value=search_out)
        agent = EvaluationAgent(
            storage_tool=EvaluationStorageTool(db_path=str(tmp_path / "evaluations.db")),
            memory_agent=mock_memory_agent,
            literature_agent=lit,
            config=EvaluationConfig(
                output_dir=str(tmp_path / "eval"),
                recommend_papers=True,
            ),
        )
        a = _record("a", {"loss": 1.0}, [_series("loss", [2.0, 1.5, 1.0, 0.8, 0.6, 0.5])])
        out = await agent.analyze(experiments=[a])
        assert out.next_experiments is not None
        assert len(out.next_experiments.paper_suggestions) >= 1


# ---------------------------------------------------------------------------
# B3: Research-output evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateResearchOutput:
    """Tests for EvaluationAgent.evaluate_research_output() (B3)."""

    @pytest.mark.asyncio
    async def test_basic_evaluation(self, evaluation_agent_no_deps):
        """Rule-based evaluation returns non-zero scores."""
        report = (
            "# Research Report: test goal\n\n"
            "## Abstract\n\nThis report investigates test goal.\n\n"
            "## Literature\n\nRelated work on test goal.\n\n"
            "## Hypothesis\n\nHypothesis: attention improves accuracy.\n\n"
            "## Experiment\n\nExperiment: exp_001 completed with loss=0.5.\n\n"
            "## Result\n\nResult: loss improved by 20% over baseline.\n\n"
            "## Conclusion\n\nConclusion: therefore, attention improves accuracy.\n"
        )
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            hypotheses=["attention improves accuracy"],
            research_goal="test goal",
        )
        assert isinstance(out, ResearchOutputEvaluationOutput)
        assert out.overall_score > 0.0
        assert out.grade in ("A", "B", "C", "D", "F")

    @pytest.mark.asyncio
    async def test_hypothesis_coverage(self, evaluation_agent_no_deps):
        """Hypothesis coverage scores higher when hypotheses are addressed."""
        report = "# Report\n\nHypothesis: attention improves accuracy. Conclusion: supported.\n"
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            hypotheses=["attention improves accuracy"],
        )
        assert out.hypothesis_coverage >= 0.5

    @pytest.mark.asyncio
    async def test_hypothesis_coverage_empty(self, evaluation_agent_no_deps):
        """When no hypotheses are given and none mentioned, score is 0.5."""
        report = "# Report\n\nSome general content."
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            hypotheses=None,
        )
        assert out.hypothesis_coverage == 0.5

    @pytest.mark.asyncio
    async def test_hypothesis_coverage_none_mentioned(self, evaluation_agent_no_deps):
        """When hypotheses don't appear in report, coverage is 0."""
        report = "# Report\n\nCompletely unrelated content about training."
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            hypotheses=["quantum entanglement for neural networks"],
        )
        assert out.hypothesis_coverage < 0.5

    @pytest.mark.asyncio
    async def test_evidence_support_with_metrics(self, evaluation_agent_no_deps):
        """Evidence support increases when metrics are mentioned."""
        report = (
            "Report with experiment result loss accuracy baseline "
            "comparison significance statistical metrics. "
            "Hypothesis: attention improves. Conclusion: supported. "
            "Experiment: exp_001 completed. Result: loss improved."
        )
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            experiment_count=2,
            analyses=[{"conclusion": "supported", "hypothesis_status": "supported"}],
        )
        assert out.evidence_support > 0.5

    @pytest.mark.asyncio
    async def test_completeness_short_report(self, evaluation_agent_no_deps):
        """Short report gets a completeness recommendation."""
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown="short",
        )
        assert out.completeness < 0.5
        assert any("abstract" in r.lower() or "expand" in r.lower() for r in out.recommendations)

    @pytest.mark.asyncio
    async def test_grade_letters(self, evaluation_agent_no_deps):
        """Letter grades follow the score thresholds."""
        assert EvaluationAgent._letter_grade(0.95) == "A"
        assert EvaluationAgent._letter_grade(0.85) == "B"
        assert EvaluationAgent._letter_grade(0.75) == "C"
        assert EvaluationAgent._letter_grade(0.65) == "D"
        assert EvaluationAgent._letter_grade(0.50) == "F"

    @pytest.mark.asyncio
    async def test_findings_populated(self, evaluation_agent_no_deps):
        """Findings list is always populated."""
        report = "# Report\n\nSome content with conclusion and results."
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
        )
        assert len(out.findings) > 0

    @pytest.mark.asyncio
    async def test_recommendations_for_low_scores(self, evaluation_agent_no_deps):
        """Low-scoring reports generate recommendations."""
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown="x",
        )
        assert len(out.recommendations) > 0

    @pytest.mark.asyncio
    async def test_summary_populated(self, evaluation_agent_no_deps):
        """Summary is always populated."""
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown="# Report\n\nSome content.",
        )
        assert len(out.summary) > 0

    @pytest.mark.asyncio
    async def test_no_llm_provider_still_works(self, evaluation_agent_no_deps):
        """Evaluation works without an LLM provider (rule-based only)."""
        report = "# Test report with experiment loss result"
        out = await evaluation_agent_no_deps.evaluate_research_output(
            report_markdown=report,
            research_goal="test",
        )
        assert out.overall_score >= 0.0
        assert "Report quality" in out.summary
