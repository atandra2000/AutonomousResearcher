"""Tests for Phase 15 - Autonomous research workflows."""

from __future__ import annotations

from typing import Any

import pytest

from research_engineer.llm import LLMProvider, LLMRequest, LLMResponse
from research_engineer.models.research import (
    ExperimentDesign,
    ExperimentOutcome,
    Hypothesis,
    HypothesisStatus,
    KnowledgeSynthesis,
    PaperFinding,
    ResearchResult,
    ResearchStageRecord,
    ResearchStageStatus,
    ResearchStageType,
    ResearchWorkflowStatus,
    ResultAnalysis,
    SharedResearchContext,
)
from research_engineer.tools.terminal import TerminalOutput


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, default_model: str = "fake-model") -> None:
        self.default_model = default_model
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            content="Key findings:\n- Finding A\n- Finding B\n\nGaps:\n- Gap 1\n\nH: Test hypothesis | R: Because | E: Better results",
            model=self.default_model,
            provider=self.name,
        )


class _FakeTerminal:
    async def execute(self, input: Any) -> TerminalOutput:
        return TerminalOutput(
            operation=input.operation,
            success=True,
            exit_code=0,
            stdout="Training complete. loss=0.05",
            stderr="",
        )


# ---------------------------------------------------------------------------
# Research models
# ---------------------------------------------------------------------------


class TestResearchModels:
    def test_stage_type_values(self):
        assert ResearchStageType.LITERATURE_DISCOVERY == "literature_discovery"
        assert ResearchStageType.HYPOTHESIS_GENERATION == "hypothesis_generation"
        assert ResearchStageType.REPORT_GENERATION == "report_generation"

    def test_stage_status_values(self):
        assert ResearchStageStatus.PENDING == "pending"
        assert ResearchStageStatus.COMPLETED == "completed"
        assert ResearchStageStatus.SKIPPED == "skipped"

    def test_workflow_status_values(self):
        assert ResearchWorkflowStatus.RUNNING == "running"
        assert ResearchWorkflowStatus.COMPLETED == "completed"

    def test_hypothesis_status_values(self):
        assert HypothesisStatus.PROPOSED == "proposed"
        assert HypothesisStatus.SUPPORTED == "supported"
        assert HypothesisStatus.REFUTED == "refuted"

    def test_paper_finding(self):
        p = PaperFinding(
            paper_id="2503.12345",
            title="Test Paper",
            year=2025,
            abstract="An abstract",
        )
        assert p.paper_id == "2503.12345"
        assert p.relevance_score == 0.0

    def test_knowledge_synthesis(self):
        s = KnowledgeSynthesis(
            key_findings=["Finding A"],
            research_gaps=["Gap 1"],
        )
        assert len(s.key_findings) == 1
        assert len(s.research_gaps) == 1

    def test_hypothesis(self):
        h = Hypothesis(
            hypothesis_id="hyp_1",
            statement="Test hypothesis",
            rationale="Because",
            expected_outcome="Better results",
        )
        assert h.status == HypothesisStatus.PROPOSED
        assert h.priority == 5

    def test_experiment_design(self):
        d = ExperimentDesign(
            experiment_id="exp_1",
            hypothesis_id="hyp_1",
            title="Test experiment",
        )
        assert d.command == "python train.py"
        assert d.expected_duration_hours == 1.0

    def test_experiment_outcome(self):
        o = ExperimentOutcome(
            experiment_id="exp_1",
            status="completed",
            exit_code=0,
        )
        assert o.status == "completed"

    def test_result_analysis(self):
        a = ResultAnalysis(
            experiment_id="exp_1",
            hypothesis_id="hyp_1",
            conclusion="Supported",
        )
        assert a.hypothesis_status == HypothesisStatus.INCONCLUSIVE

    def test_shared_research_context(self):
        ctx = SharedResearchContext(research_goal="Test goal")
        assert ctx.max_papers == 20
        assert ctx.dry_run_experiments is True
        assert ctx.workflow_id.startswith("rw_")

    def test_shared_research_context_add_paper(self):
        ctx = SharedResearchContext(research_goal="g")
        p1 = PaperFinding(paper_id="1", title="A")
        p2 = PaperFinding(paper_id="1", title="A dup")
        ctx.add_paper(p1)
        ctx.add_paper(p2)
        assert len(ctx.papers) == 1

    def test_shared_research_context_add_hypothesis(self):
        ctx = SharedResearchContext(research_goal="g")
        h1 = Hypothesis(hypothesis_id="h1", statement="s1")
        h2 = Hypothesis(hypothesis_id="h1", statement="dup")
        ctx.add_hypothesis(h1)
        ctx.add_hypothesis(h2)
        assert len(ctx.hypotheses) == 1

    def test_research_result(self):
        r = ResearchResult(
            workflow_id="rw_1",
            research_goal="g",
            status=ResearchWorkflowStatus.COMPLETED,
        )
        assert r.papers_found == 0
        assert r.hypotheses_generated == 0

    def test_stage_record(self):
        r = ResearchStageRecord(
            stage_id="s1",
            stage_type=ResearchStageType.LITERATURE_DISCOVERY,
        )
        assert r.status == ResearchStageStatus.PENDING


# ---------------------------------------------------------------------------
# Stage agents
# ---------------------------------------------------------------------------


class TestLiteratureDiscoveryAgent:
    @pytest.mark.asyncio
    async def test_rule_based_discovery(self, tmp_path):
        from research_engineer.agents import LiteratureDiscoveryAgent

        agent = LiteratureDiscoveryAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(
            research_goal="Efficient transformers", repo_path=str(tmp_path)
        )
        result = await agent.execute(ctx)
        assert "summary" in result
        assert len(ctx.literature_review) > 0
        assert "Literature Review" in ctx.literature_review

    @pytest.mark.asyncio
    async def test_llm_based_discovery(self, tmp_path):
        from research_engineer.agents import LiteratureDiscoveryAgent

        agent = LiteratureDiscoveryAgent(llm=_FakeProvider())
        ctx = SharedResearchContext(
            research_goal="Efficient transformers", repo_path=str(tmp_path)
        )
        result = await agent.execute(ctx)
        assert "summary" in result
        assert len(ctx.literature_review) > 0


class TestKnowledgeSynthesisAgent:
    @pytest.mark.asyncio
    async def test_rule_based_synthesis(self, tmp_path):
        from research_engineer.agents import KnowledgeSynthesisAgent

        agent = KnowledgeSynthesisAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(research_goal="g", repo_path=str(tmp_path))
        result = await agent.execute(ctx)
        assert ctx.synthesis is not None
        assert len(ctx.synthesis.key_findings) > 0
        assert len(ctx.synthesis.research_gaps) > 0

    @pytest.mark.asyncio
    async def test_llm_synthesis(self, tmp_path):
        from research_engineer.agents import KnowledgeSynthesisAgent

        agent = KnowledgeSynthesisAgent(llm=_FakeProvider())
        ctx = SharedResearchContext(research_goal="g", repo_path=str(tmp_path))
        ctx.literature_review = "Some review text"
        result = await agent.execute(ctx)
        assert ctx.synthesis is not None

    def test_parse_synthesis(self):
        from research_engineer.agents import KnowledgeSynthesisAgent

        content = "Key findings:\n- Finding A\n- Finding B\n\nGaps:\n- Gap 1"
        syn = KnowledgeSynthesisAgent._parse_synthesis(content)
        assert len(syn.key_findings) == 2
        assert len(syn.research_gaps) == 1


class TestHypothesisGeneratorAgent:
    @pytest.mark.asyncio
    async def test_rule_based_hypotheses(self, tmp_path):
        from research_engineer.agents import HypothesisGeneratorAgent

        agent = HypothesisGeneratorAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(
            research_goal="g", repo_path=str(tmp_path), max_hypotheses=3
        )
        ctx.synthesis = KnowledgeSynthesis(
            key_findings=["f1"],
            research_gaps=["gap1", "gap2", "gap3"],
        )
        result = await agent.execute(ctx)
        assert len(ctx.hypotheses) == 3
        assert all(h.statement for h in ctx.hypotheses)

    @pytest.mark.asyncio
    async def test_llm_hypotheses(self, tmp_path):
        from research_engineer.agents import HypothesisGeneratorAgent

        agent = HypothesisGeneratorAgent(llm=_FakeProvider())
        ctx = SharedResearchContext(
            research_goal="g", repo_path=str(tmp_path), max_hypotheses=5
        )
        ctx.synthesis = KnowledgeSynthesis(
            key_findings=["f1"], research_gaps=["gap1"]
        )
        result = await agent.execute(ctx)
        assert len(ctx.hypotheses) > 0

    def test_parse_hypotheses(self):
        from research_engineer.agents import HypothesisGeneratorAgent

        content = "H: Test hyp 1 | R: Because | E: Better\nH: Test hyp 2 | R: Reason | E: Outcome"
        ctx = SharedResearchContext(research_goal="g")
        hyps = HypothesisGeneratorAgent._parse_hypotheses(content, ctx)
        assert len(hyps) == 2
        assert hyps[0].statement == "Test hyp 1"


class TestResearchExperimentPlannerAgent:
    @pytest.mark.asyncio
    async def test_designs_experiments(self, tmp_path):
        from research_engineer.agents import ResearchExperimentPlannerAgent

        agent = ResearchExperimentPlannerAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(research_goal="g", repo_path=str(tmp_path))
        ctx.hypotheses = [
            Hypothesis(hypothesis_id="h1", statement="Hypothesis 1"),
            Hypothesis(hypothesis_id="h2", statement="Hypothesis 2"),
        ]
        result = await agent.execute(ctx)
        assert len(ctx.experiment_designs) == 2
        assert ctx.experiment_designs[0].hypothesis_id == "h1"


class TestExperimentExecutorAgent:
    @pytest.mark.asyncio
    async def test_dry_run_execution(self, tmp_path):
        from research_engineer.agents import ExperimentExecutorAgent

        agent = ExperimentExecutorAgent(
            terminal_tool=_FakeTerminal(),  # type: ignore[arg-type]
            llm=None,
        )
        agent.llm_provider = None
        ctx = SharedResearchContext(
            research_goal="g", repo_path=str(tmp_path),
            dry_run_experiments=True,
        )
        ctx.experiment_designs = [
            ExperimentDesign(
                experiment_id="exp1", hypothesis_id="h1",
                title="Test", command="python train.py",
            )
        ]
        result = await agent.execute(ctx)
        assert len(ctx.experiment_outcomes) == 1
        assert ctx.experiment_outcomes[0].status == "dry_run"

    @pytest.mark.asyncio
    async def test_live_execution(self, tmp_path):
        from research_engineer.agents import ExperimentExecutorAgent

        agent = ExperimentExecutorAgent(
            terminal_tool=_FakeTerminal(),  # type: ignore[arg-type]
            llm=None,
        )
        agent.llm_provider = None
        ctx = SharedResearchContext(
            research_goal="g", repo_path=str(tmp_path),
            dry_run_experiments=False,
        )
        ctx.experiment_designs = [
            ExperimentDesign(
                experiment_id="exp1", hypothesis_id="h1",
                title="Test", command="echo hello",
            )
        ]
        result = await agent.execute(ctx)
        assert len(ctx.experiment_outcomes) == 1
        assert ctx.experiment_outcomes[0].status == "completed"


class TestResultAnalyzerAgent:
    @pytest.mark.asyncio
    async def test_analyzes_dry_run(self, tmp_path):
        from research_engineer.agents import ResultAnalyzerAgent

        agent = ResultAnalyzerAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(research_goal="g", repo_path=str(tmp_path))
        ctx.hypotheses = [Hypothesis(hypothesis_id="h1", statement="s1")]
        ctx.experiment_designs = [
            ExperimentDesign(
                experiment_id="exp1", hypothesis_id="h1", title="t",
            )
        ]
        ctx.experiment_outcomes = [
            ExperimentOutcome(
                experiment_id="exp1", status="dry_run", exit_code=0,
            )
        ]
        result = await agent.execute(ctx)
        assert len(ctx.analyses) == 1
        assert ctx.analyses[0].hypothesis_status == HypothesisStatus.INCONCLUSIVE

    @pytest.mark.asyncio
    async def test_analyzes_success(self, tmp_path):
        from research_engineer.agents import ResultAnalyzerAgent

        agent = ResultAnalyzerAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(research_goal="g", repo_path=str(tmp_path))
        ctx.hypotheses = [Hypothesis(hypothesis_id="h1", statement="s1")]
        ctx.experiment_designs = [
            ExperimentDesign(
                experiment_id="exp1", hypothesis_id="h1", title="t",
            )
        ]
        ctx.experiment_outcomes = [
            ExperimentOutcome(
                experiment_id="exp1", status="completed", exit_code=0,
                metrics={"loss": 0.05},
            )
        ]
        result = await agent.execute(ctx)
        assert ctx.analyses[0].hypothesis_status == HypothesisStatus.SUPPORTED
        assert ctx.hypotheses[0].status == HypothesisStatus.SUPPORTED


class TestReportGeneratorAgent:
    @pytest.mark.asyncio
    async def test_generates_report(self, tmp_path):
        from research_engineer.agents import ReportGeneratorAgent

        agent = ReportGeneratorAgent(llm=None)
        agent.llm_provider = None
        ctx = SharedResearchContext(
            research_goal="Test goal", repo_path=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        ctx.literature_review = "# Review\nSome text."
        ctx.synthesis = KnowledgeSynthesis(
            key_findings=["f1"], research_gaps=["g1"],
        )
        ctx.hypotheses = [
            Hypothesis(hypothesis_id="h1", statement="Hyp 1"),
        ]
        ctx.experiment_outcomes = [
            ExperimentOutcome(experiment_id="exp1", status="dry_run", exit_code=0),
        ]
        ctx.analyses = [
            ResultAnalysis(
                experiment_id="exp1", hypothesis_id="h1",
                conclusion="Inconclusive",
            )
        ]
        result = await agent.execute(ctx)
        assert len(ctx.final_report) > 0
        assert "Research Report" in ctx.final_report
        assert ctx.report_path != ""
        assert "Test goal" in ctx.final_report


# ---------------------------------------------------------------------------
# ResearchWorkflowFramework
# ---------------------------------------------------------------------------


class TestResearchWorkflowFramework:
    @pytest.mark.asyncio
    async def test_full_workflow_dry_run(self, tmp_path):
        from research_engineer.agents import ResearchConfig, ResearchWorkflowFramework

        framework = ResearchWorkflowFramework(
            terminal_tool=_FakeTerminal(),  # type: ignore[arg-type]
            config=ResearchConfig(
                max_papers=5,
                max_hypotheses=3,
                dry_run_experiments=True,
                output_dir=str(tmp_path / "out"),
            ),
        )
        result = await framework.run(
            research_goal="Efficient diffusion transformers",
            repo_path=str(tmp_path),
        )
        assert result.status == ResearchWorkflowStatus.COMPLETED
        assert len(result.stages) == 7
        assert all(
            s.status == ResearchStageStatus.COMPLETED for s in result.stages
        )
        assert result.papers_found >= 0
        assert result.hypotheses_generated > 0
        assert result.experiments_run > 0
        assert len(result.final_report) > 0
        assert result.report_path != ""

    @pytest.mark.asyncio
    async def test_workflow_with_skipped_stages(self, tmp_path):
        from research_engineer.agents import ResearchConfig, ResearchWorkflowFramework

        framework = ResearchWorkflowFramework(
            config=ResearchConfig(
                skip_stages=[
                    ResearchStageType.EXPERIMENT_EXECUTION,
                    ResearchStageType.RESULT_ANALYSIS,
                ],
                output_dir=str(tmp_path / "out"),
            ),
        )
        result = await framework.run(
            research_goal="Test", repo_path=str(tmp_path),
        )
        assert result.status == ResearchWorkflowStatus.COMPLETED
        # 7 stages, 2 skipped.
        skipped = [s for s in result.stages if s.status == ResearchStageStatus.SKIPPED]
        assert len(skipped) == 2

    @pytest.mark.asyncio
    async def test_workflow_stage_failure(self, tmp_path):
        from research_engineer.agents import ResearchConfig, ResearchWorkflowFramework

        class _FailAgent:
            agent_name = "FailAgent"

            async def execute(self, ctx, **kwargs):
                raise RuntimeError("Stage failed")

        framework = ResearchWorkflowFramework(
            config=ResearchConfig(
                output_dir=str(tmp_path / "out"),
            ),
            stage_agents={
                ResearchStageType.LITERATURE_DISCOVERY: _FailAgent(),
            },
        )
        result = await framework.run(
            research_goal="Test", repo_path=str(tmp_path),
        )
        assert result.status == ResearchWorkflowStatus.PARTIAL
        assert result.stages[0].status == ResearchStageStatus.FAILED
        assert result.error is not None


# ---------------------------------------------------------------------------
# ResearchOrchestrator
# ---------------------------------------------------------------------------


class TestResearchOrchestrator:
    def test_orchestrator_construction(self):
        from research_engineer.agents import ResearchOrchestrator

        orch = ResearchOrchestrator()
        assert orch.agent_name == "ResearchOrchestrator"
        assert hasattr(orch, "llm_provider")

    def test_orchestrator_in_router(self):
        from research_engineer.llm import get_router

        assert "ResearchOrchestrator" in get_router().KNOWN_AGENTS

    @pytest.mark.asyncio
    async def test_orchestrator_run(self, tmp_path):
        from research_engineer.agents import ResearchConfig, ResearchOrchestrator

        orch = ResearchOrchestrator(
            terminal_tool=_FakeTerminal(),  # type: ignore[arg-type]
        )
        config = ResearchConfig(
            max_papers=3,
            max_hypotheses=2,
            dry_run_experiments=True,
            output_dir=str(tmp_path / "out"),
        )
        result = await orch.run(
            research_goal="Efficient attention",
            repo_path=str(tmp_path),
            config=config,
        )
        assert result.status == ResearchWorkflowStatus.COMPLETED
        assert len(result.stages) == 7
        assert len(result.final_report) > 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestResearchCLI:
    def test_research_help(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["research", "--help"])
        assert result.exit_code == 0
        assert "research" in result.output.lower()
        assert "goal" in result.output.lower()

    def test_research_missing_goal(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["research"])
        assert result.exit_code != 0

    def test_research_dry_run_flag(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["research", "--help"])
        assert "dry-run" in result.output.lower() or "dry_run" in result.output.lower()

    def test_research_max_papers_flag(self):
        from typer.testing import CliRunner
        from research_engineer.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["research", "--help"])
        assert "max-papers" in result.output.lower() or "max_papers" in result.output.lower()