"""Tests for Phase 3 planner tools."""


import pytest

from research_engineer.models.planner import (
    CompatibilityLevel,
    RiskLevel,
)
from research_engineer.models.repo import (
    ConfigurationAnalysis,
    FileImportance,
    KnowledgeGraph,
    RepositorySummary,
)
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.compatibility import (
    CompatibilityAnalysisTool,
    CompatibilityInput,
)
from research_engineer.tools.compute_estimator import (
    ComputeEstimatorInput,
    ComputeEstimatorTool,
)
from research_engineer.tools.experiment_design import (
    ExperimentDesignInput,
    ExperimentDesignTool,
)
from research_engineer.tools.impact_analysis import (
    ImpactAnalysisInput,
    ImpactAnalysisTool,
)
from research_engineer.tools.implementation_planner import (
    ImplementationPlannerInput,
    ImplementationPlannerTool,
)
from research_engineer.tools.result_prediction import (
    ResultPredictionInput,
    ResultPredictionTool,
)
from research_engineer.tools.risk_assessment import (
    RiskAssessmentInput,
    RiskAssessmentTool,
)
from research_engineer.tools.validation_planner import (
    ValidationPlannerInput,
    ValidationPlannerTool,
)


def _make_summary(**kwargs) -> ResearchSummary:
    defaults = dict(
        paper_id="2503.12345",
        executive_summary="Test paper about efficient attention.",
        problem_statement="Attention is O(n^2) in memory.",
        core_contributions=["FlashAttention: IO-aware attention"],
        model_architecture="Transformer with FlashAttention",
        training_methodology="Standard AdamW training",
        dataset_information="C4 corpus, 100B tokens",
        evaluation_methodology="Perplexity on validation set",
        key_results=["2-4x memory reduction", "15-20% wall clock speedup"],
        limitations=["Requires Ampere+ GPU"],
        reproduction_challenges=["Requires CUDA capability >= 8.0"],
    )
    defaults.update(kwargs)
    return ResearchSummary(**defaults)


def _make_repo_summary(**kwargs) -> RepositorySummary:
    config = ConfigurationAnalysis(
        config_files=["configs/model.yaml"],
        training_hyperparameters={"lr": 1e-4},
        model_hyperparameters={"hidden_size": 768},
        data_paths={"train": "/data/train"},
        distributed_settings={},
        checkpoint_settings={"save_interval": 1000},
        config_framework="yaml",
    )
    defaults = dict(
        repository_name="test-repo",
        project_type="LLMTrainingFramework",
        architecture_summary="Transformer training framework with attention modules",
        important_files=[
            FileImportance(
                file_path="models/attention.py",
                importance="Critical",
                reason="Contains attention implementation",
                complexity="High",
                lines_of_code=200,
                dependencies_count=5,
            )
        ],
        training_pipeline="Standard training loop",
        knowledge_graph=KnowledgeGraph(
            nodes=[], edges=[], communities=[], central_nodes=[], relationships_by_type={}
        ),
        implementation_targets=[],
        configuration_analysis=config,
    )
    defaults.update(kwargs)
    return RepositorySummary(**defaults)


@pytest.fixture
def summary():
    return _make_summary()


@pytest.fixture
def repo_summary():
    return _make_repo_summary()


# --- CompatibilityAnalysisTool ---


@pytest.mark.asyncio
async def test_compatibility_tool_import():
    from research_engineer.tools import CompatibilityAnalysisTool
    assert CompatibilityAnalysisTool is not None


@pytest.mark.asyncio
async def test_compatibility_tool_execute(summary, repo_summary):
    tool = CompatibilityAnalysisTool()
    inp = CompatibilityInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.report.paper_id == "2503.12345"
    assert result.report.overall_compatibility in [
        CompatibilityLevel.LOW,
        CompatibilityLevel.MEDIUM,
        CompatibilityLevel.HIGH,
    ]


@pytest.mark.asyncio
async def test_compatibility_validation_fails_empty_paper():
    tool = CompatibilityAnalysisTool()
    inp = CompatibilityInput(
        paper_id="",
        repo_path="/test/repo",
        summary=_make_summary(),
        repo_summary=_make_repo_summary(),
    )
    assert await tool.validate(inp) is False


# --- ImplementationPlannerTool ---


@pytest.mark.asyncio
async def test_implementation_planner_import():
    from research_engineer.tools import ImplementationPlannerTool
    assert ImplementationPlannerTool is not None


@pytest.mark.asyncio
async def test_implementation_planner_execute(summary, repo_summary):
    tool = ImplementationPlannerTool()
    inp = ImplementationPlannerInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
        compatibility_level="Medium",
    )
    result = await tool.execute(inp)
    assert result.plan.paper_id == "2503.12345"
    assert len(result.plan.steps) >= 1
    assert len(result.plan.targets) >= 1


# --- ImpactAnalysisTool ---


@pytest.mark.asyncio
async def test_impact_analysis_import():
    from research_engineer.tools import ImpactAnalysisTool
    assert ImpactAnalysisTool is not None


@pytest.mark.asyncio
async def test_impact_analysis_execute(summary, repo_summary):
    tool = ImpactAnalysisTool()
    inp = ImpactAnalysisInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.impact.paper_id == "2503.12345"
    assert result.impact.memory_impact is not None
    assert result.impact.training_speed_impact is not None


# --- ExperimentDesignTool ---


@pytest.mark.asyncio
async def test_experiment_design_import():
    from research_engineer.tools import ExperimentDesignTool
    assert ExperimentDesignTool is not None


@pytest.mark.asyncio
async def test_experiment_design_execute(summary, repo_summary):
    tool = ExperimentDesignTool()
    inp = ExperimentDesignInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.matrix.paper_id == "2503.12345"
    assert result.matrix.total_experiments > 0
    assert len(result.matrix.groups) >= 3
    assert len(result.matrix.metrics) > 0


# --- ValidationPlannerTool ---


@pytest.mark.asyncio
async def test_validation_planner_import():
    from research_engineer.tools import ValidationPlannerTool
    assert ValidationPlannerTool is not None


@pytest.mark.asyncio
async def test_validation_planner_execute(summary, repo_summary):
    tool = ValidationPlannerTool()
    inp = ValidationPlannerInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.plan.paper_id == "2503.12345"
    assert result.plan.total_test_cases > 0
    assert len(result.plan.test_suites) >= 3


# --- RiskAssessmentTool ---


@pytest.mark.asyncio
async def test_risk_assessment_import():
    from research_engineer.tools import RiskAssessmentTool
    assert RiskAssessmentTool is not None


@pytest.mark.asyncio
async def test_risk_assessment_execute(summary, repo_summary):
    tool = RiskAssessmentTool()
    inp = RiskAssessmentInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.assessment.paper_id == "2503.12345"
    assert result.assessment.overall_risk_level in [
        RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH
    ]


# --- ComputeEstimatorTool ---


@pytest.mark.asyncio
async def test_compute_estimator_import():
    from research_engineer.tools import ComputeEstimatorTool
    assert ComputeEstimatorTool is not None


@pytest.mark.asyncio
async def test_compute_estimator_execute(summary, repo_summary):
    from research_engineer.tools.experiment_design import (
        ExperimentDesignInput,
        ExperimentDesignTool,
    )

    exp_tool = ExperimentDesignTool()
    exp_input = ExperimentDesignInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    exp_output = await exp_tool.execute(exp_input)

    tool = ComputeEstimatorTool()
    inp = ComputeEstimatorInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        experiment_matrix=exp_output.matrix,
    )
    result = await tool.execute(inp)
    assert result.estimate.paper_id == "2503.12345"
    assert result.estimate.total_gpu_hours > 0
    assert result.estimate.approximate_cloud_cost_usd > 0


# --- ResultPredictionTool ---


@pytest.mark.asyncio
async def test_result_prediction_import():
    from research_engineer.tools import ResultPredictionTool
    assert ResultPredictionTool is not None


@pytest.mark.asyncio
async def test_result_prediction_execute(summary, repo_summary):
    tool = ResultPredictionTool()
    inp = ResultPredictionInput(
        paper_id="2503.12345",
        repo_path="/test/repo",
        summary=summary,
        repo_summary=repo_summary,
    )
    result = await tool.execute(inp)
    assert result.prediction.paper_id == "2503.12345"
    assert result.prediction.best_case is not None
    assert result.prediction.likely_case is not None
    assert result.prediction.worst_case is not None
    assert len(result.prediction.failure_modes) > 0
    assert len(result.prediction.success_criteria) > 0
