"""Experiment Planner Agent for Phase 3.

Orchestrates paper understanding, repository understanding,
compatibility analysis, implementation planning, experiment design,
validation planning, risk assessment, compute estimation,
impact analysis, and result prediction.
"""

import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.agents.repository_agent import RepositoryAgent
from research_engineer.agents.research_agent import ResearchAgent
from research_engineer.llm import LLMProvider
from research_engineer.models.planner import (
    CompatibilityReport,
    ComputeEstimate,
    ExperimentMatrix,
    ImplementationPlan,
    PlanResult,
    ResultPrediction,
    RiskAssessment,
    ValidationPlan,
)
from research_engineer.models.repo import (
    ConfigurationAnalysis,
    FileImportance,
    ImplementationTarget,
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


class PlannerResult(BaseModel):
    """Result of experiment planning."""

    plan_id: str = Field(..., description="Unique plan identifier")
    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    compatibility_report: dict = Field(..., description="Compatibility analysis")
    implementation_plan: dict = Field(..., description="Implementation plan")
    impact_report: dict = Field(..., description="Engineering impact analysis")
    experiment_matrix: dict = Field(..., description="Experiment matrix")
    validation_plan: dict = Field(..., description="Validation strategy")
    risk_assessment: dict = Field(..., description="Risk assessment")
    compute_estimate: dict = Field(..., description="Compute cost estimation")
    result_prediction: dict = Field(..., description="Result prediction")
    engineering_report_md: str = Field(..., description="Full engineering report in markdown")
    planning_time_seconds: float = Field(..., description="Planning duration")
    generated_files: list[str] = Field(default_factory=list, description="Generated output files")


class ExperimentPlannerAgent:
    """
    Agent that plans experiment integration of a research paper
    into a target repository.

    Orchestrates:
    1. Paper understanding (via ResearchAgent)
    2. Repository understanding (via RepositoryAgent)
    3. Compatibility analysis
    4. Implementation planning
    5. Impact analysis
    6. Experiment design
    7. Validation planning
    8. Risk assessment
    9. Compute estimation
    10. Result prediction
    11. Storage and output generation
    """

    def __init__(
        self,
        research_agent: ResearchAgent | None = None,
        repository_agent: RepositoryAgent | None = None,
        compatibility_tool: CompatibilityAnalysisTool | None = None,
        implementation_tool: ImplementationPlannerTool | None = None,
        impact_tool: ImpactAnalysisTool | None = None,
        experiment_tool: ExperimentDesignTool | None = None,
        validation_tool: ValidationPlannerTool | None = None,
        risk_tool: RiskAssessmentTool | None = None,
        compute_tool: ComputeEstimatorTool | None = None,
        prediction_tool: ResultPredictionTool | None = None,
        llm: LLMProvider | None = None,
    ):
        self.agent_name: str = "ExperimentPlannerAgent"
        self.research_agent = research_agent or ResearchAgent()
        self.repository_agent = repository_agent or RepositoryAgent()
        self.compatibility = compatibility_tool or CompatibilityAnalysisTool()
        self.implementation = implementation_tool or ImplementationPlannerTool()
        self.impact = impact_tool or ImpactAnalysisTool()
        self.experiment = experiment_tool or ExperimentDesignTool()
        self.validation = validation_tool or ValidationPlannerTool()
        self.risk = risk_tool or RiskAssessmentTool()
        self.compute = compute_tool or ComputeEstimatorTool()
        self.prediction = prediction_tool or ResultPredictionTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def plan(
        self,
        paper_input: str,
        repo_path: str,
        output_dir: str = "output",
    ) -> PlannerResult:
        """
        Main entry point for experiment planning.

        Args:
            paper_input: arXiv ID, arXiv URL, or PDF file path
            repo_path: Path to the target repository
            output_dir: Directory to save output files

        Returns:
            PlannerResult with all planning artifacts
        """
        start_time = time.time()

        # Step 1: Understand paper
        paper_result = await self.research_agent.analyze(paper_input, output_dir=output_dir)
        paper_id = paper_result["paper_id"]
        summary = ResearchSummary(**paper_result["summary"])

        # Step 2: Understand repository
        repo_result = await self.repository_agent.analyze(repo_path, output_dir=output_dir)
        repo_summary_data = repo_result
        
        # Convert important_files dicts to FileImportance objects
        important_files_data = repo_summary_data.get("important_files", [])
        important_files = []
        for f in important_files_data:
            if isinstance(f, dict):
                important_files.append(FileImportance(**f))
            elif isinstance(f, FileImportance):
                important_files.append(f)
        
        # Convert implementation_targets dicts to ImplementationTarget objects
        impl_targets_data = repo_summary_data.get("implementation_targets", [])
        impl_targets = []
        for t in impl_targets_data:
            if isinstance(t, dict):
                impl_targets.append(ImplementationTarget(**t))
            elif isinstance(t, ImplementationTarget):
                impl_targets.append(t)
        
        # Convert knowledge_graph dict to KnowledgeGraph object
        kg_data = repo_summary_data.get("knowledge_graph", {})
        if isinstance(kg_data, dict):
            kg = KnowledgeGraph(**kg_data)
        else:
            kg = kg_data
        
        # Convert configuration_analysis dict to ConfigurationAnalysis object
        config_data = repo_summary_data.get("configuration_analysis", {})
        if isinstance(config_data, dict):
            config = ConfigurationAnalysis(**config_data)
        else:
            config = config_data
        
        repo_summary = RepositorySummary(
            repository_name=repo_summary_data.get("repository_name", Path(repo_path).name),
            project_type=repo_summary_data.get("project_type", "Unknown"),
            architecture_summary=repo_summary_data.get("architecture_summary", "Unknown"),
            important_files=important_files,
            training_pipeline=str(repo_summary_data.get("training_pipeline", "")),
            knowledge_graph=kg,
            implementation_targets=impl_targets,
            configuration_analysis=config,
            analysis_timestamp=datetime.now(),
        )

        # Step 3: Compatibility analysis
        compat_input = CompatibilityInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        compat_output = await self.compatibility.execute(compat_input)

        # Step 4: Implementation planning
        impl_input = ImplementationPlannerInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
            compatibility_level=compat_output.report.overall_compatibility.value,
        )
        impl_output = await self.implementation.execute(impl_input)

        # Step 5: Impact analysis
        impact_input = ImpactAnalysisInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        impact_output = await self.impact.execute(impact_input)

        # Step 6: Experiment design
        exp_input = ExperimentDesignInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        exp_output = await self.experiment.execute(exp_input)

        # Step 7: Validation planning
        val_input = ValidationPlannerInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        val_output = await self.validation.execute(val_input)

        # Step 8: Risk assessment
        risk_input = RiskAssessmentInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        risk_output = await self.risk.execute(risk_input)

        # Step 9: Compute estimation
        compute_input = ComputeEstimatorInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            experiment_matrix=exp_output.matrix,
        )
        compute_output = await self.compute.execute(compute_input)

        # Step 10: Result prediction
        pred_input = ResultPredictionInput(
            paper_id=paper_id,
            repo_path=repo_path,
            summary=summary,
            repo_summary=repo_summary,
        )
        pred_output = await self.prediction.execute(pred_input)

        # Create full PlanResult
        plan_result = PlanResult(
            plan_id=f"{paper_id}_{Path(repo_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            paper_id=paper_id,
            repo_path=repo_path,
            timestamp=datetime.now(),
            compatibility=compat_output.report,
            implementation=impl_output.plan,
            impact=impact_output.impact,
            experiments=exp_output.matrix,
            validation=val_output.plan,
            risks=risk_output.assessment,
            compute=compute_output.estimate,
            predictions=pred_output.prediction,
        )

        # Step 11: Generate output files
        output_path = Path(output_dir) / "plans" / f"{paper_id}_{Path(repo_path).name}"
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Write individual markdown files
        md_files = {
            "compatibility_analysis.md": self._format_compatibility_md(compat_output.report),
            "implementation_plan.md": self._format_implementation_md(impl_output.plan),
            "experiment_matrix.md": self._format_experiment_md(exp_output.matrix),
            "validation_strategy.md": self._format_validation_md(val_output.plan),
            "risk_assessment.md": self._format_risk_md(risk_output.assessment),
            "cost_estimation.md": self._format_compute_md(compute_output.estimate),
            "expected_results.md": self._format_prediction_md(pred_output.prediction),
            "engineering_report.md": plan_result.to_engineering_report(),
        }

        for filename, content in md_files.items():
            filepath = output_path / filename
            filepath.write_text(content, encoding="utf-8")
            generated_files.append(str(filepath))

        # Write full JSON artifact
        plan_json_path = output_path / "plan_result.json"
        plan_json_path.write_text(
            plan_result.model_dump_json(indent=2), encoding="utf-8"
        )
        generated_files.append(str(plan_json_path))

        elapsed = time.time() - start_time

        return PlannerResult(
            plan_id=plan_result.plan_id,
            paper_id=paper_id,
            repo_path=repo_path,
            compatibility_report=compat_output.report.model_dump(),
            implementation_plan=impl_output.plan.model_dump(),
            impact_report=impact_output.impact.model_dump(),
            experiment_matrix=exp_output.matrix.model_dump(),
            validation_plan=val_output.plan.model_dump(),
            risk_assessment=risk_output.assessment.model_dump(),
            compute_estimate=compute_output.estimate.model_dump(),
            result_prediction=pred_output.prediction.model_dump(),
            engineering_report_md=plan_result.to_engineering_report(),
            planning_time_seconds=round(elapsed, 2),
            generated_files=generated_files,
        )

    def _format_compatibility_md(self, report: CompatibilityReport) -> str:
        lines = [
            f"# Compatibility Analysis: {report.paper_id}",
            "",
            f"**Repository**: {report.repo_path}",
            f"**Date**: {report.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"**Overall Compatibility**: {report.overall_compatibility.value}",
            "",
            "## Summary",
            "",
            report.overall_reasoning,
            "",
            "## Dimension Analysis",
            "",
            "| Dimension | Level | Reasoning |",
            "|-----------|-------|-----------|",
        ]
        for dim in [
            report.architecture_compatibility,
            report.training_compatibility,
            report.inference_compatibility,
            report.config_compatibility,
            report.checkpoint_compatibility,
            report.distributed_compatibility,
            report.evaluation_compatibility,
        ]:
            lines.append(f"| {dim.dimension} | {dim.level.value} | {dim.reasoning[:100]} |")

        lines.extend(["", "## Recommended Actions", ""])
        for action in report.recommended_actions:
            lines.append(f"- {action}")

        return "\n".join(lines)

    def _format_implementation_md(self, plan: ImplementationPlan) -> str:
        lines = [
            f"# Implementation Plan: {plan.paper_id}",
            "",
            f"**Repository**: {plan.repo_path}",
            f"**Date**: {plan.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"**Total Effort**: {plan.total_estimated_effort}",
            "",
            "## Steps",
            "",
        ]
        for step in plan.steps:
            lines.extend([
                f"### Step {step.step_number}: {step.title}",
                "",
                step.description,
                "",
                f"- **Difficulty**: {step.difficulty.value}",
                f"- **Effort**: {step.estimated_effort}",
                f"- **Risk**: {step.risk_level.value}",
                f"- **Dependencies**: Steps {', '.join(str(d) for d in step.dependencies)}" if step.dependencies else "- **Dependencies**: None",
                "",
            ])
        lines.extend(["", "## Modification Targets", ""])
        for tgt in plan.targets:
            lines.append(f"- **{tgt.file_path}** ({tgt.target_type}): {tgt.description} [{tgt.modification_type}]")
        lines.extend(["", "## Prerequisites", ""])
        for prereq in plan.prerequisite_knowledge:
            lines.append(f"- {prereq}")
        lines.extend(["", "## Success Criteria", ""])
        for criterion in plan.success_criteria:
            lines.append(f"- {criterion}")
        return "\n".join(lines)

    def _format_experiment_md(self, matrix: ExperimentMatrix) -> str:
        lines = [
            f"# Experiment Matrix: {matrix.paper_id}",
            "",
            f"**Repository**: {matrix.repo_path}",
            f"**Total Experiments**: {matrix.total_experiments}",
            f"**Estimated GPU Hours**: {matrix.estimated_total_gpu_hours}",
            "",
            "## Metrics to Track",
            "",
        ]
        for metric in matrix.metrics:
            lines.append(f"- **{metric.name}** ({metric.unit}): {metric.why_important}")
        lines.extend(["", "## Experiment Groups", ""])
        for group in matrix.groups:
            lines.extend([
                f"### {group.group_name}",
                "",
                group.description,
                "",
            ])
            for exp in group.experiments:
                lines.extend([
                    f"- **{exp.experiment_id}**: {exp.title}",
                    f"  - {exp.description}",
                    f"  - Duration: {exp.expected_duration}",
                    f"  - Resources: {', '.join(exp.required_resources)}",
                    "",
                ])
        return "\n".join(lines)

    def _format_validation_md(self, plan: ValidationPlan) -> str:
        lines = [
            f"# Validation Strategy: {plan.paper_id}",
            "",
            f"**Repository**: {plan.repo_path}",
            f"**Total Test Cases**: {plan.total_test_cases}",
            "",
            "## Approach",
            "",
            plan.validation_approach,
            "",
        ]
        for suite in plan.test_suites:
            lines.extend([
                f"### {suite.suite_name}",
                "",
                suite.description,
                "",
            ])
            for tc in suite.test_cases:
                lines.append(f"- [{tc.priority.value}] **{tc.test_name}**: {tc.description}")
            lines.append("")
        lines.extend(["", "## Acceptance Criteria", ""])
        for criterion in plan.acceptance_criteria:
            lines.append(f"- {criterion}")
        return "\n".join(lines)

    def _format_risk_md(self, assessment: RiskAssessment) -> str:
        lines = [
            f"# Risk Assessment: {assessment.paper_id}",
            "",
            f"**Repository**: {assessment.repo_path}",
            f"**Overall Risk**: {assessment.overall_risk_level.value}",
            "",
            assessment.overall_risk_reasoning,
            "",
        ]
        risk_categories = [
            ("Implementation Risks", assessment.implementation_risks),
            ("Training Risks", assessment.training_risks),
            ("Memory Risks", assessment.memory_risks),
            ("Performance Risks", assessment.performance_risks),
            ("Distributed Risks", assessment.distributed_risks),
            ("Inference Risks", assessment.inference_risks),
            ("Checkpoint Risks", assessment.checkpoint_risks),
        ]
        for cat_name, risk_items in risk_categories:
            if risk_items:
                lines.extend([f"### {cat_name}", ""])
                for risk in risk_items:
                    lines.extend([
                        f"- **[{risk.level.value}] {risk.risk_id}**: {risk.description}",
                        f"  - Probability: {risk.probability.value}",
                        f"  - Impact: {risk.impact}",
                        f"  - Mitigation: {risk.mitigation}",
                        f"  - Contingency: {risk.contingency}",
                        "",
                    ])
        return "\n".join(lines)

    def _format_compute_md(self, estimate: ComputeEstimate) -> str:
        lines = [
            f"# Compute Cost Estimation: {estimate.paper_id}",
            "",
            f"**Repository**: {estimate.repo_path}",
            "",
            "## Resource Estimates",
            "",
            "| Resource | Estimate |",
            "|----------|----------|",
            f"| GPU Type | {estimate.gpu_type} |",
            f"| GPUs per Experiment | {estimate.gpu_count_per_experiment} |",
            f"| GPU Hours per Experiment | {estimate.estimated_gpu_hours_per_experiment} |",
            f"| Total Experiments | {estimate.total_experiments} |",
            f"| Total GPU Hours | {estimate.total_gpu_hours} |",
            f"| Estimated Duration | {estimate.estimated_training_duration} |",
            f"| Peak Memory/GPU | {estimate.peak_memory_per_gpu_gb} GB |",
            f"| Total Storage | {estimate.total_storage_gb} GB |",
            f"| Cloud Cost (USD) | ${estimate.approximate_cloud_cost_usd:.2f} |",
            f"| Confidence | {estimate.confidence.value} |",
            "",
            "## Assumptions",
            "",
        ]
        for assumption in estimate.assumptions:
            lines.append(f"- {assumption}")
        return "\n".join(lines)

    def _format_prediction_md(self, prediction: ResultPrediction) -> str:
        lines = [
            f"# Expected Results: {prediction.paper_id}",
            "",
            f"**Repository**: {prediction.repo_path}",
            f"**Confidence**: {prediction.overall_confidence.value}",
            "",
            "## Best Case",
            "",
            prediction.best_case.description,
            "",
        ]
        for key, val in prediction.best_case.metrics.items():
            lines.append(f"- **{key}**: {val}")
        lines.extend([
            f"Probability: {prediction.best_case.probability.value}",
            "",
            "## Likely Case",
            "",
            prediction.likely_case.description,
            "",
        ])
        for key, val in prediction.likely_case.metrics.items():
            lines.append(f"- **{key}**: {val}")
        lines.extend([
            f"Probability: {prediction.likely_case.probability.value}",
            "",
            "## Worst Case",
            "",
            prediction.worst_case.description,
            "",
        ])
        for key, val in prediction.worst_case.metrics.items():
            lines.append(f"- **{key}**: {val}")
        lines.extend([
            f"Probability: {prediction.worst_case.probability.value}",
            "",
            "## Failure Modes",
            "",
        ])
        for fm in prediction.failure_modes:
            lines.extend([
                f"- **{fm.failure_id}**: {fm.description}",
                f"  - Trigger: {fm.trigger}",
                f"  - Detection: {fm.detection}",
                f"  - Recovery: {fm.recovery}",
                "",
            ])
        lines.extend(["", "## Success Criteria", ""])
        for criterion in prediction.success_criteria:
            lines.append(f"- {criterion}")
        return "\n".join(lines)
