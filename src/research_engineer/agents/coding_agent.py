"""Coding Agent for Phase 4.

Orchestrates code generation, patch creation, review, and implementation.
"""

import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.agents.repository_agent import RepositoryAgent
from research_engineer.agents.research_agent import ResearchAgent
from research_engineer.llm import LLMProvider
from research_engineer.models.coding import (
    ImplementationRequest,
    ImplementationResult,
    PatchStatus,
)
from research_engineer.models.planner import ImplementationPlan
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.code_generation import (
    CodeGenerationInput,
    CodeGenerationOutput,
    CodeGenerationTool,
)
from research_engineer.tools.implementation_report import (
    ImplementationReportInput,
    ImplementationReportOutput,
    ImplementationReportTool,
)
from research_engineer.tools.migration_planner import (
    MigrationPlannerInput,
    MigrationPlannerOutput,
    MigrationPlannerTool,
)
from research_engineer.tools.patch_generation import (
    PatchGenerationInput,
    PatchGenerationOutput,
    PatchGenerationTool,
)
from research_engineer.tools.rollback_planner import (
    RollbackPlannerInput,
    RollbackPlannerOutput,
    RollbackPlannerTool,
)
from research_engineer.tools.self_review import (
    SelfReviewInput,
    SelfReviewOutput,
    SelfReviewTool,
)
from research_engineer.tools.test_generation import (
    TestGenerationInput,
    TestGenerationOutput,
    TestGenerationTool,
)


class CodingAgentResult(BaseModel):
    """Result from coding agent."""

    implementation_id: str = Field(..., description="Unique implementation identifier")
    request_id: str = Field(..., description="Associated request ID")
    paper_id: str | None = Field(default=None, description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    task_description: str = Field(..., description="What was implemented")
    status: str = Field(default="completed", description="Implementation status")
    patches_generated: int = Field(default=0, description="Number of patches generated")
    tests_generated: int = Field(default=0, description="Number of tests generated")
    review_status: str = Field(default="pending", description="Review status")
    generated_files: list[str] = Field(default_factory=list, description="Generated files")
    implementation_time_seconds: float = Field(default=0.0, description="Duration")
    output_dir: str = Field(default="output", description="Output directory")


class CodingAgent:
    """
    Agent for implementing code changes.

    This agent orchestrates:
    1. Code generation from plans or tasks
    2. Patch generation
    3. Self-review
    4. Test generation
    5. Migration planning
    6. Rollback planning
    7. Report generation

    Follows patch-first philosophy: never directly modifies code.
    """

    def __init__(
        self,
        research_agent: ResearchAgent | None = None,
        repository_agent: RepositoryAgent | None = None,
        code_generation_tool: CodeGenerationTool | None = None,
        patch_generation_tool: PatchGenerationTool | None = None,
        self_review_tool: SelfReviewTool | None = None,
        test_generation_tool: TestGenerationTool | None = None,
        migration_planner_tool: MigrationPlannerTool | None = None,
        rollback_planner_tool: RollbackPlannerTool | None = None,
        implementation_report_tool: ImplementationReportTool | None = None,
        llm: LLMProvider | None = None,
    ):
        self.agent_name: str = "CodingAgent"
        self.research_agent = research_agent or ResearchAgent()
        self.repository_agent = repository_agent or RepositoryAgent()
        self.code_gen = code_generation_tool or CodeGenerationTool()
        self.patch_gen = patch_generation_tool or PatchGenerationTool()
        self.self_review = self_review_tool or SelfReviewTool()
        self.test_gen = test_generation_tool or TestGenerationTool()
        self.migration_planner = migration_planner_tool or MigrationPlannerTool()
        self.rollback_planner = rollback_planner_tool or RollbackPlannerTool()
        self.report_gen = implementation_report_tool or ImplementationReportTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def implement(
        self,
        task_description: str,
        repo_path: str,
        paper_input: str | None = None,
        implementation_plan: ImplementationPlan | None = None,
        output_dir: str = "output",
        constraints: list[str] | None = None,
        requirements: list[str] | None = None,
    ) -> CodingAgentResult:
        """
        Main entry point for implementation.

        Args:
            task_description: What to implement
            repo_path: Path to target repository
            paper_input: Optional paper ID/URL/PDF for context
            implementation_plan: Optional implementation plan from Phase 3
            output_dir: Directory to save outputs
            constraints: Implementation constraints
            requirements: Specific requirements

        Returns:
            CodingAgentResult with all implementation artifacts
        """
        start_time = time.time()

        # Create implementation request
        request_id = f"req_{int(time.time())}"
        request = ImplementationRequest(
            request_id=request_id,
            paper_id=None,
            repo_path=repo_path,
            task_description=task_description,
            implementation_plan_path=str(implementation_plan) if implementation_plan else None,
            constraints=constraints or [],
            requirements=requirements or [],
        )

        # Step 1: Understand paper (if provided)
        paper_summary = None
        if paper_input:
            paper_result = await self.research_agent.analyze(paper_input, output_dir=output_dir)
            paper_id = paper_result["paper_id"]
            request.paper_id = paper_id
            paper_summary = ResearchSummary(**paper_result["summary"])

        # Step 2: Understand repository
        repo_result = await self.repository_agent.analyze(repo_path, output_dir=output_dir)
        repo_summary = RepositorySummary(
            repository_name=repo_result.get("repository_name", Path(repo_path).name),
            project_type=repo_result.get("project_type", "Unknown"),
            architecture_summary=repo_result.get("architecture_summary", "Unknown"),
            important_files=repo_result.get("important_files", []),
            training_pipeline=str(repo_result.get("training_pipeline", "")),
            knowledge_graph=repo_result.get("knowledge_graph", {}) if isinstance(repo_result.get("knowledge_graph"), dict) else {},
            implementation_targets=repo_result.get("implementation_targets", []),
            configuration_analysis=repo_result.get("configuration_analysis", {}) if isinstance(repo_result.get("configuration_analysis"), dict) else {},
            analysis_timestamp=datetime.now(),
        )

        # Step 3: Generate code changes
        code_input = CodeGenerationInput(
            paper_id=request.paper_id,
            repo_path=repo_path,
            summary=paper_summary,
            repo_summary=repo_summary,
            implementation_plan=implementation_plan,
            task_description=task_description,
            constraints=constraints or [],
        )
        code_output: CodeGenerationOutput = await self.code_gen.execute(code_input)

        # Step 4: Generate patches
        patch_input = PatchGenerationInput(
            changes=code_output.changes,
            repo_path=repo_path,
            patch_id_prefix=f"patch_{request_id}",
        )
        patch_output: PatchGenerationOutput = await self.patch_gen.execute(patch_input)

        # Step 5: Self-review
        review_input = SelfReviewInput(
            patches=patch_output.patches,
            repo_path=repo_path,
            check_architecture=True,
            check_style=True,
            check_performance=True,
        )
        review_output: SelfReviewOutput = await self.self_review.execute(review_input)

        # Step 6: Generate tests
        test_input = TestGenerationInput(
            patches=patch_output.patches,
            changes=code_output.changes,
            repo_path=repo_path,
            include_edge_cases=True,
            include_performance_tests=False,
        )
        test_output: TestGenerationOutput = await self.test_gen.execute(test_input)

        # Step 7: Plan migration (if needed)
        migration_input = MigrationPlannerInput(
            patches=patch_output.patches,
            changes=code_output.changes,
            repo_path=repo_path,
            include_checkpoint_migration=True,
            include_config_migration=True,
        )
        migration_output: MigrationPlannerOutput = await self.migration_planner.execute(migration_input)

        # Step 8: Plan rollback
        rollback_input = RollbackPlannerInput(
            patches=patch_output.patches,
            repo_path=repo_path,
            include_data_recovery=True,
            include_checkpoint_recovery=True,
        )
        rollback_output: RollbackPlannerOutput = await self.rollback_planner.execute(rollback_input)

        # Step 9: Create implementation result
        implementation_id = f"impl_{request_id}"
        result = ImplementationResult(
            implementation_id=implementation_id,
            request_id=request_id,
            paper_id=request.paper_id,
            repo_path=repo_path,
            task_description=task_description,
            status="completed",
            patches_generated=patch_output.patches,
            tests_generated=test_output.test_suites,
            review_result=review_output.review_result,
            migration_plan=migration_output.migration_plan,
            rollback_plan=rollback_output.rollback_plan,
        )

        # Step 10: Generate reports
        report_input = ImplementationReportInput(
            implementation_result=result,
            output_dir=output_dir,
        )
        report_output: ImplementationReportOutput = await self.report_gen.execute(report_input)

        # Update result with report info
        result.implementation_report_md = report_output.report_md
        result.code_change_summary_md = report_output.change_summary_md
        result.patch_review_md = report_output.patch_review_md
        result.migration_plan_md = report_output.migration_plan_md
        result.test_plan_md = report_output.test_plan_md
        result.rollback_plan_md = report_output.rollback_plan_md
        result.generated_files = report_output.generated_files

        elapsed = time.time() - start_time

        return CodingAgentResult(
            implementation_id=implementation_id,
            request_id=request_id,
            paper_id=request.paper_id,
            repo_path=repo_path,
            task_description=task_description,
            status="completed",
            patches_generated=len(patch_output.patches),
            tests_generated=test_output.total_tests,
            review_status=review_output.review_result.status.value,
            generated_files=report_output.generated_files,
            implementation_time_seconds=round(elapsed, 2),
            output_dir=output_dir,
        )

    async def apply_patches(
        self,
        implementation_id: str,
        approved: bool = False,
        dry_run: bool = True,
    ) -> dict:
        """
        Apply patches from an implementation.

        Args:
            implementation_id: Implementation ID from previous run
            approved: Whether patches are approved for application
            dry_run: If True, only simulate application

        Returns:
            Application result
        """
        from research_engineer.tools.patch_application import (
            PatchApplicationInput,
            PatchApplicationTool,
        )

        # In a real implementation, would load patches from storage
        # For now, return placeholder result
        return {
            "status": "dry_run" if dry_run else "applied" if approved else "rejected",
            "message": "Patch application requires loading from storage (not implemented)",
            "dry_run": dry_run,
            "approved": approved,
        }
