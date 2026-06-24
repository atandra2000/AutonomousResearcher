"""Tests for Phase 4 Coding Agent and tools."""

import pytest
from pathlib import Path

from research_engineer.models.coding import (
    ChangeType,
    CodeChange,
    ComplexityLevel,
    GeneratedPatch,
    MigrationPlan,
    ReviewResult,
    ReviewStatus,
    RollbackPlan,
    TestSpecification,
    TestSuite,
    TestType,
)
from research_engineer.tools.code_generation import (
    CodeGenerationInput,
    CodeGenerationOutput,
    CodeGenerationTool,
)
from research_engineer.tools.patch_generation import (
    PatchGenerationInput,
    PatchGenerationOutput,
    PatchGenerationTool,
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
from research_engineer.tools.migration_planner import (
    MigrationPlannerInput,
    MigrationPlannerOutput,
    MigrationPlannerTool,
)
from research_engineer.tools.rollback_planner import (
    RollbackPlannerInput,
    RollbackPlannerOutput,
    RollbackPlannerTool,
)
from research_engineer.tools.implementation_report import (
    ImplementationReportInput,
    ImplementationReportOutput,
    ImplementationReportTool,
)
from research_engineer.agents.coding_agent import CodingAgent, CodingAgentResult


class TestCodeChange:
    """Test CodeChange model."""

    def test_create_code_change(self):
        """Test creating a code change."""
        change = CodeChange(
            file_path="models/attention.py",
            change_type=ChangeType.MODIFICATION,
            description="Add grouped query attention",
            reason="Implement GQA from paper",
            impact="medium",
            complexity=ComplexityLevel.MODERATE,
            estimated_lines_added=50,
        )

        assert change.file_path == "models/attention.py"
        assert change.change_type == ChangeType.MODIFICATION
        assert change.complexity == ComplexityLevel.MODERATE
        assert change.estimated_lines_added == 50

    def test_code_change_new_file(self):
        """Test creating a new file change."""
        change = CodeChange(
            file_path="models/mla.py",
            change_type=ChangeType.NEW_FILE,
            description="Add multi-head latent attention",
            reason="New attention mechanism",
            impact="high",
            complexity=ComplexityLevel.COMPLEX,
            estimated_lines_added=200,
        )

        assert change.change_type == ChangeType.NEW_FILE
        assert change.estimated_lines_added == 200


class TestGeneratedPatch:
    """Test GeneratedPatch model."""

    def test_create_patch(self):
        """Test creating a generated patch."""
        patch = GeneratedPatch(
            patch_id="patch_0001",
            file_path="models/attention.py",
            change_type=ChangeType.MODIFICATION,
            diff="--- a/models/attention.py\n+++ b/models/attention.py\n",
            explanation="Add GQA implementation",
            reason="Paper implementation",
            impact="medium",
            risk_level="medium",
        )

        assert patch.patch_id == "patch_0001"
        assert patch.approval_required is True
        assert patch.status.value == "generated"


class TestCodeGenerationTool:
    """Test CodeGenerationTool."""

    @pytest.mark.asyncio
    async def test_generate_changes(self, tmp_path):
        """Test code change generation."""
        # Create a temporary directory as fake repo
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        tool = CodeGenerationTool()
        input_data = CodeGenerationInput(
            repo_path=str(repo_path),
            task_description="Add new model layer",
        )

        output: CodeGenerationOutput = await tool.execute(input_data)

        assert isinstance(output, CodeGenerationOutput)
        assert len(output.changes) > 0
        assert output.generation_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_generate_from_plan(self, tmp_path):
        """Test code generation from implementation plan."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        tool = CodeGenerationTool()
        input_data = CodeGenerationInput(
            repo_path=str(repo_path),
            task_description="Implement feature",
            target_files=["models/model.py"],
        )

        output: CodeGenerationOutput = await tool.execute(input_data)

        assert isinstance(output, CodeGenerationOutput)
        assert output.total_changes > 0


class TestPatchGenerationTool:
    """Test PatchGenerationTool."""

    @pytest.mark.asyncio
    async def test_generate_patches(self, tmp_path):
        """Test patch generation."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        changes = [
            CodeChange(
                file_path="models/test.py",
                change_type=ChangeType.MODIFICATION,
                description="Test change",
                reason="Testing",
                complexity=ComplexityLevel.SIMPLE,
            )
        ]

        tool = PatchGenerationTool()
        input_data = PatchGenerationInput(
            changes=changes,
            repo_path=str(repo_path),
        )

        output: PatchGenerationOutput = await tool.execute(input_data)

        assert isinstance(output, PatchGenerationOutput)
        assert len(output.patches) > 0
        assert output.total_patches > 0


class TestSelfReviewTool:
    """Test SelfReviewTool."""

    @pytest.mark.asyncio
    async def test_review_patch(self, tmp_path):
        """Test self-review of patches."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        patches = [
            GeneratedPatch(
                patch_id="patch_0001",
                file_path="models/test.py",
                change_type=ChangeType.MODIFICATION,
                diff="--- a/models/test.py\n+++ b/models/test.py\n",
                explanation="Test",
                reason="Testing",
            )
        ]

        tool = SelfReviewTool()
        input_data = SelfReviewInput(
            patches=patches,
            repo_path=str(repo_path),
            check_architecture=True,
            check_style=True,
        )

        output: SelfReviewOutput = await tool.execute(input_data)

        assert isinstance(output, SelfReviewOutput)
        assert isinstance(output.review_result, ReviewResult)
        assert output.review_result.status in [ReviewStatus.APPROVED, ReviewStatus.CHANGES_REQUESTED]


class TestTestGenerationTool:
    """Test TestGenerationTool."""

    @pytest.mark.asyncio
    async def test_generate_tests(self, tmp_path):
        """Test test generation."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        patches = [
            GeneratedPatch(
                patch_id="patch_0001",
                file_path="models/test.py",
                change_type=ChangeType.MODIFICATION,
                diff="diff content",
                explanation="Test",
                reason="Testing",
            )
        ]

        tool = TestGenerationTool()
        input_data = TestGenerationInput(
            patches=patches,
            repo_path=str(repo_path),
            include_edge_cases=True,
        )

        output: TestGenerationOutput = await tool.execute(input_data)

        assert isinstance(output, TestGenerationOutput)
        assert len(output.test_suites) > 0
        assert output.total_tests > 0


class TestMigrationPlannerTool:
    """Test MigrationPlannerTool."""

    @pytest.mark.asyncio
    async def test_plan_migration(self, tmp_path):
        """Test migration planning."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        patches = [
            GeneratedPatch(
                patch_id="patch_0001",
                file_path="models/new_model.py",
                change_type=ChangeType.NEW_FILE,
                diff="diff content",
                explanation="New model",
                reason="Add feature",
                risk_level="high",
            )
        ]

        tool = MigrationPlannerTool()
        input_data = MigrationPlannerInput(
            patches=patches,
            repo_path=str(repo_path),
            include_checkpoint_migration=True,
        )

        output: MigrationPlannerOutput = await tool.execute(input_data)

        assert isinstance(output, MigrationPlannerOutput)
        assert output.migration_required is True
        assert isinstance(output.migration_plan, MigrationPlan)


class TestRollbackPlannerTool:
    """Test RollbackPlannerTool."""

    @pytest.mark.asyncio
    async def test_plan_rollback(self, tmp_path):
        """Test rollback planning."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()

        patches = [
            GeneratedPatch(
                patch_id="patch_0001",
                file_path="models/test.py",
                change_type=ChangeType.MODIFICATION,
                diff="diff",
                explanation="Test",
                reason="Testing",
            )
        ]

        tool = RollbackPlannerTool()
        input_data = RollbackPlannerInput(
            patches=patches,
            repo_path=str(repo_path),
        )

        output: RollbackPlannerOutput = await tool.execute(input_data)

        assert isinstance(output, RollbackPlannerOutput)
        assert isinstance(output.rollback_plan, RollbackPlan)
        assert output.rollback_plan.total_steps > 0


class TestImplementationReportTool:
    """Test ImplementationReportTool."""

    @pytest.mark.asyncio
    async def test_generate_report(self, tmp_path):
        """Test report generation."""
        from research_engineer.models.coding import ImplementationResult

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = ImplementationResult(
            implementation_id="impl_test_001",
            request_id="req_test_001",
            repo_path=str(tmp_path),
            task_description="Test implementation",
        )

        tool = ImplementationReportTool()
        input_data = ImplementationReportInput(
            implementation_result=result,
            output_dir=str(output_dir),
        )

        output: ImplementationReportOutput = await tool.execute(input_data)

        assert isinstance(output, ImplementationReportOutput)
        assert "Implementation Report" in output.report_md
        assert len(output.generated_files) > 0


class TestCodingAgentResult:
    """Test CodingAgentResult model."""

    def test_create_result(self):
        """Test creating coding agent result."""
        result = CodingAgentResult(
            implementation_id="impl_001",
            request_id="req_001",
            repo_path="./test_repo",
            task_description="Add feature",
            patches_generated=5,
            tests_generated=10,
            review_status="approved",
        )

        assert result.implementation_id == "impl_001"
        assert result.patches_generated == 5
        assert result.tests_generated == 10
        assert result.review_status == "approved"


class TestIntegration:
    """Integration tests for Phase 4."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        """Test full code generation pipeline."""
        repo_path = tmp_path / "fake_repo"
        repo_path.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Step 1: Generate code
        code_tool = CodeGenerationTool()
        code_input = CodeGenerationInput(
            repo_path=str(repo_path),
            task_description="Add test feature",
        )
        code_output = await code_tool.execute(code_input)

        # Step 2: Generate patches
        patch_tool = PatchGenerationTool()
        patch_input = PatchGenerationInput(
            changes=code_output.changes,
            repo_path=str(repo_path),
        )
        patch_output = await patch_tool.execute(patch_input)

        # Step 3: Self-review
        review_tool = SelfReviewTool()
        review_input = SelfReviewInput(
            patches=patch_output.patches,
            repo_path=str(repo_path),
        )
        review_output = await review_tool.execute(review_input)

        # Step 4: Generate tests
        test_tool = TestGenerationTool()
        test_input = TestGenerationInput(
            patches=patch_output.patches,
            repo_path=str(repo_path),
        )
        test_output = await test_tool.execute(test_input)

        # Verify pipeline
        assert len(code_output.changes) > 0
        assert len(patch_output.patches) > 0
        assert isinstance(review_output.review_result, ReviewResult)
        assert len(test_output.test_suites) > 0
