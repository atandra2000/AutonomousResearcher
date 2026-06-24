"""Migration Planner Tool for Phase 4.

Generates migration plans for major code changes.
"""

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    CodeChange,
    GeneratedPatch,
    MigrationPlan,
    MigrationStep,
    RiskLevel,
)
from research_engineer.tools.base import Tool, ToolError


class MigrationPlannerInput(BaseModel):
    """Input for migration planning."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Patches requiring migration",
    )
    changes: list[CodeChange] = Field(
        default_factory=list,
        description="Code changes",
    )
    repo_path: str = Field(..., description="Repository path")
    include_checkpoint_migration: bool = Field(
        default=True,
        description="Include checkpoint migration steps",
    )
    include_config_migration: bool = Field(
        default=True,
        description="Include configuration migration steps",
    )
    backward_compatibility_required: bool = Field(
        default=True,
        description="Whether backward compatibility is required",
    )


class MigrationPlannerOutput(BaseModel):
    """Output from migration planning."""

    migration_plan: MigrationPlan | None = Field(
        default=None,
        description="Generated migration plan",
    )
    migration_required: bool = Field(
        default=False,
        description="Whether migration is required",
    )
    complexity: str = Field(
        default="unknown",
        description="Migration complexity",
    )
    estimated_duration: str = Field(
        default="unknown",
        description="Estimated migration duration",
    )
    planning_time_seconds: float = Field(
        default=0.0,
        description="Planning duration",
    )


class MigrationPlannerTool(Tool[MigrationPlannerInput, MigrationPlannerOutput]):
    """
    Tool for planning code migrations.

    This tool:
    1. Analyzes changes requiring migration
    2. Plans migration steps
    3. Handles checkpoint migration
    4. Handles config migration
    5. Plans backward compatibility
    6. Estimates duration and risk
    """

    async def execute(self, input: MigrationPlannerInput) -> MigrationPlannerOutput:
        """Generate migration plan."""
        import time
        start_time = time.time()

        try:
            # Determine if migration is needed
            migration_required = self._assess_migration_need(input)

            if not migration_required:
                return MigrationPlannerOutput(
                    migration_required=False,
                    complexity="none",
                    estimated_duration="0s",
                    planning_time_seconds=round(time.time() - start_time, 2),
                )

            # Generate migration plan
            plan = await self._generate_migration_plan(input)

            # Assess complexity
            complexity = self._assess_complexity(plan)

            # Estimate duration
            duration = self._estimate_duration(plan)

            elapsed = time.time() - start_time

            return MigrationPlannerOutput(
                migration_plan=plan,
                migration_required=True,
                complexity=complexity,
                estimated_duration=duration,
                planning_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Migration planning failed: {e}", input, e)

    def _assess_migration_need(self, input: MigrationPlannerInput) -> bool:
        """Assess if migration is needed."""
        # Check for significant changes
        for patch in input.patches:
            # New files or major modifications need migration
            if patch.change_type in ["new_file", "deletion"]:
                return True
            if patch.risk_level in ["high", "critical"]:
                return True

        for change in input.changes:
            if change.estimated_lines_added > 100:
                return True
            if change.change_type in ["new_file", "deletion"]:
                return True

        return False

    async def _generate_migration_plan(self, input: MigrationPlannerInput) -> MigrationPlan:
        """Generate detailed migration plan."""
        plan_id = f"migration_{len(input.patches)}_patches"
        plan_title = f"Migration for {len(input.patches)} patches"

        steps = []
        step_counter = 1

        # Step 1: Preparation
        steps.append(MigrationStep(
            step_number=step_counter,
            title="Preparation",
            description="Prepare for migration by backing up current state",
            action_type="preparation",
            files_affected=[],
            prerequisites=[],
            estimated_duration="5 minutes",
            risk_level=RiskLevel.LOW,
            rollback_instructions="No changes made yet",
            validation_steps=["Verify backup exists"],
        ))
        step_counter += 1

        # Step 2: Code changes
        if input.patches:
            steps.append(MigrationStep(
                step_number=step_counter,
                title="Apply code changes",
                description="Apply generated patches to codebase",
                action_type="code_change",
                files_affected=[p.file_path for p in input.patches],
                prerequisites=[1],
                estimated_duration=f"{len(input.patches) * 2} minutes",
                risk_level=self._assess_max_risk(input.patches),
                rollback_instructions="Revert patches in reverse order",
                validation_steps=["Run unit tests", "Check imports"],
            ))
            step_counter += 1

        # Step 3: Configuration migration
        if input.include_config_migration:
            steps.append(MigrationStep(
                step_number=step_counter,
                title="Migrate configurations",
                description="Update configuration files for new code",
                action_type="config_update",
                files_affected=["config.yaml", "config.json"],
                prerequisites=[2],
                estimated_duration="10 minutes",
                risk_level=RiskLevel.MEDIUM,
                rollback_instructions="Restore previous config backup",
                validation_steps=["Validate config syntax", "Check all required fields"],
            ))
            step_counter += 1

        # Step 4: Checkpoint migration
        if input.include_checkpoint_migration:
            steps.append(MigrationStep(
                step_number=step_counter,
                title="Migrate checkpoints",
                description="Convert existing checkpoints to new format",
                action_type="checkpoint_migration",
                files_affected=["checkpoints/"],
                prerequisites=[2],
                estimated_duration="30 minutes",
                risk_level=RiskLevel.HIGH,
                rollback_instructions="Keep old checkpoint format as backup",
                validation_steps=["Load migrated checkpoint", "Verify model weights"],
            ))
            step_counter += 1

        # Step 5: Testing
        steps.append(MigrationStep(
            step_number=step_counter,
            title="Run tests",
            description="Run full test suite to validate migration",
            action_type="testing",
            files_affected=["tests/"],
            prerequisites=[step_counter - 1],
            estimated_duration="20 minutes",
            risk_level=RiskLevel.LOW,
            rollback_instructions="Tests are read-only",
            validation_steps=["All tests pass", "No regressions"],
        ))
        step_counter += 1

        # Step 6: Documentation
        steps.append(MigrationStep(
            step_number=step_counter,
            title="Update documentation",
            description="Update documentation for changes",
            action_type="doc_update",
            files_affected=["README.md", "docs/"],
            prerequisites=[step_counter - 1],
            estimated_duration="15 minutes",
            risk_level=RiskLevel.LOW,
            rollback_instructions="Revert documentation changes",
            validation_steps=["Review documentation", "Check links"],
        ))

        # Calculate total duration
        total_duration = self._sum_durations(steps)

        # Assess backward compatibility
        backward_compat = "maintained" if input.backward_compatibility_required else "not required"

        # Build plan
        plan = MigrationPlan(
            plan_id=plan_id,
            title=plan_title,
            description=f"Migration plan for {len(input.patches)} patches and {len(input.changes)} changes",
            reason="Implementation of new features",
            steps=steps,
            total_steps=len(steps),
            estimated_total_duration=total_duration,
            backward_compatibility=backward_compat,
            checkpoint_migration_notes="Run checkpoint migration script if old checkpoints exist",
            config_migration_notes="Update config files to match new parameters",
            versioning_recommendations="Increment minor version",
            risks=self._identify_migration_risks(steps),
        )

        return plan

    def _assess_max_risk(self, patches: list[GeneratedPatch]) -> str:
        """Assess maximum risk level from patches."""
        risk_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_risk = "low"

        for patch in patches:
            if risk_order.get(patch.risk_level, 2) > risk_order.get(max_risk, 2):
                max_risk = patch.risk_level

        return max_risk

    def _sum_durations(self, steps: list[MigrationStep]) -> str:
        """Sum step durations."""
        # Simplified - in reality would parse duration strings
        total_minutes = 0
        for step in steps:
            if "minute" in step.estimated_duration.lower():
                try:
                    minutes = int(step.estimated_duration.split()[0])
                    total_minutes += minutes
                except (ValueError, IndexError):
                    pass

        if total_minutes < 60:
            return f"{total_minutes} minutes"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}h {minutes}m"

    def _identify_migration_risks(self, steps: list[MigrationStep]) -> list[str]:
        """Identify migration risks."""
        risks = []

        for step in steps:
            if step.risk_level in ["high", "critical"]:
                risks.append(f"High risk in step {step.step_number}: {step.title}")

        if not risks:
            risks.append("No critical risks identified")

        return risks

    def _assess_complexity(self, plan: MigrationPlan) -> str:
        """Assess migration complexity."""
        if plan.total_steps <= 3:
            return "simple"
        elif plan.total_steps <= 6:
            return "moderate"
        else:
            return "complex"

    def _estimate_duration(self, plan: MigrationPlan) -> str:
        """Estimate total migration duration."""
        return plan.estimated_total_duration
