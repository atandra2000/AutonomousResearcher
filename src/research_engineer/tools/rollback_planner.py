"""Rollback Planner Tool for Phase 4.

Generates rollback plans for code changes.
"""

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    GeneratedPatch,
    RollbackPlan,
    RollbackStep,
)
from research_engineer.tools.base import Tool, ToolError


class RollbackPlannerInput(BaseModel):
    """Input for rollback planning."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Patches to plan rollback for",
    )
    repo_path: str = Field(..., description="Repository path")
    include_data_recovery: bool = Field(
        default=True,
        description="Include data recovery steps",
    )
    include_checkpoint_recovery: bool = Field(
        default=True,
        description="Include checkpoint recovery steps",
    )


class RollbackPlannerOutput(BaseModel):
    """Output from rollback planning."""

    rollback_plan: RollbackPlan = Field(..., description="Generated rollback plan")
    complexity: str = Field(default="unknown", description="Rollback complexity")
    estimated_duration: str = Field(
        default="unknown",
        description="Estimated rollback duration",
    )
    planning_time_seconds: float = Field(
        default=0.0,
        description="Planning duration",
    )


class RollbackPlannerTool(Tool[RollbackPlannerInput, RollbackPlannerOutput]):
    """
    Tool for planning rollback strategies.

    This tool:
    1. Analyzes patches for rollback requirements
    2. Plans rollback steps
    3. Identifies failure scenarios
    4. Plans recovery procedures
    5. Assesses data loss risk
    """

    async def execute(self, input: RollbackPlannerInput) -> RollbackPlannerOutput:
        """Generate rollback plan."""
        import time
        start_time = time.time()

        try:
            # Generate rollback plan
            plan = await self._generate_rollback_plan(input)

            # Assess complexity
            complexity = self._assess_complexity(plan)

            # Estimate duration
            duration = self._estimate_duration(plan)

            elapsed = time.time() - start_time

            return RollbackPlannerOutput(
                rollback_plan=plan,
                complexity=complexity,
                estimated_duration=duration,
                planning_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Rollback planning failed: {e}", input, e)

    async def _generate_rollback_plan(self, input: RollbackPlannerInput) -> RollbackPlan:
        """Generate detailed rollback plan."""
        plan_id = f"rollback_{len(input.patches)}_patches"
        plan_title = f"Rollback plan for {len(input.patches)} patches"

        steps = []
        step_counter = 1

        # Step 1: Identify current state
        steps.append(RollbackStep(
            step_number=step_counter,
            title="Identify current state",
            description="Document current system state before rollback",
            action="Save logs, metrics, and system state",
            files_affected=[],
            estimated_duration="2 minutes",
            validation_check="State documented",
        ))
        step_counter += 1

        # Step 2: Stop dependent services
        steps.append(RollbackStep(
            step_number=step_counter,
            title="Stop dependent services",
            description="Stop any services using the modified code",
            action="Gracefully stop training/evaluation jobs",
            files_affected=["*.py"],
            estimated_duration="5 minutes",
            validation_check="Services stopped",
        ))
        step_counter += 1

        # Step 3: Revert code changes
        for patch in reversed(input.patches):
            steps.append(RollbackStep(
                step_number=step_counter,
                title=f"Revert {patch.file_path}",
                description=f"Revert changes to {patch.file_path}",
                action=f"Restore from backup or git revert {patch.patch_id}",
                files_affected=[patch.file_path],
                estimated_duration="2 minutes",
                validation_check=f"File {patch.file_path} restored",
            ))
            step_counter += 1

        # Step 4: Restore configurations
        if input.patches:
            steps.append(RollbackStep(
                step_number=step_counter,
                title="Restore configurations",
                description="Restore previous configuration files",
                action="Restore config from backup",
                files_affected=["config.yaml", "config.json"],
                estimated_duration="3 minutes",
                validation_check="Config validated",
            ))
            step_counter += 1

        # Step 5: Restore checkpoints (if applicable)
        if input.include_checkpoint_recovery:
            steps.append(RollbackStep(
                step_number=step_counter,
                title="Restore checkpoints",
                description="Restore previous checkpoint format",
                action="Restore checkpoint from backup",
                files_affected=["checkpoints/"],
                estimated_duration="10 minutes",
                validation_check="Checkpoint loads successfully",
            ))
            step_counter += 1

        # Step 6: Restore data (if applicable)
        if input.include_data_recovery:
            steps.append(RollbackStep(
                step_number=step_counter,
                title="Restore data",
                description="Restore any modified data files",
                action="Restore data from backup",
                files_affected=["data/"],
                estimated_duration="15 minutes",
                validation_check="Data integrity verified",
            ))
            step_counter += 1

        # Step 7: Restart services
        steps.append(RollbackStep(
            step_number=step_counter,
            title="Restart services",
            description="Restart dependent services",
            action="Start training/evaluation jobs",
            files_affected=[],
            estimated_duration="5 minutes",
            validation_check="Services running",
        ))
        step_counter += 1

        # Step 8: Validate rollback
        steps.append(RollbackStep(
            step_number=step_counter,
            title="Validate rollback",
            description="Run validation tests to confirm rollback success",
            action="Run smoke tests and basic functionality tests",
            files_affected=["tests/"],
            estimated_duration="10 minutes",
            validation_check="All validation tests pass",
        ))

        # Calculate total duration
        total_duration = self._sum_durations(steps)

        # Identify failure scenarios
        failure_scenarios = [
            "Backup files corrupted",
            "Services fail to restart",
            "Checkpoint format incompatible",
            "Data integrity issues",
        ]

        # Recovery procedures
        recovery_procedures = [
            "Use version control to restore code",
            "Restore from full system backup",
            "Contact on-call engineer",
            "Escalate to platform team",
        ]

        # Build plan
        plan = RollbackPlan(
            plan_id=plan_id,
            patch_id=",".join(p.patch_id for p in input.patches),
            title=plan_title,
            trigger_conditions=[
                "Critical bug discovered",
                "Performance regression detected",
                "Test failures in production",
                "User-reported issues",
            ],
            steps=steps,
            total_steps=len(steps),
            estimated_total_duration=total_duration,
            failure_scenarios=failure_scenarios,
            recovery_procedures=recovery_procedures,
            data_loss_risk="low" if input.include_data_recovery else "medium",
        )

        return plan

    def _sum_durations(self, steps: list[RollbackStep]) -> str:
        """Sum step durations."""
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

    def _assess_complexity(self, plan: RollbackPlan) -> str:
        """Assess rollback complexity."""
        if plan.total_steps <= 4:
            return "simple"
        elif plan.total_steps <= 8:
            return "moderate"
        else:
            return "complex"

    def _estimate_duration(self, plan: RollbackPlan) -> str:
        """Estimate total rollback duration."""
        return plan.estimated_total_duration
