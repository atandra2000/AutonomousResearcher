"""Task Agent for Phase 11/12/13 - Terminal-first autonomous coding agent.

Orchestrates an autonomous coding turn driven by a natural-language
goal. Two modes:

**Legacy mode** (``delegate=False``, default — backward compatible):
1. Analyze repository.
2. Retrieve repository memory context (Phase 12).
3. Plan via LLM.
4. Generate patches (CodingAgent).
5. Show diff.
6. Optionally run tests.

**Delegation mode** (``delegate=True``, Phase 13):
1. Analyze repository (RepositoryAgentAdapter).
2. Research approach (ResearchAgentAdapter).
3. Retrieve repository memory context (Phase 12).
4. Create implementation plan (ArchitectAgent).
5. Generate code (CodingAgentAdapter).
6. Review generated code (ReviewerAgent).
7. Run tests (TestAgent).
8. Repair failures if necessary (configurable iterations).

The delegation mode uses a generic :class:`DelegationFramework` with
role/capability-based routing — no hardcoded task logic.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from research_engineer.agents._adapters import (
    CodingAgentAdapter,
    RepositoryAgentAdapter,
    ResearchAgentAdapter,
)
from research_engineer.agents._llm_support import resolve_llm
from research_engineer.agents._streaming import stream_complete
from research_engineer.agents.architect_agent import ArchitectAgent
from research_engineer.agents.coding_agent import CodingAgent
from research_engineer.agents.delegation import DelegationFramework
from research_engineer.agents.repository_agent import RepositoryAgent
from research_engineer.agents.reviewer_agent import ReviewerAgent
from research_engineer.agents.test_agent import TestAgent
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.delegation import (
    AgentCapability,
    AgentRole,
    SharedTaskContext,
)
from research_engineer.models.task import (
    TaskConfig,
    TaskResult,
    TaskStatus,
    TaskStep,
    TaskStepType,
    new_task_id,
)
from research_engineer.tools.terminal import TerminalInput, TerminalTool

try:
    from research_engineer.memory import RepositoryMemory

    _HAS_REPO_MEMORY = True
except ImportError:  # pragma: no cover
    RepositoryMemory = None  # type: ignore[assignment,misc]
    _HAS_REPO_MEMORY = False


class TaskAgent:
    """Terminal-first autonomous coding agent.

    Composes existing agents + the new :class:`TerminalTool` to turn a
    natural-language goal into reviewed code changes (and optionally a
    test run) against a target repository.

    When :class:`RepositoryMemory` (Phase 12) is available, the agent
    automatically retrieves relevant symbols, files, dependencies, and
    tests before planning and injects them into the LLM prompt — no
    manual user action required.

    When ``delegate=True`` (Phase 13), the agent becomes a coordinator
    that dispatches work to specialized agents via the
    :class:`DelegationFramework` with review/test repair loops.
    """

    def __init__(
        self,
        repository_agent: RepositoryAgent | None = None,
        coding_agent: CodingAgent | None = None,
        terminal_tool: TerminalTool | None = None,
        repository_memory: Any | None = None,
        architect_agent: ArchitectAgent | None = None,
        reviewer_agent: ReviewerAgent | None = None,
        test_agent: TestAgent | None = None,
        failure_analyzer: Any | None = None,
        repair_strategist: Any | None = None,
        self_repair_framework: Any | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "TaskAgent"
        self.repository_agent = repository_agent or RepositoryAgent(
            llm_enabled=False
        )
        self.coding_agent = coding_agent or CodingAgent()
        self.terminal = terminal_tool or TerminalTool()
        self.llm_provider = resolve_llm(self.agent_name, llm)
        # Phase 12: Repository memory (optional; auto-built if None and
        # the memory subsystem is importable).
        self._repository_memory: Any | None = repository_memory
        # Phase 13: Specialized agents for delegation mode.
        self._architect_agent = architect_agent
        self._reviewer_agent = reviewer_agent
        self._test_agent = test_agent
        # Phase 14: Self-repair components.
        self._failure_analyzer = failure_analyzer
        self._repair_strategist = repair_strategist
        self._self_repair_framework = self_repair_framework

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        goal: str,
        repo_path: str,
        config: TaskConfig | None = None,
        stream_sink: Any | None = None,
    ) -> TaskResult:
        """Execute a single autonomous coding turn.

        Args:
            goal: Natural-language coding goal.
            repo_path: Repository to operate on.
            config: Optional :class:`TaskConfig` override.
            stream_sink: Optional text sink for streamed LLM tokens
                (defaults to stdout inside :func:`stream_complete`).

        Returns:
            :class:`TaskResult` with all steps, diff, and test output.
        """
        start = time.time()
        cfg = config or TaskConfig(goal=goal, repo_path=repo_path)
        cfg = cfg.model_copy(update={"goal": goal, "repo_path": repo_path})

        # Phase 13: Branch into delegation mode if requested.
        if cfg.delegate:
            return await self._run_delegated(goal, repo_path, cfg, stream_sink, start)

        task_id = new_task_id()
        steps: list[TaskStep] = []
        impl_id: str | None = None
        patches_count = 0
        diff = ""
        test_exit: int | None = None
        test_stdout = ""
        test_stderr = ""
        generated_files: list[str] = []
        top_error: str | None = None
        status = TaskStatus.ANALYZING

        try:
            # Step 1: Analyze repository --------------------------------
            step = await self._step_analyze_repo(cfg)
            steps.append(step)
            if step.status == TaskStatus.FAILED:
                status = TaskStatus.FAILED
                top_error = step.error
                return self._finalize(
                    task_id, goal, repo_path, status, steps,
                    impl_id, patches_count, diff, test_exit,
                    test_stdout, test_stderr, generated_files,
                    start, top_error,
                )

            # Step 1b: Retrieve repository memory context (Phase 12) ---
            memory_context = await self._retrieve_memory_context(
                goal, cfg, stream_sink
            )

            # Step 2: Reason about goal + plan --------------------------
            status = TaskStatus.PLANNING
            step = await self._step_plan(goal, cfg, stream_sink, memory_context)
            steps.append(step)
            generated_files.extend(step.artifacts)

            # Step 3: Implement (generate patches) ----------------------
            status = TaskStatus.IMPLEMENTING
            step, impl_id, patches_count = await self._step_implement(
                goal, cfg, stream_sink
            )
            steps.append(step)
            generated_files.extend(step.artifacts)
            if step.status == TaskStatus.FAILED:
                status = TaskStatus.FAILED
                top_error = step.error

            # Step 4: Show diff -----------------------------------------
            if step.status != TaskStatus.FAILED:
                status = TaskStatus.DIFFING
                step, diff = await self._step_diff(cfg)
                steps.append(step)

            # Step 5: Optionally run tests ------------------------------
            if cfg.run_tests and step.status != TaskStatus.FAILED:
                status = TaskStatus.TESTING
                step, test_exit, test_stdout, test_stderr = (
                    await self._step_test(cfg)
                )
                steps.append(step)

            status = (
                TaskStatus.COMPLETED
                if all(s.status != TaskStatus.FAILED for s in steps)
                else TaskStatus.FAILED
            )
        except Exception as e:
            status = TaskStatus.FAILED
            top_error = str(e)

        return self._finalize(
            task_id, goal, repo_path, status, steps,
            impl_id, patches_count, diff, test_exit,
            test_stdout, test_stderr, generated_files,
            start, top_error,
        )

    # ------------------------------------------------------------------
    # Phase 13: Delegation pipeline
    # ------------------------------------------------------------------

    async def _run_delegated(
        self,
        goal: str,
        repo_path: str,
        cfg: TaskConfig,
        stream_sink: Any | None,
        start: float,
    ) -> TaskResult:
        """Run the multi-agent delegation pipeline (Phase 13)."""
        task_id = new_task_id()
        steps: list[TaskStep] = []

        # Build shared context.
        ctx = SharedTaskContext(
            goal=goal,
            repo_path=repo_path,
            paper_input=cfg.paper_input,
            output_dir=cfg.output_dir,
            max_repair_iterations=cfg.max_repair_iterations,
        )

        # Build delegation framework with registered agents.
        framework = self._build_delegation_framework()

        top_error: str | None = None
        repair_iterations = 0

        try:
            # 1. Repository analysis.
            step = await self._delegate_step(
                framework, AgentCapability.REPOSITORY_ANALYSIS, ctx,
                TaskStepType.ANALYZE_REPO, stream_sink,
            )
            steps.append(step)
            if step.status == TaskStatus.FAILED:
                return self._finalize_delegated(
                    task_id, goal, repo_path, TaskStatus.FAILED, steps,
                    ctx, start, step.error, repair_iterations,
                )

            # 2. Research approach.
            step = await self._delegate_step(
                framework, AgentCapability.RESEARCH, ctx,
                TaskStepType.RESEARCH, stream_sink,
            )
            steps.append(step)

            # 3. Retrieve repository memory context (Phase 12).
            ctx.memory_context = await self._retrieve_memory_context(
                goal, cfg, stream_sink
            )

            # 4. Architect: create implementation plan.
            step = await self._delegate_step(
                framework, AgentCapability.ARCHITECTURE, ctx,
                TaskStepType.PLAN, stream_sink,
            )
            steps.append(step)

            # 5. Coder: generate code.
            step = await self._delegate_step(
                framework, AgentCapability.CODE_GENERATION, ctx,
                TaskStepType.IMPLEMENT, stream_sink,
            )
            steps.append(step)
            if step.status == TaskStatus.FAILED:
                return self._finalize_delegated(
                    task_id, goal, repo_path, TaskStatus.FAILED, steps,
                    ctx, start, step.error, repair_iterations,
                )

            # 6. Get diff.
            diff_step, ctx.diff = await self._step_diff(cfg)
            steps.append(diff_step)

            # 7. Review + test + repair loop.
            review_test_steps, repair_iterations = (
                await self._run_review_test_loop(
                    framework, ctx, cfg, stream_sink
                )
            )
            steps.extend(review_test_steps)

            # Determine final status.
            all_ok = all(
                s.status != TaskStatus.FAILED for s in steps
            )
            status = TaskStatus.COMPLETED if all_ok else TaskStatus.FAILED
            if not all_ok:
                failed = [s for s in steps if s.status == TaskStatus.FAILED]
                top_error = failed[-1].error or "One or more steps failed"

        except Exception as e:
            status = TaskStatus.FAILED
            top_error = str(e)

        return self._finalize_delegated(
            task_id, goal, repo_path, status, steps,
            ctx, start, top_error, repair_iterations,
        )

    def _build_delegation_framework(self) -> DelegationFramework:
        """Build and register all agents in the delegation framework."""
        fw = DelegationFramework()
        # Existing agents via adapters.
        fw.register(
            RepositoryAgentAdapter(self.repository_agent),
            AgentRole.REPOSITORY_ANALYZER,
            [AgentCapability.REPOSITORY_ANALYSIS],
        )
        from research_engineer.agents.research_agent import ResearchAgent
        fw.register(
            ResearchAgentAdapter(ResearchAgent()),
            AgentRole.RESEARCHER,
            [AgentCapability.RESEARCH],
        )
        fw.register(
            CodingAgentAdapter(self.coding_agent),
            AgentRole.CODER,
            [AgentCapability.CODE_GENERATION, AgentCapability.REPAIR],
        )
        # New Phase 13 agents.
        fw.register(
            self._architect_agent or ArchitectAgent(),
            AgentRole.ARCHITECT,
            [AgentCapability.ARCHITECTURE],
        )
        fw.register(
            self._reviewer_agent or ReviewerAgent(),
            AgentRole.REVIEWER,
            [AgentCapability.CODE_REVIEW],
        )
        fw.register(
            self._test_agent or TestAgent(terminal_tool=self.terminal),
            AgentRole.TESTER,
            [AgentCapability.TEST_EXECUTION],
        )
        return fw

    async def _delegate_step(
        self,
        framework: DelegationFramework,
        capability: AgentCapability,
        ctx: SharedTaskContext,
        step_type: TaskStepType,
        stream_sink: Any | None,
    ) -> TaskStep:
        """Dispatch a capability and convert to a :class:`TaskStep`."""
        t0 = datetime.now()
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink
        if capability == AgentCapability.TEST_EXECUTION:
            pass
            # TestAgent reads command from kwargs or context.
        del_step = await framework.dispatch(capability, ctx, **kwargs)
        status = (
            TaskStatus.COMPLETED
            if del_step.status.value == "completed"
            else TaskStatus.FAILED
            if del_step.status.value == "failed"
            else TaskStatus.COMPLETED
        )
        return TaskStep(
            step_type=step_type,
            status=status,
            started_at=t0,
            finished_at=datetime.now(),
            duration_seconds=del_step.duration_seconds,
            summary=del_step.summary or del_step.status.value,
            artifacts=ctx.generated_files,
            error=del_step.error,
        )

    async def _run_review_test_loop(
        self,
        framework: DelegationFramework,
        ctx: SharedTaskContext,
        cfg: TaskConfig,
        stream_sink: Any | None,
    ) -> tuple[list[TaskStep], int]:
        """Run the review → test → repair loop.

        Phase 14: When the SelfRepairFramework is available, delegates to
        it for structured failure analysis, strategy generation, and
        iterative repair. Falls back to the Phase 13 basic loop for
        backward compatibility when self-repair is not configured.

        Returns the list of steps and the number of repair iterations.
        """
        # Phase 14: Use SelfRepairFramework if available.
        if self._self_repair_framework is not None:
            return await self._run_self_repair(
                framework, ctx, cfg, stream_sink
            )
        # Phase 13 fallback: basic review/test loop.
        return await self._run_basic_repair_loop(
            framework, ctx, cfg, stream_sink
        )

    async def _run_self_repair(
        self,
        framework: DelegationFramework,
        ctx: SharedTaskContext,
        cfg: TaskConfig,
        stream_sink: Any | None,
    ) -> tuple[list[TaskStep], int]:
        """Run the Phase 14 self-repair loop."""
        from research_engineer.agents.self_repair import SelfRepairFramework
        from research_engineer.models.repair import RepairConfig

        repair_config = RepairConfig(
            max_iterations=cfg.max_repair_iterations,
            require_review=True,
            require_tests=cfg.run_tests,
        )
        sr = self._self_repair_framework
        if isinstance(sr, type):
            sr = sr(
                delegation=framework,
                terminal=self.terminal,
                config=repair_config,
                failure_analyzer=self._failure_analyzer,
                repair_strategist=self._repair_strategist,
            )
        elif not isinstance(sr, SelfRepairFramework):
            sr = SelfRepairFramework(
                delegation=framework,
                terminal=self.terminal,
                config=repair_config,
                failure_analyzer=self._failure_analyzer,
                repair_strategist=self._repair_strategist,
            )
        result = await sr.run(
            ctx,
            test_command=cfg.test_command,
            timeout_seconds=cfg.timeout_seconds,
            stream_sink=stream_sink,
        )
        # Convert repair cycles to TaskSteps.
        steps: list[TaskStep] = []
        for cycle in result.cycles:
            t0 = datetime.now()
            steps.append(
                TaskStep(
                    step_type=TaskStepType.REPAIR,
                    status=(
                        TaskStatus.COMPLETED
                        if cycle.outcome.value == "success"
                        else TaskStatus.COMPLETED
                        if cycle.outcome.value == "partial"
                        else TaskStatus.FAILED
                    ),
                    started_at=t0,
                    finished_at=datetime.now(),
                    duration_seconds=cycle.duration_seconds,
                    summary=(
                        f"Cycle {cycle.cycle_number}: "
                        f"{cycle.outcome.value} "
                        f"({cycle.failure_report.category.value if cycle.failure_report else 'N/A'})"
                    ),
                    error=(
                        cycle.failure_report.root_cause
                        if cycle.failure_report and cycle.outcome.value != "success"
                        else None
                    ),
                )
            )
        return steps, result.total_cycles

    async def _run_basic_repair_loop(
        self,
        framework: DelegationFramework,
        ctx: SharedTaskContext,
        cfg: TaskConfig,
        stream_sink: Any | None,
    ) -> tuple[list[TaskStep], int]:
        """Phase 13 basic review/test loop (backward compatible)."""
        steps: list[TaskStep] = []
        iterations = 0
        kwargs: dict[str, Any] = {}
        if stream_sink is not None:
            kwargs["stream_sink"] = stream_sink

        for iteration in range(cfg.max_repair_iterations + 1):
            ctx.repair_iteration = iteration
            # Review.
            review_step = await self._delegate_step(
                framework, AgentCapability.CODE_REVIEW, ctx,
                TaskStepType.REVIEW, stream_sink,
            )
            steps.append(review_step)
            # Check review feedback.
            review_output = framework.steps[-1].output if framework.steps else {}
            feedback = review_output.get("feedback", {})
            approved = feedback.get("approved", True) if isinstance(feedback, dict) else True
            if not approved and iteration < cfg.max_repair_iterations:
                repair_step = await self._delegate_step(
                    framework, AgentCapability.REPAIR, ctx,
                    TaskStepType.REPAIR, stream_sink,
                )
                steps.append(repair_step)
                iterations += 1
                diff_step, ctx.diff = await self._step_diff(cfg)
                steps.append(diff_step)
                continue
            # Test.
            del_step = await framework.dispatch(
                AgentCapability.TEST_EXECUTION,
                ctx,
                test_command=cfg.test_command,
                timeout_seconds=cfg.timeout_seconds,
            )
            steps.append(
                TaskStep(
                    step_type=TaskStepType.TEST,
                    status=(
                        TaskStatus.COMPLETED
                        if del_step.status.value == "completed"
                        else TaskStatus.FAILED
                    ),
                    started_at=datetime.now(),
                    finished_at=datetime.now(),
                    duration_seconds=del_step.duration_seconds,
                    summary=del_step.summary or "",
                    error=del_step.error,
                )
            )
            if ctx.test_exit_code == 0:
                break
            if iteration < cfg.max_repair_iterations:
                repair_step = await self._delegate_step(
                    framework, AgentCapability.REPAIR, ctx,
                    TaskStepType.REPAIR, stream_sink,
                )
                steps.append(repair_step)
                iterations += 1
                diff_step, ctx.diff = await self._step_diff(cfg)
                steps.append(diff_step)
            else:
                break
        return steps, iterations

    @staticmethod
    def _finalize_delegated(
        task_id: str,
        goal: str,
        repo_path: str,
        status: TaskStatus,
        steps: list[TaskStep],
        ctx: SharedTaskContext,
        start: float,
        top_error: str | None,
        repair_iterations: int,
    ) -> TaskResult:
        """Assemble the final :class:`TaskResult` for delegation mode."""
        return TaskResult(
            task_id=task_id,
            goal=goal,
            repo_path=repo_path,
            status=status,
            steps=steps,
            implementation_id=ctx.implementation_id,
            patches_generated=ctx.patches_generated,
            diff=ctx.diff,
            test_exit_code=ctx.test_exit_code,
            test_stdout=ctx.test_stdout,
            test_stderr=ctx.test_stderr,
            generated_files=ctx.generated_files,
            processing_time_seconds=round(time.time() - start, 2),
            timestamp=datetime.now(),
            error=top_error,
            delegated=True,
            repair_iterations=repair_iterations,
            review_feedback=ctx.review_feedback,
            review_issues=ctx.review_issues,
            test_failures=ctx.test_failures,
        )

    # ------------------------------------------------------------------
    # Steps (legacy mode)
    # ------------------------------------------------------------------

    async def _step_analyze_repo(self, cfg: TaskConfig) -> TaskStep:
        """Step 1: analyze the target repository."""
        t0 = datetime.now()
        try:
            result = await self.repository_agent.analyze(
                cfg.repo_path, output_dir=cfg.output_dir, enable_llm=False
            )
            summary = (
                f"{result.get('repository_name', 'repo')} "
                f"({result.get('project_type', 'unknown')})"
            )
            return TaskStep(
                step_type=TaskStepType.ANALYZE_REPO,
                status=TaskStatus.COMPLETED,
                started_at=t0,
                finished_at=datetime.now(),
                duration_seconds=round(
                    (datetime.now() - t0).total_seconds(), 3
                ),
                summary=summary,
                artifacts=result.get("generated_files", []),
            )
        except Exception as e:
            return TaskStep(
                step_type=TaskStepType.ANALYZE_REPO,
                status=TaskStatus.FAILED,
                started_at=t0,
                finished_at=datetime.now(),
                duration_seconds=round(
                    (datetime.now() - t0).total_seconds(), 3
                ),
                error=str(e),
            )

    async def _retrieve_memory_context(
        self,
        goal: str,
        cfg: TaskConfig,
        stream_sink: Any | None,
    ) -> str:
        """Phase 12: retrieve repository memory context for the goal.

        Automatically builds/loads the repository memory index if not
        already present, then queries it for relevant symbols, files,
        dependencies, and tests. Returns a markdown context string
        (empty if memory is unavailable or the repo has no index).
        """
        if not _HAS_REPO_MEMORY:
            return ""
        mem = self._repository_memory
        if mem is None:
            try:
                mem = RepositoryMemory(cfg.repo_path)  # type: ignore[misc]
                if not mem.store.has_index(cfg.repo_path):
                    mem.build()
                self._repository_memory = mem
            except Exception:
                return ""
        try:
            return mem.get_context(goal, limit=8)
        except Exception:
            return ""

    async def _step_plan(
        self,
        goal: str,
        cfg: TaskConfig,
        stream_sink: Any | None,
        memory_context: str = "",
    ) -> TaskStep:
        """Step 2: reason about the goal and produce a short plan.

        Uses the orchestration model (minimax-m3:cloud by config). If the
        LLM provider is unavailable, falls back to a rule-based plan so
        the task still progresses. Repository memory context (Phase 12)
        is injected into the prompt when available.
        """
        t0 = datetime.now()
        plan_text = await self._llm_plan(goal, cfg, stream_sink, memory_context)
        # Persist the plan for traceability.
        out_dir = Path(cfg.output_dir) / "tasks"
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / f"plan_{int(time.time())}.md"
        plan_path.write_text(plan_text, encoding="utf-8")
        return TaskStep(
            step_type=TaskStepType.PLAN,
            status=TaskStatus.COMPLETED,
            started_at=t0,
            finished_at=datetime.now(),
            duration_seconds=round(
                (datetime.now() - t0).total_seconds(), 3
            ),
            summary=plan_text[:200],
            artifacts=[str(plan_path)],
        )

    async def _llm_plan(
        self,
        goal: str,
        cfg: TaskConfig,
        stream_sink: Any | None,
        memory_context: str = "",
    ) -> str:
        """Generate a short implementation plan via the LLM.

        When ``memory_context`` is non-empty (Phase 12 repository memory),
        it is appended to the user prompt so the model grounds its plan in
        the discovered symbols, files, dependencies, and tests.
        """
        provider = self.llm_provider
        if provider is None:
            return self._rule_based_plan(goal, memory_context)
        system = (
            "You are a senior ML engineer planning a focused code change. "
            "Produce a concise, ordered implementation plan (3-7 steps). "
            "Each step: one sentence. Do not write code. "
            "When repository memory context is provided, ground your plan "
            "in the discovered symbols, files, dependencies, and tests."
        )
        user = f"Goal: {goal}\nRepository: {cfg.repo_path}"
        if memory_context:
            user += f"\n\n{memory_context}"
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        if cfg.stream:
            resp = await stream_complete(
                provider, request, sink=stream_sink
            )
            if resp is not None:
                return resp.content
        try:
            resp = await provider.complete(request)
            return resp.content
        except Exception:
            return self._rule_based_plan(goal, memory_context)

    @staticmethod
    def _rule_based_plan(goal: str, memory_context: str = "") -> str:
        """Fallback plan when no LLM provider is configured."""
        plan = (
            f"# Implementation Plan\n\n"
            f"Goal: {goal}\n\n"
            f"1. Locate the relevant module(s) in the repository.\n"
            f"2. Implement the required change with minimal scope.\n"
            f"3. Add or update tests covering the change.\n"
            f"4. Run the test suite to verify.\n"
        )
        if memory_context:
            plan += f"\n## Repository Memory Context\n\n{memory_context}\n"
        return plan

    async def _step_implement(
        self,
        goal: str,
        cfg: TaskConfig,
        stream_sink: Any | None,
    ) -> tuple[TaskStep, str | None, int]:
        """Step 3: generate patches via CodingAgent."""
        t0 = datetime.now()
        try:
            result = await self.coding_agent.implement(
                task_description=goal,
                repo_path=cfg.repo_path,
                paper_input=cfg.paper_input,
                output_dir=cfg.output_dir,
            )
            return (
                TaskStep(
                    step_type=TaskStepType.IMPLEMENT,
                    status=TaskStatus.COMPLETED,
                    started_at=t0,
                    finished_at=datetime.now(),
                    duration_seconds=round(
                        (datetime.now() - t0).total_seconds(), 3
                    ),
                    summary=(
                        f"Generated {result.patches_generated} patch(es); "
                        f"review={result.review_status}"
                    ),
                    artifacts=result.generated_files,
                ),
                result.implementation_id,
                result.patches_generated,
            )
        except Exception as e:
            return (
                TaskStep(
                    step_type=TaskStepType.IMPLEMENT,
                    status=TaskStatus.FAILED,
                    started_at=t0,
                    finished_at=datetime.now(),
                    duration_seconds=round(
                        (datetime.now() - t0).total_seconds(), 3
                    ),
                    error=str(e),
                ),
                None,
                0,
            )

    async def _step_diff(self, cfg: TaskConfig) -> tuple[TaskStep, str]:
        """Step 4: show the git diff of the working tree."""
        t0 = datetime.now()
        out = await self.terminal.execute(
            TerminalInput(
                operation="git_diff",
                repo_path=cfg.repo_path,
            )
        )
        diff = out.content or out.stdout or ""
        return (
            TaskStep(
                step_type=TaskStepType.DIFF,
                status=TaskStatus.COMPLETED if out.success else TaskStatus.FAILED,
                started_at=t0,
                finished_at=datetime.now(),
                duration_seconds=out.duration_seconds,
                summary=f"{len(diff)} chars of diff",
                error=out.error,
            ),
            diff,
        )

    async def _step_test(
        self, cfg: TaskConfig
    ) -> tuple[TaskStep, int | None, str, str]:
        """Step 5: optionally run the test suite."""
        t0 = datetime.now()
        out = await self.terminal.execute(
            TerminalInput(
                operation="run_command",
                repo_path=cfg.repo_path,
                command=cfg.test_command,
                timeout_seconds=cfg.timeout_seconds,
                dry_run=False,
            )
        )
        return (
            TaskStep(
                step_type=TaskStepType.TEST,
                status=TaskStatus.COMPLETED if out.success else TaskStatus.FAILED,
                started_at=t0,
                finished_at=datetime.now(),
                duration_seconds=out.duration_seconds,
                summary=f"exit_code={out.exit_code}",
                error=out.error,
            ),
            out.exit_code,
            out.stdout,
            out.stderr,
        )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize(
        task_id: str,
        goal: str,
        repo_path: str,
        status: TaskStatus,
        steps: list[TaskStep],
        impl_id: str | None,
        patches_count: int,
        diff: str,
        test_exit: int | None,
        test_stdout: str,
        test_stderr: str,
        generated_files: list[str],
        start: float,
        top_error: str | None,
    ) -> TaskResult:
        """Assemble the final :class:`TaskResult`."""
        return TaskResult(
            task_id=task_id,
            goal=goal,
            repo_path=repo_path,
            status=status,
            steps=steps,
            implementation_id=impl_id,
            patches_generated=patches_count,
            diff=diff,
            test_exit_code=test_exit,
            test_stdout=test_stdout,
            test_stderr=test_stderr,
            generated_files=generated_files,
            processing_time_seconds=round(time.time() - start, 2),
            timestamp=datetime.now(),
            error=top_error,
        )


__all__ = ["TaskAgent"]
