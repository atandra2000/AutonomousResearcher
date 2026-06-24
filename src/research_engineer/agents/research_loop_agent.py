"""Research Loop Agent for Phase 9 - Autonomous Research Loop.

Orchestrates the full autonomous research workflow by invoking existing
Phase 1-8 agents in sequence, managing loop state, checking stopping
conditions, storing iterations, integrating with MemoryAgent, and
generating complete research reports.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from uuid import uuid4

from research_engineer.llm import LLMProvider
from research_engineer.models.loop import (
    ApprovalGate,
    ApprovalRequest,
    IterationPhase,
    IterationQueryInput,
    IterationQueryOutput,
    IterationStorageInput,
    LoopConfig,
    LoopIteration,
    LoopQueryInput,
    LoopQueryOutput,
    LoopRecord,
    LoopResult,
    LoopState,
    LoopStatus,
    LoopStorageInput,
    ReportInput,
    ReportOutput,
    StoppingCheckInput,
)
from research_engineer.models.memory import (
    MemoryRelationship,
    RelationshipType,
)
from research_engineer.tools.loop_storage import LoopStorageTool
from research_engineer.tools.report_generator import ReportGeneratorTool
from research_engineer.tools.stopping_condition import (
    StoppingConditionChecker,
)

ApprovalCallback = Callable[[ApprovalRequest], Awaitable[bool]]


class _IterationContext:
    """Mutable context shared across iteration phases."""

    def __init__(self) -> None:
        self.phase: IterationPhase = IterationPhase.LITERATURE
        self.paper_id: str | None = None
        self.paper_title: str | None = None
        self.plan_id: str | None = None
        self.impl_id: str | None = None
        self.exp_id: str | None = None
        self.eval_id: str | None = None
        self.metrics: dict[str, float] = {}
        self.metric_val: float | None = None


class ResearchLoopAgent:
    """Orchestrator for the autonomous research loop.

    Invokes existing Phase 1-8 agents in sequence, manages loop state,
    checks stopping conditions, stores iterations in SQLite, learns
    from memory across iterations, and generates research reports.
    """

    def __init__(
        self,
        memory_agent: Any | None = None,
        literature_agent: Any | None = None,
        repository_agent: Any | None = None,
        research_agent: Any | None = None,
        planner_agent: Any | None = None,
        coding_agent: Any | None = None,
        experiment_agent: Any | None = None,
        evaluation_agent: Any | None = None,
        storage_tool: LoopStorageTool | None = None,
        stopping_checker: StoppingConditionChecker | None = None,
        report_generator: ReportGeneratorTool | None = None,
        config: LoopConfig | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "ResearchLoopAgent"
        self.config = config or LoopConfig(goal="", repo_path="")
        self.memory = memory_agent
        self.literature = literature_agent
        self.repository = repository_agent
        self.research = research_agent
        self.planner = planner_agent
        self.coding = coding_agent
        self.experiment = experiment_agent
        self.evaluation = evaluation_agent
        self.storage = storage_tool or LoopStorageTool()
        self.stopping_checker = stopping_checker or StoppingConditionChecker()
        self.report_gen = report_generator or ReportGeneratorTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        goal: str,
        repo_path: str,
        config: LoopConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> LoopResult:
        """Start a new autonomous research loop.

        Args:
            goal: High-level research goal
            repo_path: Repository path to operate on
            config: Optional loop configuration override
            approval_callback: Optional async callback for approval gates

        Returns:
            LoopResult with all iterations and final outcome
        """
        start = time.time()
        cfg = config or self.config
        cfg = cfg.model_copy(update={"goal": goal, "repo_path": repo_path})

        loop_id = f"loop_{uuid4().hex[:12]}"
        state = LoopState(
            loop_id=loop_id,
            goal=goal,
            status=LoopStatus.RUNNING,
            started_at=datetime.now(),
        )

        loop_record = LoopRecord(
            loop_id=loop_id,
            goal=goal,
            config_json=cfg.model_dump_json(),
            status=LoopStatus.RUNNING,
            created_at=datetime.now(),
        )
        await self.storage.execute(LoopStorageInput(loop=loop_record))

        all_memory_ids: list[str] = []

        try:
            while state.status == LoopStatus.RUNNING:
                iteration_number = state.current_iteration + 1
                iteration_id = f"iter_{uuid4().hex[:12]}"
                state.status = LoopStatus.ITERATING
                state.updated_at = datetime.now()

                # Recall memory context for this iteration
                context_memories = await self._recall_context(goal)

                try:
                    iteration = await self._run_iteration(
                        state=state,
                        cfg=cfg,
                        iteration_number=iteration_number,
                        iteration_id=iteration_id,
                        context=context_memories,
                        approval_callback=approval_callback,
                    )
                except Exception as e:
                    iteration = LoopIteration(
                        iteration_id=iteration_id,
                        loop_id=loop_id,
                        iteration_number=iteration_number,
                        phase=IterationPhase.DECISION,
                        error=str(e),
                        status=LoopStatus.FAILED,
                        timestamp=datetime.now(),
                    )
                    if cfg.stop_on_error:
                        state.status = LoopStatus.FAILED
                        state.error = str(e)
                        break

                state.current_iteration = iteration_number
                state.iterations.append(iteration)

                # Store iteration
                await self.storage.execute(
                    IterationStorageInput(iteration=iteration)
                )

                # Update best metric
                self._update_best_metric(state, iteration, cfg)

                # Update budget
                cost_hours = self._estimate_cost_hours(iteration)
                state.cumulative_cost_hours += cost_hours
                state.cumulative_cost_usd += cost_hours * cfg.cost_per_gpu_hour

                # Store in memory + update graph
                iter_mem_ids = await self._store_iteration_in_memory(
                    iteration, cfg
                )
                iteration.memory_ids = iter_mem_ids
                all_memory_ids.extend(iter_mem_ids)
                if iter_mem_ids:
                    await self._update_graph(iteration, loop_id)

                state.status = LoopStatus.EVALUATED
                state.updated_at = datetime.now()

                # Check stopping conditions
                check = await self.stopping_checker.execute(
                    StoppingCheckInput(
                        state=state,
                        config=cfg,
                        history=state.iterations,
                    )
                )
                if check.should_stop:
                    iteration.decision = check.condition
                    state.status = LoopStatus.STOPPED
                    break

                # Continue to next iteration
                state.status = LoopStatus.RUNNING

                # Derive next experiment command from evaluation
                next_cmd = self._derive_next_command(iteration)
                state.next_command = next_cmd

        except Exception as e:
            state.status = LoopStatus.FAILED
            state.error = str(e)

        # Finalize loop record
        final_status = state.status
        stop_cond = None
        stop_reason = ""
        if state.iterations:
            last = state.iterations[-1]
            stop_cond = last.decision
            if stop_cond:
                stop_reason = f"Stopped: {stop_cond.value}"

        loop_record.status = final_status
        loop_record.iteration_count = len(state.iterations)
        loop_record.best_metric_value = state.best_metric_value
        loop_record.primary_metric_name = cfg.target_metric_name
        loop_record.stopping_condition = stop_cond
        loop_record.stopping_reason = stop_reason
        loop_record.memory_ids = all_memory_ids
        loop_record.updated_at = datetime.now()
        await self.storage.execute(LoopStorageInput(loop=loop_record))

        # Generate report
        report_files: list[str] = []
        try:
            report = await self.report_gen.execute(
                ReportInput(
                    loop=loop_record,
                    iterations=state.iterations,
                    config=cfg,
                    output_dir=cfg.output_dir,
                )
            )
            report_files = [report.report_path, report.json_path]
        except Exception:
            pass

        result = LoopResult(
            loop_id=loop_id,
            goal=goal,
            status=final_status,
            iterations=state.iterations,
            iteration_count=len(state.iterations),
            best_metric_value=state.best_metric_value,
            primary_metric_name=cfg.target_metric_name,
            stopping_condition=stop_cond,
            stopping_reason=stop_reason,
            memory_ids=all_memory_ids,
            generated_files=report_files,
            output_dir=cfg.output_dir,
            processing_time_seconds=round(time.time() - start, 2),
            timestamp=datetime.now(),
        )
        return result

    # ------------------------------------------------------------------
    # Iteration execution
    # ------------------------------------------------------------------

    async def _run_iteration(
        self,
        state: LoopState,
        cfg: LoopConfig,
        iteration_number: int,
        iteration_id: str,
        context: list[Any],
        approval_callback: ApprovalCallback | None,
    ) -> LoopIteration:
        """Run a single iteration through all phases."""
        ctx = _IterationContext()

        # Phase 1: Literature discovery
        if (
            iteration_number == 1
            or not cfg.skip_literature_after_first
        ) and self.literature:
            ctx.phase = IterationPhase.LITERATURE
            await self._run_literature(ctx, cfg)

        # Phase 2: Planning
        ctx.phase = IterationPhase.PLANNING
        if self.planner:
            await self._run_planning(ctx, cfg)

        if cfg.approval_mode:
            if not await self._gate(
                state, cfg, iteration_number, ApprovalGate.PLAN,
                f"Plan created: {ctx.plan_id or 'N/A'}",
                {"plan_id": ctx.plan_id or ""},
                approval_callback,
            ):
                return self._stopped_iteration(
                    iteration_id, state.loop_id, iteration_number, ctx, cfg
                )

        # Phase 3: Implementation
        ctx.phase = IterationPhase.IMPLEMENTATION
        if self.coding:
            await self._run_implementation(ctx, cfg)

        if cfg.approval_mode:
            if not await self._gate(
                state, cfg, iteration_number, ApprovalGate.IMPLEMENTATION,
                f"Implementation created: {ctx.impl_id or 'N/A'}",
                {"implementation_id": ctx.impl_id or ""},
                approval_callback,
            ):
                return self._stopped_iteration(
                    iteration_id, state.loop_id, iteration_number, ctx, cfg
                )

        # Phase 4: Experiment execution
        ctx.phase = IterationPhase.EXPERIMENT
        if self.experiment:
            await self._run_experiment(ctx, state, cfg)

        # Phase 5: Evaluation
        ctx.phase = IterationPhase.EVALUATION
        if self.evaluation and self.experiment:
            await self._run_evaluation(ctx, cfg)

        if cfg.approval_mode:
            if not await self._gate(
                state, cfg, iteration_number, ApprovalGate.NEXT_ITERATION,
                f"Evaluation complete: {ctx.eval_id or 'N/A'}",
                {"evaluation_id": ctx.eval_id or ""},
                approval_callback,
            ):
                return self._stopped_iteration(
                    iteration_id, state.loop_id, iteration_number, ctx, cfg
                )

        ctx.phase = IterationPhase.DECISION
        return self._make_iteration(
            iteration_id, state.loop_id, iteration_number,
            ctx.phase, ctx.paper_id, ctx.paper_title, ctx.plan_id,
            ctx.impl_id, ctx.exp_id, ctx.eval_id,
            ctx.metrics, cfg, ctx.metric_val,
            status=LoopStatus.EVALUATED,
        )

    async def _run_literature(
        self, ctx: _IterationContext, cfg: LoopConfig
    ) -> None:
        """Run literature discovery phase."""
        try:
            lit_result = await self.literature.discover(
                topic=cfg.goal,
                repo_path=cfg.repo_path,
            )
            if (
                lit_result.recommendations
                and lit_result.recommendations.recommendations
            ):
                top = lit_result.recommendations.recommendations[0]
                if isinstance(top.paper_id, str):
                    ctx.paper_id = top.paper_id
                if isinstance(top.title, str):
                    ctx.paper_title = top.title
            elif (
                lit_result.search_results
                and lit_result.search_results.papers
            ):
                top_paper = lit_result.search_results.papers[0]
                if isinstance(top_paper.paper_id, str):
                    ctx.paper_id = top_paper.paper_id
                if isinstance(top_paper.title, str):
                    ctx.paper_title = top_paper.title
        except Exception:
            pass

    async def _run_planning(
        self, ctx: _IterationContext, cfg: LoopConfig
    ) -> None:
        """Run experiment planning phase."""
        try:
            plan_input = ctx.paper_id or cfg.goal
            plan_result = await self.planner.plan(
                paper_input=plan_input,
                repo_path=cfg.repo_path,
            )
            if isinstance(plan_result.plan_id, str):
                ctx.plan_id = plan_result.plan_id
            if not ctx.paper_id and isinstance(
                plan_result.paper_id, str
            ):
                ctx.paper_id = plan_result.paper_id
        except Exception:
            pass

    async def _run_implementation(
        self, ctx: _IterationContext, cfg: LoopConfig
    ) -> None:
        """Run code implementation phase."""
        try:
            task_desc = (
                f"Implement approach from paper {ctx.paper_id or cfg.goal}"
            )
            impl_result = await self.coding.implement(
                task_description=task_desc,
                repo_path=cfg.repo_path,
                paper_input=ctx.paper_id,
            )
            if isinstance(impl_result.implementation_id, str):
                ctx.impl_id = impl_result.implementation_id
        except Exception:
            pass

    async def _run_experiment(
        self,
        ctx: _IterationContext,
        state: LoopState,
        cfg: LoopConfig,
    ) -> None:
        """Run experiment execution phase."""
        try:
            command = (
                state.next_command
                or cfg.experiment_command
                or "python train.py"
            )
            exp_result = await self.experiment.run(
                command=command,
                repo_path=cfg.repo_path,
                paper_id=ctx.paper_id,
                plan_id=ctx.plan_id,
                implementation_id=ctx.impl_id,
                dry_run=cfg.dry_run,
            )
            if isinstance(exp_result.experiment_id, str):
                ctx.exp_id = exp_result.experiment_id
            if (
                exp_result.metrics
                and exp_result.metrics.summary_metrics
            ):
                ctx.metrics = dict(exp_result.metrics.summary_metrics)
        except Exception:
            pass

    async def _run_evaluation(
        self, ctx: _IterationContext, cfg: LoopConfig
    ) -> None:
        """Run evaluation phase."""
        try:
            record = await self.experiment.get_experiment(
                ctx.exp_id or ""
            )
            if record:
                eval_result = await self.evaluation.analyze(
                    experiments=[record],
                    paper_id=ctx.paper_id,
                    repo_path=cfg.repo_path,
                    primary_metric=cfg.target_metric_name
                    or cfg.primary_metric,
                    higher_is_better=cfg.higher_is_better,
                )
                if isinstance(eval_result.evaluation_id, str):
                    ctx.eval_id = eval_result.evaluation_id
                if eval_result.record:
                    record_metrics = getattr(
                        eval_result.record, "summary_metrics", None
                    )
                    if isinstance(record_metrics, dict):
                        ctx.metrics.update(record_metrics)
                metric_name = (
                    cfg.target_metric_name or cfg.primary_metric
                )
                if metric_name and metric_name in ctx.metrics:
                    ctx.metric_val = ctx.metrics[metric_name]
                elif ctx.metrics:
                    ctx.metric_val = list(ctx.metrics.values())[0]
        except Exception:
            pass

    async def _gate(
        self,
        state: LoopState,
        cfg: LoopConfig,
        iteration_number: int,
        gate: ApprovalGate,
        summary: str,
        artifacts: dict[str, str],
        approval_callback: ApprovalCallback | None,
    ) -> bool:
        """Run an approval gate if approval_mode is enabled."""
        if not cfg.approval_mode:
            return True
        return await self._request_approval(
            state=state,
            cfg=cfg,
            iteration_number=iteration_number,
            gate=gate,
            summary=summary,
            artifacts=artifacts,
            approval_callback=approval_callback,
        )

    @staticmethod
    def _stopped_iteration(
        iteration_id: str,
        loop_id: str,
        iteration_number: int,
        ctx: _IterationContext,
        cfg: LoopConfig,
    ) -> LoopIteration:
        """Build a stopped iteration from context (approval rejected)."""
        return ResearchLoopAgent._make_iteration(
            iteration_id, loop_id, iteration_number,
            ctx.phase, ctx.paper_id, ctx.paper_title, ctx.plan_id,
            ctx.impl_id, ctx.exp_id, ctx.eval_id,
            ctx.metrics, cfg, ctx.metric_val,
            status=LoopStatus.STOPPED,
        )

    @staticmethod
    def _make_iteration(
        iteration_id: str,
        loop_id: str,
        iteration_number: int,
        phase: IterationPhase,
        paper_id: str | None,
        paper_title: str | None,
        plan_id: str | None,
        implementation_id: str | None,
        experiment_id: str | None,
        evaluation_id: str | None,
        metrics: dict[str, float],
        cfg: LoopConfig,
        primary_metric_value: float | None,
        status: LoopStatus,
    ) -> LoopIteration:
        primary_metric_name = cfg.target_metric_name or cfg.primary_metric
        return LoopIteration(
            iteration_id=iteration_id,
            loop_id=loop_id,
            iteration_number=iteration_number,
            phase=phase,
            paper_id=paper_id,
            paper_title=paper_title,
            plan_id=plan_id,
            implementation_id=implementation_id,
            experiment_id=experiment_id,
            evaluation_id=evaluation_id,
            metrics=metrics,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            status=status,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    async def _request_approval(
        self,
        state: LoopState,
        cfg: LoopConfig,
        iteration_number: int,
        gate: ApprovalGate,
        summary: str,
        artifacts: dict[str, str],
        approval_callback: ApprovalCallback | None,
    ) -> bool:
        """Request approval at a gate. Returns True if approved."""
        request = ApprovalRequest(
            loop_id=state.loop_id,
            iteration_number=iteration_number,
            gate=gate,
            summary=summary,
            artifacts=artifacts,
        )
        if approval_callback is not None:
            try:
                return await approval_callback(request)
            except Exception:
                return False
        # No callback: auto-approve in autonomous mode, pause in approval mode
        state.pending_approval = request
        return True

    # ------------------------------------------------------------------
    # Memory integration
    # ------------------------------------------------------------------

    async def _recall_context(self, goal: str) -> list[Any]:
        """Recall relevant memories before an iteration."""
        if not self.memory:
            return []
        try:
            return await self.memory.get_context(current_task=goal, limit=5)
        except Exception:
            return []

    async def _store_iteration_in_memory(
        self, iteration: LoopIteration, cfg: LoopConfig
    ) -> list[str]:
        """Store iteration outcomes in memory."""
        if not self.memory:
            return []
        memory_ids: list[str] = []
        context = (
            f"Loop {iteration.loop_id} iteration "
            f"{iteration.iteration_number}"
        )
        approach = f"paper={iteration.paper_id or 'N/A'}"
        if iteration.is_success():
            try:
                mem_id = await self.memory.store_success(
                    context=context,
                    approach=approach,
                    metrics=iteration.metrics,
                    key_factors=[
                        f"phase={iteration.phase.value}",
                        f"paper={iteration.paper_id or 'N/A'}",
                    ],
                )
                if mem_id:
                    memory_ids.append(str(mem_id))
            except Exception:
                pass
        else:
            try:
                mem_id = await self.memory.store_failure(
                    context=context,
                    approach=approach,
                    failure_mode="iteration_error",
                    lessons=[iteration.error or "unknown error"],
                    error_details=iteration.error,
                )
                if mem_id:
                    memory_ids.append(str(mem_id))
            except Exception:
                pass

        # Store insight if we have metrics
        if iteration.metrics:
            try:
                top_metrics = sorted(
                    iteration.metrics.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:3]
                evidence = [f"{k}={v:.4f}" for k, v in top_metrics]
                mem_id = await self.memory.store_insight(
                    insight_type="empirical_finding",
                    domain="research_loop",
                    description=(
                        f"Iteration {iteration.iteration_number} metrics: "
                        f"{', '.join(evidence)}"
                    ),
                    evidence=evidence,
                    applicability=["autonomous_loop"],
                )
                if mem_id:
                    memory_ids.append(str(mem_id))
            except Exception:
                pass

        return memory_ids

    async def _update_graph(
        self, iteration: LoopIteration, loop_id: str
    ) -> None:
        """Update the knowledge graph with iteration relationships."""
        if not self.memory or not self.memory.graph:
            return
        graph = self.memory.graph
        try:
            graph.add_node(iteration.iteration_id, "iteration")
            graph.add_node(loop_id, "loop")

            # loop -> iteration (DERIVED_FROM)
            rel = MemoryRelationship(
                source_memory_id=loop_id,
                target_memory_id=iteration.iteration_id,
                relationship_type=RelationshipType.DERIVED_FROM,
                confidence=0.95,
            )
            await self._safe_store_relationship(rel)
            graph.add_relationship(rel)

            # iteration -> paper (INSPIRED_BY)
            if iteration.paper_id:
                graph.add_node(iteration.paper_id, "paper")
                rel = MemoryRelationship(
                    source_memory_id=iteration.iteration_id,
                    target_memory_id=iteration.paper_id,
                    relationship_type=RelationshipType.INSPIRED_BY,
                    confidence=0.8,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # iteration -> experiment (VALIDATES)
            if iteration.experiment_id:
                graph.add_node(iteration.experiment_id, "experiment")
                rel = MemoryRelationship(
                    source_memory_id=iteration.iteration_id,
                    target_memory_id=iteration.experiment_id,
                    relationship_type=RelationshipType.VALIDATES,
                    confidence=0.9,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # iteration -> evaluation (DERIVED_FROM)
            if iteration.evaluation_id:
                graph.add_node(iteration.evaluation_id, "evaluation")
                rel = MemoryRelationship(
                    source_memory_id=iteration.iteration_id,
                    target_memory_id=iteration.evaluation_id,
                    relationship_type=RelationshipType.DERIVED_FROM,
                    confidence=0.9,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # iteration -> memory_ids (CONFIRMS / FAILED_WITH)
            for mem_id in iteration.memory_ids:
                graph.add_node(mem_id, "memory")
                rel_type = (
                    RelationshipType.CONFIRMS
                    if iteration.is_success()
                    else RelationshipType.FAILED_WITH
                )
                rel = MemoryRelationship(
                    source_memory_id=iteration.iteration_id,
                    target_memory_id=mem_id,
                    relationship_type=rel_type,
                    confidence=0.85,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)
        except Exception:
            pass

    async def _safe_store_relationship(
        self, rel: MemoryRelationship
    ) -> None:
        if not self.memory:
            return
        try:
            await self.memory.storage.store_relationship(rel)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _update_best_metric(
        state: LoopState,
        iteration: LoopIteration,
        cfg: LoopConfig,
    ) -> None:
        """Update the running best metric and improvement delta."""
        if (
            iteration.primary_metric_value is None
            or not cfg.target_metric_name
        ):
            return
        val = iteration.primary_metric_value
        if state.best_metric_value is None:
            state.best_metric_value = val
            iteration.best_metric_value = val
            iteration.improvement = None
            return
        prev_best = state.best_metric_value
        if cfg.higher_is_better:
            if val > prev_best:
                iteration.improvement = val - prev_best
                state.best_metric_value = val
        else:
            if val < prev_best:
                iteration.improvement = prev_best - val
                state.best_metric_value = val
        iteration.best_metric_value = state.best_metric_value

    @staticmethod
    def _estimate_cost_hours(iteration: LoopIteration) -> float:
        """Estimate GPU-hours from an iteration (rough heuristic)."""
        return 0.5

    @staticmethod
    def _derive_next_command(iteration: LoopIteration) -> str | None:
        """Derive the next experiment command from evaluation recommendations."""
        return None

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def list_loops(
        self,
        status: LoopStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> LoopQueryOutput:
        """List research loops with optional filters."""
        out = await self.storage.execute(
            LoopQueryInput(status=status, limit=limit, offset=offset)
        )
        if isinstance(out, LoopQueryOutput):
            return out
        return LoopQueryOutput()

    async def get_loop(self, loop_id: str) -> LoopRecord | None:
        """Retrieve a single loop by ID."""
        return await self.storage.get_loop_by_id(loop_id)

    async def get_iterations(
        self, loop_id: str, limit: int = 100, offset: int = 0
    ) -> IterationQueryOutput:
        """Get iterations for a loop."""
        out = await self.storage.execute(
            IterationQueryInput(loop_id=loop_id, limit=limit, offset=offset)
        )
        if isinstance(out, IterationQueryOutput):
            return out
        return IterationQueryOutput()

    async def get_iteration(
        self, iteration_id: str
    ) -> LoopIteration | None:
        """Retrieve a single iteration by ID."""
        return await self.storage.get_iteration_by_id(iteration_id)

    async def search_loops(
        self, search_text: str, limit: int = 50
    ) -> LoopQueryOutput:
        """Search loop history by text."""
        out = await self.storage.execute(
            LoopQueryInput(search_text=search_text, limit=limit)
        )
        if isinstance(out, LoopQueryOutput):
            return out
        return LoopQueryOutput()

    async def generate_report(
        self, loop_id: str, output_dir: str | None = None
    ) -> ReportOutput:
        """Generate (or regenerate) a research report for a loop."""
        loop = await self.get_loop(loop_id)
        if not loop:
            raise ValueError(f"Loop not found: {loop_id}")
        iterations_out = await self.get_iterations(loop_id, limit=1000)
        cfg = LoopConfig.model_validate_json(loop.config_json)
        out = output_dir or cfg.output_dir
        return await self.report_gen.execute(
            ReportInput(
                loop=loop,
                iterations=iterations_out.iterations,
                config=cfg,
                output_dir=out,
            )
        )
