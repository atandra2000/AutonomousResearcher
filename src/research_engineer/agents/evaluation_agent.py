"""Evaluation Agent for Phase 8 - Evaluation & Conclusions.

Orchestrates experiment comparison, training dynamics analysis,
statistical significance testing, next-experiment recommendations,
paper recommendations via LiteratureAgent, conclusion storage in
MemoryAgent, and automatic knowledge-graph updates.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_engineer.llm import LLMProvider
from research_engineer.models.evaluation import (
    EvaluationConfig,
    EvaluationQueryInput,
    EvaluationQueryOutput,
    EvaluationRecord,
    EvaluationResult,
    EvaluationStorageInput,
    EvaluationStorageOutput,
    ExperimentComparisonInput,
    ExperimentComparisonOutput,
    NextExperimentInput,
    NextExperimentOutput,
    PaperSuggestion,
    StatisticalSignificanceInput,
    StatisticalSignificanceOutput,
    TrainingDynamicsInput,
    TrainingDynamicsOutput,
)
from research_engineer.models.experiment import (
    ExperimentRecord,
)
from research_engineer.models.memory import (
    MemoryRelationship,
    RelationshipType,
)
from research_engineer.tools.evaluation_storage import EvaluationStorageTool
from research_engineer.tools.experiment_comparison import (
    ExperimentComparisonTool,
)
from research_engineer.tools.next_experiment import NextExperimentTool
from research_engineer.tools.statistical_significance import (
    StatisticalSignificanceTool,
)
from research_engineer.tools.training_dynamics import TrainingDynamicsTool


class EvaluationAgent:
    """Agent for evaluating experiment results and drawing conclusions.

    Orchestrates:
    1. Compare experiments (metrics, status, duration, failures)
    2. Analyze training dynamics (over/underfit, convergence, instability)
    3. Test statistical significance (Welch t-test, effect size, CIs)
    4. Recommend next experiments
    5. Recommend relevant papers via LiteratureAgent
    6. Store conclusions in MemoryAgent (insights/success/failure)
    7. Auto-update knowledge graph (evaluation nodes + edges)
    8. Persist evaluation records + generate output files
    """

    def __init__(
        self,
        memory_agent: Any | None = None,
        literature_agent: Any | None = None,
        comparison_tool: ExperimentComparisonTool | None = None,
        dynamics_tool: TrainingDynamicsTool | None = None,
        significance_tool: StatisticalSignificanceTool | None = None,
        next_tool: NextExperimentTool | None = None,
        storage_tool: EvaluationStorageTool | None = None,
        config: EvaluationConfig | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "EvaluationAgent"
        self.config = config or EvaluationConfig()
        self.memory = memory_agent
        self.literature = literature_agent
        self.comparison = comparison_tool or ExperimentComparisonTool()
        self.dynamics = dynamics_tool or TrainingDynamicsTool()
        self.significance = significance_tool or StatisticalSignificanceTool()
        self.next_tool = next_tool or NextExperimentTool()
        self.storage = storage_tool or EvaluationStorageTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def analyze(
        self,
        experiments: list[ExperimentRecord],
        paper_id: str | None = None,
        repo_path: str | None = None,
        primary_metric: str | None = None,
        higher_is_better: bool = False,
        output_dir: str | None = None,
    ) -> EvaluationResult:
        """Full evaluation workflow.

        Args:
            experiments: Experiments to evaluate (1 for single-run,
                2+ for comparison + significance)
            paper_id: Optional associated paper ID
            repo_path: Optional repository path
            primary_metric: Metric to determine the winner
            higher_is_better: True for accuracy, False for loss
            output_dir: Override config output_dir

        Returns:
            EvaluationResult with all findings
        """
        start = time.time()
        evaluation_id = f"eval_{uuid4().hex[:12]}"
        out_dir = output_dir or self.config.output_dir

        result = EvaluationResult(
            evaluation_id=evaluation_id,
            experiment_ids=[e.experiment_id for e in experiments],
            paper_id=paper_id,
            repo_path=repo_path,
        )

        # Step 1: Compare (if >=2 experiments)
        if len(experiments) >= 2:
            comparison_output = await self.compare(
                experiments,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
            )
            result.comparison = comparison_output

        # Step 2: Dynamics per experiment
        dynamics_outputs: list[TrainingDynamicsOutput] = []
        for exp in experiments:
            dyn = await self.dynamics.execute(
                TrainingDynamicsInput(
                    experiment=exp,
                    overfit_threshold=self.config.overfit_threshold,
                    convergence_window=self.config.convergence_window,
                    convergence_tolerance=self.config.convergence_tolerance,
                    instability_threshold=self.config.instability_threshold,
                )
            )
            dynamics_outputs.append(dyn)
        result.dynamics = dynamics_outputs

        # Step 3: Significance (if >=2 experiments and a metric)
        if len(experiments) >= 2 and primary_metric:
            sig_output = await self.significance_test(
                experiments,
                metric=primary_metric,
                higher_is_better=higher_is_better,
            )
            result.significance = sig_output

        # Step 4: Next-experiment recommendations
        next_output = await self.next_tool.execute(
            NextExperimentInput(
                experiments=experiments,
                dynamics=dynamics_outputs,
                significance=result.significance,
                comparison=result.comparison,
                paper_id=paper_id,
                repo_path=repo_path,
                max_recommendations=self.config.max_next_recommendations,
                recommend_papers=self.config.recommend_papers,
            )
        )

        # Step 4b: Paper recommendations via LiteratureAgent
        if (
            self.config.recommend_papers
            and self.literature
            and next_output.paper_query
        ):
            papers = await self._search_papers(next_output.paper_query)
            next_output = next_output.model_copy(
                update={"paper_suggestions": papers}
            )
        result.next_experiments = next_output

        # Step 5: Build and store evaluation record
        record = self._build_record(
            evaluation_id=evaluation_id,
            experiments=experiments,
            paper_id=paper_id,
            repo_path=repo_path,
            result=result,
        )
        store_output = await self.storage.execute(
            EvaluationStorageInput(evaluation=record)
        )
        if isinstance(store_output, EvaluationStorageOutput) and store_output.success:
            result.record = record

        # Step 6: Store conclusions in memory + update graph
        if self.config.store_conclusions and self.memory:
            memory_ids = await self._store_in_memory(result, record)
            result.memory_ids = memory_ids
            record.memory_ids = memory_ids
            if memory_ids and self.config.update_graph:
                await self._update_graph(result)

        # Step 7: Generate output files
        generated = await self._write_output_files(result, out_dir)
        result.generated_files = generated
        result.output_dir = out_dir

        result.processing_time_seconds = round(time.time() - start, 2)
        return result

    async def evaluate_single(
        self,
        experiment: ExperimentRecord,
        paper_id: str | None = None,
        repo_path: str | None = None,
        output_dir: str | None = None,
    ) -> EvaluationResult:
        """Analyze a single experiment (dynamics only)."""
        return await self.analyze(
            experiments=[experiment],
            paper_id=paper_id,
            repo_path=repo_path,
            output_dir=output_dir,
        )

    async def compare(
        self,
        experiments: list[ExperimentRecord],
        primary_metric: str | None = None,
        higher_is_better: bool = False,
        baseline_experiment_id: str | None = None,
    ) -> ExperimentComparisonOutput:
        """Compare experiments."""
        return await self.comparison.execute(
            ExperimentComparisonInput(
                experiments=experiments,
                primary_metric=primary_metric,
                higher_is_better=higher_is_better,
                baseline_experiment_id=baseline_experiment_id,
            )
        )

    async def dynamics_analysis(
        self, experiment: ExperimentRecord
    ) -> TrainingDynamicsOutput:
        """Analyze training dynamics for one experiment."""
        return await self.dynamics.execute(
            TrainingDynamicsInput(
                experiment=experiment,
                overfit_threshold=self.config.overfit_threshold,
                convergence_window=self.config.convergence_window,
                convergence_tolerance=self.config.convergence_tolerance,
                instability_threshold=self.config.instability_threshold,
            )
        )

    async def significance_test(
        self,
        experiments: list[ExperimentRecord],
        metric: str,
        higher_is_better: bool = False,
    ) -> StatisticalSignificanceOutput:
        """Test statistical significance across experiments."""
        return await self.significance.execute(
            StatisticalSignificanceInput(
                experiments=experiments,
                metric=metric,
                alpha=self.config.significance_alpha,
                min_samples=self.config.min_samples_for_stats,
                higher_is_better=higher_is_better,
            )
        )

    async def next_experiments(
        self,
        experiments: list[ExperimentRecord],
        dynamics: list[TrainingDynamicsOutput] | None = None,
        significance: StatisticalSignificanceOutput | None = None,
        comparison: ExperimentComparisonOutput | None = None,
        paper_id: str | None = None,
        repo_path: str | None = None,
    ) -> NextExperimentOutput:
        """Recommend next experiments."""
        out = await self.next_tool.execute(
            NextExperimentInput(
                experiments=experiments,
                dynamics=dynamics or [],
                significance=significance,
                comparison=comparison,
                paper_id=paper_id,
                repo_path=repo_path,
                max_recommendations=self.config.max_next_recommendations,
                recommend_papers=self.config.recommend_papers,
            )
        )
        if (
            self.config.recommend_papers
            and self.literature
            and out.paper_query
        ):
            papers = await self._search_papers(out.paper_query)
            out = out.model_copy(update={"paper_suggestions": papers})
        return out

    async def list_evaluations(
        self,
        paper_id: str | None = None,
        repo_path: str | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> EvaluationQueryOutput:
        """List evaluations with optional filters."""
        out = await self.storage.execute(
            EvaluationQueryInput(
                paper_id=paper_id,
                repo_path=repo_path,
                experiment_id=experiment_id,
                limit=limit,
                offset=offset,
            )
        )
        if isinstance(out, EvaluationQueryOutput):
            return out
        return EvaluationQueryOutput()

    async def get_evaluation(
        self, evaluation_id: str
    ) -> EvaluationRecord | None:
        """Retrieve a single evaluation by ID."""
        return await self.storage.get_by_id(evaluation_id)

    async def search_evaluations(
        self, search_text: str, limit: int = 50
    ) -> EvaluationQueryOutput:
        """Search evaluation history by text."""
        out = await self.storage.execute(
            EvaluationQueryInput(search_text=search_text, limit=limit)
        )
        if isinstance(out, EvaluationQueryOutput):
            return out
        return EvaluationQueryOutput()

    # --- LiteratureAgent Integration ---

    async def _search_papers(
        self, query: str, max_results: int = 5
    ) -> list[PaperSuggestion]:
        """Search for relevant papers via LiteratureAgent."""
        if not self.literature:
            return []
        try:
            from research_engineer.models.literature import SearchSource

            output = await self.literature.search_papers(
                query,
                max_results=max_results,
                sources=[SearchSource.LOCAL, SearchSource.ARXIV],
            )
            suggestions: list[PaperSuggestion] = []
            for r in output.papers[:max_results]:
                suggestions.append(
                    PaperSuggestion(
                        paper_id=r.paper_id,
                        title=r.title,
                        reason=f"Matched query: '{query}'",
                        relevance=min(
                            1.0,
                            max(0.0, 0.5 + 0.1 * (r.citation_count or 0)),
                        ),
                    )
                )
            return suggestions
        except Exception:
            return []

    # --- Memory & Graph Integration ---

    async def _store_in_memory(
        self, result: EvaluationResult, record: EvaluationRecord
    ) -> list[str]:
        """Store evaluation conclusions in MemoryAgent."""
        memory_ids: list[str] = []
        if not self.memory:
            return memory_ids

        try:
            memory_ids.extend(await self._store_comparison_insight(result))
            memory_ids.extend(await self._store_dynamics_insights(result))
            memory_ids.extend(await self._store_significance_insight(result))
            memory_ids.extend(
                await self._store_overall_conclusion(result, record)
            )
        except Exception:
            pass

        return memory_ids

    async def _store_comparison_insight(
        self, result: EvaluationResult
    ) -> list[str]:
        """Store the comparison findings as an insight."""
        if not result.comparison:
            return []
        mem_id = await self._safe_store_insight(
            "empirical_finding",
            "evaluation:comparison",
            (
                f"Comparison of {result.comparison.experiments_compared} "
                f"experiments. Winner: "
                f"{result.comparison.winner_experiment_id or 'none'}. "
                f"{result.comparison.recommendation}"
            ),
            evidence=result.comparison.findings[:10],
            applicability=["evaluation_comparison"],
        )
        return [mem_id] if mem_id else []

    async def _store_dynamics_insights(
        self, result: EvaluationResult
    ) -> list[str]:
        """Store dynamics patterns as insights (and failures for over/underfit)."""
        ids: list[str] = []
        for dyn in result.dynamics:
            primary = dyn.primary_pattern()
            if primary is None:
                continue
            mem_id = await self._safe_store_insight(
                "empirical_finding",
                f"evaluation:dynamics:{primary.value}",
                dyn.summary,
                evidence=[
                    p.evidence[0] if p.evidence else ""
                    for p in dyn.patterns
                    if p.detected
                ],
                applicability=["evaluation_dynamics"],
            )
            if mem_id:
                ids.append(mem_id)
            if primary.value in ("overfitting", "underfitting"):
                fid = await self._safe_store_failure(
                    context=f"Dynamics analysis for {dyn.experiment_id}",
                    approach=f"primary pattern: {primary.value}",
                    failure_mode=primary.value,
                    lessons=[dyn.summary],
                    error_details=None,
                )
                if fid:
                    ids.append(fid)
        return ids

    async def _store_significance_insight(
        self, result: EvaluationResult
    ) -> list[str]:
        """Store significance findings as an insight."""
        if not result.significance or not result.significance.results:
            return []
        sig_lines: list[str] = []
        for r in result.significance.results:
            if r.significant:
                sig_lines.append(
                    f"{r.comparison}: p={r.p_value:.4f} "
                    f"({r.effect_size_label})"
                )
        if not sig_lines:
            return []
        mem_id = await self._safe_store_insight(
            "empirical_finding",
            "evaluation:significance",
            result.significance.overall_verdict,
            evidence=sig_lines,
            applicability=["evaluation_significance"],
        )
        return [mem_id] if mem_id else []

    async def _store_overall_conclusion(
        self, result: EvaluationResult, record: EvaluationRecord
    ) -> list[str]:
        """Store the overall conclusion (success or failure memory)."""
        context = (
            f"Evaluation {result.evaluation_id} for "
            f"{result.paper_id or 'experiments'}"
        )
        if result.is_positive():
            success_id = await self._safe_store_success(
                context=context,
                approach="multi-experiment evaluation",
                metrics=self._collect_metrics(record),
                key_factors=self._collect_key_factors(result),
            )
            return [success_id] if success_id else []
        lessons = self._collect_lessons(result)
        fail_id = await self._safe_store_failure(
            context=context,
            approach="multi-experiment evaluation",
            failure_mode="poor_performance",
            lessons=lessons,
            error_details=None,
        )
        return [fail_id] if fail_id else []

    async def _safe_store_insight(
        self,
        insight_type: str,
        domain: str,
        description: str,
        evidence: list[str] | None = None,
        applicability: list[str] | None = None,
    ) -> str | None:
        if not self.memory:
            return None
        try:
            mem_id: str = await self.memory.store_insight(
                insight_type=insight_type,
                domain=domain,
                description=description,
                evidence=evidence,
                applicability=applicability,
            )
            return str(mem_id)
        except Exception:
            return None

    async def _safe_store_success(
        self,
        context: str,
        approach: str,
        metrics: dict[str, float],
        key_factors: list[str],
    ) -> str | None:
        if not self.memory:
            return None
        try:
            mem_id: str = await self.memory.store_success(
                context=context,
                approach=approach,
                metrics=metrics,
                key_factors=key_factors,
            )
            return str(mem_id)
        except Exception:
            return None

    async def _safe_store_failure(
        self,
        context: str,
        approach: str,
        failure_mode: str,
        lessons: list[str],
        error_details: str | None,
    ) -> str | None:
        if not self.memory:
            return None
        try:
            mem_id: str = await self.memory.store_failure(
                context=context,
                approach=approach,
                failure_mode=failure_mode,
                lessons=lessons,
                error_details=error_details,
            )
            return str(mem_id)
        except Exception:
            return None

    async def _update_graph(self, result: EvaluationResult) -> None:
        """Update the knowledge graph with evaluation relationships."""
        if not self.memory or not self.memory.graph:
            return
        graph = self.memory.graph
        try:
            graph.add_node(result.evaluation_id, "evaluation")

            # Link evaluation -> experiments (DERIVED_FROM)
            for exp_id in result.experiment_ids:
                graph.add_node(exp_id, "experiment")
                rel = MemoryRelationship(
                    source_memory_id=result.evaluation_id,
                    target_memory_id=exp_id,
                    relationship_type=RelationshipType.DERIVED_FROM,
                    confidence=0.95,
                    validated=False,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # Link conclusions (CONFIRMS / CONTRADICTS)
            positive = result.is_positive()
            rel_type = (
                RelationshipType.CONFIRMS
                if positive
                else RelationshipType.CONTRADICTS
            )
            for mem_id in result.memory_ids:
                graph.add_node(mem_id, "conclusion")
                rel = MemoryRelationship(
                    source_memory_id=result.evaluation_id,
                    target_memory_id=mem_id,
                    relationship_type=rel_type,
                    confidence=0.9,
                    validated=False,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # Link paper -> evaluation (IMPLEMENTS)
            if result.paper_id:
                graph.add_node(result.paper_id, "paper")
                rel = MemoryRelationship(
                    source_memory_id=result.paper_id,
                    target_memory_id=result.evaluation_id,
                    relationship_type=RelationshipType.IMPLEMENTS,
                    confidence=0.9,
                    validated=False,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # Link recommended papers -> evaluation (INSPIRED_BY)
            if result.next_experiments:
                for ps in result.next_experiments.paper_suggestions:
                    graph.add_node(ps.paper_id, "paper")
                    rel = MemoryRelationship(
                        source_memory_id=ps.paper_id,
                        target_memory_id=result.evaluation_id,
                        relationship_type=RelationshipType.INSPIRED_BY,
                        confidence=0.7,
                        validated=False,
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

    # --- Helpers ---

    @staticmethod
    def _build_record(
        evaluation_id: str,
        experiments: list[ExperimentRecord],
        paper_id: str | None,
        repo_path: str | None,
        result: EvaluationResult,
    ) -> EvaluationRecord:
        conclusions = EvaluationAgent._collect_conclusions(result)
        summary = EvaluationAgent._build_summary(result)
        tags: list[str] = ["evaluation"]
        if paper_id:
            tags.append(paper_id)
        if repo_path:
            tags.append(Path(repo_path).name)
        return EvaluationRecord(
            evaluation_id=evaluation_id,
            experiment_ids=[e.experiment_id for e in experiments],
            paper_id=paper_id,
            repo_path=repo_path,
            comparison=result.comparison,
            dynamics=result.dynamics,
            significance=result.significance,
            next_experiments=result.next_experiments,
            summary=summary,
            conclusions=conclusions,
            memory_ids=result.memory_ids,
            tags=tags,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @staticmethod
    def _collect_conclusions(result: EvaluationResult) -> list[str]:
        conclusions: list[str] = []
        if result.comparison:
            if result.comparison.winner_experiment_id:
                conclusions.append(
                    f"Winner: {result.comparison.winner_experiment_id}"
                )
            conclusions.extend(result.comparison.findings[:5])
        for dyn in result.dynamics:
            p = dyn.primary_pattern()
            if p:
                conclusions.append(
                    f"{dyn.experiment_id}: {p.value}"
                )
        if result.significance:
            sig = [
                r for r in result.significance.results if r.significant
            ]
            if sig:
                conclusions.append(
                    f"Significant differences: {len(sig)}"
                )
            else:
                conclusions.append(
                    "No statistically significant differences"
                )
        if result.next_experiments and result.next_experiments.experiment_recommendations:
            top = result.next_experiments.experiment_recommendations[0]
            conclusions.append(
                f"Next: {top.title} ({top.priority.value})"
            )
        return conclusions

    @staticmethod
    def _build_summary(result: EvaluationResult) -> str:
        parts: list[str] = []
        parts.append(
            f"Evaluated {len(result.experiment_ids)} experiment(s)."
        )
        if result.comparison and result.comparison.winner_experiment_id:
            parts.append(
                f"Winner: {result.comparison.winner_experiment_id}."
            )
        detected = [
            p
            for d in result.dynamics
            for p in [d.primary_pattern()]
            if p is not None
        ]
        if detected:
            parts.append(
                f"Dynamics: {', '.join(p.value for p in detected)}."
            )
        if result.significance:
            parts.append(result.significance.overall_verdict)
        if result.next_experiments:
            parts.append(
                f"Strategy: {result.next_experiments.overall_strategy}"
            )
        return " ".join(parts)

    @staticmethod
    def _collect_metrics(record: EvaluationRecord) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if record.comparison:
            for delta in record.comparison.metric_deltas:
                if delta.best_experiment_id and delta.best_value is not None:
                    metrics[delta.metric] = float(delta.best_value)
        return metrics

    @staticmethod
    def _collect_key_factors(
        result: EvaluationResult
    ) -> list[str]:
        factors: list[str] = []
        if result.comparison and result.comparison.winner_experiment_id:
            factors.append(
                f"winner={result.comparison.winner_experiment_id}"
            )
        if result.significance and result.significance.best_experiment_id:
            factors.append(
                f"stats_best={result.significance.best_experiment_id}"
            )
        for dyn in result.dynamics:
            p = dyn.primary_pattern()
            if p:
                factors.append(f"{dyn.experiment_id}:{p.value}")
        return factors[:10]

    @staticmethod
    def _collect_lessons(result: EvaluationResult) -> list[str]:
        lessons: list[str] = []
        for dyn in result.dynamics:
            for p in dyn.patterns:
                if p.detected and p.recommendation:
                    lessons.append(
                        f"{dyn.experiment_id} ({p.pattern_type.value}): "
                        f"{p.recommendation}"
                    )
        if result.next_experiments:
            for rec in result.next_experiments.experiment_recommendations:
                lessons.append(f"{rec.title}: {rec.rationale}")
        return lessons[:15]

    async def _write_output_files(
        self, result: EvaluationResult, output_dir: str
    ) -> list[str]:
        """Write output files for the evaluation result."""
        try:
            out_path = Path(output_dir) / result.evaluation_id
            out_path.mkdir(parents=True, exist_ok=True)
            generated: list[str] = []

            import json as _json

            # Full result JSON
            f = out_path / "evaluation_result.json"
            f.write_text(
                _json.dumps(
                    result.model_dump(), indent=2, default=str
                ),
                encoding="utf-8",
            )
            generated.append(str(f))

            # Summary markdown
            f = out_path / "evaluation_summary.md"
            f.write_text(
                self._format_summary_md(result), encoding="utf-8"
            )
            generated.append(str(f))

            # Comparison markdown
            if result.comparison:
                f = out_path / "comparison.md"
                f.write_text(
                    self._format_comparison_md(result.comparison),
                    encoding="utf-8",
                )
                generated.append(str(f))

            # Dynamics markdown
            if result.dynamics:
                f = out_path / "dynamics.md"
                f.write_text(
                    self._format_dynamics_md(result.dynamics),
                    encoding="utf-8",
                )
                generated.append(str(f))

            # Significance markdown
            if result.significance:
                f = out_path / "significance.md"
                f.write_text(
                    self._format_significance_md(result.significance),
                    encoding="utf-8",
                )
                generated.append(str(f))

            # Next experiments markdown
            if result.next_experiments:
                f = out_path / "next_experiments.md"
                f.write_text(
                    self._format_next_md(result.next_experiments),
                    encoding="utf-8",
                )
                generated.append(str(f))

            return generated
        except Exception:
            return []

    @staticmethod
    def _format_summary_md(result: EvaluationResult) -> str:
        lines = [
            f"# Evaluation: {result.evaluation_id}",
            "",
            f"**Paper**: {result.paper_id or 'N/A'}",
            f"**Repo**: {result.repo_path or 'N/A'}",
            f"**Experiments**: {', '.join(result.experiment_ids)}",
            f"**Timestamp**: {result.timestamp}",
            f"**Processing Time**: {result.processing_time_seconds}s",
            "",
        ]
        if result.record:
            lines.extend(["## Summary", "", result.record.summary, ""])
            if result.record.conclusions:
                lines.append("## Conclusions")
                lines.append("")
                for c in result.record.conclusions:
                    lines.append(f"- {c}")
                lines.append("")
        if result.memory_ids:
            lines.extend(
                ["## Memory", "", f"Memory IDs: {', '.join(result.memory_ids)}", ""]
            )
        return "\n".join(lines)

    @staticmethod
    def _format_comparison_md(
        comp: ExperimentComparisonOutput,
    ) -> str:
        lines = ["# Experiment Comparison", ""]
        lines.append(f"Experiments compared: {comp.experiments_compared}")
        lines.append(f"Winner: {comp.winner_experiment_id or 'N/A'}")
        lines.append("")
        if comp.metric_deltas:
            lines.append("## Metric Deltas")
            lines.append("")
            lines.append("| Metric | Best | Best Value | Deltas |")
            lines.append("|--------|------|------------|--------|")
            for d in comp.metric_deltas:
                deltas = ", ".join(
                    f"{k}={v:.4f}" for k, v in d.deltas.items()
                )
                lines.append(
                    f"| {d.metric} | {d.best_experiment_id} | "
                    f"{d.best_value:.4f} | {deltas} |"
                )
            lines.append("")
        if comp.findings:
            lines.append("## Findings")
            lines.append("")
            for fnd in comp.findings:
                lines.append(f"- {fnd}")
            lines.append("")
        lines.append("## Recommendation")
        lines.append("")
        lines.append(comp.recommendation)
        return "\n".join(lines)

    @staticmethod
    def _format_dynamics_md(
        dynamics: list[TrainingDynamicsOutput],
    ) -> str:
        lines = ["# Training Dynamics", ""]
        for dyn in dynamics:
            lines.append(f"## {dyn.experiment_id}")
            lines.append("")
            lines.append(f"- Summary: {dyn.summary}")
            lines.append(
                f"- Stability score: {dyn.stability_score:.2f}"
            )
            if dyn.final_train_metric is not None:
                lines.append(
                    f"- Final train: {dyn.final_train_metric:.4f}"
                )
            if dyn.final_eval_metric is not None:
                lines.append(
                    f"- Final eval: {dyn.final_eval_metric:.4f}"
                )
            if dyn.train_eval_gap is not None:
                lines.append(
                    f"- Train-eval gap: {dyn.train_eval_gap:.4f}"
                )
            lines.append("")
            lines.append("### Patterns")
            lines.append("")
            for p in dyn.patterns:
                status = "DETECTED" if p.detected else "not detected"
                lines.append(
                    f"- **{p.pattern_type.value}** ({status}, "
                    f"conf={p.confidence:.2f}): {p.recommendation}"
                )
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_significance_md(
        sig: StatisticalSignificanceOutput,
    ) -> str:
        lines = ["# Statistical Significance", ""]
        lines.append(f"Verdict: {sig.overall_verdict}")
        lines.append(f"Best: {sig.best_experiment_id or 'N/A'}")
        lines.append(f"Pairs compared: {sig.pairwise_count}")
        lines.append("")
        if sig.results:
            lines.append("## Pairwise Results")
            lines.append("")
            lines.append(
                "| Comparison | mean_a | mean_b | t | p | sig | effect |"
            )
            lines.append("|------------|--------|--------|---|---|-----|--------|")
            for r in sig.results:
                lines.append(
                    f"| {r.comparison} | {r.mean_a:.3f} | {r.mean_b:.3f} "
                    f"| {r.t_statistic:.3f} | {r.p_value:.4f} "
                    f"| {'yes' if r.significant else 'no'} "
                    f"| {r.effect_size_label} |"
                )
            lines.append("")
        if sig.insufficient_data_pairs:
            lines.append("## Insufficient Data")
            lines.append("")
            for pair in sig.insufficient_data_pairs:
                lines.append(f"- {pair}")
        return "\n".join(lines)

    @staticmethod
    def _format_next_md(next_out: NextExperimentOutput) -> str:
        lines = ["# Next Experiments", ""]
        lines.append(f"Strategy: {next_out.overall_strategy}")
        lines.append("")
        if next_out.experiment_recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in next_out.experiment_recommendations:
                lines.append(
                    f"### {rec.rank}. {rec.title} "
                    f"({rec.priority.value})"
                )
                lines.append(f"- Rationale: {rec.rationale}")
                lines.append(f"- Type: {rec.suggested_type.value}")
                lines.append(f"- Effort: {rec.estimated_effort}")
                if rec.suggested_changes:
                    lines.append("- Changes:")
                    for c in rec.suggested_changes:
                        lines.append(f"  - {c}")
                lines.append("")
        if next_out.paper_suggestions:
            lines.append("## Paper Suggestions")
            lines.append("")
            for ps in next_out.paper_suggestions:
                lines.append(
                    f"- **{ps.paper_id}**: {ps.title} "
                    f"(relevance: {ps.relevance:.2f})"
                )
                lines.append(f"  - {ps.reason}")
            lines.append("")
        if next_out.open_questions:
            lines.append("## Open Questions")
            lines.append("")
            for q in next_out.open_questions:
                lines.append(f"- {q}")
        return "\n".join(lines)
