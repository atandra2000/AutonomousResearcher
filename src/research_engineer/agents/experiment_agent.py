"""Experiment Agent for Phase 7 - Experiment Execution.

Orchestrates experiment launching, monitoring, metric collection, artifact
collection, failure detection, storage, and memory/graph integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_engineer.llm import LLMProvider
from research_engineer.models.experiment import (
    ArtifactCollectorInput,
    ExperimentConfig,
    ExperimentQueryInput,
    ExperimentQueryOutput,
    ExperimentRecord,
    ExperimentResult,
    ExperimentRunnerInput,
    ExperimentStatus,
    ExperimentType,
    FailureDetectorInput,
    MetricCollectorInput,
    MonitoringInput,
)
from research_engineer.models.memory import (
    MemoryRelationship,
    RelationshipType,
)
from research_engineer.tools.artifact_collector import ArtifactCollectorTool
from research_engineer.tools.experiment_runner import ExperimentRunnerTool
from research_engineer.tools.experiment_storage import ExperimentStorageTool
from research_engineer.tools.failure_detector import FailureDetectorTool
from research_engineer.tools.metric_collector import MetricCollectorTool
from research_engineer.tools.monitoring import MonitoringTool


class ExperimentAgent:
    """Agent for executing, monitoring, and recording ML experiments.

    Orchestrates:
    1. Launch experiments safely (dry-run default, command allowlist)
    2. Monitor running experiments (logs, metrics, checkpoints)
    3. Collect metrics from logs, JSON, CSV
    4. Collect artifacts (checkpoints, logs, plots, configs)
    5. Detect failures and classify them
    6. Store experiment results in SQLite
    7. Store findings in MemoryAgent (success/failure)
    8. Auto-update knowledge graph (paper→experiment→result)
    """

    def __init__(
        self,
        memory_agent: Any | None = None,
        runner_tool: ExperimentRunnerTool | None = None,
        monitoring_tool: MonitoringTool | None = None,
        metric_tool: MetricCollectorTool | None = None,
        artifact_tool: ArtifactCollectorTool | None = None,
        failure_tool: FailureDetectorTool | None = None,
        storage_tool: ExperimentStorageTool | None = None,
        config: ExperimentConfig | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "ExperimentAgent"
        self.config = config or ExperimentConfig()
        self.memory = memory_agent
        self.runner = runner_tool or ExperimentRunnerTool()
        self.monitoring = monitoring_tool or MonitoringTool()
        self.metrics = metric_tool or MetricCollectorTool()
        self.artifacts = artifact_tool or ArtifactCollectorTool()
        self.failure = failure_tool or FailureDetectorTool()
        self.storage = storage_tool or ExperimentStorageTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def run(
        self,
        command: str | list[str],
        repo_path: str,
        paper_id: str | None = None,
        plan_id: str | None = None,
        patch_id: str | None = None,
        implementation_id: str | None = None,
        experiment_type: ExperimentType = ExperimentType.TRAINING,
        timeout_seconds: int | None = None,
        env_vars: dict[str, str] | None = None,
        dry_run: bool | None = None,
        output_dir: str | None = None,
    ) -> ExperimentResult:
        """Full experiment execution workflow.

        Args:
            command: Command string or list to execute
            repo_path: Repository working directory
            paper_id: Optional associated paper ID
            plan_id: Optional associated plan ID
            patch_id: Optional associated patch ID
            implementation_id: Optional associated implementation ID
            experiment_type: Training, evaluation, or validation
            timeout_seconds: Max runtime (uses config default if None)
            env_vars: Additional environment variables
            dry_run: Override config dry_run_default
            output_dir: Override config output_dir

        Returns:
            ExperimentResult with all collected data
        """
        start = time.time()
        experiment_id = f"exp_{uuid4().hex[:12]}"
        cmd_list = self._parse_command(command)
        out_dir = output_dir or self.config.output_dir
        dry = dry_run if dry_run is not None else self.config.dry_run_default
        timeout = timeout_seconds or self.config.default_timeout_seconds

        result = ExperimentResult(
            experiment_id=experiment_id,
            paper_id=paper_id,
            plan_id=plan_id,
            patch_id=patch_id,
            implementation_id=implementation_id,
            repo_path=repo_path,
        )

        # Step 1: Launch the experiment
        runner_input = ExperimentRunnerInput(
            command=cmd_list,
            working_dir=repo_path,
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            timeout_seconds=timeout,
            env_vars=env_vars or {},
            dry_run=dry,
        )
        runner_output = await self.runner.execute(runner_input)
        result.run = runner_output.run

        # For dry runs, return early
        if not runner_output.launched:
            result.processing_time_seconds = round(time.time() - start, 2)
            return result

        run = runner_output.run

        # Step 2: Monitor (analyze captured output)
        monitoring_output = await self.monitoring.execute(
            MonitoringInput(
                experiment_id=experiment_id,
                stdout=run.stdout,
                stderr=run.stderr,
                working_dir=repo_path,
                log_tail_lines=self.config.default_log_tail_lines,
                status=run.status,
            )
        )
        result.monitoring = monitoring_output

        # Step 3: Collect metrics
        metric_output = await self.metrics.execute(
            MetricCollectorInput(
                experiment_id=experiment_id,
                stdout=run.stdout,
                stderr=run.stderr,
                output_dir=repo_path,
            )
        )
        result.metrics = metric_output

        # Step 4: Collect artifacts
        artifact_output = await self.artifacts.execute(
            ArtifactCollectorInput(
                experiment_id=experiment_id,
                output_dir=out_dir,
                working_dir=repo_path,
                max_artifact_size_mb=self.config.max_artifact_size_mb,
                copy_artifacts=self.config.store_results,
            )
        )
        result.artifacts = artifact_output

        # Step 5: Detect failures
        failure_output = await self.failure.execute(
            FailureDetectorInput(
                run=run,
                metrics=metric_output.metrics,
                artifacts=artifact_output.artifacts,
            )
        )
        result.failure = failure_output

        # Step 6: Build and store experiment record
        record = self._build_record(
            experiment_id=experiment_id,
            run=run,
            paper_id=paper_id,
            plan_id=plan_id,
            patch_id=patch_id,
            implementation_id=implementation_id,
            repo_path=repo_path,
            metrics=metric_output,
            artifacts=artifact_output,
            failure=failure_output,
            out_dir=out_dir,
        )
        store_output = await self.storage.execute(
            self._make_store_input(record)
        )
        if store_output.success:
            result.record = record

        # Step 7: Store in memory + update graph
        if self.config.store_results and self.memory:
            memory_ids = await self._store_in_memory(result, record)
            result.memory_ids = memory_ids
            record.memory_id = memory_ids[0] if memory_ids else None
            if memory_ids and self.config.update_graph:
                await self._update_graph(result, record)

        # Step 8: Generate output files
        generated = await self._write_output_files(result, out_dir)
        result.generated_files = generated
        result.output_dir = out_dir

        result.processing_time_seconds = round(time.time() - start, 2)
        return result

    async def run_training(
        self,
        command: str | list[str],
        repo_path: str,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Convenience wrapper for training experiments."""
        return await self.run(
            command=command,
            repo_path=repo_path,
            experiment_type=ExperimentType.TRAINING,
            **kwargs,
        )

    async def run_evaluation(
        self,
        command: str | list[str],
        repo_path: str,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Convenience wrapper for evaluation experiments."""
        return await self.run(
            command=command,
            repo_path=repo_path,
            experiment_type=ExperimentType.EVALUATION,
            **kwargs,
        )

    async def run_validation(
        self,
        command: str | list[str],
        repo_path: str,
        **kwargs: Any,
    ) -> ExperimentResult:
        """Convenience wrapper for validation experiments."""
        return await self.run(
            command=command,
            repo_path=repo_path,
            experiment_type=ExperimentType.VALIDATION,
            **kwargs,
        )

    async def list_experiments(
        self,
        paper_id: str | None = None,
        repo_path: str | None = None,
        status: ExperimentStatus | None = None,
        experiment_type: ExperimentType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ExperimentQueryOutput:
        """List experiments with optional filters."""
        return await self.storage.execute(
            ExperimentQueryInput(
                paper_id=paper_id,
                repo_path=repo_path,
                status=status,
                experiment_type=experiment_type,
                limit=limit,
                offset=offset,
            )
        )

    async def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve a single experiment by ID."""
        return await self.storage.get_by_id(experiment_id)

    async def search_experiments(
        self, search_text: str, limit: int = 50
    ) -> ExperimentQueryOutput:
        """Search experiment history by text."""
        return await self.storage.execute(
            ExperimentQueryInput(search_text=search_text, limit=limit)
        )

    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        return await self.runner.cancel(experiment_id)

    async def history(
        self, paper_id: str, limit: int = 20
    ) -> ExperimentQueryOutput:
        """Show experiment history for a paper."""
        return await self.storage.execute(
            ExperimentQueryInput(paper_id=paper_id, limit=limit)
        )

    # --- Memory & Graph Integration ---

    async def _store_in_memory(
        self, result: ExperimentResult, record: ExperimentRecord
    ) -> list[str]:
        """Store experiment results in MemoryAgent."""
        memory_ids: list[str] = []
        if not self.memory:
            return memory_ids

        try:
            if result.is_success():
                mem_id = await self.memory.store_success(
                    context=(
                        f"Experiment {record.experiment_id} "
                        f"({record.experiment_type.value}) on {record.repo_path}"
                    ),
                    approach=" ".join(record.command),
                    metrics=record.metrics,
                    key_factors=self._extract_key_factors(result),
                )
                memory_ids.append(mem_id)
            else:
                failure = result.failure
                failure_mode = (
                    failure.failure_mode if failure and failure.failure_mode else "poor_performance"
                )
                mem_id = await self.memory.store_failure(
                    context=(
                        f"Experiment {record.experiment_id} "
                        f"({record.experiment_type.value}) on {record.repo_path}"
                    ),
                    approach=" ".join(record.command),
                    failure_mode=failure_mode,
                    lessons=failure.lessons_learned if failure else [],
                    error_details=record.root_cause,
                )
                memory_ids.append(mem_id)

            # Store an insight for notable metrics
            if record.metrics:
                insight_id = await self.memory.store_insight(
                    insight_type="empirical_finding",
                    domain=f"experiment:{record.experiment_type.value}",
                    description=(
                        f"Experiment {record.experiment_id} metrics: "
                        f"{record.metrics}"
                    ),
                    evidence=list(record.metrics.keys()),
                    applicability=["experiment_result"],
                )
                memory_ids.append(insight_id)
        except Exception:
            pass

        return memory_ids

    async def _update_graph(
        self, result: ExperimentResult, record: ExperimentRecord
    ) -> None:
        """Update the knowledge graph with experiment relationships."""
        if not self.memory or not self.memory.graph:
            return
        graph = self.memory.graph
        try:
            graph.add_node(result.experiment_id, "experiment")

            # Link experiment to result memories
            for mem_id in result.memory_ids:
                graph.add_node(mem_id, "result")
                rel_type = (
                    RelationshipType.VALIDATES
                    if result.is_success()
                    else RelationshipType.FAILED_WITH
                )
                rel = MemoryRelationship(
                    source_memory_id=result.experiment_id,
                    target_memory_id=mem_id,
                    relationship_type=rel_type,
                    confidence=0.95,
                )
                await self._safe_store_relationship(rel)
                graph.add_relationship(rel)

            # Link paper/plan/patch to experiment
            for source_id in (result.paper_id, result.plan_id, result.patch_id):
                if source_id:
                    graph.add_node(source_id)
                    rel = MemoryRelationship(
                        source_memory_id=source_id,
                        target_memory_id=result.experiment_id,
                        relationship_type=RelationshipType.IMPLEMENTS,
                        confidence=0.9,
                    )
                    await self._safe_store_relationship(rel)
                    graph.add_relationship(rel)
        except Exception:
            pass

    async def _safe_store_relationship(self, rel: MemoryRelationship) -> None:
        """Store a relationship, ignoring errors."""
        if not self.memory:
            return
        try:
            await self.memory.storage.store_relationship(rel)
        except Exception:
            pass

    # --- Helpers ---

    @staticmethod
    def _parse_command(command: str | list[str]) -> list[str]:
        """Parse a command string or list into a command list."""
        if isinstance(command, list):
            return command
        return command.split()

    def _build_record(
        self,
        experiment_id: str,
        run: Any,
        paper_id: str | None,
        plan_id: str | None,
        patch_id: str | None,
        implementation_id: str | None,
        repo_path: str,
        metrics: Any,
        artifacts: Any,
        failure: Any,
        out_dir: str,
    ) -> ExperimentRecord:
        """Build an ExperimentRecord from collected outputs."""
        summary_metrics = metrics.summary_metrics if metrics else {}
        artifact_list = artifacts.artifacts if artifacts else []
        failure_mode = failure.failure_mode if failure else None
        failure_severity = (
            failure.severity if failure else ExperimentRecord.model_fields[
                "failure_severity"
            ].default
        )
        root_cause = failure.root_cause_hypothesis if failure else None
        lessons = failure.lessons_learned if failure else []

        return ExperimentRecord(
            experiment_id=experiment_id,
            paper_id=paper_id,
            plan_id=plan_id,
            patch_id=patch_id,
            implementation_id=implementation_id,
            repo_path=repo_path,
            command=run.command,
            experiment_type=run.experiment_type,
            status=run.status,
            start_time=run.start_time,
            end_time=run.end_time,
            duration_seconds=run.duration_seconds,
            exit_code=run.exit_code,
            metrics=summary_metrics,
            metric_series=metrics.metric_series if metrics else [],
            artifacts=artifact_list,
            failure_mode=failure_mode,
            failure_severity=failure_severity,
            root_cause=root_cause,
            lessons_learned=lessons,
            output_dir=out_dir,
            tags=self._extract_tags(repo_path, paper_id),
        )

    @staticmethod
    def _make_store_input(record: ExperimentRecord) -> Any:
        """Create a storage input for a record."""
        from research_engineer.models.experiment import ExperimentStorageInput

        return ExperimentStorageInput(experiment=record, operation="store")

    @staticmethod
    def _extract_key_factors(result: ExperimentResult) -> list[str]:
        """Extract key success factors from a result."""
        factors: list[str] = []
        if result.metrics and result.metrics.summary_metrics:
            for name, value in result.metrics.summary_metrics.items():
                factors.append(f"{name}={value}")
        if result.monitoring and result.monitoring.checkpoints_found:
            factors.append(
                f"checkpoints: {len(result.monitoring.checkpoints_found)}"
            )
        return factors[:10]

    @staticmethod
    def _extract_tags(repo_path: str, paper_id: str | None) -> list[str]:
        """Extract tags for an experiment."""
        tags: list[str] = [Path(repo_path).name]
        if paper_id:
            tags.append(paper_id)
        tags.append("experiment")
        return tags

    async def _write_output_files(
        self, result: ExperimentResult, output_dir: str
    ) -> list[str]:
        """Write output files for the experiment result."""
        try:
            out_path = Path(output_dir) / result.experiment_id
            out_path.mkdir(parents=True, exist_ok=True)
            generated: list[str] = []

            # Full result JSON
            f = out_path / "experiment_result.json"
            import json as _json

            f.write_text(
                _json.dumps(result.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
            generated.append(str(f))

            # Summary markdown
            f = out_path / "experiment_summary.md"
            f.write_text(self._format_summary_md(result), encoding="utf-8")
            generated.append(str(f))

            # Metrics JSON
            if result.metrics:
                f = out_path / "metrics.json"
                f.write_text(
                    result.metrics.model_dump_json(indent=2),
                    encoding="utf-8",
                )
                generated.append(str(f))

            # Metrics summary markdown
            if result.metrics:
                f = out_path / "metrics_summary.md"
                f.write_text(
                    self._format_metrics_md(result), encoding="utf-8"
                )
                generated.append(str(f))

            # Run log
            if result.run:
                f = out_path / "run_log.txt"
                log_text = (
                    f"=== STDOUT ===\n{result.run.stdout}\n\n"
                    f"=== STDERR ===\n{result.run.stderr}\n"
                )
                f.write_text(log_text, encoding="utf-8")
                generated.append(str(f))

            # Failure report
            if result.failure and result.failure.detected_failure:
                f = out_path / "failure_report.md"
                f.write_text(
                    self._format_failure_md(result), encoding="utf-8"
                )
                generated.append(str(f))

            return generated
        except Exception:
            return []

    @staticmethod
    def _format_summary_md(result: ExperimentResult) -> str:
        lines = [
            f"# Experiment: {result.experiment_id}",
            "",
            f"**Paper**: {result.paper_id or 'N/A'}",
            f"**Plan**: {result.plan_id or 'N/A'}",
            f"**Patch**: {result.patch_id or 'N/A'}",
            f"**Repository**: {result.repo_path}",
            f"**Timestamp**: {result.timestamp}",
            f"**Processing Time**: {result.processing_time_seconds}s",
            "",
        ]
        if result.run:
            lines.extend([
                "## Run",
                "",
                f"- **Command**: `{' '.join(result.run.command)}`",
                f"- **Type**: {result.run.experiment_type.value}",
                f"- **Status**: {result.run.status.value}",
                f"- **Exit Code**: {result.run.exit_code}",
                f"- **Duration**: {result.run.duration_seconds}s",
                f"- **PID**: {result.run.pid}",
                "",
            ])
        if result.monitoring:
            lines.extend([
                "## Monitoring",
                "",
                f"- **Stdout lines**: {result.monitoring.total_stdout_lines}",
                f"- **Stderr lines**: {result.monitoring.total_stderr_lines}",
                f"- **Metrics detected**: {len(result.monitoring.metrics_detected)}",
                f"- **Checkpoints found**: {len(result.monitoring.checkpoints_found)}",
                f"- **Anomalies**: {len(result.monitoring.anomalies)}",
                "",
            ])
        if result.metrics:
            lines.extend([
                "## Metrics Summary",
                "",
            ])
            for name, value in result.metrics.summary_metrics.items():
                lines.append(f"- **{name}**: {value}")
            lines.append("")
        if result.artifacts:
            lines.extend([
                "## Artifacts",
                "",
                f"- **Count**: {len(result.artifacts.artifacts)}",
                f"- **Total size**: {result.artifacts.total_size_mb} MB",
                "",
            ])
        if result.failure:
            lines.extend([
                "## Failure Analysis",
                "",
                f"- **Detected**: {result.failure.detected_failure}",
                f"- **Mode**: {result.failure.failure_mode or 'N/A'}",
                f"- **Severity**: {result.failure.severity.value}",
                f"- **Root cause**: {result.failure.root_cause_hypothesis}",
                "",
            ])
        if result.memory_ids:
            lines.extend([
                "## Memory & Graph",
                "",
                f"- **Memory IDs**: {', '.join(result.memory_ids)}",
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def _format_metrics_md(result: ExperimentResult) -> str:
        lines = ["# Metrics Summary", ""]
        if result.metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for name, value in result.metrics.summary_metrics.items():
                lines.append(f"| {name} | {value} |")
            lines.append("")
            if result.metrics.metric_series:
                lines.append("## Metric Series")
                lines.append("")
                for series in result.metrics.metric_series:
                    lines.append(f"### {series.name}")
                    lines.append(f"- Best: {series.best_value} (step {series.best_step})")
                    lines.append(f"- Final: {series.final_value}")
                    lines.append(f"- Points: {len(series.values)}")
                    lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_failure_md(result: ExperimentResult) -> str:
        failure = result.failure
        lines = [
            "# Failure Report",
            "",
            f"**Experiment**: {result.experiment_id}",
            f"**Detected**: {failure.detected_failure}",
            f"**Mode**: {failure.failure_mode or 'N/A'}",
            f"**Severity**: {failure.severity.value}",
            "",
            "## Root Cause Hypothesis",
            "",
            failure.root_cause_hypothesis,
            "",
        ]
        if failure.error_snippets:
            lines.append("## Error Snippets")
            lines.append("")
            for snippet in failure.error_snippets:
                lines.append(f"```\n{snippet}\n```")
            lines.append("")
        if failure.anomaly_indicators:
            lines.append("## Anomalies")
            lines.append("")
            for anomaly in failure.anomaly_indicators:
                lines.append(
                    f"- **{anomaly.indicator}** (conf: {anomaly.confidence:.2f}): "
                    f"{anomaly.description}"
                )
            lines.append("")
        if failure.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in failure.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        if failure.lessons_learned:
            lines.append("## Lessons Learned")
            lines.append("")
            for lesson in failure.lessons_learned:
                lines.append(f"- {lesson}")
        return "\n".join(lines)
