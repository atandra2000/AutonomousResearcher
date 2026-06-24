"""Report Generator Tool for Phase 9.

Generates a complete research report (markdown + JSON) from a finished
research loop and its iterations.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from research_engineer.models.loop import (
    LoopIteration,
    LoopRecord,
    ReportInput,
    ReportOutput,
    StoppingCondition,
)
from research_engineer.tools.base import Tool, ToolError


class ReportGeneratorTool(Tool[ReportInput, ReportOutput]):
    """Generates research reports for completed loops."""

    async def execute(self, input: ReportInput) -> ReportOutput:
        try:
            out_path = Path(input.output_dir) / input.loop.loop_id
            out_path.mkdir(parents=True, exist_ok=True)

            md = self._build_markdown(
                input.loop, input.iterations, input.config.goal
                if hasattr(input.config, "goal")
                else input.loop.goal
            )
            report_file = out_path / "research_report.md"
            report_file.write_text(md, encoding="utf-8")

            json_data = self._build_json(
                input.loop, input.iterations, input.config
            )
            json_file = out_path / "research_report.json"
            json_file.write_text(
                _json.dumps(json_data, indent=2, default=str),
                encoding="utf-8",
            )

            summary = self._build_summary(input.loop, input.iterations)
            return ReportOutput(
                report_path=str(report_file),
                json_path=str(json_file),
                summary=summary,
            )
        except Exception as e:
            raise ToolError(f"Report generation failed: {e}", input, e)

    @staticmethod
    def _build_summary(
        loop: LoopRecord, iterations: list[LoopIteration]
    ) -> str:
        parts = [
            f"Loop {loop.loop_id} '{loop.goal}'",
            f"Status: {loop.status.value}",
            f"Iterations: {loop.iteration_count}",
        ]
        if loop.best_metric_value is not None:
            parts.append(f"Best metric: {loop.best_metric_value:.6f}")
        if loop.stopping_condition:
            parts.append(f"Stopped: {loop.stopping_condition.value}")
        return ". ".join(parts) + "."

    def _build_markdown(
        self,
        loop: LoopRecord,
        iterations: list[LoopIteration],
        goal: str,
    ) -> str:
        lines: list[str] = []
        lines.extend(self._build_header(loop, goal))
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self._exec_summary(loop, iterations))
        lines.append("")

        # Methodology
        lines.append("## Methodology")
        lines.append("")
        lines.append(
            "Agents orchestrated in sequence: LiteratureAgent → "
            "RepositoryAgent → ExperimentPlannerAgent → CodingAgent → "
            "ExperimentAgent → EvaluationAgent → MemoryAgent."
        )
        lines.append("")

        # Iteration History
        lines.append("## Iteration History")
        lines.append("")
        lines.append(
            "| # | Paper | Phase | Metric | Improvement | Decision |"
        )
        lines.append(
            "|---|-------|-------|--------|-------------|----------|"
        )
        for it in iterations:
            metric = (
                f"{it.primary_metric_value:.4f}"
                if it.primary_metric_value is not None
                else "N/A"
            )
            imp = (
                f"{it.improvement:.4f}"
                if it.improvement is not None
                else "-"
            )
            decision = it.decision.value if it.decision else "continue"
            lines.append(
                f"| {it.iteration_number} | "
                f"{it.paper_id or 'N/A'} | {it.phase.value} | "
                f"{metric} | {imp} | {decision} |"
            )
        lines.append("")

        # Key Findings
        successful = [it for it in iterations if it.is_success()]
        lines.append("## Key Findings")
        lines.append("")
        if successful:
            for it in successful:
                if it.metrics:
                    top = sorted(
                        it.metrics.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:3]
                    metrics_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in top
                    )
                    lines.append(
                        f"- Iteration {it.iteration_number}: {metrics_str}"
                    )
                if it.memory_ids:
                    lines.append(
                        f"  - Memory IDs: {', '.join(it.memory_ids[:3])}"
                    )
        else:
            lines.append("- No successful iterations recorded.")
        lines.append("")

        # Failed Approaches
        failed = [it for it in iterations if not it.is_success()]
        lines.append("## Failed Approaches")
        lines.append("")
        if failed:
            for it in failed:
                lines.append(
                    f"- Iteration {it.iteration_number} "
                    f"(phase: {it.phase.value}): "
                    f"{it.error or 'unknown error'}"
                )
        else:
            lines.append("- No failed iterations.")
        lines.append("")

        # Conclusions
        lines.append("## Conclusions")
        lines.append("")
        if loop.stopping_condition:
            lines.append(self._conclusion_text(loop, iterations))
        else:
            lines.append(
                "The loop did not reach a terminal stopping condition."
            )
        lines.append("")

        # Appendix: Configuration
        lines.append("## Appendix: Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(loop.config_json)
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _build_header(loop: LoopRecord, goal: str) -> list[str]:
        """Build the report header section."""
        lines = [
            "# Autonomous Research Report",
            "",
            f"**Loop ID**: {loop.loop_id}",
            f"**Goal**: {goal}",
            f"**Status**: {loop.status.value}",
            f"**Iterations**: {loop.iteration_count}",
        ]
        if loop.best_metric_value is not None:
            lines.append(
                f"**Best Metric**: "
                f"{loop.primary_metric_name or 'metric'}="
                f"{loop.best_metric_value:.6f}"
            )
        if loop.stopping_condition:
            lines.append(
                f"**Stopping Condition**: "
                f"{loop.stopping_condition.value}"
            )
        if loop.stopping_reason:
            lines.append(f"**Stopping Reason**: {loop.stopping_reason}")
        lines.append(f"**Created**: {loop.created_at}")
        return lines

    @staticmethod
    def _exec_summary(
        loop: LoopRecord, iterations: list[LoopIteration]
    ) -> str:
        parts: list[str] = [
            f"This autonomous research loop executed "
            f"{loop.iteration_count} iteration(s) toward the goal: "
            f"'{loop.goal}'."
        ]
        if loop.stopping_condition == StoppingCondition.TARGET_ACHIEVED:
            parts.append(
                f"The target metric "
                f"({loop.primary_metric_name or 'metric'}) was achieved "
                f"with a best value of {loop.best_metric_value:.6f}."
            )
        elif loop.stopping_condition == StoppingCondition.MAX_ITERATIONS_REACHED:
            parts.append(
                "The loop reached the maximum iteration limit without "
                "achieving the target."
            )
        elif loop.stopping_condition == StoppingCondition.BUDGET_EXCEEDED:
            parts.append("The loop stopped due to budget exhaustion.")
        elif loop.stopping_condition == StoppingCondition.NO_IMPROVEMENT:
            parts.append(
                "The loop stopped because no meaningful metric improvement "
                "was observed over recent iterations."
            )
        successful_count = sum(1 for it in iterations if it.is_success())
        parts.append(
            f"{successful_count} of {len(iterations)} iteration(s) "
            f"completed successfully."
        )
        return " ".join(parts)

    @staticmethod
    def _conclusion_text(
        loop: LoopRecord, iterations: list[LoopIteration]
    ) -> str:
        if loop.stopping_condition == StoppingCondition.TARGET_ACHIEVED:
            return (
                f"The research goal was achieved. The best metric value "
                f"({loop.best_metric_value:.6f}) met the target. "
                f"Recommend archiving this configuration and exploring "
                f"related research directions."
            )
        if loop.stopping_condition == StoppingCondition.NO_IMPROVEMENT:
            return (
                "No further improvement was detected. The current "
                "approach may have plateaued. Consider exploring "
                "alternative methods identified in the literature or "
                "adjusting hyperparameters."
            )
        if loop.stopping_condition == StoppingCondition.BUDGET_EXCEEDED:
            return (
                "The compute budget was exhausted before the target was "
                "reached. Consider increasing the budget or optimizing "
                "the experiment for lower cost."
            )
        return (
            "The loop completed all iterations. Review the iteration "
            "history for insights on next steps."
        )

    @staticmethod
    def _build_json(
        loop: LoopRecord,
        iterations: list[LoopIteration],
        config: object,
    ) -> dict:
        return {
            "loop": loop.model_dump(mode="json"),
            "iterations": [it.model_dump(mode="json") for it in iterations],
            "config": (
                config.model_dump(mode="json")
                if hasattr(config, "model_dump")
                else {}
            ),
        }
