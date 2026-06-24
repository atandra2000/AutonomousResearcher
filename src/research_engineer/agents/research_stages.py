"""Phase 15 - Research stage agents.

Specialized agents for each stage of the autonomous research workflow.
Each agent conforms to the ``async execute(ctx: SharedResearchContext)
-> dict`` interface, reading structured artifacts from the context and
writing its outputs back.

Stages:
1. LiteratureDiscoveryAgent — discovers and summarizes relevant papers.
2. KnowledgeSynthesisAgent — synthesizes key findings, gaps, trends.
3. HypothesisGeneratorAgent — generates testable hypotheses from synthesis.
4. ExperimentPlannerAgent — designs experiments to test hypotheses.
5. ExperimentExecutorAgent — executes experiments (dry-run default).
6. ResultAnalyzerAgent — analyzes results and updates hypothesis status.
7. ReportGeneratorAgent — generates the final research report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.research import (
    ExperimentDesign,
    ExperimentOutcome,
    Hypothesis,
    HypothesisStatus,
    KnowledgeSynthesis,
    PaperFinding,
    ResultAnalysis,
    SharedResearchContext,
)
from research_engineer.tools.terminal import TerminalInput, TerminalTool

# ---------------------------------------------------------------------------
# Stage 1: Literature Discovery
# ---------------------------------------------------------------------------


class LiteratureDiscoveryAgent:
    """Discovers relevant papers and generates a literature review.

    Uses the existing :class:`LiteratureAgent` (Phase 6) for paper search
    and review generation when available. Falls back to a rule-based
    discovery when the literature agent is not configured.
    """

    def __init__(
        self,
        literature_agent: Any | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "LiteratureDiscoveryAgent"
        self._literature_agent = literature_agent
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Discover papers and generate a literature review."""
        if self._literature_agent is not None:
            return await self._discover_via_literature_agent(ctx)
        return await self._discover_via_llm(ctx)

    async def _discover_via_literature_agent(
        self, ctx: SharedResearchContext
    ) -> dict[str, Any]:
        """Use the existing LiteratureAgent.discover() method."""
        try:
            result = await self._literature_agent.discover(  # type: ignore[union-attr]
                topic=ctx.research_goal,
                repo_path=ctx.repo_path,
                max_papers=ctx.max_papers,
                output_dir=ctx.output_dir,
            )
            papers: list[PaperFinding] = []
            if hasattr(result, "search_results") and result.search_results:
                for p in result.search_results.papers[: ctx.max_papers]:
                    papers.append(
                        PaperFinding(
                            paper_id=p.paper_id,
                            title=p.title,
                            year=p.year,
                            abstract=p.abstract or "",
                            relevance_score=float(getattr(p, "citation_count", 0))
                            / 100,
                            citation_count=getattr(p, "citation_count", 0),
                        )
                    )
            if hasattr(result, "review") and result.review:
                ctx.literature_review = result.review.markdown or ""
            for p in papers:
                ctx.add_paper(p)
            return {
                "summary": f"Discovered {len(papers)} papers via LiteratureAgent",
                "papers": len(papers),
            }
        except Exception as e:
            return await self._discover_via_llm(ctx, error=str(e))

    async def _discover_via_llm(
        self, ctx: SharedResearchContext, error: str = ""
    ) -> dict[str, Any]:
        """LLM-based or rule-based fallback for paper discovery."""
        provider = self.llm_provider
        if provider is None:
            return self._rule_based_discovery(ctx)
        system = (
            "You are a research literature expert. Given a research goal, "
            "list 5-10 relevant papers you would search for. For each, "
            "provide a paper_id, title, and key_findings. Format as a "
            "numbered list."
        )
        user = f"Research goal: {ctx.research_goal}"
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        try:
            resp = await provider.complete(request)
            ctx.literature_review = resp.content
            return {
                "summary": f"Generated literature review via LLM ({len(resp.content)} chars)",
                "papers": 0,
            }
        except Exception:
            return self._rule_based_discovery(ctx)

    @staticmethod
    def _rule_based_discovery(ctx: SharedResearchContext) -> dict[str, Any]:
        """Rule-based fallback: generate a placeholder review."""
        review = (
            f"# Literature Review: {ctx.research_goal}\n\n"
            "## Overview\n\n"
            f"This review covers research relevant to: {ctx.research_goal}.\n\n"
            "## Key Areas\n\n"
            "1. Existing approaches and their limitations.\n"
            "2. Recent advances in the field.\n"
            "3. Open challenges and research gaps.\n\n"
            "## Note\n\n"
            "No literature agent or LLM provider was configured. "
            "This is a placeholder review. Configure the LiteratureAgent "
            "or an LLM provider for real paper discovery.\n"
        )
        ctx.literature_review = review
        return {
            "summary": "Generated placeholder literature review (no LLM)",
            "papers": 0,
        }


# ---------------------------------------------------------------------------
# Stage 2: Knowledge Synthesis
# ---------------------------------------------------------------------------


class KnowledgeSynthesisAgent:
    """Synthesizes key findings, gaps, and trends from discovered papers."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "KnowledgeSynthesisAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Synthesize knowledge from the literature review."""
        provider = self.llm_provider
        if provider is None:
            return self._rule_based_synthesis(ctx)
        system = (
            "You are a research synthesis expert. Given a literature review "
            "and research goal, produce a structured synthesis with: "
            "key_findings, research_gaps, consensus_points, contradictions, "
            "and emerging_trends. Be concise and specific."
        )
        user = (
            f"Research goal: {ctx.research_goal}\n\n"
            f"## Literature Review\n{ctx.literature_review[:6000]}\n\n"
            f"## Papers ({len(ctx.papers)} found)\n"
        )
        for p in ctx.papers[:10]:
            user += f"- [{p.paper_id}] {p.title}\n"
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        try:
            resp = await provider.complete(request)
            synthesis = self._parse_synthesis(resp.content)
            ctx.synthesis = synthesis
            return {
                "summary": f"Synthesized {len(synthesis.key_findings)} findings, "
                f"{len(synthesis.research_gaps)} gaps",
                "synthesis": synthesis.model_dump(),
            }
        except Exception:
            return self._rule_based_synthesis(ctx)

    @staticmethod
    def _parse_synthesis(content: str) -> KnowledgeSynthesis:
        """Parse LLM output into a KnowledgeSynthesis."""
        lines = content.strip().splitlines()
        findings: list[str] = []
        gaps: list[str] = []
        consensus: list[str] = []
        contradictions: list[str] = []
        trends: list[str] = []
        current = findings
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if "key finding" in lower or "findings" in lower:
                current = findings
            elif "research gap" in lower or "gaps" in lower:
                current = gaps
            elif "consensus" in lower:
                current = consensus
            elif "contradiction" in lower:
                current = contradictions
            elif "trend" in lower or "emerging" in lower:
                current = trends
            elif stripped.startswith("-") or stripped.startswith("*"):
                current.append(stripped.lstrip("-* ").strip())
        return KnowledgeSynthesis(
            key_findings=findings[:20],
            research_gaps=gaps[:10],
            consensus_points=consensus[:10],
            contradictions=contradictions[:10],
            emerging_trends=trends[:10],
            summary=content[:500],
        )

    @staticmethod
    def _rule_based_synthesis(ctx: SharedResearchContext) -> dict[str, Any]:
        """Rule-based fallback synthesis."""
        synthesis = KnowledgeSynthesis(
            key_findings=[
                "Existing approaches have known limitations.",
                "Recent work shows promising directions.",
            ],
            research_gaps=[
                "Scalability to larger models is underexplored.",
                "Efficiency-accuracy trade-offs need systematic study.",
            ],
            consensus_points=["The field is rapidly evolving."],
            contradictions=[],
            emerging_trends=["Attention mechanism optimization.", "Efficient training."],
            summary="Rule-based synthesis (no LLM configured).",
        )
        ctx.synthesis = synthesis
        return {
            "summary": "Generated rule-based synthesis (no LLM)",
            "synthesis": synthesis.model_dump(),
        }


# ---------------------------------------------------------------------------
# Stage 3: Hypothesis Generation
# ---------------------------------------------------------------------------


class HypothesisGeneratorAgent:
    """Generates testable hypotheses from the knowledge synthesis."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "HypothesisGeneratorAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate hypotheses from the synthesis."""
        provider = self.llm_provider
        if provider is None:
            return self._rule_based_hypotheses(ctx)
        system = (
            "You are a research hypothesis generator. Given a research goal, "
            "knowledge synthesis, and research gaps, generate 3-5 testable "
            "hypotheses. For each, provide: statement, rationale, "
            "expected_outcome, and testability (high/medium/low). "
            "Format each as 'H: <statement> | R: <rationale> | E: <expected>'."
        )
        synthesis_text = ""
        if ctx.synthesis:
            synthesis_text = (
                f"Key findings: {', '.join(ctx.synthesis.key_findings[:5])}\n"
                f"Research gaps: {', '.join(ctx.synthesis.research_gaps[:5])}\n"
                f"Trends: {', '.join(ctx.synthesis.emerging_trends[:5])}\n"
            )
        user = (
            f"Research goal: {ctx.research_goal}\n\n"
            f"## Synthesis\n{synthesis_text}\n"
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        try:
            resp = await provider.complete(request)
            hypotheses = self._parse_hypotheses(resp.content, ctx)
            for h in hypotheses:
                ctx.add_hypothesis(h)
            return {
                "summary": f"Generated {len(hypotheses)} hypotheses",
                "hypotheses": [h.model_dump() for h in hypotheses],
            }
        except Exception:
            return self._rule_based_hypotheses(ctx)

    @staticmethod
    def _parse_hypotheses(
        content: str, ctx: SharedResearchContext
    ) -> list[Hypothesis]:
        """Parse LLM output into Hypothesis objects."""
        hypotheses: list[Hypothesis] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Parse "H: ... | R: ... | E: ..." format.
            parts = stripped.split("|")
            statement = ""
            rationale = ""
            expected = ""
            for part in parts:
                p = part.strip()
                if p.startswith("H:"):
                    statement = p[2:].strip()
                elif p.startswith("R:"):
                    rationale = p[2:].strip()
                elif p.startswith("E:"):
                    expected = p[2:].strip()
            if statement:
                hypotheses.append(
                    Hypothesis(
                        hypothesis_id=f"hyp_{uuid4().hex[:8]}",
                        statement=statement,
                        rationale=rationale,
                        expected_outcome=expected,
                        based_on_gaps=ctx.synthesis.research_gaps[:3]
                        if ctx.synthesis
                        else [],
                        priority=7,
                        testability="medium",
                        novelty="medium",
                    )
                )
        return hypotheses[: ctx.max_hypotheses]

    @staticmethod
    def _rule_based_hypotheses(ctx: SharedResearchContext) -> dict[str, Any]:
        """Rule-based fallback hypothesis generation."""
        gaps = ctx.synthesis.research_gaps if ctx.synthesis else [
            "Efficiency is underexplored.",
            "Scalability needs study.",
        ]
        hypotheses: list[Hypothesis] = []
        for i, gap in enumerate(gaps[: ctx.max_hypotheses]):
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp_{uuid4().hex[:8]}",
                    statement=f"Addressing '{gap}' will improve performance.",
                    rationale=f"This gap was identified in the synthesis: {gap}",
                    expected_outcome="Improved efficiency or accuracy.",
                    based_on_gaps=[gap],
                    priority=8 - i,
                    testability="medium",
                    novelty="medium",
                )
            )
        for h in hypotheses:
            ctx.add_hypothesis(h)
        return {
            "summary": f"Generated {len(hypotheses)} hypotheses (rule-based)",
            "hypotheses": [h.model_dump() for h in hypotheses],
        }


# ---------------------------------------------------------------------------
# Stage 4: Experiment Planning
# ---------------------------------------------------------------------------


class ResearchExperimentPlannerAgent:
    """Designs experiments to test the generated hypotheses."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "ResearchExperimentPlannerAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Design experiments for each hypothesis."""
        designs: list[ExperimentDesign] = []
        for hyp in ctx.hypotheses:
            design = ExperimentDesign(
                experiment_id=f"exp_{uuid4().hex[:8]}",
                hypothesis_id=hyp.hypothesis_id,
                title=f"Test: {hyp.statement[:80]}",
                description=(
                    f"Experiment to test hypothesis: {hyp.statement}. "
                    f"Expected outcome: {hyp.expected_outcome}."
                ),
                command="python train.py --config config.yaml",
                expected_duration_hours=2.0,
                metrics=["loss", "accuracy", "latency"],
                resources=["1 GPU", "8GB RAM"],
            )
            designs.append(design)
        ctx.experiment_designs = designs
        return {
            "summary": f"Designed {len(designs)} experiments",
            "designs": [d.model_dump() for d in designs],
        }


# ---------------------------------------------------------------------------
# Stage 5: Experiment Execution
# ---------------------------------------------------------------------------


class ExperimentExecutorAgent:
    """Executes experiments (dry-run by default for safety)."""

    def __init__(
        self,
        terminal_tool: TerminalTool | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.agent_name: str = "ExperimentExecutorAgent"
        self.terminal = terminal_tool or TerminalTool()
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute or simulate the planned experiments."""
        outcomes: list[ExperimentOutcome] = []
        for design in ctx.experiment_designs:
            if ctx.dry_run_experiments:
                outcome = ExperimentOutcome(
                    experiment_id=design.experiment_id,
                    status="dry_run",
                    exit_code=0,
                    metrics={},
                    duration_seconds=0.0,
                    stdout=f"Dry run: would execute '{design.command}'",
                )
            else:
                outcome = await self._run_experiment(ctx, design)
            outcomes.append(outcome)
        ctx.experiment_outcomes = outcomes
        return {
            "summary": f"Executed {len(outcomes)} experiments "
            f"({'dry-run' if ctx.dry_run_experiments else 'live'})",
            "outcomes": [o.model_dump() for o in outcomes],
        }

    async def _run_experiment(
        self, ctx: SharedResearchContext, design: ExperimentDesign
    ) -> ExperimentOutcome:
        """Execute a single experiment via TerminalTool."""
        out = await self.terminal.execute(
            TerminalInput(
                operation="run_command",
                repo_path=ctx.repo_path,
                command=design.command,
                timeout_seconds=ctx.experiment_timeout,
                dry_run=False,
            )
        )
        return ExperimentOutcome(
            experiment_id=design.experiment_id,
            status="completed" if out.success else "failed",
            exit_code=out.exit_code,
            metrics={},
            duration_seconds=out.duration_seconds,
            stdout=out.stdout[:2000],
            stderr=out.stderr[:2000],
            error=out.error,
        )


# ---------------------------------------------------------------------------
# Stage 6: Result Analysis
# ---------------------------------------------------------------------------


class ResultAnalyzerAgent:
    """Analyzes experiment results and updates hypothesis status."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "ResultAnalyzerAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze experiment outcomes and update hypothesis status."""
        analyses: list[ResultAnalysis] = []
        for outcome in ctx.experiment_outcomes:
            design = next(
                (
                    d
                    for d in ctx.experiment_designs
                    if d.experiment_id == outcome.experiment_id
                ),
                None,
            )
            if design is None:
                continue
            hyp = next(
                (
                    h
                    for h in ctx.hypotheses
                    if h.hypothesis_id == design.hypothesis_id
                ),
                None,
            )
            if outcome.status == "dry_run":
                conclusion = (
                    "Experiment was dry-run; no actual results to analyze."
                )
                status = HypothesisStatus.INCONCLUSIVE
            elif outcome.status == "completed" and outcome.exit_code == 0:
                conclusion = "Experiment completed successfully."
                status = HypothesisStatus.SUPPORTED
            else:
                conclusion = f"Experiment failed: {outcome.error or 'unknown'}"
                status = HypothesisStatus.INCONCLUSIVE
            analysis = ResultAnalysis(
                experiment_id=outcome.experiment_id,
                hypothesis_id=design.hypothesis_id,
                key_metrics=outcome.metrics,
                comparison="N/A (dry-run or single experiment)",
                significance="inconclusive",
                conclusion=conclusion,
                hypothesis_status=status,
                recommendations=[
                    "Run with real data for conclusive results.",
                    "Consider ablation studies.",
                ],
            )
            analyses.append(analysis)
            if hyp is not None:
                hyp.status = status
        ctx.analyses = analyses
        return {
            "summary": f"Analyzed {len(analyses)} experiment outcomes",
            "analyses": [a.model_dump() for a in analyses],
        }


# ---------------------------------------------------------------------------
# Stage 7: Report Generation
# ---------------------------------------------------------------------------


class ReportGeneratorAgent:
    """Generates the final research report with evidence and conclusions."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "ReportGeneratorAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedResearchContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate the final research report."""
        report = self._build_report(ctx)
        ctx.final_report = report
        # Save to file.
        out_dir = Path(ctx.output_dir) / ctx.workflow_id
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "research_report.md"
        report_path.write_text(report, encoding="utf-8")
        ctx.report_path = str(report_path)
        return {
            "summary": f"Generated research report ({len(report)} chars)",
            "report_path": str(report_path),
        }

    @staticmethod
    def _build_report(ctx: SharedResearchContext) -> str:
        """Build the final research report as markdown."""
        lines = ReportGeneratorAgent._build_header(ctx)
        ReportGeneratorAgent._add_synthesis_section(ctx, lines)
        ReportGeneratorAgent._add_hypotheses_section(ctx, lines)
        ReportGeneratorAgent._add_experiments_section(ctx, lines)
        ReportGeneratorAgent._add_analyses_section(ctx, lines)
        ReportGeneratorAgent._add_footer(ctx, lines)
        return "\n".join(lines)

    @staticmethod
    def _build_header(ctx: SharedResearchContext) -> list[str]:
        return [
            f"# Research Report: {ctx.research_goal}",
            "",
            f"**Workflow ID**: {ctx.workflow_id}",
            f"**Date**: {datetime.now().isoformat()}",
            "",
            "## Executive Summary",
            "",
            f"This report presents the findings of an autonomous research "
            f"workflow investigating: **{ctx.research_goal}**.",
            f"The workflow discovered {len(ctx.papers)} papers, generated "
            f"{len(ctx.hypotheses)} hypotheses, and executed "
            f"{len(ctx.experiment_outcomes)} experiments.",
            "",
            "## Literature Review",
            "",
            ctx.literature_review[:3000] if ctx.literature_review else "No literature review generated.",
            "",
        ]

    @staticmethod
    def _add_synthesis_section(
        ctx: SharedResearchContext, lines: list[str]
    ) -> None:
        if not ctx.synthesis:
            return
        lines.extend(["## Knowledge Synthesis", "", "### Key Findings", ""])
        for f in ctx.synthesis.key_findings:
            lines.append(f"- {f}")
        lines.extend(["", "### Research Gaps", ""])
        for g in ctx.synthesis.research_gaps:
            lines.append(f"- {g}")
        lines.extend(["", "### Emerging Trends", ""])
        for t in ctx.synthesis.emerging_trends:
            lines.append(f"- {t}")
        lines.append("")

    @staticmethod
    def _add_hypotheses_section(
        ctx: SharedResearchContext, lines: list[str]
    ) -> None:
        if not ctx.hypotheses:
            return
        lines.extend(["## Hypotheses", ""])
        for i, h in enumerate(ctx.hypotheses, 1):
            lines.extend([
                f"### Hypothesis {i}: {h.statement}",
                "",
                f"- **Rationale**: {h.rationale}",
                f"- **Expected outcome**: {h.expected_outcome}",
                f"- **Testability**: {h.testability}",
                f"- **Novelty**: {h.novelty}",
                f"- **Status**: {h.status.value}",
                "",
            ])

    @staticmethod
    def _add_experiments_section(
        ctx: SharedResearchContext, lines: list[str]
    ) -> None:
        if not ctx.experiment_outcomes:
            return
        lines.extend(["## Experiments", ""])
        for outcome in ctx.experiment_outcomes:
            lines.extend([
                f"### Experiment: {outcome.experiment_id}",
                "",
                f"- **Status**: {outcome.status}",
                f"- **Exit code**: {outcome.exit_code}",
                f"- **Duration**: {outcome.duration_seconds}s",
                "",
            ])

    @staticmethod
    def _add_analyses_section(
        ctx: SharedResearchContext, lines: list[str]
    ) -> None:
        if not ctx.analyses:
            return
        lines.extend(["## Result Analysis", ""])
        for a in ctx.analyses:
            lines.extend([
                f"### Analysis: {a.experiment_id}",
                "",
                f"- **Conclusion**: {a.conclusion}",
                f"- **Hypothesis status**: {a.hypothesis_status.value}",
                f"- **Significance**: {a.significance}",
                "",
            ])

    @staticmethod
    def _add_footer(ctx: SharedResearchContext, lines: list[str]) -> None:
        lines.extend([
            "## Conclusions",
            "",
            "The autonomous research workflow completed all stages. "
            "See the hypotheses and experiment sections for detailed findings.",
            "",
            "## Reproducibility",
            "",
            f"- **Workflow ID**: {ctx.workflow_id}",
            f"- **Output directory**: {ctx.output_dir}",
            f"- **Dry-run experiments**: {ctx.dry_run_experiments}",
            "",
            "---",
            "*Generated by the Autonomous ML Research Engineer (Phase 15).*",
        ])


__all__ = [
    "LiteratureDiscoveryAgent",
    "KnowledgeSynthesisAgent",
    "HypothesisGeneratorAgent",
    "ResearchExperimentPlannerAgent",
    "ExperimentExecutorAgent",
    "ResultAnalyzerAgent",
    "ReportGeneratorAgent",
]
