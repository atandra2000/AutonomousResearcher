"""Phase 14 - Repair strategist agent.

Generates ranked repair strategies from a :class:`FailureReport`. Uses
rule-based strategy generation keyed by failure category, and optionally
augments with LLM reasoning for complex failures.

Interface: ``async execute(ctx: SharedTaskContext) -> dict``
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.delegation import SharedTaskContext
from research_engineer.models.repair import (
    FailureCategory,
    FailureReport,
    RepairActionType,
    RepairStrategy,
)

# Category -> list of (action_type, description_template, priority, confidence)
_STRATEGY_MAP: dict[FailureCategory, list[tuple[RepairActionType, str, int, float]]] = {
    FailureCategory.SYNTAX_ERROR: [
        (RepairActionType.FIX_SYNTAX,
         "Fix syntax error in {files}. Check for unbalanced parentheses, "
         "missing colons, or incorrect indentation.", 9, 0.9),
        (RepairActionType.REGENERATE,
         "Regenerate the code for {files} from scratch to avoid the syntax error.",
         5, 0.6),
    ],
    FailureCategory.IMPORT_ERROR: [
        (RepairActionType.FIX_IMPORT,
         "Add or correct the import statement for the missing module in {files}.",
         9, 0.9),
        (RepairActionType.ADD_MISSING_CODE,
         "Implement the missing module or symbol referenced in {files}.",
         6, 0.5),
    ],
    FailureCategory.ASSERTION_ERROR: [
        (RepairActionType.FIX_LOGIC,
         "Fix the logic causing assertion failure in {symbols}. Review the "
         "expected vs actual values.", 8, 0.7),
        (RepairActionType.ADJUST_CONFIG,
         "Adjust test expectations or configuration to match current behavior.",
         3, 0.3),
    ],
    FailureCategory.TYPE_ERROR: [
        (RepairActionType.FIX_TYPE,
         "Fix type mismatch in {symbols}. Ensure arguments and return types "
         "match the expected signatures.", 8, 0.7),
        (RepairActionType.REFACTOR,
         "Refactor {symbols} to use consistent types throughout.", 5, 0.5),
    ],
    FailureCategory.ATTRIBUTE_ERROR: [
        (RepairActionType.FIX_LOGIC,
         "Fix attribute access in {symbols}. Check for None or incorrect object type.",
         8, 0.7),
        (RepairActionType.ADD_MISSING_CODE,
         "Add the missing attribute or method to the relevant class in {files}.",
         6, 0.6),
    ],
    FailureCategory.RUNTIME_ERROR: [
        (RepairActionType.FIX_LOGIC,
         "Fix the runtime error in {symbols}. Add guards for edge cases.",
         7, 0.6),
        (RepairActionType.REGENERATE,
         "Regenerate {files} to avoid the runtime error.", 4, 0.4),
    ],
    FailureCategory.LOGIC_ERROR: [
        (RepairActionType.FIX_LOGIC,
         "Fix the logic error in {symbols}. Trace the data flow to find the bug.",
         8, 0.6),
    ],
    FailureCategory.MISSING_IMPLEMENTATION: [
        (RepairActionType.ADD_MISSING_CODE,
         "Implement the missing function/method/class in {files}.",
         9, 0.8),
        (RepairActionType.REGENERATE,
         "Regenerate the entire module {files}.", 5, 0.5),
    ],
    FailureCategory.REVIEW_REJECTION: [
        (RepairActionType.FIX_LOGIC,
         "Address review issues: {issues}. Modify {files} as needed.",
         7, 0.7),
        (RepairActionType.REFACTOR,
         "Refactor {files} to address review concerns about style/structure.",
         5, 0.5),
    ],
    FailureCategory.TEST_FAILURE: [
        (RepairActionType.FIX_LOGIC,
         "Fix the failing test logic in {symbols}. Review assertions and mocks.",
         7, 0.6),
        (RepairActionType.SKIP_TEST,
         "Skip the failing test if it is a pre-existing issue unrelated to the change.",
         2, 0.2),
    ],
    FailureCategory.BUILD_FAILURE: [
        (RepairActionType.FIX_SYNTAX,
         "Fix build error in {files}. Check for missing dependencies or configuration.",
         8, 0.7),
        (RepairActionType.ADJUST_CONFIG,
         "Adjust build configuration to resolve the build failure.",
         5, 0.4),
    ],
    FailureCategory.TIMEOUT: [
        (RepairActionType.FIX_LOGIC,
         "Fix potential infinite loop or resource exhaustion in {symbols}.",
         7, 0.5),
        (RepairActionType.ADJUST_CONFIG,
         "Increase timeout or optimize the operation in {files}.",
         3, 0.3),
    ],
    FailureCategory.UNKNOWN: [
        (RepairActionType.REGENERATE,
         "Regenerate {files} with a different approach.", 4, 0.3),
        (RepairActionType.MANUAL_REVIEW,
         "Manual review required for {files}. The failure could not be auto-diagnosed.",
         3, 0.2),
    ],
}


class RepairStrategist:
    """Generates ranked repair strategies from a :class:`FailureReport`.

    Reads the failure report from ``ctx`` (or from kwargs), generates
    one or more :class:`RepairStrategy` objects ranked by priority and
    confidence, and returns them in the output dict.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "RepairStrategist"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate repair strategies for the current failure."""
        report_data = kwargs.get("failure_report")
        if isinstance(report_data, dict):
            report = FailureReport.model_validate(report_data)
        else:
            report = self._build_report_from_ctx(ctx)
        strategies = self._generate_strategies(report)
        if self.llm_provider is not None:
            strategies = await self._augment_with_llm(ctx, report, strategies)
        # Sort by priority (desc) then confidence (desc).
        strategies.sort(key=lambda s: (-s.priority, -s.confidence))
        return {
            "summary": f"Generated {len(strategies)} repair strategies for {report.category.value}",
            "strategies": [s.model_dump() for s in strategies],
        }

    def _build_report_from_ctx(self, ctx: SharedTaskContext) -> FailureReport:
        """Build a minimal failure report from context if none provided."""
        from research_engineer.models.repair import FailureSeverity

        if ctx.test_failures:
            category = FailureCategory.TEST_FAILURE
        elif ctx.review_issues:
            category = FailureCategory.REVIEW_REJECTION
        else:
            category = FailureCategory.UNKNOWN
        return FailureReport(
            report_id=f"fr_ctx_{uuid4().hex[:8]}",
            category=category,
            severity=FailureSeverity.MEDIUM,
            root_cause="Failure detected from shared context.",
            evidence=ctx.test_failures[:5] + ctx.review_issues[:3],
            affected_files=list(
                {f for fail in ctx.test_failures for f in [fail.split("::")[0]] if f}
            )[:10],
            failure_source="test" if ctx.test_failures else "review",
        )

    def _generate_strategies(
        self, report: FailureReport
    ) -> list[RepairStrategy]:
        """Generate strategies based on the failure report."""
        templates = _STRATEGY_MAP.get(
            report.category, _STRATEGY_MAP[FailureCategory.UNKNOWN]
        )
        files_str = ", ".join(report.affected_files) or "the affected file(s)"
        symbols_str = ", ".join(report.affected_symbols) or "the affected symbol(s)"
        issues_str = "; ".join(report.evidence[:3]) or "see review feedback"
        strategies: list[RepairStrategy] = []
        for action_type, desc_template, priority, confidence in templates:
            desc = desc_template.format(
                files=files_str,
                symbols=symbols_str,
                issues=issues_str,
            )
            strategies.append(
                RepairStrategy(
                    strategy_id=f"rs_{uuid4().hex[:8]}",
                    action_type=action_type,
                    description=desc,
                    target_files=report.affected_files,
                    target_symbols=report.affected_symbols,
                    instructions=desc,
                    priority=priority,
                    confidence=confidence,
                )
            )
        return strategies

    async def _augment_with_llm(
        self,
        ctx: SharedTaskContext,
        report: FailureReport,
        strategies: list[RepairStrategy],
    ) -> list[RepairStrategy]:
        """Augment strategies with LLM-generated instructions."""
        provider = self.llm_provider
        if provider is None or not strategies:
            return strategies
        system = (
            "You are a repair strategy expert. For each strategy, provide "
            "concise, actionable instructions for the coding agent. "
            "Return one instruction per line, prefixed by the strategy index."
        )
        user = (
            f"Goal: {ctx.goal}\n"
            f"Failure: {report.category.value} - {report.root_cause}\n"
            f"Strategies:\n"
        )
        for i, s in enumerate(strategies):
            user += f"{i}. [{s.action_type.value}] {s.description}\n"
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.3,
            max_tokens=512,
        )
        try:
            resp = await provider.complete(request)
            lines = resp.content.strip().splitlines()
            for line in lines:
                match = __import__("re").match(r"^(\d+)\.\s*(.+)", line)
                if match:
                    idx = int(match.group(1))
                    if 0 <= idx < len(strategies):
                        strategies[idx].instructions = match.group(2).strip()
        except Exception:
            pass
        return strategies


__all__ = ["RepairStrategist"]
