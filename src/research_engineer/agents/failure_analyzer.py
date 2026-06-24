"""Phase 14 - Failure analyzer agent.

Diagnoses failures from test output, review feedback, and implementation
errors, producing a structured :class:`FailureReport` with root cause
hypothesis, affected files, and actionable evidence.

Uses rule-based pattern matching as the primary analysis engine (fast,
deterministic, no LLM dependency) and optionally augments with LLM
reasoning when a provider is configured.

Interface: ``async execute(ctx: SharedTaskContext) -> dict``
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.delegation import SharedTaskContext
from research_engineer.models.repair import (
    FailureCategory,
    FailureReport,
    FailureSeverity,
)

# Pattern -> (category, severity, root_cause_template)
_ERROR_PATTERNS: list[tuple[re.Pattern[str], FailureCategory, FailureSeverity, str]] = [
    (re.compile(r"SyntaxError", re.I), FailureCategory.SYNTAX_ERROR, FailureSeverity.HIGH,
     "Syntax error in the generated or modified code."),
    (re.compile(r"ImportError|ModuleNotFoundError", re.I), FailureCategory.IMPORT_ERROR, FailureSeverity.HIGH,
     "Missing or incorrect import statement."),
    (re.compile(r"AssertionError|assert", re.I), FailureCategory.ASSERTION_ERROR, FailureSeverity.MEDIUM,
     "Assertion failed — logic error in the implementation."),
    (re.compile(r"TypeError", re.I), FailureCategory.TYPE_ERROR, FailureSeverity.HIGH,
     "Type mismatch — incorrect argument types or return types."),
    (re.compile(r"AttributeError", re.I), FailureCategory.ATTRIBUTE_ERROR, FailureSeverity.HIGH,
     "Attribute access on wrong or missing object."),
    (re.compile(r"NameError", re.I), FailureCategory.IMPORT_ERROR, FailureSeverity.HIGH,
     "Name not defined — missing import or definition."),
    (re.compile(r"KeyError", re.I), FailureCategory.RUNTIME_ERROR, FailureSeverity.MEDIUM,
     "Dictionary key access failed — data structure mismatch."),
    (re.compile(r"ZeroDivisionError", re.I), FailureCategory.RUNTIME_ERROR, FailureSeverity.MEDIUM,
     "Division by zero — guard against zero divisor."),
    (re.compile(r"TimeoutError|timed?\s*out", re.I), FailureCategory.TIMEOUT, FailureSeverity.HIGH,
     "Operation timed out — possible infinite loop or resource exhaustion."),
    (re.compile(r"FAILED\s+(\S+)", re.I), FailureCategory.TEST_FAILURE, FailureSeverity.MEDIUM,
     "One or more tests failed."),
    (re.compile(r"ERROR\s+(\S+)", re.I), FailureCategory.TEST_FAILURE, FailureSeverity.MEDIUM,
     "One or more test errors occurred."),
    (re.compile(r"CHANGES_REQUESTED", re.I), FailureCategory.REVIEW_REJECTION, FailureSeverity.LOW,
     "Code review requested changes."),
]


class FailureAnalyzer:
    """Analyzes failures and produces structured :class:`FailureReport`.

    Reads ``ctx.test_failures``, ``ctx.test_stderr``, ``ctx.review_issues``,
    and ``ctx.diff`` from the shared context, diagnoses the root cause,
    and returns a :class:`FailureReport` in the output dict.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "FailureAnalyzer"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze the current failure state and produce a report."""
        report = self._analyze(ctx)
        if self.llm_provider is not None:
            report = await self._augment_with_llm(ctx, report)
        return {
            "summary": f"{report.category.value}: {report.root_cause[:100]}",
            "failure_report": report.model_dump(),
        }

    def _analyze(self, ctx: SharedTaskContext) -> FailureReport:
        """Rule-based failure analysis from shared context."""
        combined = " ".join(
            ctx.test_failures + [ctx.test_stderr, ctx.review_feedback]
            + ctx.review_issues
        )
        # Determine failure source.
        source = "unknown"
        if ctx.test_failures or ctx.test_exit_code not in (None, 0):
            source = "test"
        elif ctx.review_issues:
            source = "review"
        elif not ctx.diff:
            source = "implementation"

        # Pattern match.
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.MEDIUM
        root_cause = "Unknown failure."
        for pattern, cat, sev, cause in _ERROR_PATTERNS:
            if pattern.search(combined):
                category = cat
                severity = sev
                root_cause = cause
                break

        # If no pattern matched but we have review issues.
        if category == FailureCategory.UNKNOWN and ctx.review_issues:
            category = FailureCategory.REVIEW_REJECTION
            severity = FailureSeverity.LOW
            root_cause = "Code review identified issues that need addressing."

        # If no pattern matched but tests failed.
        if category == FailureCategory.UNKNOWN and ctx.test_exit_code not in (None, 0):
            category = FailureCategory.TEST_FAILURE
            severity = FailureSeverity.MEDIUM
            root_cause = "Tests failed but no specific error pattern was recognized."

        # Extract evidence.
        evidence = self._extract_evidence(ctx)

        # Extract affected files from test failures.
        affected_files = self._extract_affected_files(ctx)

        # Extract affected symbols.
        affected_symbols = self._extract_affected_symbols(ctx)

        return FailureReport(
            report_id=f"fr_{uuid4().hex[:8]}",
            category=category,
            severity=severity,
            root_cause=root_cause,
            evidence=evidence,
            affected_files=affected_files,
            affected_symbols=affected_symbols,
            failure_source=source,
            raw_error=combined[:2000],
        )

    @staticmethod
    def _extract_evidence(ctx: SharedTaskContext) -> list[str]:
        """Extract key evidence lines from failures."""
        evidence: list[str] = []
        for f in ctx.test_failures[:5]:
            evidence.append(f)
        if ctx.test_stderr:
            for line in ctx.test_stderr.splitlines():
                stripped = line.strip()
                if stripped and ("Error" in stripped or "error" in stripped):
                    evidence.append(stripped[:200])
                    if len(evidence) >= 10:
                        break
        for issue in ctx.review_issues[:3]:
            evidence.append(f"Review: {issue}")
        return evidence[:10]

    @staticmethod
    def _extract_affected_files(ctx: SharedTaskContext) -> list[str]:
        """Extract file paths from test failures and diff."""
        files: set[str] = set()
        for f in ctx.test_failures:
            # pytest format: "tests/test_foo.py::test_bar - ..."
            match = re.match(r"(?:FAILED\s+)?(\S+\.py)", f)
            if match:
                files.add(match.group(1))
        # Also extract from diff.
        for line in ctx.diff.splitlines():
            if line.startswith("+++ b/"):
                files.add(line[6:])
            elif line.startswith("--- a/"):
                files.add(line[6:])
        return sorted(files)[:10]

    @staticmethod
    def _extract_affected_symbols(ctx: SharedTaskContext) -> list[str]:
        """Extract symbol names from test failures."""
        symbols: set[str] = set()
        for f in ctx.test_failures:
            # pytest format: "...::test_bar - ..."
            match = re.search(r"::(\w+)", f)
            if match:
                symbols.add(match.group(1))
        return sorted(symbols)[:10]

    async def _augment_with_llm(
        self, ctx: SharedTaskContext, report: FailureReport
    ) -> FailureReport:
        """Augment the rule-based report with LLM reasoning."""
        provider = self.llm_provider
        if provider is None:
            return report
        system = (
            "You are a failure analysis expert. Given the following "
            "structured failure report and context, refine the root cause "
            "hypothesis and suggest which files/symbols to inspect. "
            "Be concise (3-5 sentences)."
        )
        user = (
            f"Goal: {ctx.goal}\n"
            f"Failure category: {report.category.value}\n"
            f"Evidence:\n" + "\n".join(f"- {e}" for e in report.evidence) + "\n"
            f"Current root cause: {report.root_cause}\n"
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.2,
            max_tokens=256,
        )
        try:
            resp = await provider.complete(request)
            report.root_cause = resp.content[:500]
        except Exception:
            pass
        return report


__all__ = ["FailureAnalyzer"]
