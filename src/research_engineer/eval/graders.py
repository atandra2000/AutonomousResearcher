"""E4 - Graders for the evaluation harness.

A :class:`Grader` receives a :class:`GradingRequest` (the eval task plus
the captured ``AgentExecution``) and returns a
class:`~research_engineer.eval.models.GraderResult` with a score in ``[0, 1]``.

Deterministic graders cover objective criteria (output equality,
containment, regex matching, JSON field checks, termination, budgets,
steps, tool usage, recovery). A pluggable LLM quality grader allows domain
grading through any async scoring callable; it is never constructed by
default so evaluation stays deterministic unless explicitly opted in.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from research_engineer.eval.models import EvalTask, GraderResult
from research_engineer.runtime.models import AgentExecution, AgentTermination


class GradingRequest:
    """Everything a grader may look at for one case run."""

    def __init__(
        self,
        task: EvalTask,
        execution: AgentExecution | None,
        *,
        human_interventions: int = 0,
        harness_error: str = "",
    ) -> None:
        self.task = task
        self.execution = execution
        self.human_interventions = human_interventions
        self.harness_error = harness_error

    @property
    def output(self) -> Any:
        """Final agent output (None when the run errored at harness level)."""
        if self.execution is None:
            return None
        return self.execution.output

    @property
    def context(self) -> Any:
        """The final ``AgentContext``, when available."""
        if self.execution is None:
            return None
        return self.execution.context

    def tool_call_log(self) -> list[dict[str, Any]]:
        """Tool invocations recorded by the agent during the run."""
        ctx = self.context
        raw = getattr(ctx, "metadata", {}).get("tool_call_log", [])
        return [entry for entry in raw if isinstance(entry, dict)]


class Grader(ABC):
    """Base class for all graders."""

    name: str = "grader"

    @abstractmethod
    async def grade(self, request: GradingRequest) -> GraderResult:
        """Score one run. Must never raise."""


class TerminationGrader(Grader):
    """Passes iff the underlying run terminated with SUCCESS."""

    name = "termination"

    async def grade(self, request: GradingRequest) -> GraderResult:
        term = getattr(request.context, "termination", None)
        ok = term == AgentTermination.SUCCESS
        detail = f"termination={term.value if term else 'none'}"
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class OutputEqualsGrader(Grader):
    """Exact (optionally case-insensitive) equality against an expected string."""

    name = "output_equals"

    def __init__(self) -> None:
        self.expected: str = ""
        self.case_sensitive: bool = False

    def configure(self, config: dict[str, Any]) -> OutputEqualsGrader:
        self.expected = str(config.get("expected", ""))
        self.case_sensitive = bool(config.get("case_sensitive", False))
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        actual = _stringify(request.output)
        expected = self.expected
        if not self.case_sensitive:
            actual, expected = actual.lower(), expected.lower()
        ok = actual == expected
        detail = "" if ok else f"expected={expected!r} actual={actual!r}"
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class OutputContainsGrader(Grader):
    """Configured substrings must appear (all_of) / any must appear (any_of)."""

    name = "output_contains"

    def __init__(self) -> None:
        self.any_of: list[str] = []
        self.all_of: list[str] = []

    def configure(self, config: dict[str, Any]) -> OutputContainsGrader:
        self.any_of = [str(s) for s in config.get("any_of", [])]
        self.all_of = [str(s) for s in config.get("all_of", [])]
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        text = _stringify(request.output)
        missing_all = [s for s in self.all_of if s not in text]
        matched_any = not self.any_of or any(s in text for s in self.any_of)
        ok = not missing_all and matched_any
        detail = (
            "" if ok
            else f"missing={missing_all!r}" if missing_all
            else f"none of any_of={self.any_of!r} found"
        )
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class OutputRegexGrader(Grader):
    """Output must match a regex pattern."""

    name = "output_regex"

    def __init__(self) -> None:
        self.pattern: str = ""

    def configure(self, config: dict[str, Any]) -> OutputRegexGrader:
        self.pattern = str(config.get("pattern", ""))
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        try:
            ok = re.search(self.pattern, _stringify(request.output)) is not None
        except re.error as exc:
            return GraderResult(
                grader=self.name, score=0.0, passed=False,
                detail=f"invalid pattern: {exc}",
            )
        detail = "" if ok else f"pattern={self.pattern!r} did not match"
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class OutputJSONFieldGrader(Grader):
    """Parse the output as JSON and check one field's value.

    ``op`` selects the comparison: ``eq`` (default), ``gte``/``lte``/
    ``gt``/``lt`` for numeric thresholds. Non-eq comparisons let suites
    express minimum deliverable counts without over-constraining agents.
    """

    name = "output_json_field"

    _OPS: dict[str, Any] = {
        "eq": lambda a, b: a == b,
        "gte": lambda a, b: float(a) >= float(b),
        "lte": lambda a, b: float(a) <= float(b),
        "gt": lambda a, b: float(a) > float(b),
        "lt": lambda a, b: float(a) < float(b),
    }

    def __init__(self) -> None:
        self.field: str = ""
        self.expected: Any = None
        self.op: str = "eq"

    def configure(self, config: dict[str, Any]) -> OutputJSONFieldGrader:
        self.field = str(config.get("field", ""))
        self.expected = config.get("expected")
        op = str(config.get("op", "eq"))
        if op not in self._OPS:
            raise ValueError(f"unknown output_json_field op {op!r}")
        self.op = op
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        raw = request.output
        payload = raw if isinstance(raw, dict) else None
        if payload is None:
            try:
                loaded = json.loads(_stringify(raw))
                payload = loaded if isinstance(loaded, dict) else None
            except (json.JSONDecodeError, ValueError):
                payload = None
        if payload is None or not isinstance(payload, dict):
            return GraderResult(
                grader=self.name, score=0.0, passed=False,
                detail="output is not a JSON object",
            )
        actual = payload.get(self.field)
        compare = self._OPS[self.op]
        try:
            ok = bool(compare(actual, self.expected))
        except (TypeError, ValueError) as exc:
            ok = False
            detail = f"{self.field}={actual!r}: {exc}"
        else:
            detail = "" if ok else (
                f"{self.field}={actual!r} {self.op} {self.expected!r} failed"
            )
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class BudgetGrader(Grader):
    """Passes iff no harness error and no budget/timeout forced termination."""

    name = "budget"

    async def grade(self, request: GradingRequest) -> GraderResult:
        reason = str(getattr(request.context, "termination_reason", "") or "")
        forced = ("budget" in reason.lower()) or ("timeout" in reason.lower())
        ok = not request.harness_error and not forced
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok,
            detail="" if ok else (request.harness_error or reason),
        )


class MaxStepsGrader(Grader):
    """Step count must stay within a configured maximum."""

    name = "max_steps"

    def __init__(self) -> None:
        self.max_steps: int = 10

    def configure(self, config: dict[str, Any]) -> MaxStepsGrader:
        self.max_steps = max(1, int(config.get("max_steps", 10)))
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        steps = int(getattr(request.context, "current_step", 0))
        ok = steps <= self.max_steps
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok,
            detail=f"steps={steps} limit={self.max_steps}",
        )


class ToolUsageGrader(Grader):
    """Every invoked tool must be within the case's ``allowed_tools``."""

    name = "tool_usage"

    async def grade(self, request: GradingRequest) -> GraderResult:
        allowed = request.task.allowed_tools
        if allowed is None:
            return GraderResult(
                grader=self.name, score=1.0, passed=True, detail="unrestricted"
            )
        disallowed = sorted({
            str(entry.get("tool", "?"))
            for entry in request.tool_call_log()
            if str(entry.get("tool", "?")) not in allowed
        })
        ok = not disallowed
        detail = "" if ok else f"disallowed tools used: {disallowed}"
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


class RecoveryGrader(Grader):
    """Passes iff run succeeded; flags unrecovered fatal outcomes."""

    name = "recovery"

    async def grade(self, request: GradingRequest) -> GraderResult:
        ctx = request.context
        recoverable = int(getattr(ctx, "recoverable_errors", 0) or 0)
        success = bool(getattr(ctx, "is_success", lambda: False)())
        ok = success and not _fatal(request)
        detail = f"recoverable_errors={recoverable} success={success}"
        return GraderResult(
            grader=self.name, score=1.0 if ok else 0.0, passed=ok, detail=detail
        )


def _fatal(request: GradingRequest) -> bool:
    """True when the termination indicates an unrecovered failure."""
    reason = str(getattr(request.context, "termination_reason", "") or "")
    term = getattr(request.context, "termination", None)
    return term in (
        AgentTermination.ERROR,
        AgentTermination.CANCELLED,
        AgentTermination.NO_PROGRESS,
    ) or "fatal" in reason.lower()


def _stringify(value: Any) -> str:
    """Best-effort string form of an arbitrary output."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)


DETERMINISTIC_GRADERS: dict[str, type[Grader]] = {
    TerminationGrader.name: TerminationGrader,
    OutputEqualsGrader.name: OutputEqualsGrader,
    OutputContainsGrader.name: OutputContainsGrader,
    OutputRegexGrader.name: OutputRegexGrader,
    OutputJSONFieldGrader.name: OutputJSONFieldGrader,
    BudgetGrader.name: BudgetGrader,
    MaxStepsGrader.name: MaxStepsGrader,
    ToolUsageGrader.name: ToolUsageGrader,
    RecoveryGrader.name: RecoveryGrader,
}


class LLMPromptGrader(Grader):
    """LLM-based quality grader (opt-in only).

    Wraps an injected async ``score_fn(prompt, rubric) -> float|bool``
    (typically backed by :class:`~research_engineer.llm.base.LLMProvider`).
    The harness never constructs this by default: without a callable it
    fails loudly rather than pretending to grade.
    """

    name = "llm_quality"

    def __init__(
        self, score_fn: Any | None = None, *, rubric: str = "",
        threshold: float = 0.7,
    ) -> None:
        self._score_fn = score_fn
        self.rubric = rubric
        self.threshold = threshold

    def configure(self, config: dict[str, Any]) -> LLMPromptGrader:
        self.rubric = str(config.get("rubric", self.rubric))
        self.threshold = float(config.get("threshold", self.threshold))
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        if self._score_fn is None:
            return GraderResult(
                grader=self.name, score=0.0, passed=False,
                detail="no scoring function provided",
            )
        prompt = _stringify(request.output)
        try:
            raw = self._score_fn(prompt, self.rubric)
            if asyncio.iscoroutine(raw):
                raw = await raw
            if isinstance(raw, bool):
                score = 1.0 if raw else 0.0
            elif isinstance(raw, (int, float)):
                score = min(1.0, max(0.0, float(raw)))
            else:
                score = 0.0
        except Exception as exc:  # noqa: BLE001 - graders must not raise
            return GraderResult(
                grader=self.name, score=0.0, passed=False,
                detail=f"scoring function failed: {exc}",
            )
        ok = score >= self.threshold
        return GraderResult(
            grader=self.name, score=score, passed=ok,
            detail=f"llm_score={score:.3f} threshold={self.threshold}",
        )


class CompositeGrader(Grader):
    """Apply several graders and return their weighted mean as one result."""

    name = "composite"

    def __init__(self, graders: list[Grader] | None = None) -> None:
        self.graders = graders or []

    def add(self, grader: Grader) -> CompositeGrader:
        self.graders.append(grader)
        return self

    async def grade(self, request: GradingRequest) -> GraderResult:
        results = [await g.grade(request) for g in self.graders]
        total_weight = sum(r.weight for r in results) or 1.0
        weighted = sum(r.score * r.weight for r in results) / total_weight
        passed = all(r.passed for r in results)
        failed = [r.grader for r in results if not r.passed]
        detail = "" if passed else f"failed sub-graders: {failed}"
        return GraderResult(
            grader=self.name, score=weighted, passed=passed, detail=detail
        )


def build_grader(name: str, config: dict[str, Any] | None = None) -> Grader:
    """Instantiate a registered deterministic grader by name."""
    cls = DETERMINISTIC_GRADERS.get(name)
    if cls is None:
        raise KeyError(f"unknown grader {name!r}")
    grader = cls()
    if config and hasattr(grader, "configure"):
        grader.configure(config)
    return grader


__all__ = [
    "DETERMINISTIC_GRADERS",
    "BudgetGrader",
    "CompositeGrader",
    "Grader",
    "GradingRequest",
    "LLMPromptGrader",
    "MaxStepsGrader",
    "OutputContainsGrader",
    "OutputEqualsGrader",
    "OutputJSONFieldGrader",
    "OutputRegexGrader",
    "RecoveryGrader",
    "TerminationGrader",
    "ToolUsageGrader",
    "build_grader",
]
