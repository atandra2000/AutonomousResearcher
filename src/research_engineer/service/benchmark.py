"""P1 §2 - Research benchmark suite loading and domain validation.

The benchmark itself is an ordinary E4 :class:`EvalSuite`
(``evals/research_benchmark/v1/suite.yaml``) so no competing abstraction is
introduced; this module adds only the platform-specific validation layer:
known agent kinds, known graders, category coverage, bounded budgets, and
well-formed artifact expectations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_engineer.eval.graders import DETERMINISTIC_GRADERS
from research_engineer.eval.models import EvalSuite
from research_engineer.eval.runner import load_suite
from research_engineer.service.bench_agents import (
    KIND_BENCH_FLAKY,
    KIND_BENCH_LOOP,
    KIND_BENCH_RISKY,
    KIND_BENCH_TOOL,
)
from research_engineer.service.llm_agent import KIND_LLM_REACT

#: Packaged v1 benchmark suite.
DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals" / "research_benchmark" / "v1" / "suite.yaml"
)

#: Packaged v2 (LLM-backed) benchmark suite.
DEFAULT_SUITE_V2_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals" / "research_benchmark" / "v2" / "suite.yaml"
)

#: The eight research categories the benchmark must cover.
BENCHMARK_CATEGORIES: frozenset[str] = frozenset({
    "literature_discovery",
    "hypothesis_generation",
    "experiment_design",
    "implementation",
    "debugging",
    "ablations",
    "experiment_analysis",
    "end_to_end",
})

#: Agent kinds the worker can execute for benchmark cases.
BENCHMARK_AGENT_KINDS: frozenset[str] = frozenset({
    KIND_BENCH_TOOL, KIND_BENCH_FLAKY, KIND_BENCH_LOOP, KIND_BENCH_RISKY,
    KIND_LLM_REACT,
})

#: Scenario modes a case may declare.
BENCHMARK_MODES: frozenset[str] = frozenset({
    "deterministic_sandbox", "transient_recovery", "policy_guardrail",
    "llm_agent",
})

#: Graders that consult an LLM judge; only valid inside ``llm_agent``
#: suites and always optional (never gate objective success).
LLM_JUDGED_GRADERS: frozenset[str] = frozenset({"llm_quality"})


def load_benchmark_suite(
    path: str | Path | None = None,
) -> EvalSuite:
    """Load and validate the versioned benchmark suite.

    Raises :class:`ValueError` with every structural/domain problem joined
    when the suite does not satisfy the benchmark contract.
    """
    suite = load_suite(Path(path) if path else DEFAULT_SUITE_PATH)
    problems = validate_benchmark_suite(suite)
    if problems:
        raise ValueError(
            f"benchmark suite {suite.suite_id} invalid:\n- "
            + "\n- ".join(problems)
        )
    return suite


def validate_benchmark_suite(suite: EvalSuite) -> list[str]:
    """Return a list of contract violations (empty when valid)."""
    if not suite.cases:
        return ["suite has no cases"]

    problems: list[str] = []
    seen_ids: set[str] = set()
    covered_categories: set[str] = set()
    for case in suite.cases:
        if case.case_id in seen_ids:
            problems.append(f"{case.case_id}: duplicate case_id")
        seen_ids.add(case.case_id)
        meta = case.metadata or {}
        if isinstance(meta.get("category"), str) and \
                meta["category"] in BENCHMARK_CATEGORIES:
            covered_categories.add(meta["category"])
        problems.extend(_validate_case(case))

    missing = BENCHMARK_CATEGORIES - covered_categories
    if missing:
        problems.append(
            f"categories not covered: {sorted(missing)}"
        )
    return problems


def _validate_case(case: Any) -> list[str]:
    """Contract violations for one case definition."""
    cid = case.case_id
    problems: list[str] = []
    meta = case.metadata or {}

    category = str(meta.get("category", ""))
    if category not in BENCHMARK_CATEGORIES:
        problems.append(f"{cid}: unknown category {category!r}")
    mode = str(meta.get("mode", ""))
    if mode not in BENCHMARK_MODES:
        problems.append(f"{cid}: unknown mode {mode!r}")
    kind = str(meta.get("agent_kind", ""))
    if kind not in BENCHMARK_AGENT_KINDS:
        problems.append(f"{cid}: unknown agent_kind {kind!r}")

    overrides = meta.get("agent_overrides", {})
    if not isinstance(overrides, dict):
        problems.append(f"{cid}: agent_overrides must be a mapping")

    # Unbounded runs would corrupt budget metrics.
    if case.budget.max_steps is None:
        problems.append(f"{cid}: budget.max_steps must be finite")

    problems.extend(_validate_artifacts(cid, meta))
    required = [c for c in case.criteria if c.required]
    if not required:
        problems.append(f"{cid}: at least one required criterion")

    for criterion in case.criteria:
        problem = _validate_criterion(
            cid, criterion, case.budget.max_steps, mode
        )
        if problem:
            problems.append(problem)
    return problems


def _validate_artifacts(cid: str, meta: dict) -> list[str]:
    """Validate the expected_artifacts declaration of one case."""
    artifacts = meta.get("expected_artifacts", [])
    if not isinstance(artifacts, list):
        return [f"{cid}: expected_artifacts must be a list"]
    for artifact in artifacts:
        if (not isinstance(artifact, dict)
                or "type" not in artifact
                or "min_count" not in artifact):
            return [f"{cid}: artifact entries need type+min_count"]
    return []


def _validate_criterion(
    cid: str, criterion: Any, declared_max_steps: int | None,
    mode: str = "",
) -> str | None:
    """Validate one success criterion; return a problem string or None."""
    judged = criterion.grader in LLM_JUDGED_GRADERS
    if not judged and criterion.grader not in DETERMINISTIC_GRADERS:
        return f"{cid}: unknown grader {criterion.grader!r}"
    if judged:
        # Evaluation integrity: LLM-judged quality scores are reported
        # separately and never gate objective task success.
        if mode != "llm_agent":
            return (f"{cid}: grader {criterion.grader!r} is only valid "
                    f"in llm_agent-mode cases")
        if criterion.required:
            return (f"{cid}: LLM-judged grader {criterion.grader!r} must "
                    f"be optional (required=false)")
    if criterion.grader == "max_steps":
        configured = int(criterion.config.get("max_steps", 0))
        if declared_max_steps is not None and configured != declared_max_steps:
            return (
                f"{cid}: max_steps criterion {configured} must "
                f"match budget {declared_max_steps}"
            )
    return None


__all__ = [
    "BENCHMARK_AGENT_KINDS",
    "BENCHMARK_CATEGORIES",
    "BENCHMARK_MODES",
    "DEFAULT_SUITE_PATH",
    "DEFAULT_SUITE_V2_PATH",
    "LLM_JUDGED_GRADERS",
    "load_benchmark_suite",
    "validate_benchmark_suite",
]
