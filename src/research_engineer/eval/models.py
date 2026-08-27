"""E4 - Agent Evaluation Harness models.

Typed Pydantic v2 models describing evaluation suites, cases, runs,
results, metrics, and reports. The harness drives *real*
:class:`~research_engineer.runtime.runtime.AgentRuntime` executions and
grades their outcomes, so every artifact is derived from runtime state
(contexts, terminations, tool calls) rather than duplicating it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from research_engineer.runtime.models import AgentBudget


class EvalStatus(StrEnum):
    """Lifecycle status of a single evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class SuccessCriterion(BaseModel):
    """One named grading criterion referencing a registered grader."""

    grader: str = Field(..., description="Registered grader identifier")
    weight: float = Field(default=1.0, gt=0.0, description="Criterion weight")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Grader-specific configuration overrides",
    )
    required: bool = Field(
        default=True,
        description="Whether failure of this criterion fails the whole case",
    )


class EvalTask(BaseModel):
    """A fully-specified evaluation task definition.

    This is the ``EvalCase`` payload loaded from YAML/JSON: goal,
    constraints, allowed tools, budget, success criteria, tags, and
    metadata. Cases carry a stable ``case_id`` and a ``revision`` so runs
    are reproducible and comparable across agent/runtime versions.
    """

    case_id: str = Field(
        ..., min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
        description="Stable, unique case identifier",
    )
    name: str = Field(..., description="Human-readable case name")
    description: str = Field(default="", description="What this case evaluates")
    goal: str = Field(..., description="Goal handed to the agent runtime")
    constraints: list[str] = Field(
        default_factory=list, description="Constraints the agent must respect"
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Tools the agent may use; None means unrestricted",
    )
    budget: AgentBudget = Field(
        default_factory=AgentBudget,
        description="Resource budgets enforced by the runtime",
    )
    criteria: list[SuccessCriterion] = Field(
        default_factory=list,
        description="Named grading criteria evaluated after the run",
    )
    require_runtime_success: bool = Field(
        default=True,
        description=(
            "Require AgentTermination.SUCCESS in addition to passing graders"
        ),
    )
    timeout_seconds: float | None = Field(
        default=None, gt=0.0,
        description="Wall-clock cap for the evaluation attempt itself",
    )
    tags: list[str] = Field(default_factory=list)
    revision: str = Field(
        default="r1", description="Case revision for reproducible comparisons"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata; scripted tasks put their script here",
    )

    def stable_run_id(self, suite_id: str, seed: int = 0) -> str:
        """Deterministic run id for this case within a suite."""
        return f"{suite_id}::{self.case_id}@{self.revision}#seed{seed}"


class CaseMetrics(BaseModel):
    """Quantitative measurements collected for one case."""

    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    recoverable_errors: int = Field(default=0, ge=0)
    fatal_errors: int = Field(default=0, ge=0)
    recovered: bool = Field(
        default=False,
        description="True when errors occurred yet the run still succeeded",
    )
    human_interventions: int = Field(default=0, ge=0)
    terminated: bool = Field(default=False)
    termination_reason: str = Field(
        default="", description="AgentTermination value of the underlying run"
    )


class EvalMetric(BaseModel):
    """A single aggregated scalar metric for reporting/comparison."""

    name: str
    value: float
    description: str = ""


class SuiteMetrics(BaseModel):
    """Aggregate metrics across the results of one suite."""

    cases_total: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    completion_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of cases whose underlying run terminated at all",
    )
    avg_steps: float = Field(default=0.0, ge=0.0)
    avg_tool_calls: float = Field(default=0.0, ge=0.0)
    total_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_latency_seconds: float = Field(default=0.0, ge=0.0)
    p95_latency_seconds: float = Field(default=0.0, ge=0.0)
    failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of error-affected runs that still succeeded",
    )
    human_intervention_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    termination_reasons: dict[str, int] = Field(default_factory=dict)

    def as_metrics(self) -> list[EvalMetric]:
        """Flatten into simple :class:`EvalMetric` scalars."""
        fields = self.model_dump(exclude={"termination_reasons"})
        items = [
            EvalMetric(name=k, value=float(v))
            for k, v in fields.items()
            if isinstance(v, (int, float)) and k != "cases_total"
        ]
        for reason, count in sorted(self.termination_reasons.items()):
            items.append(
                EvalMetric(
                    name=f"termination.{reason}", value=float(count),
                    description="Terminations with this reason",
                )
            )
        return items

class GraderResult(BaseModel):
    """Outcome of applying one grader.

    ``error_kind`` distinguishes *evaluator* malfunctions from agent
    failures: an empty string means the grader produced a valid verdict,
    while ``"judge_error"`` means the grader itself failed (e.g. an
    unparseable LLM-judge reply). Judge-error results are NEVER counted
    as a zero score against the agent — aggregation excludes them.
    """

    grader: str = Field(..., description="Grader identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Score in [0, 1]")
    passed: bool
    detail: str = ""
    weight: float = Field(default=1.0, gt=0.0)
    error_kind: str = Field(
        default="",
        description=(
            '"" when the verdict is valid; "judge_error" when the '
            "grader itself malfunctioned (excluded from aggregation)"
        ),
    )

    @property
    def is_judge_error(self) -> bool:
        return self.error_kind == "judge_error"


class EvalResult(BaseModel):
    """The graded outcome of running one case once."""

    run_id: str = Field(..., description="Deterministic/stable run id")
    case_id: str
    revision: str = Field(default="r1")
    status: EvalStatus = Field(default=EvalStatus.PENDING)
    success: bool = Field(
        default=False,
        description="Overall pass/fail combining runtime success + criteria",
    )
    completion: bool = Field(
        default=False,
        description="Whether the underlying run reached a terminal state",
    )
    grader_results: list[GraderResult] = Field(default_factory=list)
    weighted_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted mean grader score"
    )
    metrics: CaseMetrics = Field(default_factory=CaseMetrics)
    output: Any = Field(
        default=None, description="Final output produced by the runtime"
    )
    error: str = Field(default="", description="Harness-level error, if any")
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = Field(default=None)


class EvalSuite(BaseModel):
    """An ordered collection of :class:`EvalTask` cases."""

    suite_id: str = Field(
        ..., min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$",
        description="Stable suite identifier",
    )
    name: str = Field(default="", description="Human-readable suite name")
    description: str = Field(default="")
    version: str = Field(default="1")
    cases: list[EvalTask] = Field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]


class EvalReport(BaseModel):
    """Per-case results plus aggregate metrics for one suite execution."""

    report_id: str = Field(default_factory=lambda: f"eval_{uuid4().hex[:12]}")
    suite_id: str
    suite_version: str = Field(default="1")
    label: str = Field(
        default="", description="Label of the evaluated configuration"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    results: list[EvalResult] = Field(default_factory=list)
    aggregate: SuiteMetrics = Field(default_factory=SuiteMetrics)

    @property
    def case_ids(self) -> list[str]:
        return [r.case_id for r in self.results]


__all__ = [
    "EvalStatus",
    "SuccessCriterion",
    "EvalTask",
    "CaseMetrics",
    "EvalMetric",
    "SuiteMetrics",
    "GraderResult",
    "EvalResult",
    "EvalSuite",
    "EvalReport",
]

