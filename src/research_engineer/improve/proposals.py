"""E8 - Improvement candidate generation with hard guardrails.

Proposal generation is **pluggable**: anything implementing
:class:`ProposalSource` may emit :class:`ImprovementProposal` objects
(a deterministic rule engine ships by default; an LLM-backed generator can be
added the same way). Whatever the source, a proposal stays a pure
data/configuration artifact naming one declared component surface plus JSON
changes — never code.

``validate_changes`` enforces the mandatory safety constraints at candidate
creation time:

* only allow-listed configuration keys per component;
* autonomy/safety controls may never be disabled or have their enforcement
  action changed;
* safety-relevant thresholds may only move in the *restrictive* direction
  relative to the baseline configuration snapshot.

E3 policy enforcement and E5 safety policies remain authoritative outside
this pipeline; candidates merely tune typed configuration surfaces.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from research_engineer.improve.models import (
    CandidateComponent,
    FailurePattern,
    PatternKind,
    content_hash,
)

if TYPE_CHECKING:
    from research_engineer.improve.models import (
        Baseline,
        ImprovementCandidate,
        ImprovementProposal,
    )


class InvalidProposalError(ValueError):
    """A proposal violates component allow-lists or safety guardrails."""


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

#: JSON scalar keys each non-safety component may set (regexes over dotted keys).
_COMPONENT_KEY_ALLOWLIST: dict[CandidateComponent, tuple[str, ...]] = {
    CandidateComponent.SYSTEM_PROMPT: (r"^append$", r"^prepend$"),
    CandidateComponent.PLANNING_PARAMS: (
        r"^budget\.max_steps$",
        r"^budget\.max_tool_calls$",
        r"^budget\.max_tokens$",
        r"^policy\.max_recoverable_errors$",
        r"^policy\.stagnation_window$",
        r"^policy\.progress_threshold$",
        r"^planner\.[a-z_]+$",
    ),
    CandidateComponent.AUTONOMY_THRESHOLDS: (),
    CandidateComponent.SAFETY_POLICY_CONFIG: (),
    CandidateComponent.TOOL_SELECTION_STRATEGY: (r"^tools_strategy\.[a-z_]+$",),
    CandidateComponent.MODEL_PROVIDER_CONFIG: (r"^model_provider\.[a-z_.]+$",),
}

#: Autonomy/safety keys whose values may ONLY become more restrictive than
#: the baseline value ("restrictive" defined per key).
_SAFETY_TIGHTEN_ONLY: dict[str, str] = {
    # Smaller => triggers earlier / intervenes sooner (more restrictive).
    "autonomy.loop_threshold": "<=",
    "autonomy.duplicate_max_identical": "<=",
    "autonomy.no_progress_stagnation_limit": "<=",
    "autonomy.failure_consecutive_limit": "<=",
    "autonomy.max_replans": "<=",
    "autonomy.diminishing_threshold": "<=",
    # Larger window/more frequent warnings => more scrutiny (not weaker).
    "autonomy.loop_window": ">=",  # analysis window only, free direction
    "autonomy.duplicate_window": ">=",
    "autonomy.budget_warning_fraction": ">=",  # warn earlier
}

_FORBIDDEN_KEY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"(^|\.)\w*enabled$",         # controls can never be switched off
        r".*_action$",                # enforcement actions can never change
        r"(^|\.)mandatory$",          # mandatory decisions stay mandatory
        r"(^|\.)max_recoverable_errors$",  # E1 error tolerance is out of scope
    )
)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


_SAFETY_COMPONENTS = frozenset(
    {CandidateComponent.AUTONOMY_THRESHOLDS,
     CandidateComponent.SAFETY_POLICY_CONFIG}
)


def _problems_forbidden_keys(
    label: str, key: str,
) -> list[str]:
    """Forbidden-key violations for one change entry."""
    for pattern in _FORBIDDEN_KEY_PATTERNS:
        if pattern.search(key):
            return [f"{label}: modifying this key is forbidden"]
    return []


def _problems_safety_change(
    label: str, key: str, value: Any, current: dict[str, Any],
) -> list[str]:
    """Directional tightening-only checks for autonomy/safety knobs."""
    direction = _SAFETY_TIGHTEN_ONLY.get(key)
    if direction is None:
        return [f"{label}: not a recognized autonomy/safety knob"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return [f"{label}: must be a number"]
    base = current.get(key)
    if base is None:
        ok = value >= 0
    elif direction == "<=":
        ok = value <= base
    else:
        ok = value >= base
    if not ok:
        return [
            f"{label}: loosens safety "
            f"(baseline {base}, requires {direction} baseline)",
        ]
    return []


def _problems_plain_change(
    component: CandidateComponent, label: str, key: str, value: Any,
) -> list[str]:
    """Allow-list / scalar checks for non-safety components."""
    if component is CandidateComponent.SYSTEM_PROMPT and not isinstance(
        value, str,
    ):
        return [f"{label}: prompt patch must be a string"]
    allowed = _COMPONENT_KEY_ALLOWLIST.get(component, ())
    if not any(re.match(m, key) for m in allowed):
        return [f"{label}: key not allowed for this component"]
    if component is not CandidateComponent.SYSTEM_PROMPT and not _is_scalar(
        value,
    ):
        return [f"{label}: values must be JSON scalars"]
    return []


def validate_changes(
    component: CandidateComponent,
    changes: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Validate ``changes`` against allow-lists and directional guards.

    Args:
        component: the declared configuration surface being retuned.
        changes: proposed dotted-key -> value patch.
        current: baseline configuration snapshot (dotted-key -> value).

    Raises:
        InvalidProposalError: listing every violated rule.
    """
    problems: list[str] = []
    for key, value in sorted(changes.items()):
        label = f"{component.value}:{key}={value!r}"
        problems.extend(_problems_forbidden_keys(label, key))
        if component in _SAFETY_COMPONENTS:
            problems.extend(_problems_safety_change(label, key, value, current))
        else:
            problems.extend(
                _problems_plain_change(component, label, key, value),
            )

    if problems:
        raise InvalidProposalError("; ".join(problems))


# ---------------------------------------------------------------------------
# Pluggable proposal sources
# ---------------------------------------------------------------------------


class ProposalSource(Protocol):
    """Anything that turns mined failure patterns into proposals."""

    @property
    def name(self) -> str: ...

    def propose(
        self,
        patterns: list[FailurePattern],
        baseline: Baseline,
    ) -> list[ImprovementProposal]: ...


def _by_kind(patterns: list[FailurePattern]) -> dict[PatternKind, list[FailurePattern]]:
    grouped: dict[PatternKind, list[FailurePattern]] = {}
    for p in patterns:
        grouped.setdefault(p.kind, []).append(p)
    return grouped


class RuleBasedProposalSource:
    """Deterministic pattern -> proposal recipes.

    Each recipe maps one recurring problem class onto a conservative,
    data-only configuration change grounded in the baseline snapshot.
    Patterns without a confident deterministic fix (e.g. excessive latency)
    deliberately yield NO proposal — those go to human/LLM proposers instead.
    """

    #: Default used when the baseline snapshot lacks an explicit budget entry.
    default_budget_steps = 8
    default_budget_tool_calls = 12

    @property
    def name(self) -> str:
        return "rule_based"

    def propose(
        self,
        patterns: list[FailurePattern],
        baseline: Baseline,
    ) -> list[ImprovementProposal]:
        from research_engineer.improve.models import ImprovementProposal

        grouped = _by_kind(patterns)
        current = baseline.config_snapshot
        out: list[ImprovementProposal] = []

        if grouped.get(PatternKind.BUDGET_EXHAUSTION):
            steps = int(current.get("budget.max_steps", self.default_budget_steps))
            tools = int(
                current.get("budget.max_tool_calls", self.default_budget_tool_calls)
            )
            new_steps, new_tools = steps + max(1, steps // 2), tools + 4
            out.append(ImprovementProposal(
                title="Raise planning budgets to cut budget-exhausted runs",
                description=(
                    f"Mined BUDGET_EXHAUSTION failures; increasing "
                    f"max_steps {steps}->{new_steps} and "
                    f"max_tool_calls {tools}->{new_tools}"
                ),
                component=CandidateComponent.PLANNING_PARAMS,
                changes={
                    "budget.max_steps": new_steps,
                    "budget.max_tool_calls": new_tools,
                },
                evidence=[p.pattern_id for p in grouped[PatternKind.BUDGET_EXHAUSTION]],
                source=self.name,
            ))

        if grouped.get(PatternKind.POOR_EVALUATION_SCORE):
            out.append(ImprovementProposal(
                title="Append task-focus guidance to the system prompt",
                description=(
                    "Mined persistent poor scores; append explicit "
                    "verify-before-final-answer policy text."
                ),
                component=CandidateComponent.SYSTEM_PROMPT,
                changes={
                    "append": (
                        "Before finishing, verify your answer satisfies every "
                        "stated constraint; if not, continue working."
                    )
                },
                evidence=[
                    p.pattern_id for p in grouped[PatternKind.POOR_EVALUATION_SCORE]
                ],
                source=self.name,
            ))

        loop_pat = grouped.get(PatternKind.REPEATED_REPLANNING, [])
        if loop_pat:
            key = "autonomy.loop_threshold"
            cur = int(current.get(key, 3))
            out.append(ImprovementProposal(
                title="Detect loops one iteration earlier",
                description=f"Mined replanning churn; tighten {key} {cur}->{max(2, cur - 1)}.",
                component=CandidateComponent.AUTONOMY_THRESHOLDS,
                changes={key: max(2, cur - 1)},
                evidence=[p.pattern_id for p in loop_pat],
                source=self.name,
            ))

        dup_pat = (
            grouped.get(PatternKind.UNNECESSARY_TOOL_CALLS, [])
            + grouped.get(PatternKind.HUMAN_APPROVAL_FREQUENCY, [])
        )
        if dup_pat:
            out.append(ImprovementProposal(
                title="Dedupe identical tool calls in the selection strategy",
                description=(
                    "Mined redundant calls/approval churn; enable identical-call "
                    "memoization in the tool-selection strategy."
                ),
                component=CandidateComponent.TOOL_SELECTION_STRATEGY,
                changes={"tools_strategy.memoize_identical_calls": True},
                evidence=sorted({p.pattern_id for p in dup_pat}),
                source=self.name,
            ))

        return out


# ---------------------------------------------------------------------------
# Applying candidate configuration to an eval task (data-only)
# ---------------------------------------------------------------------------


def apply_candidate_changes(task: Any, candidate: ImprovementCandidate) -> Any:
    """Return ``task`` (an ``EvalTask``) with the candidate's changes applied.

    Pure data transformation: budget keys retune :attr:`EvalTask.budget`,
    prompt patches extend the goal text, tool-strategy flags dedupe the
    scripted tool plan, and any other allowed keys are recorded under the
    task metadata so downstream factories/controllers can honor them.
    Keys whose surface the harness cannot exercise are inert here by design
    (they still flow through to production configuration consumers).
    """

    updated = task.model_copy(deep=True)
    for key, value in sorted(candidate.changes.items()):
        if key == "budget.max_steps":
            updated.budget.max_steps = int(value)
        elif key == "budget.max_tool_calls":
            updated.budget.max_tool_calls = int(value)
        elif key == "budget.max_tokens":
            updated.budget.max_tokens = int(value)
        elif key == "policy.stagnation_window":
            updated.metadata.setdefault("script", {})["stagnation_window"] = value
        elif key in {"append", "prepend"}:
            patch = str(value)
            updated.goal = (
                f"{updated.goal}\n{patch}" if key == "append" else f"{patch}\n{updated.goal}"
            )
        elif key == "tools_strategy.memoize_identical_calls" and value:
            script = updated.metadata.setdefault("script", {})
            seen: set[tuple[Any, ...]] = set()
            kept: list[dict[str, Any]] = []
            for spec in script.get("tools", []):
                identity = tuple(sorted(spec.items()))
                if identity in seen:
                    continue
                seen.add(identity)
                kept.append(spec)
            script["tools"] = kept
        else:
            bucket = key.split(".", 1)[0]
            updated.metadata.setdefault(f"candidate_{bucket}", {})[key] = value
    return updated


def candidate_eval_runner_builder(
    base_factory_name: str = "scripted",
) -> Callable[[ImprovementCandidate], Callable[[Any], Any]]:
    """Build an ``AgentFactory`` provider honoring candidate configuration."""
    from research_engineer.eval.scripted import resolve_agent_factory

    base = resolve_agent_factory(base_factory_name)

    def builder(candidate: ImprovementCandidate) -> Callable[[Any], Any]:
        def factory(task: Any) -> Any:
            return base.build(apply_candidate_changes(task, candidate))

        return factory

    return builder


# ---------------------------------------------------------------------------
# Candidate construction (versioned / immutable / reproducible)
# ---------------------------------------------------------------------------


def build_candidate(
    proposal: ImprovementProposal,
    baseline: Baseline,
    *,
    suite_id: str,
    suite_version: str,
) -> ImprovementCandidate:
    """Build an immutable candidate with a deterministic identity.

    The id/hash depend only on content + parentage (never timestamps), so
    re-proposing the identical change against the same baseline reproduces
    the exact same candidate identifier.
    """
    from research_engineer.improve.models import ImprovementCandidate

    config_payload = [proposal.component.value, proposal.changes]
    config_hash = content_hash(config_payload)
    identity_payload = [
        proposal.component.value,
        proposal.changes,
        baseline.baseline_id,
        baseline.config_hash,
        suite_id,
        suite_version,
    ]
    full_hash = content_hash(identity_payload)
    return ImprovementCandidate(
        candidate_id=f"cand_{full_hash[:12]}",
        version=config_hash[:12],
        component=proposal.component,
        changes=dict(proposal.changes),
        config_hash=config_hash,
        parent_baseline_id=baseline.baseline_id,
        parent_config_hash=baseline.config_hash,
        suite_id=suite_id,
        suite_version=suite_version,
        proposal_source=proposal.source,
        evidence=list(proposal.evidence),
        metadata={"proposal_title": proposal.title},
    )


__all__ = [
    "InvalidProposalError",
    "ProposalSource",
    "RuleBasedProposalSource",
    "apply_candidate_changes",
    "build_candidate",
    "candidate_eval_runner_builder",
    "validate_changes",
]
