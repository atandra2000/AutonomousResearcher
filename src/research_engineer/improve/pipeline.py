"""E8 - Improvement pipeline: lifecycle, store, gate, promotion, canary.

Orchestrates the controlled improvement loop::

    Runs/Telemetry -> Failure & Pattern Mining -> Improvement Candidate
        -> E4 Evaluation Suite -> Baseline vs Candidate -> Regression Gate
            -> reject | recommend | approve/canary

Guarantees:

* explicit, enforced lifecycle transitions (:data:`LIFECYCLE_TRANSITIONS`);
* candidates are immutable records whose identity is a pure function of
  content + parentage (reproducible by construction);
* production promotion is a *decision only*: it never deploys anything and
  requires explicit ``approver`` input; every promotion stores the previous
  active version so rollback restores it exactly;
* all events flow through the existing E6 :class:`EventBus` (kind
  ``agent_improvement``) — no second telemetry system;
* no autonomous self-modification: candidates touch configuration surfaces
  validated in :mod:`research_engineer.improve.proposals`, source code is
  never touched, hard safety limits can only tighten.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from research_engineer.eval.models import EvalReport
from research_engineer.improve.gate import evaluate_gate, snapshot_from_report
from research_engineer.improve.models import (
    Baseline,
    CandidateEvaluation,
    CandidateStatus,
    GateVerdict,
    ImprovementCandidate,
    PromotionAction,
    PromotionDecision,
    RegressionGate,
    content_hash,
)
from research_engineer.improve.models import (
    ImprovementProposal as ProposalModel,
)
from research_engineer.improve.proposals import (
    build_candidate,
    validate_changes,
)

# ---------------------------------------------------------------------------
# Lifecycle state machine
# ---------------------------------------------------------------------------

#: Explicit transition table; anything not listed here is illegal.
LIFECYCLE_TRANSITIONS: dict[CandidateStatus, frozenset[CandidateStatus]] = {
    CandidateStatus.PROPOSED: frozenset(
        {CandidateStatus.EVALUATING, CandidateStatus.REJECTED}
    ),
    CandidateStatus.EVALUATING: frozenset(
        {CandidateStatus.PASSED, CandidateStatus.FAILED}
    ),
    CandidateStatus.PASSED: frozenset(
        {CandidateStatus.APPROVED, CandidateStatus.REJECTED}
    ),
    CandidateStatus.FAILED: frozenset(
        {CandidateStatus.PROPOSED, CandidateStatus.EVALUATING}
    ),
    CandidateStatus.APPROVED: frozenset(
        {
            CandidateStatus.CANARY,
            CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
        }
    ),
    CandidateStatus.CANARY: frozenset(
        {CandidateStatus.PROMOTED, CandidateStatus.REJECTED}
    ),
    # PROMOTED may only move back via an explicit ROLLED_BACK decision.
    CandidateStatus.PROMOTED: frozenset({CandidateStatus.REJECTED}),
    CandidateStatus.REJECTED: frozenset(),
}


class IllegalTransitionError(ValueError):
    """Raised when a lifecycle move is not in the transition table."""


class ApprovalRequiredError(PermissionError):
    """Raised when promotion is attempted without explicit approval."""


def transition_allowed(src: CandidateStatus, dst: CandidateStatus) -> bool:
    """True when the state machine permits ``src -> dst``."""
    return dst in LIFECYCLE_TRANSITIONS.get(src, frozenset())


# ---------------------------------------------------------------------------
# Store (atomic JSON artifacts under one root directory)
# ---------------------------------------------------------------------------


class ImprovementStore:
    """Filesystem persistence for baselines, candidates, decisions, pointers.

    Layout (all writes atomic via ``.tmp`` + ``os.rename``)::

        <root>/baselines/<baseline_id>.json
        <root>/candidates/<candidate_id>.json
        <root>/decisions.jsonl          (append-only audit log)
        <root>/active.json              per-component active/previous pointers
    """

    def __init__(self, root: str | Path = "output/improvements") -> None:
        self.root = Path(root)
        self._candidates_dir = self.root / "candidates"
        self._baselines_dir = self.root / "baselines"
        self._decisions_path = self.root / "decisions.jsonl"
        self._active_path = self.root / "active.json"

    # -- atomic write helper -------------------------------------------------
    def _write_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.rename(tmp, path)

    # -- baselines -------------------------------------------------------------
    def save_baseline(self, baseline: Baseline) -> None:
        self._write_atomic(
            self._baselines_dir / f"{baseline.baseline_id}.json",
            baseline.model_dump_json(indent=2),
        )

    def load_baseline(self, baseline_id: str) -> Baseline | None:
        path = self._baselines_dir / f"{baseline_id}.json"
        if not path.exists():
            return None
        return Baseline.model_validate_json(path.read_text(encoding="utf-8"))

    def list_baselines(self) -> list[Baseline]:
        out: list[Baseline] = []
        if self._baselines_dir.exists():
            for p in sorted(self._baselines_dir.glob("*.json")):
                out.append(Baseline.model_validate_json(p.read_text("utf-8")))
        return sorted(out, key=lambda b: b.created_at)

    # -- candidates --------------------------------------------------------------
    def save_candidate(self, candidate: ImprovementCandidate) -> None:
        self._write_atomic(
            self._candidates_dir / f"{candidate.candidate_id}.json",
            candidate.model_dump_json(indent=2),
        )

    def load_candidate(self, candidate_id: str) -> ImprovementCandidate | None:
        path = self._candidates_dir / f"{candidate_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ImprovementCandidate.model_validate(data)

    def list_candidates(self) -> list[ImprovementCandidate]:
        out: list[ImprovementCandidate] = []
        if self._candidates_dir.exists():
            for p in sorted(self._candidates_dir.glob("*.json")):
                out.append(ImprovementCandidate.model_validate_json(p.read_text("utf-8")))
        return sorted(out, key=lambda c: c.created_at)

    # -- decisions ---------------------------------------------------------------
    def record_decision(self, candidate_id: str, decision: PromotionDecision) -> None:
        entry = {"candidate_id": candidate_id} | json.loads(
            decision.model_dump_json()
        )
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self._decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def list_decisions(self) -> list[dict[str, Any]]:
        if not self._decisions_path.exists():
            return []
        lines = self._decisions_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    # -- active version pointers --------------------------------------------------
    def _load_active(self) -> dict[str, dict[str, str]]:
        if not self._active_path.exists():
            return {}
        return cast(
            "dict[str, dict[str, str]]",
            json.loads(self._active_path.read_text(encoding="utf-8")),
        )

    def _save_active(self, table: dict[str, dict[str, str]]) -> None:
        self._write_atomic(self._active_path, json.dumps(table, indent=2))

    def set_active(self, component: str, candidate_id: str) -> None:
        """Point ``component`` at ``candidate_id``, remembering the previous."""
        table = self._load_active()
        slot = table.setdefault(component, {})
        # No prior pointer => production ran the baseline default ("").
        slot["previous"] = slot.get("active", "")
        slot["active"] = candidate_id
        self._save_active(table)

    def get_active(self, component: str) -> str | None:
        return self._load_active().get(component, {}).get("active") or None

    def restore_previous(self, component: str) -> str | None:
        """Swap back to the recorded previous pointer; returns it if any.

        Returns ``""`` when production reverts to the baseline default,
        ``None`` when nothing was ever promoted for this component.
        """
        table = self._load_active()
        slot = table.get(component)
        if not slot:
            return None
        current = slot.get("active", "")
        previous = slot.get("previous", "")
        if not current or current == previous:
            return None
        return self.revert_to(component, previous)

    def revert_to(self, component: str, target: str) -> str | None:
        """Point ``component`` at ``target`` (may be ``""`` = baseline).

        Returns the restored target id, or ``None`` when there was no
        pointer or it already matched ``target``.
        """
        table = self._load_active()
        slot = table.get(component)
        if not slot:
            return None
        current = slot.get("active", "")
        if current == target:
            return None
        if not target:
            # Reverting to the baseline default drops the dangling entries.
            del table[component]
        else:
            slot["previous"] = current
            slot["active"] = target
        self._save_active(table)
        return target


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


#: Builds an E4 runner/suite invocation for this exact candidate.
EvalFn = Callable[[ImprovementCandidate], Awaitable[EvalReport]]


class ImprovementPipeline:
    """Facade over mining output -> candidates -> eval -> gate -> promotion.

    Args:
        store: :class:`ImprovementStore` root directory.
        gate: regression-gate configuration (defaults are strict).
        event_bus: optional E6 bus override; defaults to the process-wide
            :func:`research_engineer.observability.get_event_bus`.
    """

    def __init__(
        self,
        store: ImprovementStore | None = None,
        *,
        gate: RegressionGate | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.store = store or ImprovementStore()
        self.gate = gate or RegressionGate()
        self._event_bus = event_bus

    # -- telemetry (existing E6 bus only) -------------------------------------
    def _emit(self, event: str, **fields: Any) -> None:
        from research_engineer.observability import get_event_bus

        payload = {"kind": "agent_improvement", "event": event, **fields}
        bus = self._event_bus if self._event_bus is not None else get_event_bus()
        try:
            bus.emit(payload)
        except Exception:  # noqa: BLE001 - telemetry must never break the loop
            pass

    # -- baselines --------------------------------------------------------------
    def register_baseline(
        self,
        label: str,
        report: EvalReport,
        config_snapshot: dict[str, Any],
        *,
        suite_version: str = "1",
    ) -> Baseline:
        """Record production configuration + its current eval metrics."""
        metrics = snapshot_from_report(report)
        baseline_id = (
            "base_"
            + content_hash([label, report.suite_id, config_snapshot])[:12]
        )
        existing = self.store.load_baseline(baseline_id)
        if existing is not None:
            return existing
        baseline = Baseline(
            baseline_id=baseline_id,
            label=label,
            suite_id=report.suite_id,
            suite_version=suite_version,
            config_snapshot=dict(config_snapshot),
            config_hash=content_hash(config_snapshot),
            metrics=metrics,
            report_id=report.report_id,
        )
        self.store.save_baseline(baseline)
        self._emit(
            "baseline_registered",
            baseline_id=baseline_id,
            label=label,
            success_rate=metrics.success_rate,
        )
        return baseline

    # -- candidate creation --------------------------------------------------------
    def create_candidate(
        self,
        proposal: ProposalModel,
        baseline_id: str,
        *,
        proposal_source: str = "",
    ) -> ImprovementCandidate:
        """Validate + build + persist a proposed candidate (idempotent)."""
        baseline = self.store.load_baseline(baseline_id)
        if baseline is None:
            raise KeyError(f"unknown baseline {baseline_id!r}")
        validate_changes(proposal.component, proposal.changes, baseline.config_snapshot)
        candidate = build_candidate(
            proposal.model_copy(update={"source": proposal_source or proposal.source}),
            baseline,
            suite_id=baseline.suite_id,
            suite_version=baseline.suite_version,
        )
        existing = self.store.load_candidate(candidate.candidate_id)
        if existing is not None:
            return existing  # identical content => same immutable candidate
        self.store.save_candidate(candidate)
        self._emit(
            "candidate_created",
            candidate_id=candidate.candidate_id,
            component=candidate.component.value,
            parent_baseline=baseline.baseline_id,
            status=candidate.status.value,
        )
        return candidate

    # -- evaluation -------------------------------------------------------------------
    async def evaluate_candidate(
        self,
        candidate_id: str,
        run_eval: EvalFn,
        *,
        seed: int = 0,
    ) -> ImprovementCandidate:
        """Run the E4 suite for the candidate and apply the regression gate."""
        candidate = self._require(candidate_id)
        baseline = self.store.load_baseline(candidate.parent_baseline_id)
        if baseline is None:
            raise KeyError(f"missing parent baseline {candidate.parent_baseline_id!r}")
        if candidate.status == CandidateStatus.PASSED:
            assert candidate.evaluation is not None  # invariant kept below
            return candidate
        # Persist the in-flight state before spending evaluation effort.
        evaluating = self._transition(candidate, CandidateStatus.EVALUATING)
        self.store.save_candidate(evaluating)

        candidate_report = await run_eval(evaluating)
        cand_metrics = snapshot_from_report(candidate_report)
        verdict = evaluate_gate(baseline.metrics, cand_metrics, self.gate)
        deltas = baseline.metrics.deltas(cand_metrics)
        evaluation = CandidateEvaluation(
            suite_id=candidate.suite_id,
            suite_version=candidate.suite_version,
            seed=seed,
            baseline_report_id=baseline.report_id,
            candidate_report_id=candidate_report.report_id,
            baseline=baseline.metrics,
            candidate=cand_metrics,
            deltas=deltas,
            verdict=verdict,
        )
        new_status = (
            CandidateStatus.PASSED if verdict.passed else CandidateStatus.FAILED
        )
        updated = self._transition(
            evaluating,
            new_status,
            evaluation=evaluation,
        ).evolve(
            decision=PromotionDecision(
                action=verdict.recommended(),
                decided_by="system",
                reason=(
                    "regression gate passed"
                    if verdict.passed
                    else "; ".join(verdict.violations)
                ),
                details={
                    "deltas": {k: round(v, 6) for k, v in deltas.items()},
                    "violations": verdict.violations,
                },
            )
        )
        self.store.save_candidate(updated)
        self._emit(
            "candidate_evaluated",
            candidate_id=candidate_id,
            status=new_status.value,
            passed=verdict.passed,
            violations=len(verdict.violations),
        )
        self._emit_gate(candidate_id, verdict, deltas)
        return updated

    def _emit_gate(
        self,
        candidate_id: str,
        verdict: GateVerdict,
        deltas: dict[str, float],
    ) -> None:
        self._emit(
            "gate_decision",
            candidate_id=candidate_id,
            action=verdict.recommended().value,
            passed=verdict.passed,
            delta_success_rate=round(deltas.get("success_rate", 0.0), 6),
            delta_quality_score=round(deltas.get("quality_score", 0.0), 6),
            violations=list(verdict.violations),
        )

    # -- lifecycle moves ------------------------------------------------------------------
    def reject(
        self, candidate_id: str, reason: str, *, rejected_by: str = "system",
    ) -> ImprovementCandidate:
        candidate = self._require(candidate_id)
        updated = self._transition(
            candidate,
            CandidateStatus.REJECTED,
        ).evolve(
            decision=PromotionDecision(
                action=PromotionAction.REJECTED,
                decided_by=rejected_by,
                reason=reason,
            ),
        )
        self.store.save_candidate(updated)
        self.store.record_decision(candidate_id, PromotionDecision(
            action=PromotionAction.REJECTED,
            decided_by=rejected_by,
            reason=reason,
        ))
        self._emit(
            "candidate_rejected", candidate_id=candidate_id, reason=reason,
        )
        return updated

    def approve(
        self, candidate_id: str, approver: str, *, note: str = "",
    ) -> ImprovementCandidate:
        """Explicit human approval gate before anything reaches production."""
        if not approver.strip():
            raise ApprovalRequiredError("approval requires a named approver")
        candidate = self._require(candidate_id)
        if candidate.evaluation is None or candidate.evaluation.verdict.passed is False:
            raise ApprovalRequiredError(
                "candidate has not passed the regression gate evaluation"
            )
        updated = self._transition(candidate, CandidateStatus.APPROVED).evolve(
            decision=PromotionDecision(
                action=PromotionAction.APPROVED,
                decided_by=approver,
                reason=note or "explicit human approval",
            ),
        )
        self.store.save_candidate(updated)
        self.store.record_decision(candidate_id, PromotionDecision(
            action=PromotionAction.APPROVED, decided_by=approver, reason=note,
        ))
        self._emit(
            "candidate_approved", candidate_id=candidate_id, approver=approver,
        )
        return updated

    # -- canary ------------------------------------------------------------------------------
    def start_canary(
        self, candidate_id: str, percentage: float, *, started_by: str = "",
    ) -> ImprovementCandidate:
        """Begin canary exposure of an approved candidate (a decision record).

        Routing itself happens through :meth:`route_canary`; nothing here
        touches deployment infrastructure.
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError("percentage must be within [0, 100]")
        candidate = self._require(candidate_id)
        decided_by = started_by or (
            candidate.decision.decided_by if candidate.decision else "system"
        )
        updated = self._transition(candidate, CandidateStatus.CANARY).evolve(
            metadata={**candidate.metadata, "canary_percentage": percentage},
            decision=PromotionDecision(
                action=PromotionAction.CANARY_STARTED,
                decided_by=decided_by,
                reason=f"canary at {percentage:g}% of runs",
            ),
        )
        self.store.save_candidate(updated)
        self.store.record_decision(candidate_id, PromotionDecision(
            action=PromotionAction.CANARY_STARTED,
            decided_by=started_by or "system",
            reason=f"canary at {percentage:g}% of runs",
        ))
        self._emit(
            "canary_started",
            candidate_id=candidate_id,
            percentage=percentage,
        )
        return updated

    def route_canary(self, candidate: ImprovementCandidate, run_key: str) -> bool:
        """Deterministic bucket assignment for one run.

        Returns True when ``run_key`` falls inside the candidate's canary
        percentage. Stable across processes/restarts (pure hash function).
        """
        pct = float(candidate.metadata.get("canary_percentage", 0.0))
        if pct >= 100.0:
            return True
        if pct <= 0.0:
            return False
        bucket = int(content_hash(["canary", candidate.candidate_id, run_key])[:8], 16)
        return (bucket % 10_000) < pct * 100

    def record_canary_outcome(
        self, candidate_id: str, success: bool,
    ) -> ImprovementCandidate:
        candidate = self._require(candidate_id)
        outcomes = dict(candidate.metadata.get("canary_outcomes", {}))
        outcomes["total"] = int(outcomes.get("total", 0)) + 1
        outcomes["ok"] = int(outcomes.get("ok", 0)) + (1 if success else 0)
        updated = candidate.evolve(
            metadata={**candidate.metadata, "canary_outcomes": outcomes},
        )
        self.store.save_candidate(updated)
        self._emit(
            "canary_outcome",
            candidate_id=candidate_id,
            success=success,
            ok=outcomes["ok"],
            total=outcomes["total"],
        )
        return updated

    def _canary_ok(self, candidate: ImprovementCandidate) -> bool:
        outcomes = candidate.metadata.get("canary_outcomes", {})
        total = int(outcomes.get("total", 0))
        if total == 0:
            return False
        base = self.store.load_baseline(candidate.parent_baseline_id)
        min_rate = max(base.metrics.success_rate, 0.0) if base else 1.0
        return int(outcomes.get("ok", 0)) / total >= min_rate

    # -- promotion / rollback --------------------------------------------------------------
    def promote(
        self, candidate_id: str, approver: str, *, note: str = "",
    ) -> ImprovementCandidate:
        """Mark a candidate PROMOTED — records the production pointer change.

        This is deliberately a bookkeeping boundary only: it persists which
        version production should load, and always requires a fresh explicit
        approver name. Deploying that pointer stays an operator/E7 action.
        """
        if not approver.strip():
            raise ApprovalRequiredError("promotion requires a named approver")
        candidate = self._require(candidate_id)
        if candidate.evaluation is None or not candidate.evaluation.verdict.passed:
            raise ApprovalRequiredError("candidate never passed the regression gate")
        if candidate.status is CandidateStatus.CANARY and not self._canary_ok(
            candidate,
        ):
            raise ApprovalRequiredError(
                "canary outcomes do not meet the promotion threshold yet"
            )
        previous_active = self.store.get_active(candidate.component.value)
        updated = self._transition(candidate, CandidateStatus.PROMOTED).evolve(
            decision=PromotionDecision(
                action=PromotionAction.PROMOTED,
                decided_by=approver,
                reason=note or "explicit approval to promote",
                details={"previous_active": previous_active or ""},
            ),
        )
        self.store.save_candidate(updated)
        self.store.set_active(candidate.component.value, candidate_id)
        self.store.record_decision(candidate_id, PromotionDecision(
            action=PromotionAction.PROMOTED,
            decided_by=approver,
            reason=note or "explicit approval to promote",
        ))
        self._emit(
            "candidate_promoted",
            candidate_id=candidate_id,
            component=candidate.component.value,
            approver=approver,
            previous_active=previous_active or "",
        )
        return updated

    def rollback(self, candidate_id: str, *, rolled_back_by: str = "") -> tuple[
        ImprovementCandidate, str | None,
    ]:
        """Reverse a promotion: repoint production at the parent version."""
        candidate = self._require(candidate_id)
        if candidate.status is not CandidateStatus.PROMOTED:
            raise IllegalTransitionError(
                f"only PROMOTED candidates can be rolled back "
                f"(got {candidate.status.value})"
            )
        # Restore *this promotion's* recorded parent version, not a blind
        # last-change swap: rolling back an older candidate must never
        # resurrect a newer one.
        target = ""
        if candidate.decision and candidate.decision.details:
            target = str(candidate.decision.details.get("previous_active", ""))
        restored = self.store.revert_to(
            candidate.component.value, target,
        )
        if restored is None:
            # Pointer already gone/matching: fall back to the generic undo.
            restored = self.store.restore_previous(candidate.component.value)
        updated = self._transition(candidate, CandidateStatus.REJECTED).evolve(
            decision=PromotionDecision(
                action=PromotionAction.ROLLED_BACK,
                decided_by=rolled_back_by or "system",
                reason="promotion rolled back; production restored to parent",
                details={"restored_active": restored or ""},
            ),
        )
        self.store.save_candidate(updated)
        self.store.record_decision(candidate_id, PromotionDecision(
            action=PromotionAction.ROLLED_BACK,
            decided_by=rolled_back_by or "system",
            reason="promotion rolled back",
        ))
        self._emit(
            "candidate_rolled_back",
            candidate_id=candidate_id,
            restored_active=restored or "",
        )
        return updated, restored

    # -- internals -----------------------------------------------------------------------------
    def _require(self, candidate_id: str) -> ImprovementCandidate:
        candidate = self.store.load_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"unknown candidate {candidate_id!r}")
        return candidate

    def _transition(
        self,
        candidate: ImprovementCandidate,
        target: CandidateStatus,
        **extra_updates: Any,
    ) -> ImprovementCandidate:
        if not transition_allowed(candidate.status, target):
            raise IllegalTransitionError(
                f"illegal transition {candidate.status.value} -> {target.value} "
                f"for {candidate.candidate_id}"
            )
        return candidate.evolve(status=target, **extra_updates)


__all__ = [
    "ApprovalRequiredError",
    "EvalFn",
    "IllegalTransitionError",
    "ImprovementPipeline",
    "ImprovementStore",
    "LIFECYCLE_TRANSITIONS",
    "transition_allowed",
]
