"""Tests for E8 - Continuous Agent Improvement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from research_engineer.eval.metrics import build_report
from research_engineer.eval.models import (
    AgentBudget,
    CaseMetrics,
    EvalResult,
    EvalStatus,
    EvalSuite,
    EvalTask,
)
from research_engineer.eval.runner import EvalRunner, load_suite
from research_engineer.eval.scripted import ScriptedAgentFactory
from research_engineer.improve import (
    HARD_SAFETY_METRICS,
    LIFECYCLE_TRANSITIONS,
    ApprovalRequiredError,
    Baseline,
    CandidateComponent,
    CandidateStatus,
    FailurePattern,
    IllegalTransitionError,
    ImprovementCandidate,
    ImprovementPipeline,
    ImprovementProposal,
    ImprovementStore,
    MetricSnapshot,
    MiningConfig,
    PatternKind,
    PromotionAction,
    PromotionDecision,
    RegressionGate,
    RuleBasedProposalSource,
    apply_candidate_changes,
    build_candidate,
    candidate_eval_runner_builder,
    canonical_json,
    content_hash,
    evaluate_gate,
    extract_features,
    mine_patterns,
    mine_report,
    snapshot_from_report,
    transition_allowed,
)
from research_engineer.improve.gate import _EPSILON
from research_engineer.runtime.models import AgentContext

SUITE_DIR = Path(__file__).resolve().parent.parent / "evals"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    case_id: str,
    *,
    success: bool = False,
    score: float = 0.0,
    term: str = "success",
    tokens: int = 0,
    cost: float = 0.0,
    latency: float = 0.1,
    tool_calls: int = 0,
    errors: int = 0,
    human: int = 0,
) -> EvalResult:
    """A minimal completed EvalResult with hand-set metrics."""
    return EvalResult(
        run_id=f"run_{case_id}",
        case_id=case_id,
        status=EvalStatus.COMPLETED,
        success=success,
        completion=True,
        weighted_score=(score if score else float(success)),
        metrics=CaseMetrics(
            steps=2,
            tool_calls=tool_calls,
            tokens=tokens,
            cost_usd=cost,
            latency_seconds=latency,
            recoverable_errors=errors,
            human_interventions=human,
            terminated=True,
            termination_reason=term,
        ),
    )


class BusStub:
    """Minimal event-bus double matching ``bus.emit(dict)``."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


def make_ctx(run_id: str, metadata: dict[str, Any]) -> AgentContext:
    ctx = AgentContext(goal="test goal")
    ctx.metadata.update({"run_id": run_id, **metadata})
    return ctx


def make_baseline(config: dict[str, Any] | None = None) -> Baseline:
    cfg = {} if config is None else dict(config)
    return Baseline(
        baseline_id="base_fixed",
        label="v1",
        suite_id="demo_suite_v1",
        config_snapshot=cfg,
        config_hash=content_hash(cfg),
        metrics=MetricSnapshot(cases_total=4, success_rate=0.5),
    )


def prompt_proposal(text: str = "Be careful.") -> ImprovementProposal:
    return ImprovementProposal(
        title="t",
        component=CandidateComponent.SYSTEM_PROMPT,
        changes={"append": text},
    )


# ---------------------------------------------------------------------------
# Failure-pattern mining
# ---------------------------------------------------------------------------


class TestMining:
    def test_single_run_is_never_sufficient_evidence(self) -> None:
        report = build_report(
            [
                make_result("a", success=False, term="budget_exceeded"),
                make_result("b", success=True),
                make_result("c", success=True),
            ],
            suite_id="s",
        )
        kinds = {p.kind for p in mine_report(report)}
        # One budget-exhausted run alone must NOT produce a pattern...
        assert PatternKind.BUDGET_EXHAUSTION not in kinds
        # ...and neither must its single zero score.
        assert PatternKind.POOR_EVALUATION_SCORE not in kinds

    def test_budget_exhaustion_needs_two_runs(self) -> None:
        report = build_report(
            [
                make_result("a", term="budget_exceeded"),
                make_result("b", term="budget_exceeded"),
                make_result("ok"),
            ],
            suite_id="s",
        )
        budget = [
            p for p in mine_report(report)
            if p.kind is PatternKind.BUDGET_EXHAUSTION
        ]
        assert len(budget) == 1
        assert budget[0].affected_runs == 2
        assert set(budget[0].run_ids) == {"run_a", "run_b"}
        assert set(budget[0].example_case_ids) == {"a", "b"}

    def test_context_enrichment_tool_failures(self) -> None:
        results = [make_result(f"c{i}", term="error") for i in range(3)]
        report = build_report(results, suite_id="s")
        contexts = {
            f"run_c{i}": make_ctx(
                f"run_c{i}",
                {
                    "tool_call_log": [
                        {"step": 1, "tool": "shell", "status": "denied"},
                        {"step": 1, "tool": "shell", "status": "denied"},
                    ]
                },
            )
            for i in range(3)
        }
        features = extract_features(report, contexts)
        assert all(f.tool_failure_total >= 2 for f in features)
        tool_pats = [
            p for p in mine_report(report, contexts=contexts)
            if p.kind is PatternKind.REPEATED_TOOL_FAILURES
        ]
        assert len(tool_pats) == 1
        assert "shell" in tool_pats[0].signature
        assert tool_pats[0].total_events == 6

    def test_replanning_and_safety_triggers(self) -> None:
        report = build_report(
            [make_result(f"s{i}", term="error") for i in range(3)],
            suite_id="s",
        )
        contexts = {
            f"run_s{i}": make_ctx(
                f"run_s{i}",
                {
                    "safety_state": {
                        "decisions": [
                            {"step": 1, "action": "replan",
                             "trigger": "loop_detected"},
                            {"step": 2, "action": "replan",
                             "trigger": "risk_escalation"},
                            {"step": 3, "action": "pause_for_approval",
                             "trigger": "policy_violation"},
                        ]
                    }
                },
            )
            for i in range(3)
        }
        raw = mine_patterns(extract_features(report, contexts), MiningConfig())
        kinds = {v["kind"] for v in raw.values()}
        assert PatternKind.REPEATED_REPLANNING in kinds
        assert PatternKind.SAFETY_INTERVENTIONS in kinds
        replan = next(
            v for v in raw.values()
            if v["kind"] is PatternKind.REPEATED_REPLANNING
        )
        assert replan["total_events"] == 6  # 2 replans x 3 runs
        safety = next(
            v for v in raw.values()
            if v["kind"] is PatternKind.SAFETY_INTERVENTIONS
        )
        # Only recognised triggers are counted as safety events.
        assert safety["total_events"] == 6

    def test_unnecessary_and_human_frequency_patterns(self) -> None:
        report = build_report(
            [make_result(f"d{i}", term="error") for i in range(3)],
            suite_id="s",
        )
        contexts = {
            f"run_d{i}": make_ctx(
                f"run_d{i}",
                {
                    "human_interventions": 2,
                    "tool_call_log": [
                        {"step": 1, "tool": "web", "status": "allowed"},
                        {"step": 1, "tool": "web", "status": "allowed"},
                        {"step": 2, "tool": "web", "status": "error"},
                        {"step": 2, "tool": "shell", "status": "denied"},
                    ],
                },
            )
            for i in range(3)
        }
        raw = mine_patterns(extract_features(report, contexts), MiningConfig())
        kinds = {v["kind"] for v in raw.values()}
        assert PatternKind.UNNECESSARY_TOOL_CALLS in kinds
        assert PatternKind.REPEATED_TOOL_FAILURES in kinds
        assert PatternKind.HUMAN_APPROVAL_FREQUENCY in kinds

    def test_pattern_ids_are_deterministic(self) -> None:
        p1 = FailurePattern(
            pattern_id=f"pat_{content_hash(['budget_exhaustion', ''])[:12]}",
            kind=PatternKind.BUDGET_EXHAUSTION,
            affected_runs=2,
        )
        p2 = FailurePattern.model_validate_json(p1.model_dump_json())
        assert p2.pattern_id == p1.pattern_id
        assert p2.pattern_id.startswith("pat_")


# ---------------------------------------------------------------------------
# Candidate identity / immutability / versioning
# ---------------------------------------------------------------------------


class TestCandidateIdentity:
    def test_deterministic_ids_and_hashes(self) -> None:
        prop = ImprovementProposal(
            title="t",
            component=CandidateComponent.PLANNING_PARAMS,
            changes={"budget.max_steps": 6},
        )
        c1 = build_candidate(prop, make_baseline(), suite_id="demo_suite_v1",
                             suite_version="1")
        c2 = build_candidate(prop, make_baseline(), suite_id="demo_suite_v1",
                             suite_version="1")
        assert c1.candidate_id == c2.candidate_id
        assert c1.config_hash == c2.config_hash
        assert c1.version == c1.config_hash[:12]
        assert c1.parent_baseline_id == "base_fixed"

    def test_different_changes_give_different_identity(self) -> None:
        prop = ImprovementProposal(
            title="t",
            component=CandidateComponent.PLANNING_PARAMS,
            changes={"budget.max_steps": 6},
        )
        c1 = build_candidate(prop, make_baseline(), suite_id="demo_suite_v1",
                             suite_version="1")
        other = prop.model_copy(update={"changes": {"budget.max_steps": 8}})
        c2 = build_candidate(other, make_baseline(), suite_id="demo_suite_v1",
                             suite_version="1")
        assert c2.candidate_id != c1.candidate_id
        assert c2.config_hash != c1.config_hash

    def test_content_hash_is_order_insensitive(self) -> None:
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
        assert content_hash(["x", 3]) == content_hash(["x", 3])
        assert canonical_json({"b": [1], "a": 2}) == '{"a":2,"b":[1]}'

    def test_candidates_are_immutable(self) -> None:
        cand = build_candidate(prompt_proposal(), make_baseline(),
                               suite_id="demo_suite_v1", suite_version="1")
        with pytest.raises(Exception):
            cand.status = CandidateStatus.APPROVED

    def test_evolve_preserves_identity_updates_state_only(self) -> None:
        cand = build_candidate(prompt_proposal(), make_baseline(),
                               suite_id="demo_suite_v1", suite_version="1")
        evolved = cand.evolve(status=CandidateStatus.EVALUATING)
        assert evolved.candidate_id == cand.candidate_id
        assert evolved.status is CandidateStatus.EVALUATING
        assert cand.status is CandidateStatus.PROPOSED
        assert evolved.updated_at >= evolved.created_at


# ---------------------------------------------------------------------------
# Guardrails on proposal validation
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_prompt_patch_allowed(self) -> None:
        from research_engineer.improve.proposals import validate_changes

        validate_changes(CandidateComponent.SYSTEM_PROMPT,
                         {"append": "verify twice"}, {})

    def test_disabling_controls_rejected(self) -> None:
        from research_engineer.improve.proposals import validate_changes

        with pytest.raises(ValueError, match="forbidden"):
            validate_changes(
                CandidateComponent.AUTONOMY_THRESHOLDS,
                {"autonomy.loop_enabled": False}, {},
            )

    def test_loosening_thresholds_rejected_tightening_allowed(self) -> None:
        from research_engineer.improve.proposals import validate_changes

        current = {"autonomy.loop_threshold": 3}
        with pytest.raises(ValueError, match="loosens"):
            validate_changes(CandidateComponent.AUTONOMY_THRESHOLDS,
                             {"autonomy.loop_threshold": 10}, current)
        validate_changes(CandidateComponent.AUTONOMY_THRESHOLDS,
                         {"autonomy.loop_threshold": 2}, current)

    def test_forbidden_keys_are_all_reported(self) -> None:
        from research_engineer.improve.proposals import validate_changes

        with pytest.raises(ValueError) as err:
            validate_changes(
                CandidateComponent.SAFETY_POLICY_CONFIG,
                {
                    "gate.max_recoverable_errors": 99,
                    "gate.approval_action": "auto_approve",
                },
                {},
            )
        text = str(err.value)
        assert "max_recoverable_errors" in text
        assert "_action" in text

    def test_unknown_keys_rejected(self) -> None:
        from research_engineer.improve.proposals import validate_changes

        with pytest.raises(ValueError, match="not allowed"):
            validate_changes(CandidateComponent.PLANNING_PARAMS,
                             {"evil.key": True}, {})


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------

BASE_SNAP = MetricSnapshot(
    cases_total=4,
    success_rate=0.75,
    quality_score=0.8,
    failure_rate=0.25,
    safety_interventions=0.0,
    human_interventions=0.25,
    total_cost_usd=1.0,
    total_tokens=1000,
    avg_latency_seconds=2.0,
)


class TestGate:
    def test_strict_improvement_passes(self) -> None:
        cand = BASE_SNAP.model_copy(update={
            "success_rate": 1.0, "quality_score": 0.9, "failure_rate": 0.0,
        })
        verdict = evaluate_gate(BASE_SNAP, cand, RegressionGate())
        assert verdict.passed, verdict.violations
        assert verdict.recommended() is PromotionAction.RECOMMENDED

    def test_success_regression_rejected(self) -> None:
        verdict = evaluate_gate(
            BASE_SNAP, BASE_SNAP.model_copy(update={"success_rate": 0.5}),
            RegressionGate(),
        )
        assert not verdict.passed
        assert any("success_rate" in v for v in verdict.violations)

    def test_quality_drop_beyond_tolerance_rejected(self) -> None:
        cand = BASE_SNAP.model_copy(update={"quality_score": 0.7})
        verdict = evaluate_gate(BASE_SNAP, cand,
                                RegressionGate(quality_tolerance=0.01))
        assert not verdict.passed
        assert any("quality_score" in v for v in verdict.violations)

    def test_hard_safety_regression_always_rejects(self) -> None:
        cand = BASE_SNAP.model_copy(update={
            "success_rate": 1.0, "quality_score": 1.0, "failure_rate": 0.0,
            "human_interventions": 0.0, "total_cost_usd": 0.5,
            "safety_interventions": 2.0,
        })
        verdict = evaluate_gate(BASE_SNAP, cand, RegressionGate())
        assert not verdict.passed
        assert any("HARD SAFETY" in v for v in verdict.violations)

    def test_hard_safety_survives_gate_configuration(self) -> None:
        cand = BASE_SNAP.model_copy(update={"safety_interventions": 1.0})
        verdict = evaluate_gate(
            BASE_SNAP, cand, RegressionGate(no_regression_metrics=()),
        )
        assert verdict.rejected
        assert any(v.startswith("HARD SAFETY") for v in verdict.violations)

    def test_absolute_safety_cap(self) -> None:
        cap = RegressionGate(max_safety_interventions_abs=0.5)
        cand = BASE_SNAP.model_copy(update={"safety_interventions": 1.0})
        verdict = evaluate_gate(BASE_SNAP, cand, cap)
        assert not verdict.passed
        assert any("absolute cap" in v for v in verdict.violations)

    def test_cost_budget_violation(self) -> None:
        cand = BASE_SNAP.model_copy(update={
            "success_rate": 1.0, "total_cost_usd": 2.0,
        })
        verdict = evaluate_gate(
            BASE_SNAP, cand, RegressionGate(max_cost_increase_fraction=0.05),
        )
        assert not verdict.passed
        assert any("cost" in v.lower() for v in verdict.violations)

    def test_latency_budget_violation(self) -> None:
        cand = BASE_SNAP.model_copy(update={
            "avg_latency_seconds": 5.0, "success_rate": 1.0,
        })
        verdict = evaluate_gate(
            BASE_SNAP, cand,
            RegressionGate(max_latency_increase_fraction=0.25),
        )
        assert not verdict.passed
        assert any("latency" in v.lower() for v in verdict.violations)

    def test_zero_baseline_resources_allow_absolute_growth(self) -> None:
        zero = BASE_SNAP.model_copy(update={"total_tokens": 0.0})
        cand = zero.model_copy(update={
            "success_rate": 1.0, "total_tokens": 42.0,
        })
        assert evaluate_gate(zero, cand, RegressionGate()).passed

    def test_snapshot_from_report(self) -> None:
        report = build_report(
            [
                make_result("a", success=True),
                make_result("b", term="budget_exceeded"),
                make_result("c", term="safety_terminated", human=1),
            ],
            suite_id="s",
        )
        snap = snapshot_from_report(report)
        assert snap.cases_total == 3
        assert snap.success_rate == pytest.approx(1 / 3)
        assert snap.safety_interventions == 1.0
        assert snap.human_interventions == pytest.approx(1 / 3)
        assert snap.quality_score == pytest.approx((1.0 + 0.0 + 0.0) / 3)
        assert _EPSILON < 1e-6


# ---------------------------------------------------------------------------
# Lifecycle state machine
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_happy_chain_is_connected(self) -> None:
        chain = [
            CandidateStatus.PROPOSED, CandidateStatus.EVALUATING,
            CandidateStatus.PASSED, CandidateStatus.APPROVED,
            CandidateStatus.CANARY, CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
        ]
        for src, dst in zip(chain, chain[1:], strict=False):
            assert transition_allowed(src, dst), f"{src}->{dst} missing"
        # Terminal states are closed.
        assert LIFECYCLE_TRANSITIONS[CandidateStatus.PROMOTED] == frozenset(
            {CandidateStatus.REJECTED}
        )
        assert LIFECYCLE_TRANSITIONS[CandidateStatus.REJECTED] == frozenset()

    def test_shortcuts_are_illegal(self) -> None:
        illegal = [
            (CandidateStatus.PROPOSED, CandidateStatus.PROMOTED),
            (CandidateStatus.PROPOSED, CandidateStatus.APPROVED),
            (CandidateStatus.FAILED, CandidateStatus.APPROVED),
            (CandidateStatus.FAILED, CandidateStatus.CANARY),
            (CandidateStatus.EVALUATING, CandidateStatus.APPROVED),
            (CandidateStatus.REJECTED, CandidateStatus.PASSED),
        ]
        for src, dst in illegal:
            assert not transition_allowed(src, dst)

    def test_failed_candidates_may_be_re_evaluated(self) -> None:
        assert transition_allowed(CandidateStatus.FAILED,
                                  CandidateStatus.EVALUATING)

    def test_illegal_transition_raises_in_pipeline(self, tmp_path: Path) -> None:
        pipe = ImprovementPipeline(ImprovementStore(tmp_path))
        pipe.store.save_candidate(
            ImprovementCandidate(
                candidate_id="cand_x",
                version="v",
                component=CandidateComponent.SYSTEM_PROMPT,
                changes={"append": "x"},
                config_hash="h",
                parent_baseline_id="base_x",
                parent_config_hash="ph",
                suite_id="demo_suite_v1",
            )
        )
        with pytest.raises(IllegalTransitionError):
            pipe.rollback("cand_x")


# ---------------------------------------------------------------------------
# Scripted improvement-target suite (E4-real, deterministic)
# ---------------------------------------------------------------------------

DEMO_CONFIG = {"budget.max_steps": 3, "budget.max_tool_calls": 6}


def demo_suite_data() -> dict[str, Any]:
    """Two budget-starved cases (+ noisy helpers).

    Probed empirically against the runtime: a ``steps=2`` scripted case
    completes only at ``max_steps >= steps + 2`` (the budget check fires
    before the evaluator's done flag, and equality terminates), so
    starving at max_steps=3 fails deterministically while +50% => 4 lets
    it finish.
    """
    starved_budget = {"max_steps": 3, "max_tool_calls": 6}
    cases = [
        {
            "case_id": "budget_starved_a",
            "name": "starved A",
            "goal": "finish within tiny budget",
            "budget": dict(starved_budget),
            "metadata": {"script":
                         {"steps": 2, "final_output": "done-a"}},
        },
        {
            "case_id": "budget_starved_b",
            "name": "starved B",
            "goal": "finish within tiny budget",
            "budget": dict(starved_budget),
            "metadata": {"script":
                         {"steps": 2, "final_output": "done-b"}},
        },
        {
            "case_id": "recovers_from_error",
            "name": "transient failure",
            "goal": "recover from transient errors",
            "metadata": {"script": {
                "steps": 2, "error_at_step": 1,
                "final_output": "recovered",
            }},
        },
        {
            "case_id": "noisy_duplicate_tools",
            "name": "dup tools",
            "goal": "complete without redundant calls",
            "metadata": {"script": {
                "steps": 2,
                "tools": [
                    {"step": 1, "name": "search", "status": "allowed"},
                    {"step": 1, "name": "search", "status": "allowed"},
                    {"step": 2, "name": "search", "status": "allowed"},
                    {"step": 2, "name": "search", "status": "allowed"},
                ],
                "final_output": "noisy done",
            }},
        },
    ]
    return {
        "suite_id": "improve_demo_v1",
        "name": "E8 demo improvement suite",
        "version": "1",
        "cases": cases,
    }


async def run_scripted_suite(suite_data: dict[str, Any], label: str) -> Any:
    runner = EvalRunner(ScriptedAgentFactory(), label=label)
    return await runner.run_suite(EvalSuite.model_validate(suite_data))


async def candidate_eval_fn(
    suite_data: dict[str, Any],
) -> Any:
    """Build the E8 EvalFn wiring candidate config into scripted factories."""
    builder = candidate_eval_runner_builder("scripted")

    async def run_eval(candidate: ImprovementCandidate) -> Any:
        runner = EvalRunner(builder(candidate), label=candidate.version)
        return await runner.run_suite(EvalSuite.model_validate(suite_data))

    return run_eval


# Micro-benchmark harness: latencies are nanosecond-scale and jitter dominates
# any relative threshold, so eval-driven tests waive ONLY the latency fraction
# while every functional constraint stays strict.
LATENCY_TOLERANT_GATE = RegressionGate(max_latency_increase_fraction=1000.0)


class TestFullCycle:
    @pytest.mark.asyncio
    async def test_proposal_to_rollback(self, tmp_path: Path) -> None:
        bus = BusStub()
        pipe = ImprovementPipeline(ImprovementStore(tmp_path),
                                   gate=LATENCY_TOLERANT_GATE, event_bus=bus)
        suite_data = demo_suite_data()

        # --- baseline ------------------------------------------------------
        base_report = await run_scripted_suite(suite_data, "baseline-v1")
        assert base_report.aggregate.success_rate == pytest.approx(0.5)
        baseline = pipe.register_baseline("baseline-v1", base_report,
                                          DEMO_CONFIG)
        assert baseline.baseline_id == "base_" + content_hash(
            ["baseline-v1", base_report.suite_id, DEMO_CONFIG]
        )[:12]
        assert baseline.suite_id == "improve_demo_v1"
        assert baseline.metrics.quality_score == pytest.approx(0.5)

        # --- mine + propose --------------------------------------------------
        patterns = mine_report(base_report)
        assert any(p.kind is PatternKind.BUDGET_EXHAUSTION for p in patterns)
        proposals = RuleBasedProposalSource().propose(patterns, baseline)
        budget_props = [
            p for p in proposals
            if p.component is CandidateComponent.PLANNING_PARAMS
            and "budget.max_steps" in p.changes
        ]
        assert len(budget_props) == 1
        # 3 + max(1, 3 // 2) = 4: crosses the probed completion threshold.
        assert budget_props[0].changes["budget.max_steps"] == 4

        # --- create candidate (idempotent identity) ---------------------------
        cand = pipe.create_candidate(budget_props[0], baseline.baseline_id)
        assert cand.status is CandidateStatus.PROPOSED
        again = pipe.create_candidate(budget_props[0], baseline.baseline_id)
        assert again.candidate_id == cand.candidate_id

        # promotion/rollback impossible before evaluation -----------------------
        with pytest.raises(ApprovalRequiredError):
            pipe.promote(cand.candidate_id, "operator")

        # --- evaluate vs baseline through the regression gate -----------------
        run_eval = await candidate_eval_fn(suite_data)
        evaluated = await pipe.evaluate_candidate(cand.candidate_id, run_eval)
        ev = evaluated.evaluation
        assert ev is not None
        assert evaluated.status is CandidateStatus.PASSED
        assert ev.verdict.passed
        assert ev.deltas["success_rate"] == pytest.approx(0.5)
        assert ev.deltas["quality_score"] == pytest.approx(0.5)
        assert evaluated.decision is not None
        assert evaluated.decision.action is PromotionAction.RECOMMENDED

        # PASSED short-circuits re-evaluation -----------------------------------
        recached = await pipe.evaluate_candidate(cand.candidate_id, run_eval)
        assert recached.evaluation is not None
        assert recached.evaluation.verdict.passed

        # --- approval gate ------------------------------------------------------
        with pytest.raises(ApprovalRequiredError):
            pipe.approve(cand.candidate_id, "")
        approved = pipe.approve(cand.candidate_id, "alice", note="looks good")
        assert approved.status is CandidateStatus.APPROVED
        assert approved.decision is not None
        assert approved.decision.decided_by == "alice"
        with pytest.raises(IllegalTransitionError):
            pipe.approve(cand.candidate_id, "bob")

        # --- promotion requires a named approver --------------------------------
        with pytest.raises(ApprovalRequiredError):
            pipe.promote(cand.candidate_id, " ")
        promoted = pipe.promote(cand.candidate_id, "carol")
        assert promoted.status is CandidateStatus.PROMOTED
        comp = promoted.component.value
        assert pipe.store.get_active(comp) == promoted.candidate_id

        # --- rollback restores production ----------------------------------------
        rolled, restored = pipe.rollback(promoted.candidate_id,
                                         rolled_back_by="carol")
        assert rolled.status is CandidateStatus.REJECTED
        assert rolled.decision is not None
        assert rolled.decision.action is PromotionAction.ROLLED_BACK
        # First-ever promotion: previous pointer was "" (baseline default).
        assert restored == ""
        assert pipe.store.get_active(comp) is None
        with pytest.raises(IllegalTransitionError):
            pipe.promote(rolled.candidate_id, "carol")
        with pytest.raises(IllegalTransitionError):
            pipe.rollback(rolled.candidate_id)

        # --- telemetry stayed on the single provided bus --------------------------
        emitted = [e["event"] for e in bus.events]
        assert bus.events[0]["kind"] == "agent_improvement"
        for expected in (
            "baseline_registered", "candidate_created", "candidate_evaluated",
            "gate_decision", "candidate_approved", "candidate_promoted",
            "candidate_rolled_back",
        ):
            assert expected in emitted

    @pytest.mark.asyncio
    async def test_superseding_promotion_restores_previous_candidate(
        self, tmp_path: Path,
    ) -> None:
        pipe = ImprovementPipeline(ImprovementStore(tmp_path),
                                   gate=LATENCY_TOLERANT_GATE)
        suite_data = demo_suite_data()
        base_report = await run_scripted_suite(suite_data, "v1")
        baseline = pipe.register_baseline("v1", base_report, DEMO_CONFIG)
        budget_prop = next(
            p for p in RuleBasedProposalSource().propose(
                mine_report(base_report), baseline,
            )
            if p.component is CandidateComponent.PLANNING_PARAMS
        )
        run_eval = await candidate_eval_fn(suite_data)

        first = pipe.create_candidate(budget_prop, baseline.baseline_id)
        await pipe.evaluate_candidate(first.candidate_id, run_eval)
        comp = first.component.value
        pipe.approve(first.candidate_id, "op")
        pipe.promote(first.candidate_id, "op")

        # A second, different-but-passing candidate supersedes it.
        tweak = budget_prop.model_copy(
            update={"changes": {"budget.max_steps": 5}},
        )
        second = pipe.create_candidate(tweak, baseline.baseline_id)
        await pipe.evaluate_candidate(second.candidate_id, run_eval)
        pipe.approve(second.candidate_id, "op")
        pipe.promote(second.candidate_id, "op")
        assert pipe.store.get_active(comp) == second.candidate_id

        _, restored = pipe.rollback(second.candidate_id)
        assert restored == first.candidate_id
        assert pipe.store.get_active(comp) == first.candidate_id
        _, restored_again = pipe.rollback(first.candidate_id)
        assert restored_again == ""
        assert pipe.store.get_active(comp) is None


class TestCanaryAndApproval:
    @pytest.mark.asyncio
    async def test_outcomes_gate_promotion_and_routing_is_stable(
        self, tmp_path: Path,
    ) -> None:
        pipe = ImprovementPipeline(ImprovementStore(tmp_path),
                                   gate=LATENCY_TOLERANT_GATE)
        suite_data = demo_suite_data()
        base_report = await run_scripted_suite(suite_data, "v1")
        baseline = pipe.register_baseline("v1", base_report, DEMO_CONFIG)
        prop = next(
            p for p in RuleBasedProposalSource().propose(
                mine_report(base_report), baseline,
            )
            if p.component is CandidateComponent.PLANNING_PARAMS
        )
        cand = pipe.create_candidate(prop, baseline.baseline_id)
        run_eval = await candidate_eval_fn(suite_data)
        await pipe.evaluate_candidate(cand.candidate_id, run_eval)
        pipe.approve(cand.candidate_id, "dana")
        canary = pipe.start_canary(cand.candidate_id, 50.0)
        assert canary.status is CandidateStatus.CANARY
        assert canary.metadata["canary_percentage"] == 50.0

        # Deterministic assignment: same key -> same bucket, stably.
        assert pipe.route_canary(canary, "run-42") == pipe.route_canary(
            canary, "run-42",
        )
        full = canary.model_copy(update={"metadata":
                                         {"canary_percentage": 100.0}})
        none = canary.model_copy(update={"metadata":
                                         {"canary_percentage": 0.0}})
        assert pipe.route_canary(full, "k") is True
        assert pipe.route_canary(none, "k") is False
        with pytest.raises(ValueError):
            pipe.start_canary(cand.candidate_id, 150.0)

        # Canary outcomes below the bar block promotion.
        failing = pipe.record_canary_outcome(cand.candidate_id, success=False)
        assert failing.metadata["canary_outcomes"] == {"total": 1, "ok": 0}
        with pytest.raises(ApprovalRequiredError):
            pipe.promote(cand.candidate_id, "erin")
        pipe.record_canary_outcome(cand.candidate_id, success=True)
        pipe.record_canary_outcome(cand.candidate_id, success=True)
        promoted = pipe.promote(cand.candidate_id, "erin")
        assert promoted.status is CandidateStatus.PROMOTED

    @pytest.mark.asyncio
    async def test_approve_or_promote_before_evaluation_is_refused(
        self, tmp_path: Path,
    ) -> None:
        pipe = ImprovementPipeline(ImprovementStore(tmp_path))
        suite_data = demo_suite_data()
        base_report = await run_scripted_suite(suite_data, "v1")
        baseline = pipe.register_baseline("v1", base_report, DEMO_CONFIG)
        prop = next(
            p for p in RuleBasedProposalSource().propose(
                mine_report(base_report), baseline,
            )
            if p.component is CandidateComponent.PLANNING_PARAMS
        )
        cand = pipe.create_candidate(prop, baseline.baseline_id)
        with pytest.raises(ApprovalRequiredError):
            pipe.approve(cand.candidate_id, "someone")
        with pytest.raises(ApprovalRequiredError):
            pipe.promote(cand.candidate_id, "someone")


# ---------------------------------------------------------------------------
# Store behaviour + reproducibility
# ---------------------------------------------------------------------------


class TestStoreAndReproducibility:
    def test_roundtrip_and_atomic_files(self, tmp_path: Path) -> None:
        store = ImprovementStore(tmp_path)
        cand = build_candidate(prompt_proposal("verify"), make_baseline(),
                               suite_id="demo_suite_v1", suite_version="1")
        store.save_candidate(cand)
        path = tmp_path / "candidates" / f"{cand.candidate_id}.json"
        assert path.exists()
        assert not (tmp_path / "candidates" / (path.name + ".tmp")).exists()
        loaded = store.load_candidate(cand.candidate_id)
        assert loaded == cand
        assert store.load_candidate("missing") is None
        assert len(store.list_candidates()) == 1
        store.record_decision(
            cand.candidate_id,
            PromotionDecision(action=PromotionAction.APPROVED,
                              decided_by="zoe"),
        )
        entries = store.list_decisions()
        assert entries[-1]["decided_by"] == "zoe"

    def test_json_roundtrip_fidelity(self) -> None:
        cand = build_candidate(
            ImprovementProposal(
                title="t",
                component=CandidateComponent.PLANNING_PARAMS,
                changes={"budget.max_steps": 6},
            ),
            make_baseline({"budget.max_steps": 2}),
            suite_id="demo_suite_v1",
            suite_version="1",
        ).evolve(status=CandidateStatus.EVALUATING)
        clone = ImprovementCandidate.model_validate_json(cand.model_dump_json())
        assert clone == cand
        assert clone.changes == {"budget.max_steps": 6}
        assert clone.parent_config_hash == cand.parent_config_hash

    def test_pointer_semantics_first_promotion_defaults_to_baseline(
        self, tmp_path: Path,
    ) -> None:
        store = ImprovementStore(tmp_path)
        assert store.get_active("planning_params") is None
        # First-ever promotion points at the candidate; "" was implicit.
        store.set_active("planning_params", "cand_a")
        assert store.get_active("planning_params") == "cand_a"
        assert store.restore_previous("planning_params") == ""
        assert store.get_active("planning_params") is None
        assert store.restore_previous("planning_params") is None
        # Chained promotions restore their immediate predecessor.
        store.set_active("planning_params", "cand_b")
        store.set_active("planning_params", "cand_c")
        assert store.restore_previous("planning_params") == "cand_b"
        assert store.get_active("planning_params") == "cand_b"
        assert store.restore_previous("missing_component") is None


# ---------------------------------------------------------------------------
# Applying candidate configuration onto eval tasks (data-only)
# ---------------------------------------------------------------------------


def simple_task() -> EvalTask:
    return EvalTask(
        case_id="c",
        name="n",
        goal="g",
        budget=AgentBudget(max_steps=2),
        metadata={
            "script": {
                "steps": 2,
                "tools": [
                    {"step": 1, "name": "search", "status": "allowed"},
                    {"step": 1, "name": "search", "status": "allowed"},
                ],
            }
        },
    )


class TestApplyCandidateChanges:
    def test_budget_key_is_applied(self) -> None:
        cand = build_candidate(
            ImprovementProposal(
                title="t",
                component=CandidateComponent.PLANNING_PARAMS,
                changes={"budget.max_steps": 9},
            ),
            make_baseline(),
            suite_id="demo_suite_v1",
            suite_version="1",
        )
        updated = apply_candidate_changes(simple_task(), cand)
        assert updated.budget.max_steps == 9
        # Everything else untouched.
        assert updated.metadata["script"]["tools"][0]["name"] == "search"

    def test_duplicate_tools_are_deduped(self) -> None:
        cand = build_candidate(
            ImprovementProposal(
                title="t",
                component=CandidateComponent.TOOL_SELECTION_STRATEGY,
                changes={"tools_strategy.memoize_identical_calls": True},
            ),
            make_baseline(),
            suite_id="demo_suite_v1",
            suite_version="1",
        )
        updated = apply_candidate_changes(simple_task(), cand)
        tools = updated.metadata["script"]["tools"]
        assert len(tools) == 1

    def test_prompt_append_extends_goal_without_breaking_case_id(self) -> None:
        cand = build_candidate(prompt_proposal("be careful"), make_baseline(),
                               suite_id="demo_suite_v1", suite_version="1")
        updated = apply_candidate_changes(simple_task(), cand)
        assert updated.case_id == "c"
        assert updated.goal.startswith("g\nbe careful")

    def test_unmapped_but_allowed_keys_recorded_in_metadata(self) -> None:
        cand = build_candidate(
            ImprovementProposal(
                title="t",
                component=CandidateComponent.MODEL_PROVIDER_CONFIG,
                changes={"model_provider.tier": "light"},
            ),
            make_baseline(),
            suite_id="demo_suite_v1",
            suite_version="1",
        )
        updated = apply_candidate_changes(simple_task(), cand)
        bucket = updated.metadata["candidate_model_provider"]
        assert bucket["model_provider.tier"] == "light"


# ---------------------------------------------------------------------------
# The real E4 sample suite flowing through the improvement pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_eval_suite_through_pipeline(tmp_path: Path) -> None:
    pipe = ImprovementPipeline(ImprovementStore(tmp_path),
                               gate=LATENCY_TOLERANT_GATE)
    suite = load_suite(SUITE_DIR / "sample_suite.yaml")
    factory = ScriptedAgentFactory()

    base_report = await EvalRunner(factory, label="sample-base").run_suite(
        suite,
    )
    assert base_report.aggregate.cases_total == len(suite.cases)

    baseline = pipe.register_baseline("sample-suite", base_report, {})
    assert baseline.baseline_id.startswith("base_")

    prompt_prop = ImprovementProposal(
        title="Add verification guidance",
        description="Append explicit verify-before-final policy text.",
        component=CandidateComponent.SYSTEM_PROMPT,
        changes={"append": "Verify every constraint before finishing."},
        evidence=["manual-review"],
    )
    cand = pipe.create_candidate(prompt_prop, baseline.baseline_id)
    builder = candidate_eval_runner_builder("scripted")

    async def run_eval(c: ImprovementCandidate) -> Any:
        runner = EvalRunner(builder(c), label=c.version)
        return await runner.run_suite(suite)

    evaluated = await pipe.evaluate_candidate(cand.candidate_id, run_eval)
    assert evaluated.status is CandidateStatus.PASSED, (
        evaluated.evaluation.verdict.violations
        if evaluated.evaluation else "?"
    )
    # Identical behaviour => every functional delta is exactly zero.
    ev = evaluated.evaluation
    assert ev is not None
    assert ev.deltas["success_rate"] == 0.0
    assert ev.deltas["failure_rate"] == 0.0
    assert ev.deltas["human_interventions"] == 0.0
    pipe.approve(evaluated.candidate_id, "maintainer")
    promoted = pipe.promote(evaluated.candidate_id, "maintainer")
    assert pipe.store.get_active("system_prompt") == promoted.candidate_id
    _, restored = pipe.rollback(promoted.candidate_id)
    assert restored == ""
    assert pipe.store.get_active("system_prompt") is None


def test_hard_safety_metrics_constant() -> None:
    assert HARD_SAFETY_METRICS == frozenset({"safety_interventions"})
