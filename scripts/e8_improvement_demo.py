#!/usr/bin/env python
"""E8 - representative improvement experiment, end to end.

Baseline -> mine failures -> create candidate -> evaluate both through the
E4 harness -> regression gate -> explicit approval -> promotion -> rollback.

Every stage asserts its preconditions; any drift fails loudly. Run:

    uv run python scripts/e8_improvement_demo.py [--store output/improvements/demo]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from typing import Any

from research_engineer.eval.models import EvalSuite
from research_engineer.eval.runner import EvalRunner
from research_engineer.eval.scripted import ScriptedAgentFactory
from research_engineer.improve import (
    CandidateComponent,
    CandidateStatus,
    ImprovementPipeline,
    ImprovementStore,
    PatternKind,
    PromotionAction,
    RegressionGate,
    RuleBasedProposalSource,
    candidate_eval_runner_builder,
    mine_report,
)

# Baseline production configuration. A steps=2 scripted case completes only
# at max_steps >= steps + 2 (probed: the budget check fires before the
# evaluator's done flag, and equality terminates), so at baseline the two
# starved cases terminate with budget_exceeded deterministically and +50%
# (3 -> 4) crosses exactly that completion threshold.
BASELINE_CONFIG = {"budget.max_steps": 3, "budget.max_tool_calls": 6}

# Micro-benchmark harness: latencies are sub-millisecond and dominated by
# scheduler jitter, so the demo waives only the latency fraction while every
# functional constraint (success/quality/failure/human/safety) stays strict.
GATE = RegressionGate(max_latency_increase_fraction=1000.0)


def demo_suite() -> dict[str, Any]:
    starved_budget = {"max_steps": 3, "max_tool_calls": 6}
    return {
        "suite_id": "improve_demo_v1",
        "name": "E8 demo improvement suite",
        "version": "1",
        "cases": [
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
        ],
    }


def banner(stage: str) -> None:
    print(f"\n=== {stage} {'=' * max(0, 66 - len(stage))}")


async def main(store_dir: str) -> None:
    # Fresh experiment: a deterministic candidate id resurrects any prior
    # run's advanced lifecycle from a persistent store, so wipe it first.
    shutil.rmtree(store_dir, ignore_errors=True)
    pipe = ImprovementPipeline(ImprovementStore(store_dir), gate=GATE)
    suite_data = demo_suite()
    suite = EvalSuite.model_validate(suite_data)
    builder = candidate_eval_runner_builder("scripted")

    async def run_eval(candidate: Any) -> Any:
        runner = EvalRunner(builder(candidate), label=candidate.version)
        return await runner.run_suite(suite)

    # 1 - baseline ---------------------------------------------------------
    banner("1. BASELINE evaluation (real E4 runner + scripted agents)")
    base_report = await EvalRunner(
        ScriptedAgentFactory(), label="baseline-v1",
    ).run_suite(suite)
    for r in base_report.results:
        print(f"   {r.case_id:<24} success={r.success!s:<5} "
              f"score={r.weighted_score:.2f} term={r.metrics.termination_reason}")
    agg = base_report.aggregate
    print(f"   success_rate={agg.success_rate:.2f} "
          f"failure_rate={agg.failure_rate:.2f} "
          f"quality={(sum(r.weighted_score for r in base_report.results) / len(base_report.results)):.2f}")
    assert agg.success_rate == 0.5

    baseline = pipe.register_baseline("baseline-v1", base_report,
                                      BASELINE_CONFIG)
    print(f"   baseline_id={baseline.baseline_id}")

    # 2 - mining -------------------------------------------------------------
    banner("2. FAILURE MINING (>=2 distinct runs per pattern)")
    patterns = mine_report(base_report)
    for p in patterns:
        print(f"   {p.kind.value:<28} runs={p.affected_runs} "
              f"events={p.total_events}")
    assert any(p.kind is PatternKind.BUDGET_EXHAUSTION for p in patterns)

    # 3 - proposal ------------------------------------------------------------
    banner("3. IMPROVEMENT PROPOSAL (rule-based source)")
    proposals = RuleBasedProposalSource().propose(patterns, baseline)
    prop = next(p for p in proposals
                if p.component is CandidateComponent.PLANNING_PARAMS)
    print(f"   [{prop.component.value}] {prop.title}")
    print(f"   changes: {prop.changes}")
    assert prop.changes["budget.max_steps"] == 4  # crosses the >=4 threshold

    # 4 - candidate -------------------------------------------------------------
    banner("4. CANDIDATE CREATION (deterministic id/hash, immutable)")
    cand = pipe.create_candidate(prop, baseline.baseline_id)
    again = pipe.create_candidate(prop, baseline.baseline_id)
    print(f"   candidate_id={cand.candidate_id} version={cand.version}")
    print(f"   config_hash={cand.config_hash[:16]}...")
    assert again.candidate_id == cand.candidate_id
    assert cand.status is CandidateStatus.PROPOSED

    # 5 - offline evaluation vs baseline ----------------------------------------
    banner("5. OFFLINE EVALUATION (candidate suite run vs baseline metrics)")
    evaluated = await pipe.evaluate_candidate(cand.candidate_id, run_eval)
    ev = evaluated.evaluation
    assert ev is not None and ev.verdict.passed
    print(f"   {'metric':<24}{'baseline':>10}{'candidate':>11}{'delta':>9}")
    for name in ("success_rate", "quality_score", "failure_rate",
                 "human_interventions", "total_tokens"):
        b = getattr(ev.baseline, name)
        c = getattr(ev.candidate, name)
        print(f"   {name:<24}{b:>10.2f}{c:>11.2f}"
              f"{ev.deltas[name]:>+9.2f}")
    for v in ev.verdict.violations or ["(none)"]:
        print(f"   violation: {v}")

    # 6 - gate -----------------------------------------------------------------
    banner("6. REGRESSION GATE")
    assert evaluated.status is CandidateStatus.PASSED
    assert evaluated.decision is not None
    print(f"   verdict=PASS action={evaluated.decision.action.value}")
    print(f"   warnings={ev.verdict.warnings}")

    # 7 - explicit approval -----------------------------------------------------
    banner("7. EXPLICIT HUMAN APPROVAL")
    try:
        pipe.promote(cand.candidate_id, "")
        raise SystemExit("promotion without approver must fail")
    except PermissionError:
        print("   empty-approver promotion correctly refused")
    approved = pipe.approve(cand.candidate_id, "atandra",
                            note="demo approval")
    assert approved.status is CandidateStatus.APPROVED
    assert approved.decision is not None
    print(f"   status={approved.status.value} "
          f"decided_by={approved.decision.decided_by}")

    # 8 - promotion --------------------------------------------------------------
    banner("8. PROMOTION (production pointer decision only)")
    promoted = pipe.promote(cand.candidate_id, "atandra")
    comp = promoted.component.value
    print(f"   active[{comp}] = {pipe.store.get_active(comp)}")
    assert promoted.status is CandidateStatus.PROMOTED
    assert pipe.store.get_active(comp) == promoted.candidate_id

    # 9 - rollback ---------------------------------------------------------------
    banner("9. ROLLBACK (reversible promotion)")
    rolled, restored = pipe.rollback(promoted.candidate_id,
                                     rolled_back_by="atandra")
    assert rolled.status is CandidateStatus.REJECTED
    assert rolled.decision is not None
    assert rolled.decision.action is PromotionAction.ROLLED_BACK
    assert restored == ""
    print(f"   pointer restored to baseline default ({restored!r})")
    print(f"   final status={rolled.status.value}")
    assert pipe.store.get_active(comp) is None

    banner("DEMO COMPLETE")
    print("   full loop verified: mine -> propose -> evaluate -> gate ->")
    print("   approve -> promote -> rollback, all on real E4 runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="output/improvements/demo",
                        help="ImprovementStore root directory")
    args = parser.parse_args()
    asyncio.run(main(args.store))
