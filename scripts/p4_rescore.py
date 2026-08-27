#!/usr/bin/env python
"""P4 - Rescore persisted P2/P3 benchmark artifacts with corrected graders.

Evaluation-semantics changes applied (suite v2 case revision r2):

1. design_02 / e2e_03: brittle exact-string numeric requirements
   ("50000" / "8000") replaced by the format-tolerant ``output_number``
   grader ("$50,000", "8,000/month", ... are equally valid renderings).
2. impl_01: brittle exact substring "(batch, seq" replaced by a
   case/spacing-tolerant ``output_regex`` requiring explicit batch/seq dims.
3. Malformed/failed LLM-judge responses are reclassified as explicit
   JUDGE_ERROR results (``error_kind="judge_error"``) and EXCLUDED from
   weighted scores and agent-failure accounting - an evaluator malfunction
   is never a genuine score=0 against the agent.

Only persisted artifacts are touched: run payloads are re-graded locally,
no LLM is invoked, and the original P2/P3 files are never modified. All
corrected outputs land under ``artifacts/p4_rescore/``.

    uv run python scripts/p4_rescore.py [--artifacts artifacts]
        [--out artifacts/p4_rescore] [--skip-gate]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from research_engineer.eval.models import (
    CaseMetrics,
    EvalReport,
    EvalResult,
    EvalStatus,
    SuiteMetrics,
)
from research_engineer.service.benchmark import (
    DEFAULT_SUITE_V2_PATH,
    load_benchmark_suite,
)
from research_engineer.service.benchmark_runner import _StoredExecutionView

#: Cases whose criteria changed in r2 (brittle exact strings removed).
RESCORED_CASES: tuple[str, ...] = (
    "design_02_data_scaling_study",
    "impl_01_attention_module_plan",
    "e2e_03_build_decision_memorandum",
)

#: Detail markers identifying a judge malfunction in OLD persisted criteria
#: (pre-JUDGE_ERROR artifacts).
_JUDGE_FAIL_MARKERS = ("unparseable", "scoring function failed")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_judge_error(criterion: dict[str, Any]) -> bool:
    if str(criterion.get("error_kind", "")) == "judge_error":
        return True
    detail = str(criterion.get("detail", ""))
    return any(m in detail for m in _JUDGE_FAIL_MARKERS)


def _carry_judge_criterion(
    old_criteria: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Carry the persisted llm_quality verdict forward under r2 semantics.

    A judge malfunction becomes an explicit judge_error (excluded from
    aggregation); a valid judged score is preserved verbatim.
    """
    for c in old_criteria:
        if c.get("grader") != "llm_quality":
            continue
        carried = dict(c)
        if _is_judge_error(c):
            carried["error_kind"] = "judge_error"
            carried["detail"] = (
                "JUDGE_ERROR: malformed judge reply (reclassified from "
                f"{c.get('detail', '')!r})"
            )
        return carried
    return None


async def _regrade_criteria(
    case: Any, view: _StoredExecutionView
) -> list[dict[str, Any]]:
    """Re-run the deterministic (non-LLM) graders for one case."""
    from research_engineer.eval.graders import GradingRequest, build_grader

    request = GradingRequest(task=case, execution=view)  # type: ignore[arg-type]
    outcomes: list[dict[str, Any]] = []
    for criterion in case.criteria:
        if criterion.grader == "llm_quality":
            continue  # carried from the persisted artifact instead
        result = await build_grader(criterion.grader, criterion.config).grade(
            request
        )
        outcomes.append({
            "grader": criterion.grader,
            "weight": criterion.weight,
            "required": criterion.required,
            "passed": result.passed,
            "score": result.score,
            "detail": result.detail,
            "error_kind": getattr(result, "error_kind", "") or "",
        })
    return outcomes


async def _regrade_case(
    case: Any,
    payload: dict[str, Any] | None,
    old_criteria: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, float]:
    """Corrected criteria / graded_success / weighted for one outcome.

    ``payload`` is the worker result payload (the run record's ``result``
    field); ``None`` means no payload was found, in which case deterministic
    criteria are kept as persisted and only the JUDGE_ERROR reclassification
    is applied.
    """
    judge = _carry_judge_criterion(old_criteria)
    if payload is None or case is None:
        criteria = [
            deepcopy(c) for c in old_criteria
            if c.get("grader") != "llm_quality"
        ]
        if judge is not None:
            criteria.append(judge)
    else:
        view = _StoredExecutionView(payload)
        criteria = await _regrade_criteria(case, view)
        if judge is not None:
            criteria.append(judge)
    required_ok = all(c["passed"] for c in criteria if c["required"])
    graded = [c for c in criteria if c.get("error_kind") != "judge_error"]
    total = sum(c["weight"] for c in graded) or 1.0
    weighted = sum(c["score"] * c["weight"] for c in graded) / total
    return criteria, required_ok, round(weighted, 4)


# ---------------------------------------------------------------------------
# Persisted-payload access
# ---------------------------------------------------------------------------


def _find_runs_dbs(report_path: Path) -> list[Path]:
    """runs.db candidates for a v2_*.json report (same dir + repeat_*)."""
    base = report_path.parent
    candidates = [base / "runs.db", *sorted(base.glob("repeat_*/runs.db"))]
    return [p for p in candidates if p.exists()]


def _load_payloads(db_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """(normalized goal, worker result payload) pairs from one runs.db."""
    out: list[tuple[str, dict[str, Any]]] = []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        con.row_factory = sqlite3.Row
        for row in con.execute(
            "SELECT goal, payload_json, created_at FROM runs "
            "ORDER BY created_at"
        ):
            try:
                record = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            payload = record.get("result") or {}
            if isinstance(payload, dict) and payload:
                out.append((_norm(str(row["goal"])), payload))
    except sqlite3.Error:
        pass
    finally:
        con.close()
    return out


# ---------------------------------------------------------------------------
# Report rescoring
# ---------------------------------------------------------------------------

_MIRROR_COUNTER: dict[str, int] = {}


def _mirror_name(report_path: Path, artifacts_root: Path) -> str:
    """Unique, traceable name for a corrected report copy.

    Encodes the artifact-relative path so identically-named reports from
    different arms never collide (``p3/baseline_repro`` vs
    ``p3short/glm/rep2/baseline_repro``).
    """
    try:
        rel = report_path.relative_to(artifacts_root)
    except ValueError:
        rel = Path(*report_path.parts[-3:])
    parts = list(rel.parts[:-1]) + [report_path.stem]
    return "_".join(parts) + ".corrected.json"


def rescore_report(
    report_path: Path,
    suite_by_id: dict[str, Any],
    cases_by_goal: dict[str, str],
    out_root: Path,
    stats: dict[str, Any],
    artifacts_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Rescore one persisted v2 report; write the corrected copy."""
    data = json.loads(report_path.read_text(encoding="utf-8"))
    old_success: dict[tuple[str, int], bool] = {
        (str(o.get("case_id")), int(o.get("repeat") or 1)):
            bool(o.get("graded_success"))
        for o in data.get("outcomes", [])
        if o.get("mode") == "llm_agent"
    }
    goals_in_dbs: list[tuple[str, dict[str, Any]]] = []
    for db in _find_runs_dbs(report_path):
        goals_in_dbs.extend(_load_payloads(db))

    async def _rescore() -> list[dict[str, Any]]:
        flips: list[dict[str, Any]] = []
        for outcome in data.get("outcomes", []):
            if outcome.get("mode") != "llm_agent":
                continue
            case_id = str(outcome.get("case_id"))
            old_criteria = list(outcome.get("criteria", []))
            judge_before = any(
                _is_judge_error(c) for c in old_criteria
                if c.get("grader") == "llm_quality"
            )
            if case_id not in RESCORED_CASES and not judge_before:
                continue  # nothing changes under r2 semantics
            payload: dict[str, Any] | None = None
            if case_id in RESCORED_CASES:
                goal = cases_by_goal.get(case_id, "")
                matches = [p for g, p in goals_in_dbs if g == goal]
                if matches:
                    payload = matches[-1]  # latest attempt wins
            criteria, required_ok, weighted = await _regrade_case(
                suite_by_id.get(case_id), payload, old_criteria
            )
            key = (case_id, int(outcome.get("repeat") or 1))
            new_success = bool(required_ok and outcome.get("runtime_success"))
            flips.append({
                "report": str(report_path),
                "case_id": case_id,
                "repeat": key[1],
                "old_success": old_success.get(key),
                "new_success": new_success,
                "payload_found": payload is not None,
                "judge_error_reclassified": judge_before,
                "old_weighted_score": outcome.get("weighted_score"),
                "new_weighted_score": weighted,
            })
            outcome["criteria"] = criteria
            outcome["weighted_score"] = weighted
            outcome["graded_success"] = new_success
            outcome["rescored"] = {
                "applied_revision": "r2",
                "payload_found": payload is not None,
                "judge_error_reclassified": judge_before,
            }
        return flips

    flips = asyncio.run(_rescore())
    stats["flips"].extend(flips)

    out_path = out_root / _mirror_name(
        report_path, artifacts_root or report_path.parent.parent
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return flips


# ---------------------------------------------------------------------------
# Corrected E8 gate recomputation (DeepSeek candidate)
# ---------------------------------------------------------------------------


def _arm_report(arm_dir: Path, report_id: str, label: str) -> Any:
    """Rebuild an E4 EvalReport from corrected outcome rows in an arm dir.

    Mirrors scripts/p3/e8_gate.py: every field is computed from the
    corrected measured outcomes; nothing is synthesized.
    """
    results: list[EvalResult] = []
    for path in sorted(arm_dir.glob("*.corrected.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for o in data.get("outcomes", []):
            if o.get("mode") != "llm_agent":
                continue
            lat = float(o.get("latency_seconds") or 0.0)
            metrics = CaseMetrics(
                steps=int(o.get("steps") or 0),
                tool_calls=int(o.get("tool_calls") or 0),
                tokens=int(o.get("tokens") or 0),
                cost_usd=float(o.get("cost_usd") or 0.0),
                latency_seconds=lat,
                recoverable_errors=int(o.get("recoverable_errors") or 0),
                fatal_errors=int(o.get("fatal_errors") or 0),
                recovered=bool(
                    o.get("recoverable_errors") and o.get("graded_success")
                ),
                terminated=True,
                termination_reason=str(o.get("termination") or "unknown"),
            )
            results.append(EvalResult(
                run_id=f"{arm_dir.name}:{path.stem}:{o.get('case_id')}:"
                       f"{o.get('repeat')}",
                case_id=str(o.get("case_id")),
                status=(EvalStatus.COMPLETED if o.get("graded_success")
                        else EvalStatus.FAILED),
                success=bool(o.get("graded_success")),
                weighted_score=float(o.get("weighted_score") or 0.0),
                metrics=metrics,
            ))
    total = len(results)
    successes = sum(1 for r in results if r.success)
    latencies = sorted(r.metrics.latency_seconds for r in results)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] \
        if latencies else 0.0
    return EvalReport(
        report_id=report_id,
        suite_id="research_benchmark_v2_llm",
        suite_version="2",
        label=label,
        results=results,
        aggregate=SuiteMetrics(
            cases_total=total,
            success_rate=(successes / total) if total else 0.0,
            completion_rate=1.0 if total else 0.0,
            failure_rate=(1 - successes / total) if total else 0.0,
            total_tokens=sum(r.metrics.tokens for r in results),
            total_cost_usd=sum(r.metrics.cost_usd for r in results),
            avg_latency_seconds=(
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            p95_latency_seconds=p95,
            termination_reasons={
                "success": sum(1 for r in results if r.success),
                "failure": sum(1 for r in results if not r.success),
            },
        ),
    )


def recompute_e8_gate(store_root: Path) -> dict[str, Any]:
    """Corrected E8 gate for the DeepSeek candidate. NO promotion.

    The corrected measured evidence (rescored baseline_repro and
    p3short_ds_evidence) is registered as baseline/candidate reports and
    the standard RegressionGate decides PASS/FAIL. On PASS the candidate
    is left in the APPROVED state (approved-candidate) awaiting explicit
    human promotion; promotion itself is never executed.
    """
    from research_engineer.improve import (
        CandidateComponent,
        ImprovementPipeline,
        ImprovementStore,
    )
    from research_engineer.improve.models import ImprovementProposal

    p4_root = Path("artifacts/p4_rescore")
    baseline = _arm_report(
        p4_root / "arms" / "baseline_repro",
        "p4_rescored_baseline_repro", "p4-rescored-frozen-baseline",
    )
    candidate_ev = _arm_report(
        p4_root / "arms" / "p3short_ds",
        "p4_rescored_p3short_ds", "p4-rescored-deepseek-evidence",
    )
    pipe = ImprovementPipeline(store=ImprovementStore(root=store_root))
    snap = {
        "arm": "baseline_repro",
        "provider": "ollama",
        "model": "glm-5.3-flash",
        "judge": "glm-5.3-flash",
        "rescored": "p4/r2-graders",
    }
    base = pipe.register_baseline(
        "p4-rescored-p2-baseline", baseline, snap, suite_version="2",
    )
    proposal = ImprovementProposal(
        title=(
            "P4 rescored candidate: p3short_ds_evidence "
            "(deepseek-v4-pro:0813)"
        ),
        description=(
            "DeepSeek candidate re-evaluated with corrected r2 graders "
            "over persisted P3-short measured outcomes; no LLM re-run."
        ),
        component=CandidateComponent("model_provider_config"),
        changes={"model_provider.model": "deepseek-v4-pro:0813"},
        evidence=[
            "p3/arm=p3short_ds_evidence",
            "p3/baseline=baseline_repro",
            "p4/rescore=artifacts/p4_rescore",
        ],
        source="p4_rescore",
    )
    cand = pipe.create_candidate(proposal, base.baseline_id)

    async def run_eval(_c: Any) -> Any:
        return candidate_ev

    evaluated = asyncio.run(pipe.evaluate_candidate(
        cand.candidate_id, run_eval=run_eval,
    ))
    ev = evaluated.evaluation
    assert ev is not None
    result: dict[str, Any] = {
        "candidate_id": evaluated.candidate_id,
        "status": evaluated.status.value,
        "gate_passed": ev.verdict.passed,
        "violations": list(ev.verdict.violations),
        "warnings": list(ev.verdict.warnings),
        "promotion_executed": False,
        "baseline_metrics": ev.baseline.model_dump(),
        "candidate_metrics": ev.candidate.model_dump(),
        "deltas": {k: round(v, 6) for k, v in ev.deltas.items()},
        "provenance": {
            "baseline_report_id": baseline.report_id,
            "candidate_report_id": candidate_ev.report_id,
            "built_from_measured_outcomes": True,
            "graders": "r2 (corrected)",
        },
        "note": (
            "Corrected-grader E8 gate decision only. On pass the candidate "
            "is held in APPROVED-CANDIDATE state awaiting explicit human "
            "promotion; production configuration is untouched."
        ),
    }
    if ev.verdict.passed:
        approved = pipe.approve(
            evaluated.candidate_id,
            approver=(
                "P4-closure directive (approved-candidate; promotion "
                "withheld pending explicit human promotion)"
            ),
            note="rescored evidence still passes the E8 regression gate",
        )
        result["status"] = approved.status.value
        result["approved_candidate"] = True
    gate_path = p4_root / "e8_gate" / "deepseek_candidate_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="artifacts/p4_rescore")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    out_root = Path(args.out)
    suite = load_benchmark_suite(str(DEFAULT_SUITE_V2_PATH))
    suite_by_id = {c.case_id: c for c in suite.cases}
    cases_by_goal = {cid: _norm(str(c.goal)) for cid, c in suite_by_id.items()}

    stats: dict[str, Any] = {"flips": []}
    for rp in sorted(artifacts.glob("**/v2_*.json")):
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not any(
            o.get("mode") == "llm_agent" for o in data.get("outcomes", [])
        ):
            continue
        rescore_report(
            rp, suite_by_id, cases_by_goal, out_root, stats,
            artifacts_root=artifacts,
        )

    summary = {
        "rescored_reports": len(_MIRROR_COUNTER),
        "semantics": (
            "r2: output_number/output_regex replace brittle exact strings "
            "on design_02/impl_01/e2e_03; JUDGE_ERROR reclassification and "
            "exclusion from weighted scores and agent-failure accounting"
        ),
        "outcome_changes": stats["flips"],
    }
    (out_root / "corrected_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"rescored {len(_MIRROR_COUNTER)} reports -> {out_root}")
    for f in stats["flips"]:
        if f["old_success"] != f["new_success"]:
            print(
                f"  flip: {Path(f['report']).parent.name}/"
                f"{f['case_id']} r{f['repeat']} "
                f"{f['old_success']} -> {f['new_success']} "
                f"(payload_found={f['payload_found']})"
            )

    if not args.skip_gate:
        # Collect the corrected evidence for the two gate arms. The
        # baseline is the P3 frozen-baseline arm; the candidate is the
        # DeepSeek P3-Short evidence (payload-bearing runs under
        # artifacts/p3short/ds, whose payloads allow full deterministic
        # regrading, unlike the payload-less copies in
        # artifacts/p3/p3short_ds_evidence).
        for prefix, arm in (
            ("p3_baseline_repro", "baseline_repro"),
            ("p3short_ds_rep", "p3short_ds"),
        ):
            dst = out_root / "arms" / arm
            dst.mkdir(parents=True, exist_ok=True)
            for corrected in sorted(out_root.glob(f"{prefix}*.corrected.json")):
                (dst / corrected.name).write_text(
                    corrected.read_text(encoding="utf-8"), encoding="utf-8"
                )
        gate = recompute_e8_gate(out_root / "e8_gate" / "store")
        print(
            f"corrected E8 gate: candidate={gate['candidate_id']} "
            f"status={gate['status']} passed={gate['gate_passed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
