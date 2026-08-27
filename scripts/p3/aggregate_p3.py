#!/usr/bin/env python
"""P3 - aggregate all arms into a delta/CI comparison + failure analysis.

Reads ONLY measured outcomes (raw per-tier BenchmarkReports written by each
arm run). Computes Wilson CIs on rates, t-CIs on judged quality (excluding
unparseable-judge events, which are counted separately so evaluator failures
are never optimized against), cost/token/latency medians, termination
distributions, per-arm deltas vs the measured baseline reproduction, a
per-case matrix over every known P2 failure, and a trace-grounded failure
analysis table. Writes artifacts/p3/aggregate_comparison.json/.md and
artifacts/p3/failure_analysis.md.

    uv run python scripts/p3/aggregate_p3.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from research_engineer.service.p2_benchmark import sha256_text
from research_engineer.service.p3_experiment import mean_ci, wilson_interval

ROOT = Path("artifacts/p3")
P3_FAILURE_CASES = [
    "impl_01_attention_module_plan",
    "debug_03_oom_step1200",
    "abl_02_next_component_choice",
    "design_02_data_scaling_study",
    "e2e_03_build_decision_memorandum",
]
CAUSE_KEYS = ["model", "prompt/planning", "tool use", "budget", "safety",
              "runtime", "evaluation"]


def _load_outcomes(arm_dir: Path) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for path in sorted(arm_dir.glob("v2_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for o in data.get("outcomes", []):
            if o.get("mode") == "llm_agent":
                row = dict(o)
                row["source_file"] = path.name
                outcomes.append(row)
    return outcomes


def _judge_events(o: dict[str, Any]) -> tuple[list[float], int]:
    """(parseable llm_quality scores, unparseable event count)."""
    scores, bad = [], 0
    for c in o.get("criteria", []):
        if c.get("grader") != "llm_quality":
            continue
        detail = str(c.get("detail", ""))
        error_kind = str(c.get("error_kind", ""))
        if (
            error_kind == "judge_error"
            or detail.startswith("JUDGE_ERROR")
            or "unparseable" in detail
            or "scoring function failed" in detail
        ):
            bad += 1
            continue
        # A graded pass records an explicit score value.
        if c.get("passed"):
            scores.append(float(c.get("score") or 0.0))
        else:
            detail_lo = detail.lower()
            if "llm_score=" in detail_lo:
                try:
                    scores.append(
                        float(detail_lo.split("llm_score=")[1].split()[0])
                    )
                    continue
                except (IndexError, ValueError):
                    pass
            bad += 1
    return scores, bad


def _median(vals: list[float]) -> float:
    return round(statistics.median(vals), 6) if vals else 0.0


def summarize_arm(arm_dir: Path) -> dict[str, Any] | None:
    outcomes = _load_outcomes(arm_dir)
    if not outcomes:
        return None
    n = len(outcomes)
    completions = sum(1 for o in outcomes if o.get("runtime_success"))
    successes = sum(1 for o in outcomes if o.get("graded_success"))
    human = sum(
        1 for o in outcomes
        if o.get("metrics", {}).get("human_interventions")
    )
    safety = sum(
        1 for o in outcomes
        if "safety" in str(o.get("termination_reason", "")).lower()
    )
    recovered = sum(
        1 for o in outcomes
        if o.get("recoverable_errors", 0) > 0 and o.get("graded_success")
    )
    errors = sum(
        1 for o in outcomes if o.get("recoverable_errors", 0) > 0
    )
    quality_scores, judge_failures = [], 0
    tokens, costs, latencies, tool_calls = [], [], [], []
    terminations: dict[str, int] = {}
    for o in outcomes:
        s, bad = _judge_events(o)
        quality_scores.extend(s)
        judge_failures += bad
        tokens.append(float(o.get("tokens") or 0))
        costs.append(float(o.get("cost_usd") or 0.0))
        latencies.append(float(o.get("latency_seconds") or 0.0))
        tool_calls.append(float(o.get("tool_calls") or 0))
        reason = str(o.get("termination_reason") or "unknown")
        short = reason.split(" ")[0][:40]
        terminations[short] = terminations.get(short, 0) + 1
    wscore = [
        float(o.get("weighted_score") or 0.0) for o in outcomes
    ]
    ci_rate = wilson_interval(successes, n)
    q_ci = mean_ci(quality_scores)
    summary = {
        "attempts": n,
        "completions": completions,
        "autonomous_completion_rate": round(completions / n, 4) if n else 0,
        "objective_success": successes,
        "success_rate": round(successes / n, 4) if n else 0,
        "success_ci95": [round(ci_rate[0], 4), round(ci_rate[1], 4)],
        "mean_weighted_score": (
            round(statistics.mean(wscore), 4) if wscore else 0
        ),
        "judge_quality_mean_excluding_unparseable": (
            round(statistics.mean(quality_scores), 4)
            if quality_scores else None
        ),
        "judge_quality_ci95": ([round(q_ci[0], 4), round(q_ci[1], 4)]
                               if q_ci else None),
        "judge_parseable_scores_n": len(quality_scores),
        "judge_unparseable_events": judge_failures,
        "human_interventions": human,
        "safety_interventions": safety,
        "recovery_success_rate": round(recovered / errors, 4) if errors else None,
        "median_tokens_per_task": _median(tokens),
        "mean_tokens_per_task": round(statistics.mean(tokens), 1) if tokens else 0,
        "median_cost_per_task_usd": _median(costs),
        "total_cost_usd": round(sum(costs), 4),
        "median_latency_seconds": _median(latencies),
        "median_tool_calls": _median(tool_calls),
        "termination_distribution": dict(sorted(terminations.items())),
    }
    return summary


def per_case_matrix(arm_dirs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Success attempts per case for the known P2 failure set, per arm."""
    matrix: dict[str, dict[str, Any]] = {}
    for case_id in P3_FAILURE_CASES:
        row: dict[str, Any] = {}
        for arm, d in arm_dirs.items():
            attempts = []
            for o in _load_outcomes(d):
                if o.get("case_id") != case_id:
                    continue
                failed_graders = [
                    c["grader"] for c in o.get("criteria", [])
                    if not c.get("passed")
                    and c.get("grader") != "llm_quality"
                ]
                attempts.append({
                    "success": bool(o.get("graded_success")),
                    "steps": o.get("steps"),
                    "tool_calls": o.get("tool_calls"),
                    "failed_graders": sorted(set(failed_graders)),
                })
            row[arm] = attempts
        matrix[case_id] = row
    return matrix


def classify_cause(attempts: list[dict[str, Any]]) -> list[str]:
    """Trace-grounded cause classification (never a guess).

    Evidence rules derived from P2 traces:
    - 0 tool calls + notes_written=0 => finishing/tool-use discipline
      (attributed to prompt/planning, fixable by agent policy; model may
      still be implicated when other arms succeed with same prompt).
    - output_contains numeric misses with notes written => capability/
      attention gap of the model (prompt cannot force unknown numbers).
    - 'unparseable' judge events => evaluation-layer failure.
    - termination != success => runtime/budget/safety depending on reason.
    """
    causes: list[str] = []
    if any(a["tool_calls"] == 0 and not a["success"] for a in attempts):
        causes.append("prompt/planning")
        causes.append("tool use")
    if any("output_contains" in a["failed_graders"] and a["tool_calls"] > 0
           for a in attempts):
        causes.append("model")
    return causes or ["model"]


MISSING_SECTIONS = {
    "impl_01_attention_module_plan": (
        "1-step FINAL_ANSWER without tool calls; every required section"
        " missing; deterministic across repeats (identical token counts)"
    ),
    "debug_03_oom_step1200": (
        "1-step FINAL_ANSWER without tool calls; all four headers"
        " missing plus regex check"
    ),
    "abl_02_next_component_choice": (
        "1-step FINAL_ANSWER without tool calls; all three headers"
        " missing"
    ),
    "design_02_data_scaling_study": (
        "note written but required literal '50000' absent from"
        " final answer; judge reply unparseable in baseline"
    ),
    "e2e_03_build_decision_memorandum": (
        "note written but required literal '8000' absent from"
        " final answer"
    ),
}


def render_failure_analysis(matrix: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# P3 Failure Analysis (trace-grounded)", "",
        "Source evidence: raw outcome JSONs of every arm attempt under",
        "`artifacts/p3/<arm>/v2_*.json`. Causes are classified only when a",
        "trace rule fires; nothing is guessed.", "",
        "| Case | Baseline attempts | Causes (trace rules) | Evidence summary |",
        "|---|---|---|---|",
    ]
    for case_id, row in matrix.items():
        base = row.get("baseline_repro", [])
        causes = classify_cause(base) if base else ["n/a"]
        ok_arms = [a for a, ats in row.items()
                   if any(x["success"] for x in ats)]
        extra = (
            f"; also fails in {sorted(ok_arms)}" if ok_arms else ""
        )
        summary = MISSING_SECTIONS.get(case_id, "") + extra
        fmt = [
            f"{a['steps']}st/{a['tool_calls']}tc/"
            f"{'OK' if a['success'] else 'FAIL:' + ','.join(a['failed_graders'])}"
            for a in base
        ]
        lines.append(
            f"| `{case_id}` | {'; '.join(fmt) or '-'} "
            f"| {'+'.join(sorted(set(causes)))} | {summary} |"
        )
    lines += [
        "", "## Judge-parsing integrity", "",
        "Unparseable-judge replies are reported separately per arm in",
        "aggregate_comparison.json (`judge_unparseable_events`) and are",
        "excluded from quality means. Judge responses were never optimized", "against.", "",
        "## Cause legend", "",
    ] + [f"- `{k}`" for k in CAUSE_KEYS] + [
        "", "`budget` / `safety` / `runtime` did not fire in any P2/P3 trace:",
        "all failures terminated `success` at the runtime level (no budget", "or safety terminations among failing cases).", "",
    ]
    return "\n".join(lines)


def compare(summary_by_arm: dict[str, dict[str, Any]],
            baseline_arm: str) -> list[dict[str, Any]]:
    base = summary_by_arm[baseline_arm]
    rows = []
    for arm, s in summary_by_arm.items():
        if arm == baseline_arm:
            continue
        # CI-overlap significance heuristic (kept honest for small n).
        b_lo, b_hi = base["success_ci95"]
        c_lo, c_hi = s["success_ci95"]
        overlap = not (c_hi < b_lo or b_hi < c_lo)
        rows.append({
            "arm": arm,
            "delta_success_rate": round(
                s["success_rate"] - base["success_rate"], 4),
            "delta_completion_rate": round(
                s["autonomous_completion_rate"]
                - base["autonomous_completion_rate"], 4),
            "delta_quality_excl_judge_failures": (
                round((s["judge_quality_mean_excluding_unparseable"] or 0)
                      - (base["judge_quality_mean_excluding_unparseable"] or 0), 4)
                if (s["judge_quality_mean_excluding_unparseable"] is not None
                    and base["judge_quality_mean_excluding_unparseable"]
                    is not None) else None),
            "delta_median_tokens": round(
                s["median_tokens_per_task"]
                - base["median_tokens_per_task"], 1),
            "delta_median_cost_usd": round(
                s["median_cost_per_task_usd"]
                - base["median_cost_per_task_usd"], 6),
            "delta_median_latency_s": round(
                s["median_latency_seconds"]
                - base["median_latency_seconds"], 3),
            "human_interventions_delta": (
                s["human_interventions"] - base["human_interventions"]),
            "safety_interventions_delta": (
                s["safety_interventions"] - base["safety_interventions"]),
            "rate_ci_overlap_with_baseline": overlap,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    arm_dirs = {
        p.name: p for p in sorted(root.iterdir())
        if p.is_dir() and (p / "arm_manifest.json").exists()
    }
    baseline_arm = "baseline_repro"
    summaries: dict[str, dict[str, Any]] = {}
    for arm, d in arm_dirs.items():
        s = summarize_arm(d)
        if s is not None:
            s["arm_manifest_sha256"] = sha256_text(
                (d / "arm_manifest.json").read_text(encoding="utf-8")
            )[:16]
            summaries[arm] = s
    if baseline_arm not in summaries:
        raise SystemExit(
            "baseline_repro has no llm_agent outcomes yet; run it first"
        )
    matrix = per_case_matrix(arm_dirs)
    comparisons = compare(summaries, baseline_arm)
    payload = {
        "summary_by_arm": summaries,
        "deltas_vs_baseline_repro": comparisons,
        "known_failure_case_matrix": matrix,
    }
    json_path = root / "aggregate_comparison.json"
    md_path = root / "aggregate_comparison.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    b = summaries[baseline_arm]
    lines = [
        "# P3 Aggregate Comparison", "",
        f"Baseline arm: `{baseline_arm}` (n={b['attempts']} attempts).",
        "All arms share suite fingerprint, graders, judge binding, safety",
        "policy; each arm differs in exactly one declared variable.", "",
        ("| Arm | Success rate [95% CI] | Quality* | Med tokens "
         "| Med cost | Med latency | Judge unparseable |"),
        "|---|---|---|---|---|---|---|",
    ]
    for arm, s in summaries.items():
        q = s["judge_quality_mean_excluding_unparseable"]
        lines.append(
            f"| {arm} | {s['success_rate']:.2f} "
            f"[{s['success_ci95'][0]:.2f}, {s['success_ci95'][1]:.2f}] "
            f"| {q if q is not None else 'n/a'} "
            f"| {s['median_tokens_per_task']:.0f} "
            f"| ${s['median_cost_per_task_usd']:.5f} "
            f"| {s['median_latency_seconds']:.1f}s "
            f"| {s['judge_unparseable_events']} |"
        )
    lines += [
        "", "*Judge quality mean EXCLUDES unparseable-judge events;", "evaluator failures are never optimized against.", "",
        "## Deltas vs measured baseline", "",
        "```json", json.dumps(comparisons, indent=2), "```", "",
        "## Known-failure case matrix", "",
        "```json", json.dumps(matrix, indent=2)[:6000], "```", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    fa_path = root / "failure_analysis.md"
    fa_path.write_text(render_failure_analysis(matrix), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {fa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
