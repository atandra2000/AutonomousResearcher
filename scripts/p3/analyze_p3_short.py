#!/usr/bin/env python
"""P3-Short - model selection analysis.

Aggregates six full-suite passes (3 models x 2 independent repeats x 20
cases), separates genuine agent failures from judge/evaluator malfunctions,
computes Wilson CIs / t-CIs, per-case flips vs the GLM baseline, and writes
artifacts/p3short/p3_short_report.{md,json}. Read-only w.r.t. production
configuration: nothing here promotes a candidate or mutates config.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from research_engineer.service.p3_experiment import (
    mean_ci,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "artifacts" / "p3short"

#: (model_label, repeat_no, path relative to repo root)
PASSES: list[tuple[str, int, str]] = [
    ("glm", 1, "artifacts/p3/baseline_repro/v2_full_suite.json"),
    ("glm", 2, "artifacts/p3short/glm/rep2/baseline_repro/"
               "v2_full_suite.json"),
    ("kimi", 1, "artifacts/p3/exp_a1_kimi_k27_code/v2_full_suite.json"),
    ("kimi", 2, "artifacts/p3short/kimi/rep2/exp_a1_kimi_k27_code/"
                "v2_full_suite.json"),
    ("ds", 1, "artifacts/p3short/ds/rep1/exp_f_deepseek_v4_pro/"
              "v2_full_suite.json"),
    ("ds", 2, "artifacts/p3short/ds/rep2/exp_f_deepseek_v4_pro/"
              "v2_full_suite.json"),
]

MODEL_NAMES = {
    "glm": "ollama/glm-5.3-flash (baseline)",
    "kimi": "ollama/kimi-k2.7-code",
    "ds": "ollama/deepseek-v4-pro:0813",
}

_PARSE_MARKERS = (
    "unparseable", "unparsable", "parse_error", "parse error", "malformed",
)


def _load_outcomes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["outcomes"])


def _judge_malfunction(outcome: dict[str, Any]) -> bool:
    """True when the only failing required criterion is the LLM judge
    producing an unparseable/absent response - evaluator failure, not a
    genuine agent failure. Excluded from agent success accounting."""
    criteria = outcome.get("criteria") or []
    other_failed = [
        c for c in criteria
        if c.get("required") and c.get("grader") != "llm_quality"
        and not c.get("passed")
    ]
    lq = next((c for c in criteria if c.get("grader") == "llm_quality"), None)
    if other_failed or lq is None or lq.get("passed"):
        return False
    detail = str(lq.get("detail", "")).lower()
    if "llm_score=" not in detail:
        return True
    return any(m in detail for m in _PARSE_MARKERS)


def _summarize(model: str,
               passes: dict[int, list[dict[str, Any]]],
               run_metrics: dict[int, dict[str, Any]]) -> dict:
    flat = [o for outs in passes.values() for o in outs]
    n = len(flat)
    completed = sum(1 for o in flat if o.get("runtime_success"))
    successes = sum(1 for o in flat if o.get("graded_success"))
    agent_failures = [
        o for o in flat
        if not o.get("graded_success") and not _judge_malfunction(o)
    ]
    judge_failures = [
        o for o in flat
        if not o.get("graded_success") and _judge_malfunction(o)
    ]
    latencies = sorted(float(o["latency_seconds"]) for o in flat)
    p95_idx = max(0, min(n - 1, int(0.95 * (n - 1))))
    fail_cats: Counter[str] = Counter()
    for o in agent_failures:
        for cat in (o.get("failure_categories") or []):
            fail_cats[str(cat)] += 1
    term = Counter(str(o.get("termination")) for o in flat)
    scores = [float(o["weighted_score"]) for o in flat]
    costs = [float(o["cost_usd"]) for o in flat]
    tokens = [float(o["tokens"]) for o in flat]
    tool_eff = []
    for outs in passes.values():
        tool_calls: float = 0
        steps_total = 0
        for o in outs:
            tc = o.get("tool_calls") or 0
            tool_calls += tc if isinstance(tc, (int, float)) else len(tc)
            steps_total += int(o.get("steps") or 0)
        if steps_total:
            tool_eff.append(tool_calls / steps_total)
    success_lo, success_hi = wilson_interval(successes, n)
    q_lo, q_hi = mean_ci(scores) or (0.0, 0.0)
    recoverable = [o for o in flat if o.get("recoverable_errors")]
    recovered = [o for o in recoverable if o.get("recovered")]
    run_rec = [
        float(m.get("recovery_success_rate") or 0.0)
        for m in run_metrics.values() if m
    ]
    recovery_rate = (
        statistics.fmean(run_rec) if run_rec
        else (len(recovered) / len(recoverable) if recoverable else None)
    )
    return {
        "model": MODEL_NAMES[model],
        "repeats": {
            rep: {
                "successes": sum(1 for o in outs if o.get("graded_success")),
                "total": len(outs),
                "quality_mean": round(statistics.fmean(
                    float(o["weighted_score"]) for o in outs), 4),
            }
            for rep, outs in sorted(passes.items())
        },
        "n_cases_total": n,
        "autonomous_completion_rate": round(completed / n, 4),
        "objective_success_rate": round(successes / n, 4),
        "success_rate_wilson95": [round(success_lo, 4),
                                  round(success_hi, 4)],
        "research_quality_score_mean": round(statistics.fmean(scores), 4),
        "quality_t_ci95": [round(q_lo, 4), round(q_hi, 4)],
        "human_interventions": sum(int(o.get("human_interventions") or 0)
                                   for o in flat),
        "safety_interventions": sum(int(o.get("safety_interventions") or 0)
                                    for o in flat),
        "gateway_denials": sum(int(o.get("gateway_denials") or 0)
                               for o in flat),
        "recoverable_errors": len(recoverable),
        "recovery_success_rate": recovery_rate,
        "run_level_recovery_success_rate": run_rec,
        "median_latency_s": round(statistics.median(latencies), 2),
        "p95_latency_s": round(latencies[p95_idx], 2),
        "median_tokens_per_task": round(statistics.median(tokens)),
        "median_cost_per_task_usd": round(statistics.median(costs), 6),
        "total_cost_usd": round(sum(costs), 4),
        "tool_efficiency_mean": round(statistics.fmean(tool_eff), 4)
        if tool_eff else None,
        "termination_distribution": dict(term),
        "agent_failure_categories": dict(fail_cats),
        "judge_malfunction_cases": [str(o.get("case_id"))
                                    for o in judge_failures],
        "agent_failure_cases": [str(o.get("case_id"))
                                for o in agent_failures],
    }


def _repeat_consistency(summary: dict) -> str:
    reps = list(summary["repeats"].values())
    totals = {r["total"] for r in reps}
    if len(reps) == 2 and len(totals) == 1:
        total = totals.pop()
        succ = {r["successes"] for r in reps}
        if len(succ) == 1:
            return f"identical ({succ.pop()}/{total} both repeats)"
        vals = sorted(r["successes"] for r in reps)
        return f"varies across repeats: {vals[0]}/{total} vs {vals[1]}/{total}"
    return f"unexpected repeat layout: {summary['repeats']}"


def _flips_vs_glm(by_case: dict[str, dict[int, dict[str, dict]]],
                  model: str) -> dict[str, list[str]]:
    base = by_case["glm"]
    cand = by_case[model]
    better: list[str] = []
    worse: list[str] = []
    mixed: list[str] = []
    cids = sorted(set(base[1]) | set(base[2]) | set(cand[1]) | set(cand[2]))
    for cid in cids:
        b = [bool(base[r].get(cid, {}).get("graded_success"))
             for r in (1, 2)]
        c = [bool(cand[r].get(cid, {}).get("graded_success"))
             for r in (1, 2)]
        if all(c) and not any(b):
            better.append(cid)
        elif not any(c) and all(b):
            worse.append(cid)
        elif b != c:
            mixed.append(f"{cid} base={b} candidate={c}")
    return {"won_both_repeats": better, "lost_both_repeats": worse,
            "inconsistent_across_repeats": mixed}


def _estimate_zero_cost_models(
    summaries: dict[str, dict],
    by_case: dict[str, dict[int, dict[str, dict]]],
) -> None:
    """Models added without a pricing entry record cost_usd == 0.0.

    For such models, estimate cost from measured tokens using the blended
    $/token rate measured on the Kimi arm (closest priced model) and expose
    it as ``median_cost_per_task_usd_estimated``; the raw zero is kept for
    auditability. Pricing entry for deepseek-v4-pro was added to cost.py
    AFTER the P3-Short runs, so completed passes cannot be repriced.
    """
    kimi_tokens = [
        float(o["tokens"])
        for rep in by_case.get("kimi", {}).values()
        for o in rep.values()
    ]
    kimi_costs = [
        float(o["cost_usd"])
        for rep in by_case.get("kimi", {}).values()
        for o in rep.values()
    ]
    if not kimi_tokens or not any(c > 0 for c in kimi_costs):
        return
    blended = sum(kimi_costs) / sum(kimi_tokens)
    for model, s in summaries.items():
        if s["median_cost_per_task_usd"] == 0.0:
            med_tok = float(s["median_tokens_per_task"])
            s["median_cost_per_task_usd_estimated"] = round(
                med_tok * blended, 6)
            s["total_cost_usd_estimated"] = round(
                s["total_cost_usd"] + med_tok * blended
                * (s["n_cases_total"] / 2), 6)
            s["cost_note"] = (
                "recorded cost was 0.0 (model unpriced at run time); "
                f"estimate = measured tokens x kimi blended rate "
                f"({blended:.3e} USD/token)"
            )
            s["median_cost_per_task_usd"] = None
            s["total_cost_usd"] = None


def _fmt_pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0%}"


def main() -> None:
    passes: dict[str, dict[int, list[dict[str, Any]]]] = {}
    by_case: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    run_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    for model, rep, rel in PASSES:
        path = REPO / rel
        if not path.exists():
            raise SystemExit(
                f"missing pass artifact ({model} rep{rep}): {path} - run "
                "scripts/p3/run_p3_short.sh first"
            )
        outs = _load_outcomes(path)
        passes.setdefault(model, {})[rep] = outs
        by_case.setdefault(model, {})[rep] = {
            str(o["case_id"]): o for o in outs
        }
        run_metrics.setdefault(model, {})[rep] = (
            json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        )

    summaries = {
        m: _summarize(m, passes[m], run_metrics[m])
        for m in ("glm", "kimi", "ds")
    }
    for s in summaries.values():
        s["repeat_consistency"] = _repeat_consistency(s)
    _estimate_zero_cost_models(summaries, by_case)
    flips = {m: _flips_vs_glm(by_case, m) for m in ("kimi", "ds")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "P3-Short model selection",
        "judge_malfunction_policy": (
            "cases whose only failing required criterion is an unparseable/"
            "absent llm_quality response are classified as evaluator "
            "failures and excluded from agent-failure accounting"
        ),
        "models": summaries,
        "per_case_flips_vs_glm": flips,
    }
    json_path = OUT_DIR / "p3_short_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")

    lines = ["# P3-Short - Model Selection Report", ""]
    for m in ("glm", "kimi", "ds"):
        s = summaries[m]
        lines += [
            f"## {s['model']}",
            f"- autonomous completion: {s['autonomous_completion_rate']:.0%}",
            f"- objective success: {s['objective_success_rate']:.0%} "
            f"(Wilson95 {s['success_rate_wilson95']})",
            f"- quality mean: {s['research_quality_score_mean']} "
            f"(t-CI95 {s['quality_t_ci95']})",
            f"- repeat consistency: {s['repeat_consistency']}",
            f"- human/safety interventions: {s['human_interventions']}/"
            f"{s['safety_interventions']}, recovery "
            f"{_fmt_pct(s['recovery_success_rate'])} "
            f"(run-level: {s['run_level_recovery_success_rate']})",
            f"- latency median/p95: {s['median_latency_s']}s/"
            f"{s['p95_latency_s']}s",
            f"- tokens/task median: {s['median_tokens_per_task']}, "
            f"cost/task median: {s['median_cost_per_task_usd']}"
            + (f" (estimated: "
               f"{s['median_cost_per_task_usd_estimated']})"
               if s.get('median_cost_per_task_usd_estimated') else ""),
            f"- termination: {s['termination_distribution']}",
            f"- tool efficiency mean: {s['tool_efficiency_mean']}",
            f"- agent failures ({len(s['agent_failure_cases'])}): "
            f"{s['agent_failure_cases']} cats="
            f"{s['agent_failure_categories']}",
            f"- judge malfunctions: {s['judge_malfunction_cases']}",
            "",
        ]
    for m in ("kimi", "ds"):
        f = flips[m]
        lines += [
            f"## Per-case flips vs GLM - {MODEL_NAMES[m]}",
            f"- won both repeats: {f['won_both_repeats']}",
            f"- lost both repeats: {f['lost_both_repeats']}",
            f"- inconsistent: {f['inconsistent_across_repeats']}",
            "",
        ]
    md_path = OUT_DIR / "p3_short_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()


