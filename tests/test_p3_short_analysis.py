"""Tests for the P3-Short analysis helpers (scripts/p3/analyze_p3_short.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3" / \
    "analyze_p3_short.py"
_spec = importlib.util.spec_from_file_location("analyze_p3_short", _SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("analyze_p3_short", mod)
mod.loader = _spec.loader  # type: ignore[attr-defined]
_spec.loader.exec_module(mod)

from analyze_p3_short import _flips_vs_glm, _judge_malfunction  # noqa: E402


def _outcome(criteria: list[dict[str, object]],
             **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "x", "status": "completed", "termination": "success",
        "graded_success": False, "runtime_success": True,
        "latency_seconds": 1.0, "weighted_score": 0.5, "steps": 2,
        "tool_calls": [1], "tokens": 100, "cost_usd": 0.001,
        "recoverable_errors": 0, "recovered": False,
        "human_interventions": 0, "safety_interventions": 0,
        "gateway_denials": 0, "failure_categories": ["evaluation"],
    }
    base.update(extra)
    base["criteria"] = criteria
    return base


def _crit(grader: str, passed: bool, detail: str = "") -> dict[str, object]:
    return {"grader": grader, "required": True, "passed": passed,
            "score": 1.0 if passed else 0.0, "detail": detail}


def test_judge_malfunction_unparseable_only() -> None:
    o = _outcome([
        _crit("termination", True),
        _crit("llm_quality", False, "unparseable judge response"),
    ])
    assert _judge_malfunction(o) is True


def test_judge_malfunction_missing_score() -> None:
    o = _outcome([
        _crit("termination", True),
        _crit("llm_quality", False, ""),
    ])
    assert _judge_malfunction(o) is True


def test_genuine_failure_not_reclassified() -> None:
    o = _outcome([
        _crit("termination", True),
        _crit("output_contains", False, "missing=['INTERFACE:']"),
        _crit("llm_quality", True, "llm_score=0.920 threshold=0.5"),
    ])
    assert _judge_malfunction(o) is False


def test_real_low_judge_score_is_genuine() -> None:
    o = _outcome([
        _crit("termination", True),
        _crit("llm_quality", False, "llm_score=0.100 threshold=0.5"),
    ])
    assert _judge_malfunction(o) is False


def test_flips_both_repeats() -> None:
    def rep(ok: bool) -> dict[str, dict[str, object]]:
        return {"a": _outcome([], case_id="a", graded_success=ok),
                "b": _outcome([], case_id="b", graded_success=False)}
    by_case = {
        "glm": {1: rep(False), 2: rep(False)},
        "kimi": {1: rep(True), 2: rep(True)},
    }
    flips = _flips_vs_glm(by_case, "kimi")
    assert flips["won_both_repeats"] == ["a"]
    assert flips["lost_both_repeats"] == []
    assert flips["inconsistent_across_repeats"] == []


def test_flips_inconsistent_across_repeats() -> None:
    by_case = {
        "glm": {1: {"a": _outcome([], case_id="a", graded_success=False)},
                2: {"a": _outcome([], case_id="a", graded_success=False)}},
        "ds": {1: {"a": _outcome([], case_id="a", graded_success=True)},
               2: {"a": _outcome([], case_id="a", graded_success=False)}},
    }
    flips = _flips_vs_glm(by_case, "ds")
    assert flips["won_both_repeats"] == []
    assert len(flips["inconsistent_across_repeats"]) == 1
