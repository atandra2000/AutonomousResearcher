"""P1 §5/§6 tests — benchmark runner: payload views, taxonomy, E2E run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_engineer.runtime.models import AgentTermination
from research_engineer.service.benchmark import DEFAULT_SUITE_PATH
from research_engineer.service.benchmark_runner import (
    FAILURE_TAXONOMY,
    BenchmarkRunner,
    CaseOutcome,
    classify_failure,
)
from research_engineer.service.models import RunStatus


def _outcome(**overrides: object) -> CaseOutcome:
    """A completed, passing, clean outcome to mutate per test."""
    base = dict(
        case_id="case_x", repeat=1, category="implementation",
        mode="deterministic_sandbox", agent_kind="bench_tool",
        submitted_at=__import__("datetime").datetime.now(),
        latency_seconds=0.01, status=RunStatus.COMPLETED.value,
        runtime_success=True, graded_success=True, weighted_score=1.0,
        criteria=[], steps=3, tool_calls=6, tokens=0, cost_usd=0.0,
        recoverable_errors=0, fatal_errors=0, recovered=False,
        human_interventions=0, safety_interventions=0, gateway_denials=0,
        termination="success", termination_reason="Evaluator signalled",
    )
    base.update(overrides)
    return CaseOutcome(**base)


class TestStoredViewsAndTaxonomy:
    def test_payload_view_parses_worker_result(self) -> None:
        from research_engineer.service.benchmark_runner import (
            _StoredExecutionView,
        )

        view = _StoredExecutionView({
            "termination": "success",
            "reason": "Evaluator signalled completion",
            "output": {"notes_written": 2},
            "context": {
                "current_step": 3, "tool_calls": 6, "tokens": 42,
                "cost_usd": 0.001, "recoverable_errors": 1,
                "fatal_errors": ["x"], "duration_seconds": 0.5,
                "human_interventions": 1,
                "tool_call_log": [
                    {"tool": "a", "status": "success"}, "junk-entry",
                ],
                "safety_state": {"decisions": [{"action": "continue"}]},
            },
        })
        ctx = view.context
        assert ctx.is_success()
        assert ctx.current_step == 3 and ctx.tool_calls == 6
        assert ctx.recoverable_errors == 1 and ctx.fatal_error_count == 1
        # Non-dict log entries are filtered; junk cannot break graders.
        assert [e["tool"] for e in ctx.metadata["tool_call_log"]] == ["a"]
        assert view.output == {"notes_written": 2}

    def test_unknown_termination_maps_to_none(self) -> None:
        from research_engineer.service.benchmark_runner import (
            _StoredContextView,
        )

        ctx = _StoredContextView({"termination": "warp_drive"})
        assert ctx.termination is None
        assert not ctx.is_success()

    def test_taxonomy_covers_every_label(self) -> None:
        assert set(FAILURE_TAXONOMY) >= {
            "planning", "tool_use", "coding", "experiment",
            "research_grounding", "analysis", "safety", "budget",
            "infrastructure", "recovery", "evaluation",
        }

    def test_budget_and_timeout_labels(self) -> None:
        o = _outcome(status=RunStatus.FAILED.value,
                     termination="budget_exhausted",
                     termination_reason="max steps budget exhausted")
        assert classify_failure(o) == ["budget"]
        o = _outcome(status=RunStatus.FAILED.value,
                     termination="timeout")
        assert classify_failure(o) == ["budget", "infrastructure"]

    def test_safety_and_replan_labels(self) -> None:
        o = _outcome(
            status=RunStatus.FAILED.value,
            termination="safety_terminated",
            termination_reason="safety:replan_limit.replans=2 reached",
        )
        assert classify_failure(o) == ["safety"]

    def test_incomplete_without_recovery_is_recovery_failure(self) -> None:
        o = _outcome(status=RunStatus.FAILED.value,
                     termination="error",
                     termination_reason="unrecoverable boom",
                     recoverable_errors=3, recovered=False)
        labels = classify_failure(o)
        assert "recovery" in labels
        assert "infrastructure" in labels

    def test_gateway_denials_label_tool_use_plus_safety(self) -> None:
        o = _outcome(graded_success=False, runtime_success=False,
                     gateway_denials=1, status=RunStatus.FAILED.value,
                     termination="safety_terminated",
                     termination_reason="safety:policy_violation ...")
        labels = classify_failure(o)
        assert "tool_use" in labels and "safety" in labels

    def test_clean_success_has_no_labels(self) -> None:
        assert classify_failure(_outcome()) == []

    def test_failed_grading_after_success_runtime_labeled_evaluation(
        self,
    ) -> None:
        from research_engineer.service.benchmark_runner import (
            CriterionOutcome,
        )

        criteria = [CriterionOutcome(
            grader="output_json_field", weight=1.0, required=True,
            passed=False, score=0.0, detail="missing field",
        )]
        o = _outcome(graded_success=False, criteria=criteria)
        assert classify_failure(o) == ["evaluation"]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Real 3-case x2-repeat stack run (gateway+safety+checkpointing)."""
    base = tmp_path_factory.mktemp("bench_e2e")
    runner = BenchmarkRunner(base)
    report = _run = __import__("asyncio").run(runner.run_suite(
        DEFAULT_SUITE_PATH, repeat=2, report_name="e2e_mini",
        case_ids={
            "litdisc_01_kv_cache_survey",
            "impl_03_flaky_registry_load",
            "e2e_04_external_fetch_guardrail",
        },
    ))
    return {"report": report, "base": base}


class TestEndToEndStackRun:
    def test_all_outcomes_recorded(self, e2e: dict) -> None:
        report = e2e["report"]
        assert len(report.outcomes) == 6
        assert report.cases_total == 6

    def test_tool_case_passes_deterministically(self, e2e: dict) -> None:
        rows = [o for o in e2e["report"].outcomes
                if o.case_id.startswith("litdisc")]
        assert len(rows) == 2
        for o in rows:
            assert o.status == RunStatus.COMPLETED.value
            assert o.graded_success and o.weighted_score == 1.0
            # write + list per checklist item through the real gateway.
            assert o.tool_calls >= 2 * o.steps > 0
            assert o.artifacts_added >= 1

    def test_flaky_case_recovers_transient_failures(self, e2e: dict) -> None:
        rows = [o for o in e2e["report"].outcomes
                if o.case_id.startswith("impl_03")]
        assert all(o.status == RunStatus.COMPLETED.value for o in rows)
        assert all(o.graded_success for o in rows)
        assert all(o.recoverable_errors >= 1 for o in rows)
        assert all(o.recovered for o in rows)

    def test_guardrail_case_is_by_design_stop(self, e2e: dict) -> None:
        rows = [o for o in e2e["report"].outcomes
                if o.case_id.startswith("e2e_04")]
        assert rows
        for o in rows:
            assert o.status == RunStatus.FAILED.value
            assert o.termination == AgentTermination.SAFETY_TERMINATED.value
            assert o.gateway_denials == 1
            assert o.safety_interventions >= 1
            assert o.expected_failed_by_design
            # Enforcement behaved (bounded stop) so no defect labels; the
            # by-design flag + safety termination tell the real story.
            assert o.failure_categories == []

    def test_by_design_rows_excluded_from_rate_denominators(
        self, e2e: dict,
    ) -> None:
        m = e2e["report"].metrics
        assert m["by_design_exclusions"] == 2
        n_eff = 4  # 2 cases x 2 repeats remain
        completed = sum(
            1 for o in e2e["report"].outcomes
            if not o.expected_failed_by_design and o.graded_success
            and o.status == RunStatus.COMPLETED.value
            and o.human_interventions == 0
        )
        assert m["autonomous_completion_rate"] == pytest.approx(
            completed / n_eff
        )

    def test_repeat_spreads_cover_cases_consistently(self, e2e: dict) -> None:
        spreads = {s.case_id: s for s in e2e["report"].repeat_spreads}
        assert len(spreads) == 3
        assert all(s.consistent_success for s in spreads.values())
        tool = spreads["litdisc_01_kv_cache_survey"]
        assert tool.successes == [1, 1]

    def test_report_artifacts_written(self, e2e: dict) -> None:
        base: Path = e2e["base"]
        md = base / "e2e_mini.md"
        js = base / "e2e_mini.json"
        assert md.exists() and js.exists()
        text = md.read_text(encoding="utf-8")
        assert "Autonomous task completion rate" in text
        assert "(by design)" in text  # guardrail row annotated
        parsed = json.loads(js.read_text(encoding="utf-8"))
        assert parsed["cases_total"] == 6
        assert "metrics" in parsed

    def test_unknown_case_ids_rejected(self, tmp_path: Path) -> None:
        import asyncio

        runner = BenchmarkRunner(tmp_path)
        with pytest.raises(ValueError, match="unknown case_ids"):
            asyncio.run(runner.run_suite(
                DEFAULT_SUITE_PATH, case_ids={"no_such_case"},
            ))
