"""Tests for E4 - Agent Evaluation Harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML

from research_engineer.eval import (
    BudgetGrader,
    CompositeGrader,
    EvalReport,
    EvalResult,
    EvalRunner,
    EvalStatus,
    EvalSuite,
    EvalTask,
    GraderResult,
    LLMPromptGrader,
    MaxStepsGrader,
    ScriptedAgentFactory,
    aggregate,
    build_grader,
    compare_reports,
    load_report,
    load_suite,
    save_report,
)
from research_engineer.eval.graders import (
    GradingRequest,
    OutputEqualsGrader,
)
from research_engineer.runtime.models import (
    AgentBudget,
    AgentExecution,
    AgentTermination,
)

SUITE_DIR = Path(__file__).resolve().parent.parent / "evals"


# ---------------------------------------------------------------------------
# Suite loading / validation
# ---------------------------------------------------------------------------


class TestSuiteLoading:
    def test_load_sample_yaml(self) -> None:
        suite = load_suite(SUITE_DIR / "sample_suite.yaml")
        assert suite.suite_id == "sample_suite_v1"
        assert len(suite.cases) == 3
        assert all(c.goal for c in suite.cases)
        assert any(c.allowed_tools == ["read_doc"] for c in suite.cases)

    def test_load_json_roundtrip(self, tmp_path: Path) -> None:
        suite = load_suite(SUITE_DIR / "sample_suite.yaml")
        path = tmp_path / "suite.json"
        path.write_text(suite.model_dump_json(), encoding="utf-8")
        loaded = load_suite(path)
        assert loaded.model_dump() == suite.model_dump()

    def test_rejects_empty_and_duplicate(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text(yaml.safe_dump({"suite_id": "s", "cases": []}))
        with pytest.raises(ValueError):
            load_suite(empty)
        dup_case = {
            "case_id": "dup",
            "name": "d",
            "goal": "g",
            "metadata": {"script": {"steps": 1}},
        }
        dup = tmp_path / "dup.yaml"
        dup.write_text(yaml.safe_dump(
            {"suite_id": "s", "cases": [dup_case, dup_case]}
        ))
        with pytest.raises(ValueError):
            load_suite(dup)

    def test_rejects_bad_ids(self) -> None:
        with pytest.raises(Exception):
            EvalTask(case_id="bad id!", name="x", goal="g")
        with pytest.raises(Exception):
            EvalSuite(suite_id="", cases=[])

    def test_unknown_grader_rejected_by_registry(self) -> None:
        build_grader("termination")
        with pytest.raises(KeyError):
            build_grader("nope")


def _fake_execution(output: Any, *, reason: str = "ok") -> AgentExecution:
    from research_engineer.runtime.models import AgentContext

    ctx = AgentContext(goal="g")
    ctx.termination = AgentTermination.SUCCESS
    ctx.output = output
    exec_ = AgentExecution(context=ctx, termination=AgentTermination.SUCCESS)
    exec_.output = output
    exec_.reason = reason
    return exec_


class TestGraders:
    @pytest.mark.asyncio
    async def test_output_equals(self) -> None:
        grader = OutputEqualsGrader().configure({"expected": "hello"})
        request = GradingRequest(task=_task(), execution=_fake_execution("Hello"))
        result = await grader.grade(request)
        assert result.passed and result.score == 1.0
        strict = OutputEqualsGrader().configure(
            {"expected": "hello", "case_sensitive": True}
        )
        result2 = await strict.grade(request)
        assert not result2.passed

    @pytest.mark.asyncio
    async def test_output_contains_and_regex(self) -> None:
        from research_engineer.eval import OutputContainsGrader, OutputRegexGrader

        exec_ = _fake_execution({"answer": "the loss decreased to 0.42"})
        contains = OutputContainsGrader().configure(
            {"all_of": ["loss"], "any_of": ["0.42", "0.99"]}
        )
        assert (await contains.grade(GradingRequest(_task(), exec_))).passed
        regex = OutputRegexGrader().configure({"pattern": r"0\.\d+"})
        assert (await regex.grade(GradingRequest(_task(), exec_))).passed

    @pytest.mark.asyncio
    async def test_budget_grader_flags_forced_reason(self) -> None:
        from research_engineer.runtime.models import AgentContext

        ctx = AgentContext(goal="g")
        ctx.termination = AgentTermination.BUDGET_EXCEEDED
        ctx.termination_reason = "step budget exceeded"
        result = await BudgetGrader().grade(
            GradingRequest(_task(), AgentExecution(context=ctx, termination=ctx.termination))
        )
        assert not result.passed

    @pytest.mark.asyncio
    async def test_max_steps_grader(self) -> None:
        from research_engineer.runtime.models import AgentContext

        ctx = AgentContext(goal="g")
        ctx.current_step = 12
        grader = MaxStepsGrader().configure({"max_steps": 5})
        result = await grader.grade(GradingRequest(_task(), AgentExecution(context=ctx, termination=AgentTermination.ERROR)))
        assert not result.passed and "steps=12" in result.detail

    @pytest.mark.asyncio
    async def test_tool_usage_rejects_disallowed(self) -> None:
        from research_engineer.runtime.models import AgentContext

        ctx = AgentContext(goal="g")
        ctx.metadata["tool_call_log"] = [
            {"tool": "shell", "status": "allowed"},
        ]
        task = _task()
        task.allowed_tools = ["read_doc"]
        result = await build_grader("tool_usage").grade(
            GradingRequest(task, AgentExecution(context=ctx, termination=AgentTermination.ERROR))
        )
        assert not result.passed and "shell" in result.detail

    @pytest.mark.asyncio
    async def test_composite_weighted(self) -> None:
        exec_ = _fake_execution("correct")
        composite = CompositeGrader([
            OutputEqualsGrader().configure({"expected": "correct"}),
        ])
        result = await composite.grade(GradingRequest(_task(), exec_))
        assert result.passed and result.score == 1.0

    @pytest.mark.asyncio
    async def test_llm_grader_requires_scoring_fn(self) -> None:
        result = await LLMPromptGrader(rubric="quality").grade(
            GradingRequest(_task(), _fake_execution("text"))
        )
        assert not result.passed and "no scoring function" in result.detail
        grader = LLMPromptGrader(lambda p, r2: 0.9, rubric="quality")
        ok_result = await grader.grade(
            GradingRequest(_task(), _fake_execution("text"))
        )
        assert ok_result.passed and ok_result.score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_llm_grader_failure_is_contained(self) -> None:
        async def boom(prompt: str, rubric: str) -> float:
            raise RuntimeError("provider down")

        result = await LLMPromptGrader(boom).grade(
            GradingRequest(_task(), _fake_execution("t"))
        )
        assert not result.passed and "provider down" in result.detail

    def test_registry_build(self) -> None:
        for name in (
            "termination", "output_equals", "output_contains",
            "output_regex", "output_json_field", "budget", "max_steps",
            "tool_usage", "recovery",
        ):
            assert isinstance(build_grader(name), object) is True


def _task(**overrides: Any) -> EvalTask:
    data: dict[str, Any] = {"case_id": "case_a", "name": "Case A", "goal": "go"}
    data.update(overrides)
    task: EvalTask = EvalTask.model_validate(data)
    return task


r = ""  # rubric placeholder used by llm test above


# ---------------------------------------------------------------------------
# Runtime integration / metric collection / failed runs / budget termination
# ---------------------------------------------------------------------------


def _script(steps: int, **extra: Any) -> dict[str, Any]:
    return {"steps": steps, **extra}


class TestRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_success_case_end_to_end(self) -> None:
        task = _task(metadata={"script": _script(1, final_output="done")})
        report = await EvalRunner(ScriptedAgentFactory()).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert result.success and result.status == EvalStatus.COMPLETED
        assert result.metrics.steps >= 1
        assert result.metrics.termination_reason == "success"
        assert result.run_id.startswith("s::case_a@r1#seed0")

    @pytest.mark.asyncio
    async def test_failed_run_reported_not_raised(self) -> None:
        # Impossible goal: budget exhausted, graded failure but no crash.
        task = _task(
            case_id="impossible",
            metadata={"script": _script(50)},
            budget=AgentBudget(max_steps=2),
        )
        report = await EvalRunner(ScriptedAgentFactory()).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert not result.success
        assert result.completion
        assert result.metrics.termination_reason == "budget_exceeded"

    @pytest.mark.asyncio
    async def test_timeout_termination(self) -> None:
        from collections.abc import AsyncIterator

        from research_engineer.runtime.models import AgentContext
        from research_engineer.runtime.runtime import AgentRuntime

        del AsyncIterator

        async def hang_planner(ctx: AgentContext) -> Any:
            await __import__("asyncio").sleep(30)
            return {}

        factory_calls: list[int] = []

        def factory(task: EvalTask) -> AgentRuntime:
            factory_calls.append(1)
            return AgentRuntime(
                planner=hang_planner,
                actor=lambda c, p: _async_noop(),
                observer=lambda c, a: _async_noop(),
                evaluator=lambda c, o: _async_noop(),
            )

        task = _task(case_id="hang", timeout_seconds=0.05)
        report = await EvalRunner(factory).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert result.status == EvalStatus.FAILED
        assert "timeout" in result.error
        assert result.metrics.termination_reason in ("", "none")
        assert not result.success
        assert factory_calls == [1]

    @pytest.mark.asyncio
    async def test_recoverable_error_counts_as_recovery(self) -> None:
        task = _task(
            case_id="recovery",
            metadata={"script": _script(
                2, error_at_step=1, final_output="still fine",
            )},
            criteria=[{"grader": "recovery"}],
        )
        report = await EvalRunner(ScriptedAgentFactory()).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert result.metrics.recoverable_errors >= 1
        assert result.metrics.recovered is True
        assert result.success

    @pytest.mark.asyncio
    async def test_human_interventions_tracked(self) -> None:
        task = _task(
            case_id="approval",
            metadata={"script": _script(
                1, final_output="ok",
                tools=[{"step": 1, "name": "shell", "status": "approval_required"}],
            )},
        )
        report = await EvalRunner(ScriptedAgentFactory()).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert result.metrics.human_interventions == 1
        agg = aggregate(report.results)
        assert agg.human_intervention_rate == 1.0

    @pytest.mark.asyncio
    async def test_tool_usage_through_gateway_style_log(self) -> None:
        task = _task(
            case_id="tools",
            allowed_tools=["read_doc"],
            metadata={"script": _script(
                1, final_output="ok",
                tools=[{"step": 1, "name": "read_doc", "status": "allowed"}],
            )},
            criteria=[{"grader": "tool_usage"}],
        )
        report = await EvalRunner(ScriptedAgentFactory()).run_suite(
            EvalSuite(suite_id="s", cases=[task])
        )
        (result,) = report.results
        assert result.success
        assert result.metrics.tool_calls == 1


async def _async_noop() -> None:
    return None


# ---------------------------------------------------------------------------
# Aggregation, regression comparison, serialization, reproducibility
# ---------------------------------------------------------------------------


def _result(case: str, success: bool, score: float, **m: Any) -> EvalResult:
    return EvalResult(
        run_id=f"s::{case}", case_id=case, success=success,
        weighted_score=score, status=EvalStatus.COMPLETED,
        metrics={"termination_reason": "success" if success else "error"} | m,
    )


class TestAggregationAndRegression:
    def test_aggregate_metrics(self) -> None:
        results = [
            _result("a", True, 1.0, steps=3, tool_calls=2, tokens=100),
            _result("b", False, 0.4, steps=5, tool_calls=4, tokens=300,
                    recoverable_errors=1),
        ]
        agg = aggregate(results)
        assert agg.cases_total == 2
        assert agg.success_rate == 0.5
        assert agg.failure_rate == 0.5
        assert agg.avg_steps == 4.0
        assert agg.total_tokens == 400
        assert agg.recovery_rate == 0.0  # errored run did not succeed
        assert agg.termination_reasons == {"success": 1, "error": 1}
        metrics = agg.as_metrics()
        assert any(m.name == "success_rate" for m in metrics)
        assert any(m.name == "termination.error" for m in metrics)

    def test_regression_comparison_detects_flip(self) -> None:
        base = EvalReport(
            suite_id="s", results=[
                _result("a", True, 1.0), _result("b", False, 0.3),
            ],
        )
        cand = EvalReport(
            suite_id="s", results=[
                _result("a", False, 0.2), _result("b", True, 0.9),
            ],
        )
        cmp_res = compare_reports(base, cand)
        statuses = {d.case_id: d.status for d in cmp_res.case_diffs}
        assert statuses == {"a": "regressed", "b": "improved"}
        assert not cmp_res.is_regression_free
        assert any("a" in reg for reg in cmp_res.regressions)

    def test_regression_free_when_identical(self) -> None:
        base = EvalReport(suite_id="s", results=[_result("a", True, 1.0)])
        cmp_res = compare_reports(base, base.model_copy(deep=True))
        assert cmp_res.is_regression_free

    def test_serialization_roundtrip(self, tmp_path: Path) -> None:
        report = EvalReport(
            suite_id="s", label="v1",
            results=[_result("a", True, 1.0, steps=2)],
        )
        path = tmp_path / "report.json"
        save_report(report, path)
        loaded = load_report(path)
        assert isinstance(loaded, EvalReport)
        assert loaded.aggregate == report.aggregate
        data = json.loads(path.read_text())
        assert data["results"][0]["metrics"]["termination_reason"] == "success"

    @pytest.mark.asyncio
    async def test_run_ids_are_stable_and_seed_dependent(self) -> None:
        task = _task(metadata={"script": _script(1)})
        r1 = await EvalRunner(ScriptedAgentFactory(), seed=7).run_case(task, "s")
        r2 = await EvalRunner(ScriptedAgentFactory(), seed=7).run_case(task, "s")
        r3 = await EvalRunner(ScriptedAgentFactory(), seed=8).run_case(task, "s")
        assert r1.run_id == r2.run_id
        assert r1.run_id != r3.run_id
        # Deterministic payload across identical runs.
        assert r1.weighted_score == r2.weighted_score


class TestEvalModels:
    def test_grader_result_bounds(self) -> None:
        with pytest.raises(Exception):
            GraderResult(grader="x", score=1.5, passed=True)

    def test_stable_run_id_format(self) -> None:
        assert _task().stable_run_id("suite", seed=3).endswith("#seed3")
