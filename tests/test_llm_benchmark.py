"""P2 - LLM-backed benchmark tests.

Covers the v2 suite contract, the ``llm_react`` agent adapter (through a
scripted provider and a fake gateway chain), the LLM judge wiring,
budget-override plumbing, P2 aggregation helpers, and an end-to-end
production-stack run driven entirely by a scripted LLM (no credentials).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from research_engineer.benchmark.bench_agents import (
    NoteWriteInput,
    NoteWriteOutput,
)
from research_engineer.benchmark.benchmark_runner import (
    CaseOutcome,
    _grade_case,
    _StoredExecutionView,
)
from research_engineer.benchmark.llm_agent import (
    KIND_LLM_REACT,
    LLMReActAdapter,
)
from research_engineer.eval.graders import LLMPromptGrader
from research_engineer.eval.models import EvalTask, SuccessCriterion
from research_engineer.llm.base import (
    LLMResponse,
    LLMUsage,
    ToolCall,
)
from research_engineer.runtime.models import AgentTermination

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Minimal provider standing in for any Phase-10 backend."""

    name = "scripted"
    default_model = "fake-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []
        self.tools_seen: list[list[str]] = []

    async def complete_with_tools(
        self, request: Any, tools: list[Any],
    ) -> LLMResponse:
        self.requests.append(request)
        self.tools_seen.append([t.name for t in tools])
        if not self._responses:
            raise AssertionError("scripted provider ran out of responses")
        return self._responses.pop(0)


class SimpleResult:
    def __init__(self, *, status: str, output: Any = None,
                 error: str = "") -> None:
        self.status = status
        self.output = output
        self.error = error


class FakeGatewayRuntime:
    """Duck-typed runtime routing tool calls like ``AgentRuntime.call_tool``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def call_tool(
        self, tool_name: str, input: Any, *, agent_name: str = "",
    ) -> Any:
        self.calls.append((tool_name, input))
        if tool_name == "research_note_write":
            payload: NoteWriteInput = input
            return SimpleResult(
                status="success",
                output=NoteWriteOutput(
                    path=f"/tmp/{payload.name}.txt",
                    bytes_written=len(payload.content.encode()),
                ),
            )
        return SimpleResult(status="success", output=None)


class FakeContext:
    """Duck-typed AgentContext surface the adapter touches."""

    def __init__(self, goal: str = "research the thing") -> None:
        self.goal = goal
        self.execution_id = "exec_test"
        self.tokens = 0
        self.cost_usd = 0.0
        self.current_step = 0
        self.steps: list[Any] = []


def _resp(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    tokens: int = 20,
    cost: float = 0.0002,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake-model",
        provider="scripted",
        usage=LLMUsage(
            prompt_tokens=tokens - 5,
            completion_tokens=5,
            total_tokens=tokens,
            cost_usd=cost,
        ),
        finish_reason="stop" if not tool_calls else "tool_calls",
        tool_calls=tool_calls,
    )


def _write_call(note: str = "note_a") -> ToolCall:
    return ToolCall(
        id="c1", name="research_note_write",
        arguments={"name": note, "content": "structured findings"},
    )


# ---------------------------------------------------------------------------
# Suite contract
# ---------------------------------------------------------------------------


def test_v2_suite_loads_and_validates() -> None:
    from research_engineer.benchmark.benchmark import (
        BENCHMARK_CATEGORIES,
        DEFAULT_SUITE_V2_PATH,
        load_benchmark_suite,
    )

    suite = load_benchmark_suite(DEFAULT_SUITE_V2_PATH)
    assert len(suite.cases) == 20
    covered = {c.metadata["category"] for c in suite.cases}
    assert covered == set(BENCHMARK_CATEGORIES)
    assert all(c.metadata["mode"] == "llm_agent" for c in suite.cases)
    assert all(c.metadata["agent_kind"] == KIND_LLM_REACT
               for c in suite.cases)
    # Every judge criterion must be optional: judged quality never gates
    # objective success.
    for case in suite.cases:
        for crit in case.criteria:
            if crit.grader == "llm_quality":
                assert crit.required is False


# ---------------------------------------------------------------------------
# Adapter behaviour
# ---------------------------------------------------------------------------


def test_adapter_runs_react_loop_through_gateway() -> None:
    provider = ScriptedProvider([
        _resp(tool_calls=[_write_call()]),
        _resp(content="SURVEY: done\nFINAL_ANSWER:\nsummary text"),
    ])
    adapter = LLMReActAdapter(
        provider, model="fake-model", temperature=0.3,
        max_tokens_per_call=256,
    )
    runtime = FakeGatewayRuntime()
    adapter.attach_runtime(runtime)
    ctx = FakeContext()

    obs1 = asyncio.run(adapter.actor(ctx, None))
    assert obs1["finished"] is False
    assert obs1["notes_written"] == 1
    assert len(runtime.calls) == 1

    obs2 = asyncio.run(adapter.actor(ctx, None))
    assert obs2["finished"] is True
    evaluation = asyncio.run(adapter.evaluator(ctx, obs2))
    assert evaluation["done"] is True
    output = evaluation["output"]
    assert output["notes_written"] == 1
    assert output["model"] == "fake-model"
    assert "SURVEY:" in output["final_answer"]
    # Token/cost accounting flowed onto the runtime context.
    assert ctx.tokens == 40
    assert ctx.cost_usd == pytest.approx(0.0004)
    # Sampling parameters were applied per request.
    assert provider.requests[0].temperature == 0.3
    assert provider.requests[0].max_tokens == 256
    # The tool surface mirrors the gateway registrations.
    assert set(provider.tools_seen[0]) == {
        "research_note_write", "research_note_list", "echo_probe",
    }


def test_adapter_without_runtime_fails_closed() -> None:
    adapter = LLMReActAdapter(ScriptedProvider([]))
    with pytest.raises(RuntimeError):
        asyncio.run(adapter._execute_tool(_write_call()))


def test_adapter_surfaces_denials_as_errors() -> None:
    provider = ScriptedProvider([
        _resp(tool_calls=[_write_call()]),
        _resp(content="FINAL_ANSWER: gave up"),
    ])
    adapter = LLMReActAdapter(provider)
    runtime = FakeGatewayRuntime()
    adapter.attach_runtime(runtime)

    async def denying_call(name: str, payload: Any, **_: Any) -> Any:
        return SimpleResult(status="denied_by_policy", error="not approved")

    runtime.call_tool = denying_call  # type: ignore[method-assign]
    ctx = FakeContext()
    obs = asyncio.run(adapter.actor(ctx, None))
    assert obs["finished"] is False
    assert obs["notes_written"] == 0
    obs2 = asyncio.run(adapter.actor(ctx, None))
    assert obs2["finished"] is True
    final = asyncio.run(adapter.evaluator(ctx, obs2))["output"]
    assert final["tool_calls_failed"] == 1
    assert final["notes_written"] == 0


def test_adapter_rejects_unknown_tools_before_gateway() -> None:
    provider = ScriptedProvider([
        _resp(tool_calls=[
            ToolCall(id="c1", name="rm_rf", arguments={"path": "/"})
        ]),
        _resp(content="FINAL_ANSWER: x"),
    ])
    adapter = LLMReActAdapter(provider)
    runtime = FakeGatewayRuntime()
    adapter.attach_runtime(runtime)
    asyncio.run(adapter.actor(FakeContext(), None))
    assert runtime.calls == []  # unknown tool never reached the gateway


def test_adapter_records_invalid_arguments_as_failed_calls() -> None:
    provider = ScriptedProvider([
        _resp(tool_calls=[
            ToolCall(id="c1", name="research_note_write",
                     arguments={"name": "bad name!;", "content": "x"})
        ]),
        _resp(content="FINAL_ANSWER: x"),
    ])
    adapter = LLMReActAdapter(provider)
    runtime = FakeGatewayRuntime()
    adapter.attach_runtime(runtime)
    ctx = FakeContext()
    asyncio.run(adapter.actor(ctx, None))
    assert runtime.calls == []  # invalid input never reached the gateway


def test_resolve_provider_prefers_injected_instance() -> None:
    from research_engineer.benchmark.llm_agent import resolve_llm_provider

    injected = ScriptedProvider([])
    provider, model = resolve_llm_provider({}, injected=injected)
    assert provider is injected
    assert model is None


# ---------------------------------------------------------------------------
# Budget override plumbing
# ---------------------------------------------------------------------------


def test_registry_honors_extended_budget_overrides() -> None:
    from research_engineer.benchmark.p2_benchmark import build_p2_factories

    registry = build_p2_factories(provider=ScriptedProvider([]))
    adapter, policy = asyncio.run(registry.build(
        KIND_LLM_REACT,
        {
            "max_steps": 6,
            "max_tool_calls": 12,
            "max_tokens": 9000,
            "max_cost_usd": 0.4,
            "max_recoverable_errors": 5,
            "llm_temperature": 0.05,
        },
    ))
    assert policy.budget.max_steps == 6
    assert policy.budget.max_tool_calls == 12
    assert policy.budget.max_tokens == 9000
    assert policy.budget.max_cost_usd == pytest.approx(0.4)
    assert policy.max_recoverable_errors == 5
    assert isinstance(adapter, LLMReActAdapter)
    assert adapter.temperature == pytest.approx(0.05)


def test_default_registry_still_builds_deterministic_kinds() -> None:
    from research_engineer.benchmark.agents import AgentFactoryRegistry
    from research_engineer.benchmark.bench_agents import (
        KIND_BENCH_TOOL,
        register_benchmark_kinds,
    )

    registry = AgentFactoryRegistry(config_max_steps=12)
    register_benchmark_kinds(registry)
    adapter, _policy = asyncio.run(registry.build(KIND_BENCH_TOOL, {}))
    assert adapter is not None  # unchanged P1 behavior


# ---------------------------------------------------------------------------
# Grading layer
# ---------------------------------------------------------------------------


def test_unknown_grader_fails_closed_in_grade_case(tmp_path: Path) -> None:
    task = EvalTask(
        case_id="t1", name="t", goal="g",
        criteria=[SuccessCriterion(grader="nonexistent_grader",
                                   required=False)],
        require_runtime_success=False,
    )
    payload = {
        "termination": AgentTermination.SUCCESS.value,
        "reason": "",
        "output": {"x": 1},
        "context": {"current_step": 1, "tool_calls": 0, "tokens": 0,
                    "cost_usd": 0.0, "recoverable_errors": 0,
                    "duration_seconds": 0.1},
    }
    view = _StoredExecutionView(payload)
    outcomes, graded_success, weighted = asyncio.run(_grade_case(task, view))
    assert outcomes[0].passed is False
    assert outcomes[0].score == 0.0
    assert "unknown grader" in outcomes[0].detail
    assert graded_success is True  # non-required failure cannot fail the run


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content


def test_llm_quality_grader_through_extra_graders(tmp_path: Path) -> None:
    task = EvalTask(
        case_id="t2", name="t", goal="g",
        criteria=[SuccessCriterion(grader="llm_quality", required=False,
                                   config={"threshold": 0.7,
                                           "rubric": "be excellent"})],
        require_runtime_success=False,
    )
    payload = {
        "termination": AgentTermination.SUCCESS.value,
        "reason": "",
        "output": {"final_answer": "rich research content"},
        "context": {"current_step": 2, "tool_calls": 2, "tokens": 40,
                    "cost_usd": 0.001, "recoverable_errors": 0,
                    "duration_seconds": 1.0},
    }
    view = _StoredExecutionView(payload)

    seen_prompts: list[str] = []

    class JudgeProvider:
        async def complete(self, request: Any) -> Any:
            seen_prompts.append(request.messages[1].content)
            return _Reply('{"score": 0.9, "rationale": "excellent"}')

    from research_engineer.benchmark.llm_judge import make_judge_score_fn

    grader = LLMPromptGrader(make_judge_score_fn(JudgeProvider()))
    grader.configure({"threshold": 0.7, "rubric": "be excellent"})
    outcomes, _, _ = asyncio.run(
        _grade_case(task, view, extra_graders={"llm_quality": grader})
    )
    assert outcomes[0].grader == "llm_quality"
    assert outcomes[0].score == pytest.approx(0.9)
    # Rubric travels verbatim; candidate output is the only extra input.
    assert "be excellent" in seen_prompts[0]
    assert "rich research content" in seen_prompts[0]


def test_judge_fail_closed_on_unparseable_reply() -> None:
    from research_engineer.benchmark.llm_judge import make_judge_score_fn

    class BadProvider:
        async def complete(self, request: Any) -> Any:
            return _Reply("I cannot score this")

    score_fn = make_judge_score_fn(BadProvider())
    # Fail closed: the error propagates (LLMPromptGrader records the cause
    # in criterion detail) rather than returning a fabricated 0.0.
    try:
        asyncio.run(score_fn("candidate text", "rubric"))
        raised = ""
    except ValueError as exc:
        raised = str(exc)
    assert "unparseable judge reply" in raised


def test_judge_requires_rubric() -> None:
    from research_engineer.benchmark.llm_judge import make_judge_score_fn

    class NeverProvider:
        async def complete(self, request: Any) -> Any:  # pragma: no cover
            raise AssertionError("must not be called without rubric")

    score_fn = make_judge_score_fn(NeverProvider())
    assert asyncio.run(score_fn("candidate", "")) == 0.0


def test_extra_grader_receives_per_criterion_config(tmp_path: Path) -> None:
    """Regression: shared extra graders must pick up each criterion's
    config (rubric/threshold) instead of grading with defaults."""
    task = EvalTask(
        case_id="t3", name="t", goal="g",
        criteria=[SuccessCriterion(grader="llm_quality", required=False,
                                   config={"threshold": 0.6,
                                           "rubric": "case rubric"})],
        require_runtime_success=False,
    )
    payload = {
        "termination": AgentTermination.SUCCESS.value,
        "reason": "",
        "output": {"final_answer": "answer text"},
        "context": {"current_step": 1, "tool_calls": 1, "tokens": 10,
                    "cost_usd": 0.001, "recoverable_errors": 0,
                    "duration_seconds": 1.0},
    }
    view = _StoredExecutionView(payload)
    seen_rubrics: list[str] = []

    class JudgeProvider:
        async def complete(self, request: Any) -> Any:
            seen_rubrics.append(request.messages[1].content)
            return _Reply('{"score": 0.8, "rationale": "good"}')

    from research_engineer.benchmark.llm_judge import make_judge_score_fn

    # Deliberately UNCONFIGURED instance (as run_p2 wires it).
    grader = LLMPromptGrader(make_judge_score_fn(JudgeProvider()))
    outcomes, _, _ = asyncio.run(
        _grade_case(task, view, extra_graders={"llm_quality": grader})
    )
    assert outcomes[0].score == pytest.approx(0.8)
    assert outcomes[0].passed is True
    assert "case rubric" in seen_rubrics[0]


# ---------------------------------------------------------------------------
# P2 aggregation helpers
# ---------------------------------------------------------------------------


def _outcome(**overrides: object) -> CaseOutcome:
    defaults: dict[str, Any] = dict(
        case_id="c1",
        repeat=1,
        category="literature_discovery",
        mode="llm_agent",
        agent_kind=KIND_LLM_REACT,
        submitted_at="2026-01-01T00:00:00Z",
        latency_seconds=1.0,
        status="completed",
        runtime_success=True,
        graded_success=True,
        weighted_score=0.9,
        criteria=[],
        steps=3,
        tool_calls=4,
        tokens=120,
        cost_usd=0.01,
        recoverable_errors=0,
        fatal_errors=0,
        recovered=False,
        human_interventions=0,
        safety_interventions=0,
        gateway_denials=0,
        termination="success",
        termination_reason="success",
    )
    defaults.update(overrides)
    return CaseOutcome(**defaults)


def test_tier_metrics_kept_separate_per_mode() -> None:
    from research_engineer.benchmark.p2_benchmark import tier_metrics_for

    outcomes = [
        _outcome(case_id=f"llm{i}", repeat=1)
        for i in range(2)
    ] + [
        _outcome(case_id=f"llm{i}", repeat=2, weighted_score=0.7)
        for i in range(2)
    ] + [
        _outcome(mode="deterministic_sandbox",
                 agent_kind="bench_tool", case_id=f"d{i}")
        for i in range(3)
    ]
    llm_tier = tier_metrics_for(outcomes, "llm_agent")
    det_tier = tier_metrics_for(outcomes, "deterministic_sandbox")
    assert llm_tier is not None and det_tier is not None
    assert llm_tier.cases_total == 4
    assert det_tier.cases_total == 3
    assert llm_tier.mode != det_tier.mode
    assert llm_tier.task_success_rate == 1.0
    assert llm_tier.median_tokens_per_task == 120.0


def test_variance_stats_detect_instability() -> None:
    from research_engineer.benchmark.benchmark_runner import BenchmarkReport
    from research_engineer.benchmark.p2_benchmark import (
        _variance_from_reports,
    )

    def report(repeats: list[CaseOutcome]) -> BenchmarkReport:
        return BenchmarkReport(
            suite_id="s", suite_version="2",
            generated_at="2026-01-01T00:00:00Z",
            repeat_count=len(repeats), cases_total=len(repeats),
            outcomes=repeats, metrics={},
            termination_distribution={}, failure_summary={},
        )

    outcomes: list[CaseOutcome] = []
    for rep in (1, 2):
        outcomes.append(_outcome(
            case_id="stable", repeat=rep))
        outcomes.append(_outcome(
            case_id="flaky", repeat=rep,
            graded_success=(rep == 1), weighted_score=0.9 if rep == 1
            else 0.2))
    stats = {
        s.case_id: s for s in _variance_from_reports(
            [report(outcomes)], {"stable", "flaky"})
    }
    assert stats["stable"].success_consistent is True
    assert stats["flaky"].success_consistent is False
    assert stats["flaky"].attempts == 2
    assert stats["flaky"].score_spread == pytest.approx(0.7)


def test_suite_fingerprint_changes_with_revision() -> None:
    from research_engineer.benchmark.benchmark import (
        DEFAULT_SUITE_V2_PATH,
        load_benchmark_suite,
    )
    from research_engineer.benchmark.p2_benchmark import suite_fingerprint

    suite = load_benchmark_suite(DEFAULT_SUITE_V2_PATH)
    fp1 = suite_fingerprint(suite)
    tweaked = suite.model_copy(deep=True)
    tweaked.cases[0].revision = "r2"
    fp2 = suite_fingerprint(tweaked)
    assert fp1["content_sha256"] != fp2["content_sha256"]
    assert len(fp1["case_revisions"]) == 20


def test_regression_comparison_flags_config_mismatch(tmp_path: Path) -> None:
    import json as jsonlib

    from research_engineer.benchmark.p2_benchmark import (
        P2Report,
        compare_reports,
        tier_metrics_for,
    )

    prev = {
        "report_id": "p2_prev",
        "configuration": {"provider": "ollama", "model": "old-model"},
        "tier_metrics": [
            t.model_dump() for t in [
                tier_metrics_for([_outcome()], "llm_agent")
            ] if t is not None
        ],
    }
    path = tmp_path / "prev.json"
    path.write_text(jsonlib.dumps(prev))

    candidate = P2Report(
        configuration={"provider": "ollama", "model": "new-model"},
        tier_metrics=[
            t for t in [tier_metrics_for(
                [_outcome(weighted_score=0.8)], "llm_agent")] if t
        ],
    )
    comparison = compare_reports(path, candidate)
    assert comparison.compared_against == "p2_prev"
    assert "CONFIG MISMATCH" in comparison.note


# ---------------------------------------------------------------------------
# End-to-end runtime path with a scripted LLM (no credentials)
# ---------------------------------------------------------------------------


def test_e2e_llm_agent_through_runtime_path(tmp_path: Path) -> None:
    """Direct runtime execution: AgentRuntime -> gateway -> safety chain."""
    from datetime import datetime

    from research_engineer.benchmark.benchmark_runner import (
        _payload_from_context,
    )
    from research_engineer.benchmark.p2_benchmark import build_p2_factories
    from research_engineer.benchmark.safety import (
        build_default_safety_chain,
    )
    from research_engineer.runtime.checkpoint_stores import (
        SQLiteCheckpointStore,
    )
    from research_engineer.runtime.runtime import AgentRuntime

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway, controller = build_default_safety_chain(workspace)
    responses = [
        _resp(tool_calls=[_write_call("e2e_note")]),
        _resp(content="SURVEY: ok\nFINAL_ANSWER:\ndone"),
    ]
    registry = build_p2_factories(provider=ScriptedProvider(responses))

    async def scenario() -> Any:
        adapter, policy = await registry.build(
            KIND_LLM_REACT,
            {"max_steps": 5, "max_tool_calls": 8, "max_tokens": 5000},
        )
        runtime = AgentRuntime(
            planner=adapter.planner,
            actor=adapter.actor,
            observer=adapter.observer,
            evaluator=adapter.evaluator,
            policy=policy,
            checkpoint_store=SQLiteCheckpointStore(str(tmp_path / "cp.db")),
            tool_gateway=gateway,
            safety_controller=controller,
        )
        attach = getattr(adapter, "attach_runtime", None)
        if attach is not None:
            attach(runtime)
        execution = await runtime.run(
            "Survey the provided digest and record findings.",
            metadata={"case_id": "e2e_llm"},
        )
        return execution.context

    context = asyncio.run(scenario())
    payload = _payload_from_context(context, datetime.now())
    assert payload["termination"] == "success"
    output = payload["output"]
    assert output["llm_agent"] is True
    assert output["notes_written"] == 1
    summary = payload["context"]
    # Real usage accounting reached the persisted payload.
    assert summary["tokens"] >= 20
    assert summary["tool_calls"] >= 1
    assert summary["cost_usd"] > 0.0
    # The note physically exists under the approved sandbox root.
    notes = sorted((workspace / "sandbox" / "notes").glob("*.txt"))
    assert [p.name for p in notes] == ["e2e_note.txt"]
    assert notes[0].read_text(encoding="utf-8") == "structured findings"





def test_tier_metrics_antiguardrail_only_mode_does_not_crash() -> None:
    """Regression: a mode family consisting only of guardrail anti-cases
    must aggregate to neutral metrics, not raise StatisticsError."""
    from research_engineer.benchmark.p2_benchmark import tier_metrics_for

    outcomes = [
        _outcome(mode="policy_guardrail", agent_kind="bench_tool",
                 case_id=f"anti{i}", expected_failed_by_design=True)
        for i in range(3)
    ]
    tier = tier_metrics_for(outcomes, "policy_guardrail")
    assert tier is not None
    assert tier.cases_total == 3
    assert tier.mean_weighted_score == 0.0
    assert tier.task_success_rate == 0.0


def test_judge_grade_surfaces_unparseable_cause() -> None:
    """Judge failures must be auditable in criterion detail, not silently
    conflated with a legitimate 0.0 quality score."""

    from research_engineer.benchmark.llm_judge import make_judge_score_fn
    from research_engineer.eval.graders import GradingRequest, LLMPromptGrader

    class BadProvider:
        async def complete(self, request: Any) -> Any:
            return _Reply("I cannot score this")

    grader = LLMPromptGrader(make_judge_score_fn(BadProvider()))
    task = EvalTask(
        case_id="j1", name="t", goal="g",
        criteria=[SuccessCriterion(grader="llm_quality", required=False,
                                   config={"threshold": 0.5,
                                           "rubric": "be excellent"})],
        require_runtime_success=False,
    )
    payload = {
        "termination": AgentTermination.SUCCESS.value,
        "reason": "",
        "output": {"final_answer": "candidate text"},
        "context": {"current_step": 2, "tool_calls": 1, "tokens": 20,
                    "cost_usd": 0.0005, "recoverable_errors": 0,
                    "duration_seconds": 1.0},
    }
    view = _StoredExecutionView(payload)
    grader.configure({"threshold": 0.5, "rubric": "be excellent"})
    req = GradingRequest(task, view, human_interventions=0)
    result = asyncio.run(grader.grade(req))
    assert result.score == 0.0
    assert "unparseable judge reply" in result.detail
