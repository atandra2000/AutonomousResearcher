"""Tests for E6 - Production Observability & Telemetry.

Covers event correlation, trace/span relationships across
Runtime -> LLM -> Gateway -> Safety -> Evaluation, concurrent-run
isolation, metrics derivation, run-summary reconstruction, privacy/
redaction/payload limits, exporter-failure isolation, optional
OpenTelemetry support, and existing JSONL/SQLite sinks.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

import pytest

from research_engineer.gateway import (
    CallbackApprovalHandler,
    RiskLevel,
    ToolCallStatus,
    ToolGateway,
    ToolGatewayConfig,
)
from research_engineer.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
)
from research_engineer.observability import (
    EventBus,
    JSONLSink,
    SQLiteSink,
    TelemetryConfig,
    get_correlation,
    hash_text,
    opentelemetry_available,
    stamp_correlation,
)
from research_engineer.observability.context import (
    CorrelationContext,
    reset_correlation,
    set_correlation,
)
from research_engineer.observability.metrics import MetricsRegistry, MetricsSink
from research_engineer.observability.otel import (
    OTelMetricsBridge,
    OTelSpanSink,
)
from research_engineer.observability.summary import RunAggregator, RunSummary
from research_engineer.runtime import AgentRuntime, AgentTermination
from research_engineer.safety import AutonomyPolicy, SafetyController
from research_engineer.tools.base import Tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CapturingSink:
    """Sink that records every event into a shared list."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class ExplodingSink:
    """A sink whose backend is broken - must never break callers."""

    def emit(self, event: dict[str, Any]) -> None:
        raise RuntimeError("backend down")

    def close(self) -> None:
        raise RuntimeError("shutdown blew up")


class EchoTool(Tool[Any, Any]):
    async def execute(self, input: Any) -> Any:
        return {"echo": input}

    async def validate(self, input: Any) -> bool:
        return True


async def _noop_planner(ctx: Any) -> Any:
    return {"goal": ctx.goal}


async def _noop_observer(ctx: Any, action: Any) -> Any:
    return {"obs": action}


def _runtime(
    capturing: CapturingSink | None = None,
    *,
    actor: Any = None,
    evaluator: Any = None,
    **kwargs: Any,
) -> tuple[AgentRuntime, CapturingSink]:
    cap = capturing or CapturingSink()
    bus = EventBus([cap])
    rt = AgentRuntime(
        planner=_noop_planner,
        actor=actor or _default_actor,
        observer=_noop_observer,
        evaluator=evaluator or _done_evaluator,
        event_bus=bus,
        **kwargs,
    )
    return rt, cap


async def _default_actor(ctx: Any, plan: Any) -> Any:
    return {"action": "a"}


async def _done_evaluator(ctx: Any, observation: Any) -> Any:
    return {"done": True, "output": observation}


async def _deny(request: Any) -> bool:
    return False


def _llm_response(model: str = "fake-model") -> LLMResponse:
    return LLMResponse(
        content="ok",
        model=model,
        provider="fake",
        usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        finish_reason="stop",
    )


class TestCorrelationContext:
    def test_stamp_adds_ambient_ids(self) -> None:
        ctx = CorrelationContext(run_id="r1", execution_id="e1", trace_id="t1")
        token = set_correlation(ctx)
        try:
            event: dict[str, Any] = {"kind": "x"}
            assert stamp_correlation(event)["trace_id"] == "t1"
            assert event["run_id"] == "r1"
        finally:
            reset_correlation(token)

    def test_stamp_preserves_existing_fields(self) -> None:
        ctx = CorrelationContext(run_id="ambient")
        token = set_correlation(ctx)
        try:
            event = {"run_id": "explicit"}
            assert stamp_correlation(event)["run_id"] == "explicit"
        finally:
            reset_correlation(token)

    def test_no_context_is_noop(self) -> None:
        assert stamp_correlation({"kind": "x"}) == {"kind": "x"}

    def test_child_span_relationships(self) -> None:
        root = CorrelationContext(
            run_id="r", execution_id="e", trace_id="tr", span_id="sp-root"
        )
        child = root.child(step_id="e:step:1")
        grandchild = child.child(tool_call_id="c1")
        assert child.parent_span_id == "sp-root"
        assert grandchild.parent_span_id == child.span_id
        assert grandchild.trace_id == root.trace_id

    @pytest.mark.asyncio
    async def test_concurrent_contexts_are_isolated(self) -> None:
        seen: list[str] = []

        async def worker(name: str) -> None:
            ctx = CorrelationContext(run_id=name, trace_id=f"trace-{name}")
            token = set_correlation(ctx)
            try:
                await asyncio.sleep(0.01)
                current = get_correlation()
                seen.append(current.run_id)  # type: ignore[union-attr]
                await asyncio.sleep(0.01)
                current2 = get_correlation()
                assert current2 is not None  # type narrowing
                assert current2.trace_id == f"trace-{name}"
            finally:
                reset_correlation(token)

        await asyncio.gather(worker("alpha"), worker("beta"))
        assert sorted(seen) == ["alpha", "beta"]


class TestRuntimeCorrelation:
    @pytest.mark.asyncio
    async def test_run_events_share_trace_and_run_ids(self) -> None:
        rt, cap = _runtime()
        execution = await rt.run("test goal")
        assert execution.termination == AgentTermination.SUCCESS

        kinds = [e["event"] for e in cap.events if e.get("kind") == "agent_runtime"]
        for expected in ("start", "step", "evaluation", "terminate", "end"):
            assert expected in kinds, f"missing {expected}"

        run_ids = {e["run_id"] for e in cap.events if "run_id" in e}
        trace_ids = {
            e["trace_id"] for e in cap.events if "trace_id" in e and e["trace_id"]
        }
        assert len(run_ids) == 1
        assert len(trace_ids) == 1
        ctx = execution.context
        assert ctx.metadata["run_id"] in run_ids
        assert ctx.metadata["trace_id"] in trace_ids

    @pytest.mark.asyncio
    async def test_step_events_have_step_ids(self) -> None:
        rt, cap = _runtime()
        await rt.run("goal")
        step_events = [
            e for e in cap.events if e.get("event") == "step" and "step_id" in e
        ]
        assert step_events, "no step events with step_id"
        evals = [e for e in cap.events if e.get("event") == "evaluation"]
        assert evals
        assert all(e["trace_id"] == step_events[0]["trace_id"] for e in evals)

    @pytest.mark.asyncio
    async def test_concurrent_runs_are_isolated(self) -> None:
        caps = [CapturingSink(), CapturingSink()]
        runs = []
        bus_a, bus_b = EventBus([caps[0]]), EventBus([caps[1]])
        for bus in (bus_a, bus_b):
            rt = AgentRuntime(
                planner=_noop_planner,
                actor=_default_actor,
                observer=_noop_observer,
                evaluator=_done_evaluator,
                event_bus=bus,
            )
            runs.append(rt)
        executions = await asyncio.gather(runs[0].run("g1"), runs[1].run("g2"))
        assert executions[0].termination == AgentTermination.SUCCESS

        traces = []
        for cap in caps:
            ids = {e["trace_id"] for e in cap.events if e.get("trace_id")}
            runs_seen = {e["run_id"] for e in cap.events if e.get("run_id")}
            assert len(ids) == 1, f"cross-run leakage: {ids}"
            assert len(runs_seen) == 1, f"cross-run run_id leakage: {runs_seen}"
            traces.append(ids.pop())
        assert len(set(traces)) == 2, "two concurrent runs shared a trace id"


# ---------------------------------------------------------------------------
# End-to-end Runtime -> LLM -> Gateway -> Safety -> Evaluation
# ---------------------------------------------------------------------------


class TestEndToEndTrace:
    @pytest.mark.asyncio
    async def test_full_chain_single_trace(self) -> None:
        cap = CapturingSink()
        bus = EventBus([cap])
        gateway = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]), event_bus=bus)
        gateway.register_tool(EchoTool(), name="echo", risk_level=RiskLevel.LOW)
        safety = SafetyController(AutonomyPolicy(enabled=False), event_bus=bus)
        rt_ref: list[AgentRuntime] = []

        async def actor(ctx: Any, plan: Any) -> Any:
            rt = rt_ref[0]
            bus.emit_llm_call(
                agent_name="actor",
                request=LLMRequest(
                    messages=[LLMMessage(role=LLMRole.USER, content="hi")]
                ),
                response=_llm_response(),
                latency_seconds=0.04,
            )
            result = await rt.call_tool("echo", {"x": 1})
            assert result.status == ToolCallStatus.SUCCESS
            return result

        async def scoring_evaluator(ctx: Any, observation: Any) -> Any:
            return {"done": True, "output": observation, "score": 0.9}

        from research_engineer.runtime import AgentPolicy

        rt = AgentRuntime(
            planner=_noop_planner,
            actor=actor,
            observer=_noop_observer,
            evaluator=scoring_evaluator,
            policy=AgentPolicy(),
            event_bus=bus,
            tool_gateway=gateway,
            safety_controller=safety,
        )
        rt_ref.append(rt)
        execution = await rt.run("e2e goal")
        assert execution.termination == AgentTermination.SUCCESS

        trace_ids = {e["trace_id"] for e in cap.events if e.get("trace_id")}
        assert len(trace_ids) == 1, cap.events
        trace = trace_ids.pop()

        llm_events = [e for e in cap.events if e["kind"] == "llm_call"]
        tool_end = [e for e in cap.events if e.get("event") == "tool_call_end"]
        evals = [e for e in cap.events if e.get("event") == "evaluation"]
        assert llm_events and tool_end and evals
        for ev in (*llm_events, *tool_end, *evals):
            assert ev["trace_id"] == trace
        assert tool_end[0]["call_id"]

    @pytest.mark.asyncio
    async def test_human_intervention_recorded_on_approval(self) -> None:
        cap = CapturingSink()
        bus = EventBus([cap])
        gateway = ToolGateway(
            ToolGatewayConfig(workspace=["/tmp"]),
            approval_handler=CallbackApprovalHandler(_deny),
            event_bus=bus,
        )
        gateway.register_tool(EchoTool(), name="risky", risk_level=RiskLevel.HIGH)

        async def actor(ctx: Any, plan: Any) -> Any:
            return {"action": "attempted"}

        rt = AgentRuntime(
            planner=_noop_planner,
            actor=actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
            event_bus=bus,
            tool_gateway=gateway,
        )
        result = await rt.call_tool("risky", {})
        assert not result.ok
        approvals = [
            e for e in cap.events if e.get("event") == "tool_approval_requested"
        ]
        assert approvals


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _labels(reg: MetricsRegistry, name: str) -> list[tuple[str, float]]:
    snap = reg.snapshot()[name]
    return [(k, float(v)) for k, v in snap.items()]


class TestMetrics:
    @pytest.mark.asyncio
    async def test_runs_termination_and_steps(self) -> None:
        reg = MetricsRegistry()
        sink = MetricsSink(reg)
        cap = CapturingSink()
        bus = EventBus([cap])

        async def scoring(ctx: Any, observation: Any) -> Any:
            return {"done": True, "output": observation, "score": 0.8}

        rt = AgentRuntime(
            planner=_noop_planner,
            actor=_default_actor,
            observer=_noop_observer,
            evaluator=scoring,
            event_bus=bus,
        )
        await rt.run("m goal")
        for event in cap.events:
            sink.emit(event)
        assert reg.counter("runs_started") == 1.0
        success = reg.counter(
            "runs_total", labels={"termination": "success", "outcome": "success"}
        )
        failures = sum(
            v for k, v in _labels(reg, "runs_total") if "failure" in k
        )
        assert success == 1.0 and failures == 0.0
        assert reg.counter("steps_total") >= 1
        assert reg.values("evaluation_score")

    def test_llm_token_cost_latency_metrics(self) -> None:
        reg = MetricsRegistry()
        bus = EventBus([MetricsSink(reg)])
        request = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        bus.emit_llm_call(
            agent_name="a",
            request=request,
            response=_llm_response(),
            latency_seconds=0.25,
        )
        assert (
            reg.counter(
                "llm_calls_total", labels={"model": "fake-model", "provider": "fake"}
            )
            == 1.0
        )
        assert reg.counter("llm_tokens", labels={"kind": "total"}) == 15.0
        assert len(reg.values("llm_latency_seconds")) == 1

    def test_tool_failure_rate_metrics(self) -> None:
        reg = MetricsRegistry()
        sink = MetricsSink(reg)
        sink.emit(
            {
                "kind": "tool_gateway",
                "event": "tool_call_end",
                "tool_name": "echo",
                "status": "success",
                "duration_seconds": 0.01,
            }
        )
        sink.emit(
            {
                "kind": "tool_gateway",
                "event": "tool_call_end",
                "tool_name": "boom",
                "status": "error",
                "duration_seconds": 0.02,
            }
        )
        assert (
            reg.counter("tool_calls_total", labels={"tool": "echo", "status": "success"})
            == 1.0
        )
        assert (
            reg.counter("tool_failures_total", labels={"tool": "boom", "status": "error"})
            == 1.0
        )

    def test_safety_and_human_intervention_metrics(self) -> None:
        reg = MetricsRegistry()
        sink = MetricsSink(reg)
        base = {"kind": "agent_runtime", "event": "safety_decision"}
        sink.emit({**base, "action": "continue", "trigger": "none"})
        sink.emit({**base, "action": "pause_for_approval", "trigger": "risk"})
        assert reg.counter("safety_decisions_total") == 2.0
        assert reg.counter("human_interventions_total") == 1.0

    def test_checkpoint_resume_termination_metrics(self) -> None:
        reg = MetricsRegistry()
        sink = MetricsSink(reg)
        base = {"kind": "agent_runtime"}
        sink.emit({**base, "event": "checkpoint"})
        sink.emit({**base, "event": "checkpoint_failed", "error": "x"})
        sink.emit({**base, "event": "resume"})
        sink.emit({**base, "event": "terminate", "termination": "no_progress"})
        sink.emit({**base, "event": "terminate", "termination": "budget_exceeded"})


# ---------------------------------------------------------------------------
# Run summary / reconstruction API
# ---------------------------------------------------------------------------


class TestRunSummaryAPI:
    @pytest.mark.asyncio
    async def test_reconstruct_run_from_events(self) -> None:
        cap = CapturingSink()
        agg = RunAggregator()

        from research_engineer.runtime import AgentPolicy

        rt_ref: list[AgentRuntime] = []
        bus = EventBus([cap, agg])
        gateway = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]), event_bus=bus)
        gateway.register_tool(EchoTool(), name="echo", risk_level=RiskLevel.LOW)
        safety = SafetyController(AutonomyPolicy(enabled=False), event_bus=bus)

        async def actor(ctx: Any, plan: Any) -> Any:
            bus.emit_llm_call(
                agent_name="actor",
                request=LLMRequest(
                    messages=[LLMMessage(role=LLMRole.USER, content="?")]
                ),
                response=_llm_response(),
                latency_seconds=0.03,
            )
            return await rt_ref[0].call_tool("echo", {"k": 2})

        rt = AgentRuntime(
            planner=_noop_planner,
            actor=actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
            policy=AgentPolicy(),
            event_bus=bus,
            tool_gateway=gateway,
            safety_controller=safety,
        )
        rt_ref.append(rt)
        execution = await rt.run("summary goal")
        run_id = execution.context.metadata["run_id"]

        summary = agg.get_summary(run_id)
        assert summary is not None
        assert isinstance(summary, RunSummary)
        assert summary.termination == "success"
        assert summary.step_count >= 1
        assert summary.llm_call_count == 1
        assert summary.tool_call_count == 1
        assert summary.total_tokens == 15
        assert summary.started_ts is not None and summary.ended_ts is not None
        assert summary.trace_ids
        json.loads(summary.to_json())

    @pytest.mark.asyncio
    async def test_unknown_run_returns_none(self) -> None:
        agg = RunAggregator()
        agg.emit({"kind": "agent_runtime", "event": "start", "execution_id": "e"})
        assert agg.get_summary("missing") is None
        assert len(agg.all_run_ids()) >= 1


# ---------------------------------------------------------------------------
# Privacy / redaction / payload limits
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_secrets_redacted_at_bus_boundary(self) -> None:
        cap = CapturingSink()
        bus = EventBus([cap])
        bus.emit(
            {
                "kind": "test",
                "api_key": "sk-super-secret",
                "nested": {"Authorization": "Bearer abc", "safe": 1},
                "list": [{"SECRET_TOKEN": "zzz"}],
            }
        )
        ev = cap.events[0]
        assert ev["api_key"] == "[REDACTED]"
        assert ev["nested"]["Authorization"] == "[REDACTED]"
        assert ev["nested"]["safe"] == 1
        assert ev["list"][0]["SECRET_TOKEN"] == "[REDACTED]"

    def test_payload_limits_enforced(self) -> None:
        cap = CapturingSink()
        config = TelemetryConfig(
            max_payload_chars=100, capture_prompts=True, hash_prompts=False
        )
        bus = EventBus([cap], config=config)
        bus.emit({"kind": "test", "blob": "x" * 500})
        ev = cap.events[0]
        assert len(ev["blob"]) == 100 + len("...[truncated]")
        assert ev["blob"].endswith("...[truncated]")

    def test_hash_text_stable(self) -> None:
        assert hash_text("abc") == hash_text("abc")
        assert hash_text("abc") != hash_text("abd")

    @pytest.mark.asyncio
    async def test_runtime_goal_survives_config(self) -> None:
        # Privacy scrubbing should not break normal runtime events.
        cap = CapturingSink()
        bus = EventBus([cap])
        rt = AgentRuntime(
            planner=_noop_planner,
            actor=_default_actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
            event_bus=bus,
        )
        execution = await rt.run("goal with api_key mention")
        assert execution.termination == AgentTermination.SUCCESS
        # 'goal' key itself is not a sensitive fragment.
        assert any(e.get("goal") for e in cap.events)


# ---------------------------------------------------------------------------
# Exporter failure isolation + existing sinks
# ---------------------------------------------------------------------------


class TestExporterFailureIsolation:
    def test_failing_sink_does_not_break_bus(self) -> None:
        ok = CapturingSink()
        bus = EventBus([ExplodingSink(), ok])
        bus.emit({"kind": "x", "ts": "t"})
        assert ok.events == [{"kind": "x", "ts": "t"}]

    @pytest.mark.asyncio
    async def test_failing_sink_does_not_break_run(self) -> None:
        ok = CapturingSink()
        bus = EventBus([ExplodingSink(), ok])
        rt = AgentRuntime(
            planner=_noop_planner,
            actor=_default_actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
            event_bus=bus,
        )
        execution = await rt.run("resilient goal")
        assert execution.termination == AgentTermination.SUCCESS
        assert ok.events


class TestExistingSinksStillWork:
    def test_jsonl_sink_records_enriched_event(self, tmp_path: Any) -> None:
        path = tmp_path / "events.jsonl"
        ctx = CorrelationContext(run_id="rj", trace_id="tj")
        token = set_correlation(ctx)
        try:
            EventBus([JSONLSink(path)]).emit({"kind": "llm_call"})
        finally:
            reset_correlation(token)
        event = json.loads(path.read_text().splitlines()[0])
        assert event["trace_id"] == "tj"
        assert event["run_id"] == "rj"

    def test_sqlite_sink_roundtrip(self, tmp_path: Any) -> None:
        db = tmp_path / "obs.db"
        sink = SQLiteSink(db)
        EventBus([sink]).emit({"kind": "x", "ts": "t1"})
        sink.close()
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT ts, kind FROM events").fetchall()
        conn.close()
        assert rows == [("t1", "x")]


# ---------------------------------------------------------------------------
# Optional OpenTelemetry
# ---------------------------------------------------------------------------


class TestOptionalOtel:
    def test_feature_detection_is_bool(self) -> None:
        assert isinstance(opentelemetry_available(), bool)

    def test_start_span_noop_when_unavailable(self, monkeypatch: Any) -> None:
        import asyncio

        import research_engineer.observability.otel as otel_mod

        monkeypatch.setattr(otel_mod, "opentelemetry_available", lambda: False)

        async def main() -> None:
            with otel_mod.start_span("noop") as span:
                assert span is None

        asyncio.run(main())

    def test_span_sink_raises_when_unavailable(self, monkeypatch: Any) -> None:
        import research_engineer.observability.otel as otel_mod

        monkeypatch.setattr(otel_mod, "opentelemetry_available", lambda: False)
        with pytest.raises(ImportError):
            OTelSpanSink()

    def test_metrics_bridge_disabled_without_otel(self, monkeypatch: Any) -> None:
        import research_engineer.observability.otel as otel_mod

        monkeypatch.setattr(otel_mod, "opentelemetry_available", lambda: False)
        reg = MetricsRegistry()
        reg.increment("counter")
        assert OTelMetricsBridge(reg).export() is False

    @pytest.mark.skipif(
        not opentelemetry_available(), reason="OpenTelemetry not installed"
    )
    def test_span_sink_exports_events(self) -> None:
        sink = OTelSpanSink()
        sink.emit(
            {"kind": "tool_gateway", "event": "tool_call_end", "status": "success"}
        )

    @pytest.mark.asyncio
    async def test_run_works_with_otel_span_wrapper(self) -> None:
        sinks: list[Any] = [CapturingSink()]
        if opentelemetry_available():
            sinks.append(OTelSpanSink())
        cap = CapturingSink()
        bus = EventBus([cap, *sinks[1:]])
        rt = AgentRuntime(
            planner=_noop_planner,
            actor=_default_actor,
            observer=_noop_observer,
            evaluator=_done_evaluator,
            event_bus=bus,
        )
        execution = await rt.run("with-otel")
        assert execution.termination == AgentTermination.SUCCESS
        assert cap.events
