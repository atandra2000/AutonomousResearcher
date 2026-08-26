"""Tests for D2 - Observability (observability/ + LLM event emission)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from research_engineer.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
    ModelRouter,
    ProviderFactory,
    register_provider_type,
    reset_factory,
    reset_router,
    reset_usage_tracker,
)
from research_engineer.llm.base import LLMProvider
from research_engineer.observability import (
    EventBus,
    JSONLSink,
    NullSink,
    SQLiteSink,
    get_event_bus,
    reset_event_bus,
)


class _CapturingSink:
    """Test sink that captures events into a list."""

    def __init__(self, store: list[dict]) -> None:
        self._store = store

    def emit(self, event: dict) -> None:
        self._store.append(event)

    def close(self) -> None:
        return None


class _FakeProviderObs(LLMProvider):
    name = "fake"

    def __init__(self, default_model: str = "fake-model") -> None:
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model
        return LLMResponse(
            content="ok",
            model=model,
            provider=self.name,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )


class TestNullSink:
    def test_emit_is_noop(self):
        NullSink().emit({"kind": "llm_call"})


class TestJSONLSink:
    def test_writes_one_line_per_event(self, tmp_path: Path):
        path = tmp_path / "events.jsonl"
        sink = JSONLSink(path)
        sink.emit({"kind": "llm_call", "ts": "2026-01-01T00:00:00"})
        sink.emit({"kind": "stage", "ts": "2026-01-01T00:00:01"})
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["kind"] == "llm_call"
        assert json.loads(lines[1])["kind"] == "stage"

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "sub" / "dir" / "events.jsonl"
        JSONLSink(path).emit({"kind": "x"})
        assert path.exists()

    def test_emit_swallows_errors(self, tmp_path: Path):
        JSONLSink(tmp_path).emit({"kind": "x"})  # tmp_path is a directory


class TestSQLiteSink:
    def test_writes_events_to_table(self, tmp_path: Path):
        db = tmp_path / "obs.db"
        sink = SQLiteSink(db)
        sink.emit({"kind": "llm_call", "ts": "t1", "extra": 1})
        sink.emit({"kind": "stage", "ts": "t2"})
        sink.close()
        conn = sqlite3.connect(str(db))
        rows = conn.execute("SELECT ts, kind, payload_json FROM events ORDER BY id").fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "t1"
        assert rows[0][1] == "llm_call"
        assert json.loads(rows[0][2])["extra"] == 1
        assert rows[1][1] == "stage"

    def test_close_is_idempotent(self, tmp_path: Path):
        sink = SQLiteSink(tmp_path / "obs.db")
        sink.emit({"kind": "x"})
        sink.close()
        sink.close()


class TestEventBus:
    def test_fans_out_to_all_sinks(self, tmp_path: Path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        bus = EventBus([JSONLSink(p1), JSONLSink(p2)])
        bus.emit({"kind": "x", "ts": "t"})
        assert len(p1.read_text().splitlines()) == 1
        assert len(p2.read_text().splitlines()) == 1

    def test_sink_failure_does_not_break_bus(self):
        class BadSink:
            def emit(self, event):
                raise RuntimeError("boom")

        bus = EventBus([BadSink()])  # type: ignore[arg-type]
        bus.emit({"kind": "x"})

    def test_add_sink_and_clear(self):
        bus = EventBus()
        s = NullSink()
        bus.add_sink(s)
        assert bus.sinks() == [s]
        bus.clear_sinks()
        assert bus.sinks() == []

    def test_emit_llm_call_shape(self):
        events: list[dict] = []
        bus = EventBus([_CapturingSink(events)])
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        resp = LLMResponse(
            content="ok", model="gpt-4o", provider="fake",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.001),
            finish_reason="stop",
        )
        bus.emit_llm_call(agent_name="fake/gpt-4o", request=req, response=resp, latency_seconds=1.5)
        ev = events[0]
        assert ev["kind"] == "llm_call"
        assert ev["model"] == "gpt-4o"
        assert ev["latency_seconds"] == 1.5
        assert ev["tokens"]["prompt"] == 10
        assert ev["cost_usd"] == 0.001
        assert ev["prompt_hash"].startswith("ph_")

    def test_emit_stage_shape(self):
        events: list[dict] = []
        bus = EventBus([_CapturingSink(events)])
        bus.emit_stage(stage_id="s1", stage_type="lit", status="completed", duration_seconds=2.5, workflow_id="wf1")
        ev = events[0]
        assert ev["kind"] == "stage"
        assert ev["stage_id"] == "s1"
        assert ev["duration_seconds"] == 2.5


class TestGlobalEventBus:
    def test_singleton_is_stable(self):
        reset_event_bus()
        a = get_event_bus()
        b = get_event_bus()
        assert a is b

    def test_reset_clears_sinks(self):
        bus = get_event_bus()
        bus.add_sink(NullSink())
        assert len(bus.sinks()) >= 1
        reset_event_bus()
        assert get_event_bus().sinks() == []


class TestRouterEmitsEvent:
    def setup_method(self):
        reset_factory()
        reset_router()
        reset_usage_tracker()
        reset_event_bus()

    def teardown_method(self):
        reset_factory()
        reset_router()
        reset_usage_tracker()
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_complete_emits_event(self):
        events: list[dict] = []
        get_event_bus().add_sink(_CapturingSink(events))
        register_provider_type("fake", _FakeProviderObs)
        cfg = {
            "default_provider": "fake",
            "default_model": "gpt-4o",
            "providers": {"fake": {"type": "fake"}},
            "agents": {"CodingAgent": {"provider": "fake", "model": "gpt-4o"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        prov = router.for_agent("CodingAgent")
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        resp = await prov.complete(req)
        assert resp.content == "ok"
        llm_events = [e for e in events if e["kind"] == "llm_call"]
        assert len(llm_events) == 1
        assert llm_events[0]["model"] == "gpt-4o"
        assert llm_events[0]["tokens"]["prompt"] == 10
        assert llm_events[0]["prompt_hash"].startswith("ph_")
