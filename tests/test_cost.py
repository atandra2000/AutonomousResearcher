"""Tests for D1 - Cost accounting (llm/cost.py)."""

from __future__ import annotations

import pytest

from research_engineer.llm import LLMUsage
from research_engineer.llm.cost import (
    PricingTable,
    UsageTracker,
    compute_usage_cost,
    default_pricing_table,
    get_usage_tracker,
    load_pricing_table,
    reset_usage_tracker,
)


class TestPricingTable:
    def test_builtin_table_has_known_models(self):
        t = default_pricing_table()
        assert "gpt-4o" in t
        assert "glm-5.2:cloud" in t

    def test_price_for_exact_match(self):
        t = default_pricing_table()
        assert t.price_for("gpt-4o") == (2.50, 10.00)

    def test_price_for_prefix_match(self):
        t = default_pricing_table()
        assert t.price_for("gpt-4o-2024-08-06") == (2.50, 10.00)

    def test_price_for_case_insensitive(self):
        t = default_pricing_table()
        assert t.price_for("GPT-4O") == (2.50, 10.00)

    def test_price_for_unknown_returns_none(self):
        t = default_pricing_table()
        assert t.price_for("totally-unknown-model") is None

    def test_price_for_empty_returns_none(self):
        t = default_pricing_table()
        assert t.price_for("") is None

    def test_add_overrides(self):
        t = default_pricing_table()
        t.add("gpt-4o", 9.99, 9.99)
        assert t.price_for("gpt-4o") == (9.99, 9.99)

    def test_longest_prefix_wins(self):
        t = PricingTable({"gpt": (1.0, 1.0), "gpt-4o": (2.5, 10.0)})
        assert t.price_for("gpt-4o-mini") == (2.5, 10.0)

    def test_compute_cost_known_model(self):
        t = default_pricing_table()
        usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        assert t.compute_cost("gpt-4o", usage) == pytest.approx(12.50, rel=1e-6)

    def test_compute_cost_unknown_model_is_zero(self):
        t = default_pricing_table()
        usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        assert t.compute_cost("unknown-model", usage) == 0.0

    def test_compute_cost_zero_tokens(self):
        t = default_pricing_table()
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert t.compute_cost("gpt-4o", usage) == 0.0


class TestLoadPricingTable:
    def test_none_config_returns_builtins(self):
        t = load_pricing_table(None)
        assert "gpt-4o" in t

    def test_empty_config_returns_builtins(self):
        t = load_pricing_table({})
        assert "gpt-4o" in t

    def test_list_entries_override(self):
        cfg = {"pricing": {"gpt-4o": [1.0, 2.0]}}
        t = load_pricing_table(cfg)
        assert t.price_for("gpt-4o") == (1.0, 2.0)

    def test_dict_entries_override(self):
        cfg = {"pricing": {"new-model": {"prompt_per_1m": 5.0, "completion_per_1m": 7.0}}}
        t = load_pricing_table(cfg)
        assert t.price_for("new-model") == (5.0, 7.0)

    def test_no_pricing_key_returns_builtins(self):
        t = load_pricing_table({"providers": {}})
        assert "gpt-4o" in t


class TestComputeUsageCost:
    def test_returns_copy_with_cost(self):
        usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        result = compute_usage_cost(usage, "gpt-4o")
        assert result.cost_usd == pytest.approx(12.50, rel=1e-6)
        assert usage.cost_usd == 0.0  # original not mutated

    def test_unknown_model_zero_cost(self):
        usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        result = compute_usage_cost(usage, "unknown-model")
        assert result.cost_usd == 0.0
class TestUsageTracker:
    def test_record_accumulates(self):
        tracker = UsageTracker()
        tracker.record("CodingAgent", LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.01))
        tracker.record("CodingAgent", LLMUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost_usd=0.02))
        rec = tracker.get("CodingAgent")
        assert rec is not None
        assert rec.calls == 2
        assert rec.prompt_tokens == 300
        assert rec.completion_tokens == 150
        assert rec.total_tokens == 450
        assert rec.cost_usd == pytest.approx(0.03, rel=1e-9)

    def test_separate_agents_separate_records(self):
        tracker = UsageTracker()
        tracker.record("CodingAgent", LLMUsage(prompt_tokens=10, cost_usd=0.01))
        tracker.record("ResearchAgent", LLMUsage(prompt_tokens=20, cost_usd=0.02))
        assert tracker.get("CodingAgent").prompt_tokens == 10
        assert tracker.get("ResearchAgent").prompt_tokens == 20
        assert tracker.total_calls() == 2
        assert tracker.total_cost_usd() == pytest.approx(0.03, rel=1e-9)

    def test_empty_agent_name_becomes_unknown(self):
        tracker = UsageTracker()
        tracker.record("", LLMUsage(prompt_tokens=5, cost_usd=0.001))
        assert tracker.get("unknown") is not None

    def test_total_cost_and_calls(self):
        tracker = UsageTracker()
        tracker.record("a", LLMUsage(cost_usd=1.5))
        tracker.record("b", LLMUsage(cost_usd=2.5))
        assert tracker.total_cost_usd() == pytest.approx(4.0, rel=1e-9)
        assert tracker.total_calls() == 2

    def test_reset_clears(self):
        tracker = UsageTracker()
        tracker.record("a", LLMUsage(cost_usd=1.0))
        tracker.reset()
        assert tracker.total_calls() == 0
        assert tracker.total_cost_usd() == 0.0
        assert tracker.records() == []

    def test_to_dict_serializable(self):
        tracker = UsageTracker()
        tracker.record("a", LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.5))
        d = tracker.to_dict()
        assert d["total_calls"] == 1
        assert d["total_cost_usd"] == pytest.approx(0.5, rel=1e-9)
        assert d["per_agent"][0]["agent_name"] == "a"
        assert d["per_agent"][0]["prompt_tokens"] == 10

    def test_consume_iterable(self):
        tracker = UsageTracker()
        tracker.consume([
            ("a", LLMUsage(cost_usd=0.1)),
            ("b", LLMUsage(cost_usd=0.2)),
        ])
        assert tracker.total_calls() == 2


class TestGlobalUsageTracker:
    def test_singleton_is_stable(self):
        reset_usage_tracker()
        a = get_usage_tracker()
        b = get_usage_tracker()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_usage_tracker()
        a.record("x", LLMUsage(cost_usd=1.0))
        reset_usage_tracker()
        b = get_usage_tracker()
        assert a is not b
        assert b.total_calls() == 0


class TestLLMUsageCostField:
    def test_default_cost_zero(self):
        u = LLMUsage()
        assert u.cost_usd == 0.0

    def test_cost_round_trips(self):
        u = LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.012345)
        d = u.model_dump()
        assert d["cost_usd"] == 0.012345

    def test_custom_table(self):
        t = PricingTable({"foo": (1.0, 1.0)})
        usage = LLMUsage(prompt_tokens=500_000, completion_tokens=500_000, total_tokens=1_000_000)
        result = compute_usage_cost(usage, "foo", t)
        assert result.cost_usd == pytest.approx(1.0, rel=1e-6)
        usage = LLMUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        assert t.compute_cost("unknown-model", usage) == 0.0

    def test_compute_cost_zero_tokens(self):
        t = default_pricing_table()
        usage = LLMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert t.compute_cost("gpt-4o", usage) == 0.0
