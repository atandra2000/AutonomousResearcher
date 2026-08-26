"""Per-model token pricing and cumulative usage/cost tracking.

The pricing table is loaded from the ``pricing`` section of
``llm_config.yaml`` (see :func:`load_pricing_table`). Each entry maps a model
id to a ``(prompt_per_1m, completion_per_1m)`` pair, where both values are the
*USD cost per 1 million tokens* (the convention used by OpenAI and Anthropic).
Model ids are matched as a case-insensitive prefix so that versioned or
quantized variants (``gpt-4o-2024-08-06``, ``gpt-4o-mini``) resolve to the
base price.

A built-in fallback table covers common commercial models so the platform
reports cost even when the config omits a ``pricing`` section.

The :class:`UsageTracker` accumulates usage and cost per agent name across
the lifetime of a run; the router records every completion into the
process-wide tracker exposed by :func:`get_usage_tracker`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

from research_engineer.llm.base import LLMUsage

#: Built-in fallback pricing (USD per 1M tokens): prompt, completion.
#: This is intentionally a small, conservative table; production deployments
#: should override via the ``pricing`` section of ``llm_config.yaml``.
_BUILTIN_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
    # Ollama Cloud models used by this project (conservative placeholders).
    "glm-5.2:cloud": (0.50, 1.50),
    "kimi-k2.7-code": (0.60, 2.20),
    "minimax-m3:cloud": (0.40, 1.20),
    # Local models are free.
    "llama3": (0.0, 0.0),
}

class PricingTable:
    """Per-model token price table (USD per 1M tokens)."""

    def __init__(self, entries: dict[str, tuple[float, float]] | None = None) -> None:
        # Normalize keys to lowercase for case-insensitive matching.
        self._entries: dict[str, tuple[float, float]] = {}
        if entries:
            for k, v in entries.items():
                self._entries[k.lower()] = (float(v[0]), float(v[1]))

    def add(self, model: str, prompt_per_1m: float, completion_per_1m: float) -> None:
        """Register or override a model's pricing."""
        self._entries[model.lower()] = (float(prompt_per_1m), float(completion_per_1m))

    def price_for(self, model: str) -> tuple[float, float] | None:
        """Return ``(prompt_per_1m, completion_per_1m)`` or ``None`` if unknown.

        Matching is case-insensitive prefix: the longest configured key that
        is a prefix of ``model`` wins, so ``gpt-4o-2024-08-06`` resolves to
        the ``gpt-4o`` entry.
        """
        if not model:
            return None
        m = model.lower()
        best: tuple[float, float] | None = None
        best_len = -1
        for key, price in self._entries.items():
            if m == key or m.startswith(key):
                if len(key) > best_len:
                    best = price
                    best_len = len(key)
        return best

    def compute_cost(self, model: str, usage: LLMUsage) -> float:
        """Compute the USD cost of ``usage`` under this pricing table.

        Returns ``0.0`` when the model is unknown (so unpriced models never
        produce misleading costs).
        """
        price = self.price_for(model)
        if price is None:
            return 0.0
        prompt_per_1m, completion_per_1m = price
        return round(
            (usage.prompt_tokens / _MILLION) * prompt_per_1m
            + (usage.completion_tokens / _MILLION) * completion_per_1m,
            6,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, model: str) -> bool:
        return self.price_for(model) is not None


def default_pricing_table() -> PricingTable:
    """Return a :class:`PricingTable` seeded with the built-in fallbacks."""
    return PricingTable(dict(_BUILTIN_PRICING))

def load_pricing_table(config: dict[str, Any] | None) -> PricingTable:
    """Build a :class:`PricingTable` from a config dict's ``pricing`` section.

    The built-in fallbacks are always included; config entries override or
    extend them. Config entries may use either a 2-element list
    ``[prompt, completion]`` or a mapping ``{prompt_per_1m, completion_per_1m}``.
    """
    table = default_pricing_table()
    if not config:
        return table
    pricing = config.get("pricing")
    if not isinstance(pricing, dict):
        return table
    for model, spec in pricing.items():
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            table.add(model, float(spec[0]), float(spec[1]))
        elif isinstance(spec, dict):
            p = spec.get("prompt_per_1m", spec.get("prompt", 0.0)) or 0.0
            c = spec.get("completion_per_1m", spec.get("completion", 0.0)) or 0.0
            table.add(model, float(p), float(c))
    return table


def compute_usage_cost(
    usage: LLMUsage, model: str, table: PricingTable | None = None
) -> LLMUsage:
    """Return a *copy* of ``usage`` with ``cost_usd`` populated.

    Callers (providers, router) use this to stamp cost onto a response's
    usage object without mutating the original. If ``table`` is ``None``
    the built-in fallback table is used.
    """
    tbl = table if table is not None else default_pricing_table()
    cost = tbl.compute_cost(model, usage)
    copied: LLMUsage = usage.model_copy(update={"cost_usd": cost})
    return copied

class UsageRecord:
    """A single accumulated per-agent usage record."""

    __slots__ = (
        "agent_name",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    )

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0

    def add(self, usage: LLMUsage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.cost_usd = round(self.cost_usd + usage.cost_usd, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class UsageTracker:
    """Thread-safe accumulator of per-agent token usage and USD cost.

    A single process-wide instance is exposed by :func:`get_usage_tracker`.
    The router records every completion (model + usage) into the tracker so
    that orchestration code (the research loop) can derive real cost
    estimates instead of a hardcoded constant.
    """

    def __init__(self) -> None:
        self._records: dict[str, UsageRecord] = {}
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_calls = 0

    def record(self, agent_name: str, usage: LLMUsage) -> None:
        """Accumulate ``usage`` under ``agent_name``."""
        if not agent_name:
            agent_name = "unknown"
        with self._lock:
            rec = self._records.get(agent_name)
            if rec is None:
                rec = UsageRecord(agent_name)
                self._records[agent_name] = rec
            rec.add(usage)
            self._total_cost = round(self._total_cost + usage.cost_usd, 6)
            self._total_calls += 1

    def get(self, agent_name: str) -> UsageRecord | None:
        with self._lock:
            return self._records.get(agent_name)

    def records(self) -> list[UsageRecord]:
        """Return a snapshot of all per-agent records."""
        with self._lock:
            return list(self._records.values())

    def total_cost_usd(self) -> float:
        with self._lock:
            return round(self._total_cost, 6)

    def total_calls(self) -> int:
        with self._lock:
            return self._total_calls

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_calls = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        with self._lock:
            return {
                "per_agent": [r.to_dict() for r in self._records.values()],
                "total_cost_usd": round(self._total_cost, 6),
                "total_calls": self._total_calls,
            }

    def consume(self, iterable: Iterable[tuple[str, LLMUsage]]) -> None:
        """Convenience: record each ``(agent_name, usage)`` from an iterable."""
        for agent_name, usage in iterable:
            self.record(agent_name, usage)


_tracker: UsageTracker | None = None
_tracker_lock = threading.Lock()


def get_usage_tracker() -> UsageTracker:
    """Return the process-wide :class:`UsageTracker` singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = UsageTracker()
    return _tracker


def reset_usage_tracker() -> None:
    """Clear the process-wide tracker (tests use this)."""
    global _tracker
    with _tracker_lock:
        if _tracker is not None:
            _tracker.reset()
        _tracker = None


__all__ = [
    "PricingTable",
    "UsageRecord",
    "UsageTracker",
    "compute_usage_cost",
    "default_pricing_table",
    "load_pricing_table",
    "get_usage_tracker",
    "reset_usage_tracker",
]
_MILLION = 1_000_000.0
