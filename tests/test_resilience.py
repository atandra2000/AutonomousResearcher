"""Tests for resilient LLM completions and truncation handling (A4).

Covers:
- ``LLMResponse.truncated`` convenience property.
- ``complete_with_retry`` escalating ``max_tokens`` once when a response is
  truncated (empty or non-empty), then returning the final response.
- Non-truncated, non-empty responses returning immediately.
"""

from __future__ import annotations

import pytest

from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
)
from research_engineer.llm.resilience import complete_with_retry


class _TruncatingProvider(LLMProvider):
    """Fake provider that returns truncated responses until told otherwise.

    ``responses`` is a queue of ``(content, finish_reason)`` tuples consumed
    one per ``complete`` call. When exhausted, the last entry repeats.
    """

    name = "fake"

    def __init__(
        self,
        responses: list[tuple[str, str | None]],
        default_model: str = "fake-model",
    ) -> None:
        self.default_model = default_model
        self._responses = responses
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        content, finish_reason = self._responses[
            min(len(self.calls) - 1, len(self._responses) - 1)
        ]
        return LLMResponse(
            content=content,
            model=request.model or self.default_model,
            provider=self.name,
            finish_reason=finish_reason,
        )


# ---------------------------------------------------------------------------
# LLMResponse.truncated property
# ---------------------------------------------------------------------------


class TestTruncatedProperty:
    def test_truncated_true_when_finish_reason_length(self):
        r = LLMResponse(
            content="partial answer",
            model="m",
            provider="p",
            finish_reason="length",
        )
        assert r.truncated is True

    def test_truncated_false_when_finish_reason_stop(self):
        r = LLMResponse(
            content="full answer",
            model="m",
            provider="p",
            finish_reason="stop",
        )
        assert r.truncated is False

    def test_truncated_false_when_finish_reason_none(self):
        r = LLMResponse(content="ok", model="m", provider="p")
        assert r.truncated is False

    def test_truncated_false_when_other_finish_reason(self):
        r = LLMResponse(
            content="ok",
            model="m",
            provider="p",
            finish_reason="tool_calls",
        )
        assert r.truncated is False


# ---------------------------------------------------------------------------
# complete_with_retry truncation handling
# ---------------------------------------------------------------------------


class TestCompleteWithRetryTruncation:
    def _request(self, max_tokens: int = 100) -> LLMRequest:
        return LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
            max_tokens=max_tokens,
        )

    @pytest.mark.asyncio
    async def test_non_truncated_non_empty_returns_immediately(self):
        prov = _TruncatingProvider([("full answer", "stop")])
        resp = await complete_with_retry(prov, self._request())
        assert resp.content == "full answer"
        assert resp.truncated is False
        # Only one call: no retry for a clean completion.
        assert len(prov.calls) == 1

    @pytest.mark.asyncio
    async def test_empty_truncated_escalates_budget_once(self):
        # First call: empty content, finish_reason='length' -> escalate.
        # Second call: full answer, finish_reason='stop' -> return.
        prov = _TruncatingProvider([("", "length"), ("full answer", "stop")])
        resp = await complete_with_retry(prov, self._request(max_tokens=100))
        assert resp.content == "full answer"
        assert resp.truncated is False
        assert len(prov.calls) == 2
        # The retry doubled the max_tokens budget.
        assert prov.calls[1].max_tokens == 200

    @pytest.mark.asyncio
    async def test_non_empty_truncated_escalates_budget_once(self):
        # First call: partial content, finish_reason='length' -> escalate.
        # Second call: full answer, finish_reason='stop' -> return.
        prov = _TruncatingProvider(
            [("partial answer...", "length"), ("full answer", "stop")]
        )
        resp = await complete_with_retry(prov, self._request(max_tokens=100))
        assert resp.content == "full answer"
        assert resp.truncated is False
        assert len(prov.calls) == 2
        assert prov.calls[1].max_tokens == 200

    @pytest.mark.asyncio
    async def test_truncated_escalation_capped_at_max_tokens(self):
        # max_tokens already at the cap -> no escalation possible. The retry
        # loop still runs to max_attempts, but the budget is never doubled.
        prov = _TruncatingProvider([("partial", "length")])
        resp = await complete_with_retry(
            prov, self._request(max_tokens=16_384)
        )
        assert resp.content == "partial"
        assert resp.truncated is True
        # Default max_attempts=3; all three calls keep the capped budget.
        assert len(prov.calls) == 3
        assert all(c.max_tokens == 16_384 for c in prov.calls)

    @pytest.mark.asyncio
    async def test_truncated_escalation_applied_only_once(self):
        # Both calls truncated: escalate once, then keep retrying without
        # further escalation until max_attempts is exhausted.
        prov = _TruncatingProvider(
            [("partial one", "length"), ("partial two", "length")]
        )
        resp = await complete_with_retry(prov, self._request(max_tokens=100))
        assert resp.content == "partial two"
        assert resp.truncated is True
        # Default max_attempts=3; escalation applied once (call 2), then the
        # budget stays at 200 for call 3 (no second doubling).
        assert len(prov.calls) == 3
        assert prov.calls[1].max_tokens == 200
        assert prov.calls[2].max_tokens == 200

    @pytest.mark.asyncio
    async def test_empty_not_truncated_does_not_escalate(self):
        # Empty content with finish_reason='stop' is not a truncation; no
        # budget escalation, but the retry loop still runs (empty content is
        # not returned). With max_attempts=2 the second call returns content.
        prov = _TruncatingProvider([("", "stop"), ("ok", "stop")])
        resp = await complete_with_retry(
            prov, self._request(max_tokens=100), max_attempts=2
        )
        assert resp.content == "ok"
        assert len(prov.calls) == 2
        # Budget was NOT escalated for a non-truncated empty response.
        assert prov.calls[1].max_tokens == 100

