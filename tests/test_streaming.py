"""Tests for streaming-first LLM completions (v2.1).

Covers the Anthropic native streaming parser, the :func:`collect_stream` and
:func:`stream_with_retry` helpers, the router's cost/observability-stamped
``stream_response``, and ReAct-loop streaming mode.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from research_engineer.llm import (
    AnthropicProvider,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    ProviderError,
    ReActLoopConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
    collect_stream,
    run_react_loop,
    stream_with_retry,
)
from research_engineer.llm.streaming import StreamChunk


def _req() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])


# ---------------------------------------------------------------------------
# Anthropic native streaming parser
# ---------------------------------------------------------------------------


class TestAnthropicStreamParser:
    def _make(self) -> AnthropicProvider:
        return AnthropicProvider(
            base_url="https://api.anthropic.com",
            api_key="test-key",
            default_model="claude-3-5-sonnet-latest",
        )

    def test_parse_text_delta(self) -> None:
        prov = self._make()
        line = (
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"Hello"}}'
        )
        assert prov._parse_stream_event(line) == "Hello"

    def test_parse_ignores_non_text_events(self) -> None:
        prov = self._make()
        # message_start, content_block_start, content_block_stop, message_delta,
        # message_stop, and ping events carry no visible text.
        assert prov._parse_stream_event('data: {"type":"message_start"}') is None
        assert (
            prov._parse_stream_event(
                'data: {"type":"content_block_start","content_block":{"type":"text"}}'
            )
            is None
        )
        assert (
            prov._parse_stream_event(
                'data: {"type":"content_block_delta","delta":{"type":"input_json_delta"}}'
            )
            is None
        )
        assert (
            prov._parse_stream_event(
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}'
            )
            is None
        )
        assert prov._parse_stream_event("event: message_stop") is None
        assert prov._parse_stream_event("data: [DONE]") is None

    def test_parse_handles_malformed_lines(self) -> None:
        prov = self._make()
        assert prov._parse_stream_event("") is None
        assert prov._parse_stream_event("not a data line") is None
        assert prov._parse_stream_event("data: not-json") is None
        assert prov._parse_stream_event("data: 42") is None

class _StreamTransport(httpx.MockTransport):
    """Mock transport returning a canned SSE body for streaming tests."""

    def __init__(self, body: str, status: int = 200) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.last = request  # type: ignore[attr-defined]
            return httpx.Response(
                status,
                content=body.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )

        super().__init__(handler)
        self.last: httpx.Request | None = None


class _MockJsonTransport(httpx.MockTransport):
    """Mock transport returning a canned JSON payload (for non-streamed calls)."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.last = request  # type: ignore[attr-defined]
            return httpx.Response(status, json=payload)

        super().__init__(handler)
        self.last: httpx.Request | None = None


class TestAnthropicStream:
    def _make(self, body: str, status: int = 200) -> AnthropicProvider:
        client = httpx.AsyncClient(transport=_StreamTransport(body, status))
        return AnthropicProvider(
            base_url="https://api.anthropic.com",
            api_key="test-key",
            default_model="claude-3-5-sonnet-latest",
            client=client,
        )

    @pytest.mark.asyncio
    async def test_stream_yields_text_deltas(self) -> None:
        body = (
            'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-3-5-sonnet-latest"}}\n\n'
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n'
            'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
            'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )
        prov = self._make(body)
        req = _req().model_copy(update={"stream": True})
        chunks = [c async for c in prov.stream(req)]
        assert chunks == ["Hello", " world"]
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_stream_http_error_raises(self) -> None:
        prov = self._make("error", status=500)
        with pytest.raises(ProviderError):
            async for _ in prov.stream(_req()):
                pass
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_stream_false_falls_back_to_complete(self) -> None:
        # When request.stream is False, stream() delegates to complete(), which
        # expects a non-streamed JSON payload.
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        client = httpx.AsyncClient(transport=_MockJsonTransport(payload))
        prov = AnthropicProvider(
            base_url="https://api.anthropic.com",
            api_key="test-key",
            default_model="claude-3-5-sonnet-latest",
            client=client,
        )
        req = _req().model_copy(update={"stream": False})
        chunks = [c async for c in prov.stream(req)]
        assert chunks == ["hello"]
        await prov.aclose()


# ---------------------------------------------------------------------------
# collect_stream
# ---------------------------------------------------------------------------


class _ChunkProvider(LLMProvider):
    """Provider whose stream() yields a fixed sequence of chunks."""

    name = "chunky"

    def __init__(self, chunks: list[Any], default_model: str = "m") -> None:
        self._chunks = list(chunks)
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def stream(self, request: LLMRequest) -> Any:
        for c in self._chunks:
            yield c


class _NoStreamProvider(LLMProvider):
    """Provider that does not implement streaming."""

    name = "nostream"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def stream(self, request: LLMRequest) -> Any:
        raise NotImplementedError(f"{self.name} does not support streaming")


class TestCollectStream:
    @pytest.mark.asyncio
    async def test_assembles_str_chunks(self) -> None:
        prov = _ChunkProvider(["Hello", " ", "world"])
        resp = await collect_stream(prov, _req())
        assert resp.content == "Hello world"
        assert resp.provider == "chunky"
        assert resp.model == "m"

    @pytest.mark.asyncio
    async def test_handles_stream_chunks_with_model(self) -> None:
        prov = _ChunkProvider(
            [
                StreamChunk(text="a"),
                StreamChunk(text="b", model="gpt-4o"),
                StreamChunk(text="c"),
            ]
        )
        resp = await collect_stream(prov, _req())
        assert resp.content == "abc"
        assert resp.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_uses_request_model(self) -> None:
        prov = _ChunkProvider(["x"])
        req = _req().model_copy(update={"model": "override"})
        resp = await collect_stream(prov, req)
        assert resp.model == "override"

    @pytest.mark.asyncio
    async def test_raises_when_streaming_unsupported(self) -> None:
        prov = _NoStreamProvider()
        with pytest.raises(ProviderError, match="does not support streaming"):
            await collect_stream(prov, _req())


# ---------------------------------------------------------------------------
# stream_with_retry
# ---------------------------------------------------------------------------


class _FlakyProvider(LLMProvider):
    """Provider that fails ``n`` times before streaming successfully."""

    name = "flaky"

    def __init__(self, failures: int, chunks: list[str]) -> None:
        self._failures = failures
        self._chunks = list(chunks)
        self.attempts = 0
        self.default_model = "m"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def stream(self, request: LLMRequest) -> Any:
        self.attempts += 1
        if self.attempts <= self._failures:
            raise ProviderError("transient 500", provider=self.name)
        for c in self._chunks:
            yield c


class _PermanentProvider(LLMProvider):
    """Provider that always fails with a permanent (4xx) error."""

    name = "perm"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def stream(self, request: LLMRequest) -> Any:
        raise ProviderError("HTTP 401 unauthorized", provider=self.name)
        yield  # pragma: no cover - makes this an async generator


class TestStreamWithRetry:
    @pytest.mark.asyncio
    async def test_retries_transient_then_succeeds(self) -> None:
        prov = _FlakyProvider(failures=2, chunks=["ok"])
        resp = await stream_with_retry(prov, _req(), max_attempts=4, base_delay=0.0)
        assert resp.content == "ok"
        assert prov.attempts == 3

    @pytest.mark.asyncio
    async def test_fails_fast_on_permanent_error(self) -> None:
        prov = _PermanentProvider()
        with pytest.raises(ProviderError, match="401"):
            await stream_with_retry(prov, _req(), max_attempts=3, base_delay=0.0)

    @pytest.mark.asyncio
    async def test_exhausts_attempts(self) -> None:
        prov = _FlakyProvider(failures=5, chunks=["never"])
        with pytest.raises(ProviderError, match="failed after 3 attempts"):
            await stream_with_retry(prov, _req(), max_attempts=3, base_delay=0.0)

    @pytest.mark.asyncio
    async def test_invokes_on_complete(self) -> None:
        prov = _ChunkProvider(["hi"])
        seen: list[tuple[LLMRequest, LLMResponse, float]] = []

        def hook(req: LLMRequest, resp: LLMResponse, latency: float) -> None:
            seen.append((req, resp, latency))

        resp = await stream_with_retry(
            prov, _req(), agent_name="test", on_complete=hook
        )
        assert resp.content == "hi"
        assert len(seen) == 1
        assert seen[0][1].content == "hi"


# ---------------------------------------------------------------------------
# Router stream_response (cost + observability stamping)
# ---------------------------------------------------------------------------


class _BoundLikeProvider(LLMProvider):
    """Minimal provider exposing a ``stream_response``-style assembled path."""

    name = "boundlike"

    def __init__(self, delegate: LLMProvider, model: str | None) -> None:
        self._delegate = delegate
        self._model = model
        self.default_model = model or delegate.default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def stream_response(self, request: LLMRequest) -> LLMResponse:
        from research_engineer.llm.streaming import stream_with_retry

        return await stream_with_retry(
            self._delegate,
            request,
            agent_name=f"{self._delegate.name}/{self._model or 'default'}",
            on_complete=self._stamp,
        )

    def _stamp(
        self, request: LLMRequest, response: LLMResponse, latency_seconds: float
    ) -> None:
        from research_engineer.llm.cost import compute_usage_cost

        response.usage = compute_usage_cost(response.usage, response.model)


class TestBoundStreamResponse:
    @pytest.mark.asyncio
    async def test_stream_response_assembles_and_stamps_cost(self) -> None:
        delegate = _ChunkProvider(["hello"], default_model="gpt-4o")
        bound = _BoundLikeProvider(delegate, "gpt-4o")
        resp = await bound.stream_response(_req())
        assert resp.content == "hello"
        assert resp.model == "gpt-4o"
        # gpt-4o is priced; cost is stamped even with zero tokens.
        assert resp.usage.cost_usd == 0.0


# ---------------------------------------------------------------------------
# ReAct loop streaming mode
# ---------------------------------------------------------------------------


class _StreamingReActProvider(LLMProvider):
    """Provider that streams via ``stream_response`` and supports tools."""

    name = "streamreact"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.stream_calls = 0
        self.default_model = "m"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("complete() not used")

    async def complete_with_tools(
        self, request: LLMRequest, tools: list[ToolDefinition]
    ) -> LLMResponse:
        raise AssertionError("complete_with_tools() should not be used in stream mode")

    async def stream_response(self, request: LLMRequest) -> LLMResponse:
        self.stream_calls += 1
        if not self.responses:
            raise AssertionError("No more scripted responses")
        return self.responses.pop(0)


def _resp(content: str, tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="m",
        provider="streamreact",
        tool_calls=tool_calls,
    )


async def _echo_executor(call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id, name=call.name, content=f"result:{call.name}"
    )


TOOLS = [ToolDefinition(name="f", description="", parameters={"type": "object"})]


class TestReActStreaming:
    @pytest.mark.asyncio
    async def test_stream_mode_uses_stream_response(self) -> None:
        provider = _StreamingReActProvider(
            [
                _resp("", [ToolCall(id="c1", name="f", arguments={})]),
                _resp("FINAL_ANSWER: done"),
            ]
        )
        config = ReActLoopConfig(stream=True)
        result = await run_react_loop(
            provider,
            [LLMMessage(role=LLMRole.USER, content="go")],
            TOOLS,
            _echo_executor,
            config,
        )
        assert result.termination.value == "final_answer"
        assert result.response.content == "FINAL_ANSWER: done"
        assert provider.stream_calls == 2

    @pytest.mark.asyncio
    async def test_stream_mode_falls_back_to_tools(self) -> None:
        # A provider without stream_response falls back to complete_with_tools.
        provider = _StreamingReActProvider(
            [_resp("FINAL_ANSWER: done")]
        )
        # Remove stream_response to simulate a non-streaming provider.
        del type(provider).stream_response  # type: ignore[attr-defined]
        config = ReActLoopConfig(stream=True)
        # complete_with_tools raises here, proving the fallback path is taken.
        with pytest.raises(AssertionError, match="complete_with_tools"):
            await run_react_loop(
                provider,
                [LLMMessage(role=LLMRole.USER, content="go")],
                TOOLS,
                _echo_executor,
                config,
            )

    def test_react_loop_config_stream_default(self) -> None:
        assert ReActLoopConfig().stream is False

