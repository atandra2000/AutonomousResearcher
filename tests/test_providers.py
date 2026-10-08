"""Tests for the multi-provider LLM layer (A3).

Covers the OpenAI-compatible, Anthropic, and local-Ollama providers with
mock httpx transports, plus factory health-check/failover ordering and the
router's failover binding.
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
    LLMRole,
    LocalOllamaProvider,
    ModelRouter,
    OpenAIProvider,
    ProviderError,
    ProviderFactory,
    ToolDefinition,
    register_provider_type,
    reset_factory,
    reset_router,
)


class _MockTransport(httpx.MockTransport):
    """Mock transport returning a canned JSON payload."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.last = request  # type: ignore[attr-defined]
            return httpx.Response(status, json=payload)

        super().__init__(handler)
        self.last: httpx.Request | None = None


def _req() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _make(self, payload: dict[str, Any], status: int = 200) -> OpenAIProvider:
        client = httpx.AsyncClient(transport=_MockTransport(payload, status))
        return OpenAIProvider(
            base_url="https://api.openai.com",
            api_key="test-key",
            default_model="gpt-4o",
            client=client,
        )

    @pytest.mark.asyncio
    async def test_complete_success(self):
        payload = {
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        prov = self._make(payload)
        resp = await prov.complete(_req())
        assert resp.content == "hello"
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai"
        assert resp.usage.total_tokens == 5
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_http_error_raises(self):
        prov = self._make({"error": "bad"}, status=401)
        with pytest.raises(ProviderError):
            await prov.complete(_req())
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_headers_include_auth(self):
        prov = self._make({"choices": [{"message": {"content": "ok"}}]})
        await prov.complete(_req())
        transport = prov._client._transport  # type: ignore[attr-defined]
        sent: httpx.Request | None = getattr(transport, "last", None)
        assert sent is not None
        assert sent.headers.get("Authorization") == "Bearer test-key"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_with_tools_parses_tool_calls(self):
        payload = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q": "x"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        prov = self._make(payload)
        tools = [ToolDefinition(name="search", description="", parameters={"type": "object"})]
        resp = await prov.complete_with_tools(_req(), tools)
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "x"}
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_ok(self):
        prov = self._make({"data": []})
        assert await prov.health() is True
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_unhealthy_status(self):
        prov = self._make({"error": "down"}, status=503)
        assert await prov.health() is False
        await prov.aclose()

    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/")
        monkeypatch.setenv("OPENAI_API_KEY", "envkey")
        monkeypatch.setenv("OPENAI_MODEL", "env-model")
        prov = OpenAIProvider()
        assert prov.base_url == "https://env.example"
        assert prov.api_key == "envkey"
        assert prov.default_model == "env-model"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _make(self, payload: dict[str, Any], status: int = 200) -> AnthropicProvider:
        client = httpx.AsyncClient(transport=_MockTransport(payload, status))
        return AnthropicProvider(
            base_url="https://api.anthropic.com",
            api_key="test-key",
            default_model="test-model",
            client=client,
        )

    @pytest.mark.asyncio
    async def test_complete_success(self):
        payload = {
            "model": "test-model",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        prov = self._make(payload)
        resp = await prov.complete(_req())
        assert resp.content == "hello"
        assert resp.model == "test-model"
        assert resp.provider == "anthropic"
        assert resp.usage.total_tokens == 5
        assert resp.finish_reason == "end_turn"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_http_error_raises(self):
        prov = self._make({"error": "bad"}, status=401)
        with pytest.raises(ProviderError):
            await prov.complete(_req())
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_headers_include_api_key(self):
        prov = self._make({"content": [{"type": "text", "text": "ok"}]})
        await prov.complete(_req())
        transport = prov._client._transport  # type: ignore[attr-defined]
        sent: httpx.Request | None = getattr(transport, "last", None)
        assert sent is not None
        assert sent.headers.get("x-api-key") == "test-key"
        assert sent.headers.get("anthropic-version") == "2023-06-01"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_with_tools_parses_tool_use(self):
        payload = {
            "model": "test-model",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}}
            ],
            "stop_reason": "tool_use",
        }
        prov = self._make(payload)
        tools = [ToolDefinition(name="search", description="", parameters={"type": "object"})]
        resp = await prov.complete_with_tools(_req(), tools)
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].id == "toolu_1"
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "x"}
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_build_payload_hoists_system_and_defaults_max_tokens(self):
        prov = self._make({"content": [{"type": "text", "text": "ok"}]})
        req = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content="be brief"),
                LLMMessage(role=LLMRole.USER, content="hi"),
            ]
        )
        await prov.complete(req)
        transport = prov._client._transport  # type: ignore[attr-defined]
        sent: httpx.Request | None = getattr(transport, "last", None)
        assert sent is not None
        import json

        data = json.loads(sent.read())
        assert data["system"] == "be brief"
        assert data["max_tokens"] == AnthropicProvider.DEFAULT_MAX_TOKENS
        assert data["messages"] == [{"role": "user", "content": "hi"}]
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_ok(self):
        prov = self._make({"data": []})
        assert await prov.health() is True
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_unhealthy_status(self):
        prov = self._make({"error": "down"}, status=500)
        assert await prov.health() is False
        await prov.aclose()

    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.example/")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "envkey")
        monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
        prov = AnthropicProvider()
        assert prov.base_url == "https://env.example"
        assert prov.api_key == "envkey"
        assert prov.default_model == "env-model"


# ---------------------------------------------------------------------------
# Local Ollama provider
# ---------------------------------------------------------------------------


class TestLocalOllamaProvider:
    def _make(self, payload: dict[str, Any], status: int = 200) -> LocalOllamaProvider:
        client = httpx.AsyncClient(transport=_MockTransport(payload, status))
        return LocalOllamaProvider(
            base_url="http://localhost:11434",
            default_model="llama3",
            client=client,
        )

    @pytest.mark.asyncio
    async def test_complete_success(self):
        payload = {
            "model": "llama3",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }
        prov = self._make(payload)
        resp = await prov.complete(_req())
        assert resp.content == "hi"
        assert resp.model == "llama3"
        assert resp.provider == "local_ollama"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_http_error_raises(self):
        prov = self._make({"error": "bad"}, status=500)
        with pytest.raises(ProviderError):
            await prov.complete(_req())
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_no_auth_header_by_default(self):
        prov = self._make({"choices": [{"message": {"content": "ok"}}]})
        await prov.complete(_req())
        transport = prov._client._transport  # type: ignore[attr-defined]
        sent: httpx.Request | None = getattr(transport, "last", None)
        assert sent is not None
        assert sent.headers.get("Authorization") is None
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_ok(self):
        prov = self._make({"data": []})
        assert await prov.health() is True
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_health_unhealthy_status(self):
        prov = self._make({"error": "down"}, status=503)
        assert await prov.health() is False
        await prov.aclose()

    def test_default_base_url(self):
        prov = LocalOllamaProvider()
        assert prov.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Factory health-check + failover ordering
# ---------------------------------------------------------------------------


class _HealthyProvider(LLMProvider):
    name = "healthy"

    def __init__(self, default_model: str = "m") -> None:
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> Any:
        return None

    async def health(self) -> bool:
        return True


class _UnhealthyProvider(LLMProvider):
    name = "unhealthy"

    def __init__(self, default_model: str = "m") -> None:
        self.default_model = default_model

    async def complete(self, request: LLMRequest) -> Any:
        return None

    async def health(self) -> bool:
        return False


class TestFactoryFailover:
    def setup_method(self):
        reset_factory()
        reset_router()

    def teardown_method(self):
        reset_factory()
        reset_router()

    @pytest.mark.asyncio
    async def test_healthy_provider_order_puts_healthy_first(self):
        register_provider_type("healthy", _HealthyProvider)
        register_provider_type("unhealthy", _UnhealthyProvider)
        cfg = {
            "default_provider": "unhealthy",
            "providers": {
                "unhealthy": {"type": "unhealthy"},
                "healthy": {"type": "healthy"},
            },
            "agents": {},
        }
        f = ProviderFactory(cfg)
        order = await f.healthy_provider_order()
        assert order[0] == "healthy"
        assert order[1] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_default_provider(self):
        register_provider_type("healthy", _HealthyProvider)
        cfg = {"default_provider": "healthy", "providers": {"healthy": {"type": "healthy"}}}
        f = ProviderFactory(cfg)
        assert await f.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        register_provider_type("unhealthy", _UnhealthyProvider)
        cfg = {"default_provider": "unhealthy", "providers": {"unhealthy": {"type": "unhealthy"}}}
        f = ProviderFactory(cfg)
        assert await f.health_check() is False

    @pytest.mark.asyncio
    async def test_router_failover_binds_healthy_provider(self):
        register_provider_type("healthy", _HealthyProvider)
        register_provider_type("unhealthy", _UnhealthyProvider)
        cfg = {
            "default_provider": "unhealthy",
            "default_model": "base",
            "providers": {
                "unhealthy": {"type": "unhealthy"},
                "healthy": {"type": "healthy"},
            },
            "agents": {"CodingAgent": {"provider": "unhealthy", "model": "coder"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        prov = await router.for_agent_with_failover("CodingAgent")
        # The configured provider is unhealthy, so failover picks the healthy one.
        assert prov.name == "healthy"
        assert prov.default_model == "coder"

    @pytest.mark.asyncio
    async def test_router_failover_falls_back_when_all_unhealthy(self):
        register_provider_type("unhealthy", _UnhealthyProvider)
        cfg = {
            "default_provider": "unhealthy",
            "default_model": "base",
            "providers": {"unhealthy": {"type": "unhealthy"}},
            "agents": {"CodingAgent": {"provider": "unhealthy", "model": "coder"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        prov = await router.for_agent_with_failover("CodingAgent")
        # No healthy provider; fall back to the configured one.
        assert prov.name == "unhealthy"
        assert prov.default_model == "coder"

    @pytest.mark.asyncio
    async def test_router_health_check(self):
        register_provider_type("healthy", _HealthyProvider)
        cfg = {
            "default_provider": "healthy",
            "providers": {"healthy": {"type": "healthy"}},
            "agents": {"CodingAgent": {"provider": "healthy", "model": "coder"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        assert await router.health_check("CodingAgent") is True
