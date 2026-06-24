"""Tests for the provider-agnostic LLM layer (Phase 10)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from research_engineer.llm import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
    ModelRouter,
    OllamaCloudProvider,
    ProviderError,
    ProviderFactory,
    get_factory,
    get_router,
    load_config,
    register_provider_type,
    reset_factory,
    reset_router,
)

# ---------------------------------------------------------------------------
# Base models
# ---------------------------------------------------------------------------


class TestLLMModels:
    def test_llm_role_values(self):
        assert LLMRole.SYSTEM == "system"
        assert LLMRole.USER == "user"
        assert LLMRole.ASSISTANT == "assistant"
        assert LLMRole.TOOL == "tool"

    def test_llm_message_roundtrip(self):
        m = LLMMessage(role=LLMRole.USER, content="hello")
        d = m.model_dump(exclude_none=True)
        assert d == {"role": "user", "content": "hello"}

    def test_llm_request_defaults(self):
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        assert req.model is None
        assert req.temperature == 0.2
        assert req.stream is False
        assert req.extra == {}

    def test_llm_request_temperature_bounds(self):
        with pytest.raises(Exception):
            LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content="hi")],
                temperature=3.0,
            )

    def test_llm_response_usage(self):
        r = LLMResponse(content="x", model="m", provider="p", usage=LLMUsage(prompt_tokens=4, completion_tokens=6))
        assert r.usage.total_tokens == 0  # explicitly not auto-computed
        assert r.provider == "p"


# ---------------------------------------------------------------------------
# Fake provider for unit tests (no network)
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, default_model: str = "fake-model", *, echo: bool = True) -> None:
        self.default_model = default_model
        self.calls: list[LLMRequest] = []
        self.echo = echo

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        model = request.model or self.default_model
        content = "\n".join(m.content for m in request.messages) if self.echo else "ok"
        return LLMResponse(content=content, model=model, provider=self.name)


@pytest.mark.asyncio
async def test_fake_provider_basic():
    p = _FakeProvider()
    req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
    resp = await p.complete(req)
    assert resp.content == "hi"
    assert resp.model == "fake-model"
    assert resp.provider == "fake"


# ---------------------------------------------------------------------------
# Ollama provider with an injected mock httpx client
# ---------------------------------------------------------------------------


class _MockTransport(httpx.MockTransport):
    """Mock transport returning a canned OpenAI-style chat completion."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.last = request  # type: ignore[attr-defined]
            return httpx.Response(status, json=payload)

        super().__init__(handler)
        self.last: httpx.Request | None = None


class TestOllamaCloudProvider:
    def _make(self, payload: dict[str, Any], status: int = 200) -> OllamaCloudProvider:
        client = httpx.AsyncClient(transport=_MockTransport(payload, status))
        return OllamaCloudProvider(
            base_url="https://api.olama.cloud",
            api_key="test-key",
            default_model="llama3",
            client=client,
        )

    @pytest.mark.asyncio
    async def test_complete_success(self):
        payload = {
            "model": "llama3",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        prov = self._make(payload)
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        resp = await prov.complete(req)
        assert resp.content == "hello there"
        assert resp.model == "llama3"
        assert resp.provider == "ollama"
        assert resp.usage.total_tokens == 5
        assert resp.finish_reason == "stop"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_uses_default_model(self):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        prov = self._make(payload)
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        resp = await prov.complete(req)
        assert resp.model == "llama3"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_request_model_override(self):
        payload = {"model": "qwen2.5-coder", "choices": [{"message": {"content": "x"}}]}
        prov = self._make(payload)
        req = LLMRequest(
            messages=[LLMMessage(role=LLMRole.USER, content="hi")],
            model="qwen2.5-coder",
        )
        resp = await prov.complete(req)
        assert resp.model == "qwen2.5-coder"
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_invalid_messages_raises(self):
        prov = self._make({})
        req = LLMRequest(messages=[])
        with pytest.raises(ProviderError):
            await prov.complete(req)
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_http_error_raises(self):
        prov = self._make({"error": "bad"}, status=401)
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        with pytest.raises(ProviderError):
            await prov.complete(req)
        await prov.aclose()

    @pytest.mark.asyncio
    async def test_complete_headers_include_auth(self):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        prov = self._make(payload)
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        await prov.complete(req)
        # MockTransport sets self.last on the transport instance.
        transport = prov._client._transport  # type: ignore[attr-defined]
        sent: httpx.Request | None = getattr(transport, "last", None)
        assert sent is not None
        assert sent.headers.get("Authorization") == "Bearer test-key"
        await prov.aclose()

    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://env.example/")
        monkeypatch.setenv("OLLAMA_API_KEY", "envkey")
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        prov = OllamaCloudProvider()
        assert prov.base_url == "https://env.example"  # trailing slash stripped
        assert prov.api_key == "envkey"
        assert prov.default_model == "env-model"

    def test_stream_line_parser(self):
        prov = OllamaCloudProvider()
        assert prov._parse_stream_line("") is None
        assert prov._parse_stream_line("data: [DONE]") is None
        line = 'data: {"choices":[{"delta":{"content":"foo"}}]}'
        assert prov._parse_stream_line(line) == "foo"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestProviderFactory:
    def setup_method(self):
        reset_factory()
        reset_router()

    def teardown_method(self):
        reset_factory()
        reset_router()

    def test_load_missing_config_returns_empty(self, tmp_path):
        cfg = load_config(tmp_path / "does_not_exist.yaml")
        assert cfg == {}

    def test_factory_builds_provider_from_config(self):
        cfg = {
            "default_provider": "ollama",
            "default_model": "llama3",
            "providers": {
                "ollama": {
                    "type": "ollama",
                    "base_url": "https://api.olama.cloud",
                    "api_key": "secret",
                    "default_model": "llama3",
                    "timeout": "120",
                }
            },
            "agents": {
                "CodingAgent": {"provider": "ollama", "model": "qwen2.5-coder"},
            },
        }
        f = ProviderFactory(cfg)
        prov = f.get_provider()
        assert isinstance(prov, OllamaCloudProvider)
        assert prov.default_model == "llama3"
        assert prov.base_url == "https://api.olama.cloud"
        assert prov.api_key == "secret"
        spec = f.get_spec("CodingAgent")
        assert spec.provider_name == "ollama"
        assert spec.model == "qwen2.5-coder"

    def test_factory_env_expansion(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "envsecret")
        cfg = {
            "providers": {"ollama": {"type": "ollama", "api_key": "${OLLAMA_API_KEY}"}},
            "agents": {},
        }
        f = ProviderFactory(cfg)
        prov = f.get_provider()
        assert prov.api_key == "envsecret"

    def test_factory_unknown_provider_type_raises(self):
        cfg = {"providers": {"weird": {"type": "unknown"}}}
        f = ProviderFactory(cfg)
        with pytest.raises(ProviderError):
            f.get_provider("weird")

    def test_factory_agent_spec_falls_back_to_default(self):
        cfg = {"default_provider": "ollama", "default_model": "llama3", "providers": {"ollama": {"type": "ollama"}}}
        f = ProviderFactory(cfg)
        spec = f.get_spec("NotConfigured")
        assert spec.provider_name == "ollama"
        assert spec.model == "llama3"

    def test_register_provider_type(self):
        class Custom(_FakeProvider):
            name = "custom"

        register_provider_type("custom", Custom)
        cfg = {"providers": {"c": {"type": "custom", "default_model": "x"}}}
        f = ProviderFactory(cfg)
        prov = f.get_provider("c")
        assert isinstance(prov, Custom)

    def test_factory_singleton(self):
        f1 = get_factory()
        f2 = get_factory()
        assert f1 is f2


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TestModelRouter:
    def setup_method(self):
        reset_factory()
        reset_router()

    def teardown_method(self):
        reset_factory()
        reset_router()

    def test_router_binds_model(self):
        cfg = {
            "default_provider": "fake",
            "default_model": "base-model",
            "providers": {"fake": {"type": "fake"}},
            "agents": {"CodingAgent": {"provider": "fake", "model": "coder"}},
        }
        register_provider_type("fake", _FakeProvider)
        try:
            f = ProviderFactory(cfg)
            router = ModelRouter(f)
            prov = router.for_agent("CodingAgent")
            assert prov.default_model == "coder"
            assert router.model_for("CodingAgent") == "coder"
            assert router.provider_name_for("CodingAgent") == "fake"
        finally:
            # restore registry (leave _FakeProvider registered; harmless)
            pass

    def test_router_caches_provider(self):
        register_provider_type("fake", _FakeProvider)
        cfg = {
            "default_provider": "fake",
            "default_model": "base-model",
            "providers": {"fake": {"type": "fake"}},
            "agents": {"ResearchAgent": {"provider": "fake", "model": "r"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        a = router.for_agent("ResearchAgent")
        b = router.for_agent("ResearchAgent")
        assert a is b

    @pytest.mark.asyncio
    async def test_router_pinned_model_applied(self):
        register_provider_type("fake", _FakeProvider)
        cfg = {
            "default_provider": "fake",
            "default_model": "base-model",
            "providers": {"fake": {"type": "fake"}},
            "agents": {"CodingAgent": {"provider": "fake", "model": "coder"}},
        }
        f = ProviderFactory(cfg)
        router = ModelRouter(f)
        prov = router.for_agent("CodingAgent")
        req = LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
        resp = await prov.complete(req)
        assert resp.model == "coder"

    def test_get_router_singleton(self):
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------


class TestAgentLLMWiring:
    def setup_method(self):
        reset_factory()
        reset_router()

    def teardown_method(self):
        reset_factory()
        reset_router()

    def test_agents_expose_llm_provider_attribute(self):
        from research_engineer.agents import (
            CodingAgent,
            EvaluationAgent,
            ExperimentAgent,
            ExperimentPlannerAgent,
            LiteratureAgent,
            MemoryAgent,
            ResearchAgent,
            ResearchLoopAgent,
        )

        for cls in (
            ResearchAgent,
            ExperimentPlannerAgent,
            CodingAgent,
            MemoryAgent,
            LiteratureAgent,
            ExperimentAgent,
            EvaluationAgent,
            ResearchLoopAgent,
        ):
            agent = cls()
            assert hasattr(agent, "agent_name")
            assert hasattr(agent, "llm_provider")

    def test_explicit_provider_takes_precedence(self):
        from research_engineer.agents import CodingAgent

        custom = _FakeProvider(default_model="custom-model")
        agent = CodingAgent(llm=custom)
        assert agent.llm_provider is custom

    def test_repository_agent_llm_disabled_by_default(self):
        from research_engineer.agents import RepositoryAgent

        agent = RepositoryAgent()
        assert agent._llm_enabled is False
        # llm_provider None because llm_enabled=False
        assert agent.llm_provider is None
        # legacy attribute still present and None for back-compat
        assert not hasattr(agent, "llm") or agent.llm is None

    def test_repository_agent_llm_enabled_routes_via_provider(self):
        from research_engineer.agents import RepositoryAgent

        custom = _FakeProvider(default_model="custom-model")
        agent = RepositoryAgent(llm_enabled=True, llm=custom)
        assert agent.llm_provider is custom
