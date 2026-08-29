"""Tests for the LangChain-backed provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from research_engineer.llm import (
    LangChainChatProvider,
    LLMMessage,
    LLMRequest,
    LLMRole,
    ProviderFactory,
    ToolDefinition,
)
from research_engineer.observability.context import (
    CorrelationContext,
    reset_correlation,
    set_correlation,
)


class _FakeChatModel:
    """Network-free ChatModel substitute that records LangChain calls."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.messages: list[Any] = []
        self.kwargs: dict[str, Any] = {}
        self.bound_tools: list[dict[str, Any]] | None = None

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.messages = messages
        self.kwargs = {"stop": stop, **kwargs}
        return self.response

    def bind_tools(self, tools: list[dict[str, Any]]) -> _FakeChatModel:
        self.bound_tools = tools
        return self

    async def astream(
        self,
        messages: list[Any],
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        self.messages = messages
        self.kwargs = {"stop": stop, **kwargs}
        yield AIMessageChunk(content="first ")
        yield AIMessageChunk(content="second")


def _request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(role=LLMRole.SYSTEM, content="Use the repository evidence."),
            LLMMessage(role=LLMRole.USER, content="Summarize the worker."),
        ],
        model="test-model",
        temperature=0.4,
        max_tokens=64,
        top_p=0.8,
        stop=["END"],
    )


@pytest.mark.asyncio
async def test_complete_maps_platform_contract_to_langchain() -> None:
    model = _FakeChatModel(
        AIMessage(
            content="The worker owns execution.",
            response_metadata={"model_name": "served-model", "finish_reason": "stop"},
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )
    )
    provider = LangChainChatProvider(chat_model=model, default_model="fallback")

    token = set_correlation(CorrelationContext(run_id="run_1", trace_id="trace_1"))
    try:
        response = await provider.complete(_request())
    finally:
        reset_correlation(token)

    assert response.content == "The worker owns execution."
    assert response.model == "served-model"
    assert response.provider == "langchain"
    assert response.usage.total_tokens == 18
    assert response.finish_reason == "stop"
    assert [message.type for message in model.messages] == ["system", "human"]
    assert model.kwargs["temperature"] == 0.4
    assert model.kwargs["max_completion_tokens"] == 64
    assert model.kwargs["top_p"] == 0.8
    assert model.kwargs["stop"] == ["END"]
    assert model.kwargs["config"]["tags"] == ["research-engineer"]
    assert model.kwargs["config"]["metadata"] == {
        "provider": "langchain",
        "run_id": "run_1",
        "trace_id": "trace_1",
    }


@pytest.mark.asyncio
async def test_complete_with_tools_returns_requested_tool_calls() -> None:
    model = _FakeChatModel(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "search_papers", "args": {"topic": "agents"}}
            ],
        )
    )
    provider = LangChainChatProvider(chat_model=model)

    response = await provider.complete_with_tools(
        _request(),
        [
            ToolDefinition(
                name="search_papers",
                description="Search approved paper sources.",
                parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
            )
        ],
    )

    assert response.tool_calls is not None
    assert response.tool_calls[0].model_dump() == {
        "id": "call_1",
        "name": "search_papers",
        "arguments": {"topic": "agents"},
    }
    assert model.bound_tools is not None
    assert model.bound_tools[0]["function"]["name"] == "search_papers"


@pytest.mark.asyncio
async def test_stream_yields_langchain_content_chunks() -> None:
    model = _FakeChatModel(AIMessage(content="unused"))
    provider = LangChainChatProvider(chat_model=model)

    request = _request().model_copy(update={"stream": True})
    chunks = [chunk async for chunk in provider.stream(request)]

    assert chunks == ["first ", "second"]


def test_factory_builds_the_langchain_provider_from_yaml_style_config() -> None:
    factory = ProviderFactory(
        {
            "default_provider": "framework",
            "providers": {
                "framework": {
                    "type": "langchain",
                    "default_model": "test-model",
                    "api_key": "not-a-real-key",
                }
            },
        }
    )

    provider = factory.get_provider()

    assert isinstance(provider, LangChainChatProvider)
    assert provider.default_model == "test-model"
