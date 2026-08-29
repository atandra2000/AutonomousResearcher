"""LangChain-backed adapter for the platform LLM provider contract.

The adapter lets existing agents keep using :class:`LLMProvider` while
LangChain supplies model invocation, tool-schema binding, and LangSmith-native
tracing.  Tool execution deliberately remains outside this module so every
call still crosses the platform's gateway.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMUsage,
    ProviderError,
    ToolCall,
    ToolDefinition,
)
from research_engineer.observability.context import get_correlation


class LangChainChatProvider(LLMProvider):
    """Adapt a LangChain chat model to the platform's typed provider API.

    The default model is :class:`~langchain_openai.ChatOpenAI`, which also
    supports the project's existing OpenAI-compatible gateways through
    ``base_url``.  Tests and alternative integrations can inject any object
    implementing ``ainvoke``, ``astream``, and ``bind_tools``.
    """

    name = "langchain"

    def __init__(
        self,
        *,
        chat_model: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.default_model = (
            default_model
            or model
            or os.environ.get("LANGCHAIN_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o"
        )
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._chat_model = chat_model or ChatOpenAI(
            model=self.default_model,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one LangChain chat invocation and restore the local contract."""
        self._assert_valid(request)
        message = await self._invoke(self._chat_model, request)
        return self._to_response(message, request)

    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        """Bind schemas for model selection; execution stays gateway-owned."""
        self._assert_valid(request)
        try:
            model = self._chat_model.bind_tools([tool.to_openai() for tool in tools])
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise ProviderError(
                f"LangChain tool binding failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc
        message = await self._invoke(model, request)
        return self._to_response(message, request)

    async def stream(self, request: LLMRequest) -> Any:
        """Yield non-empty text deltas from the LangChain streaming API."""
        self._assert_valid(request)
        if not request.stream:
            yield (await self.complete(request)).content
            return
        try:
            async for chunk in self._chat_model.astream(
                self._to_messages(request.messages),
                config=self._runnable_config(),
                stop=request.stop,
                **self._invoke_options(request),
            ):
                content = _content_text(getattr(chunk, "content", ""))
                if content:
                    yield content
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise ProviderError(
                f"LangChain stream failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

    async def _invoke(self, model: Any, request: LLMRequest) -> Any:
        try:
            return await model.ainvoke(
                self._to_messages(request.messages),
                config=self._runnable_config(),
                stop=request.stop,
                **self._invoke_options(request),
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise ProviderError(
                f"LangChain completion failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

    @staticmethod
    def _invoke_options(request: LLMRequest) -> dict[str, Any]:
        options = dict(request.extra)
        options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["max_completion_tokens"] = request.max_tokens
        if request.top_p != 1.0:
            options["top_p"] = request.top_p
        return options

    def _runnable_config(self) -> RunnableConfig:
        """Attach correlation-only metadata for optional LangSmith tracing."""
        metadata: dict[str, str] = {"provider": self.name}
        context = get_correlation()
        if context is not None:
            for key in ("run_id", "execution_id", "trace_id", "span_id"):
                value = getattr(context, key)
                if value:
                    metadata[key] = value
        return {"tags": ["research-engineer"], "metadata": metadata}

    def _assert_valid(self, request: LLMRequest) -> None:
        if not request.messages or any(
            not message.content and not message.tool_calls
            for message in request.messages
        ):
            raise ProviderError(
                "Invalid request: messages must include content or tool calls",
                provider=self.name,
            )

    @staticmethod
    def _to_messages(messages: list[LLMMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for message in messages:
            if message.role == LLMRole.SYSTEM:
                out.append(SystemMessage(content=message.content, name=message.name))
            elif message.role == LLMRole.USER:
                out.append(HumanMessage(content=message.content, name=message.name))
            elif message.role == LLMRole.ASSISTANT:
                tool_calls = [
                    {
                        "id": call.id,
                        "name": call.name,
                        "args": call.arguments,
                        "type": "tool_call",
                    }
                    for call in message.tool_calls or []
                ]
                out.append(
                    AIMessage(
                        content=message.content,
                        name=message.name,
                        tool_calls=tool_calls,
                    )
                )
            elif message.tool_call_id:
                out.append(
                    ToolMessage(
                        content=message.content,
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                )
            else:
                raise ProviderError(
                    "Tool messages require a tool_call_id",
                    provider=LangChainChatProvider.name,
                )
        return out

    def _to_response(self, message: Any, request: LLMRequest) -> LLMResponse:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage_data = getattr(message, "usage_metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(usage_data, dict):
            usage_data = {}
        model = str(
            metadata.get("model_name")
            or metadata.get("model")
            or request.model
            or self.default_model
        )
        return LLMResponse(
            content=_content_text(getattr(message, "content", "")),
            model=model,
            provider=self.name,
            usage=LLMUsage(
                prompt_tokens=_token_count(usage_data, "input_tokens", "prompt_tokens"),
                completion_tokens=_token_count(
                    usage_data, "output_tokens", "completion_tokens"
                ),
                total_tokens=_token_count(usage_data, "total_tokens"),
            ),
            finish_reason=_string_or_none(metadata.get("finish_reason")),
            tool_calls=_tool_calls(getattr(message, "tool_calls", [])),
            raw=metadata,
        )


def _content_text(content: Any) -> str:
    """Return text for LangChain's string or content-block message payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    if content is None:
        return ""
    return json.dumps(content, default=str)


def _token_count(data: dict[str, Any], *keys: str) -> int:
    """Return the first non-negative integer token count in ``data``."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _tool_calls(calls: Any) -> list[ToolCall] | None:
    """Convert LangChain tool-call dictionaries to local typed calls."""
    if not isinstance(calls, list):
        return None
    out: list[ToolCall] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        call_id = call.get("id")
        name = call.get("name")
        arguments = call.get("args", {})
        if isinstance(call_id, str) and isinstance(name, str) and isinstance(arguments, dict):
            out.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return out or None


__all__ = ["LangChainChatProvider"]
