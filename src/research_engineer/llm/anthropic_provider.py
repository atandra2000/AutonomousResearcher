"""Anthropic LLM provider.

Talks to the native Anthropic Messages API (``POST {base_url}/v1/messages``)
over httpx — any Messages-API-compatible endpoint. Connection settings are
sourced from constructor arguments with environment-variable fallbacks:

================ ==================== ==============================
Setting          Env var              Default
================ ==================== ==============================
``base_url``      ``ANTHROPIC_BASE_URL``      ``https://api.anthropic.com``
``api_key``       ``ANTHROPIC_API_KEY``       (none)
``default_model`` ``ANTHROPIC_MODEL`` / ``ANTHROPIC_DEFAULT_MODEL``  (none — set explicitly)
``timeout``       ``ANTHROPIC_TIMEOUT``       ``60``
================ ==================== ==============================

The provider never imports the ``anthropic`` SDK: it speaks the native
Messages API JSON directly, keeping the dependency surface minimal and
making it trivially testable with a mock httpx transport.

The Messages API differs from the OpenAI wire format in a few ways that
this provider normalises:

- ``system`` is a top-level field rather than a ``system``-role message.
- ``max_tokens`` is required by the API; a sensible default is applied
  when the request omits it.
- Tool definitions use ``input_schema`` instead of ``parameters``.
- Tool calls are returned as ``content`` blocks with ``type ==
  \"tool_use\"`` rather than a ``tool_calls`` array.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ToolCall,
    ToolDefinition,
)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the native Anthropic Messages API."""

    name = "anthropic"

    DEFAULT_BASE_URL = "https://api.anthropic.com"

    #: Anthropic requires an explicit ``max_tokens``; used when the request
    #: does not specify one. Thinking counts toward this ceiling on the current
    #: Claude generation even when its text is not returned, so a thinking-off
    #: sized default truncates replies mid-thought. Callers that need a
    #: deliberately short output set their own cap.
    DEFAULT_MAX_TOKENS = 16000

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("ANTHROPIC_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.default_model = (
            default_model
            or os.environ.get("ANTHROPIC_MODEL")
            or os.environ.get("ANTHROPIC_DEFAULT_MODEL")
            or ""
        )
        timeout_s = timeout or float(os.environ.get("ANTHROPIC_TIMEOUT", "60"))
        self._timeout = timeout_s
        # Allow callers (e.g. tests) to inject a mock client.
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a chat completion via the Anthropic Messages API."""
        if not await self.validate(request):
            raise ProviderError(
                "Invalid request: messages must be non-empty with content",
                provider=self.name,
            )
        model = request.model or self.default_model
        payload = self._build_payload(request, model)

        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Anthropic request failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Anthropic returned HTTP {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Anthropic returned non-JSON body: {resp.text}",
                provider=self.name,
                cause=exc,
            ) from exc

        return self._parse_response(data, model)

    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        """Generate a completion that may request tool calls.

        Sends the native ``tools`` array (with ``input_schema``) and parses
        any ``tool_use`` content blocks into :class:`ToolCall` objects on the
        returned :class:`LLMResponse`.
        """
        if not await self.validate(request):
            raise ProviderError(
                "Invalid request: messages must be non-empty with content",
                provider=self.name,
            )
        model = request.model or self.default_model
        payload = self._build_payload(request, model)
        payload["tools"] = [self._tool_to_anthropic(t) for t in tools]

        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Anthropic request failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Anthropic returned HTTP {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Anthropic returned non-JSON body: {resp.text}",
                provider=self.name,
                cause=exc,
            ) from exc

        response = self._parse_response(data, model)
        response.tool_calls = self._parse_tool_calls(data)
        return response

    async def stream(self, request: LLMRequest) -> Any:
        """Stream completion chunks from the Anthropic Messages API.

        Yields ``str`` delta-content chunks parsed from the native SSE event
        stream (``content_block_delta`` events with ``text_delta`` blocks).
        Falls back to a single non-streamed call when the server does not
        support streaming.
        """
        if request.stream is False:
            response = await self.complete(request.model_copy(update={"stream": False}))
            yield response.content
            return

        model = request.model or self.default_model
        payload = self._build_payload(request, model)
        payload["stream"] = True
        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"Anthropic returned HTTP {resp.status_code}: {body.decode(errors='replace')}",
                        provider=self.name,
                    )
                async for line in resp.aiter_lines():
                    text = self._parse_stream_event(line)
                    if text is not None:
                        yield text
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Anthropic stream failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

    async def health(self) -> bool:
        """Probe liveness by pinging the Messages API.

        Sends a minimal ``GET /v1/models`` request; returns ``True`` when the
        endpoint responds with HTTP 2xx. Any transport error or non-2xx
        status is treated as unhealthy.
        """
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=self._timeout,
            )
            return int(resp.status_code) < 400
        except httpx.HTTPError:
            return False

    @property
    def models(self) -> list[str]:
        """Best-effort list of advertised models (cached per instance)."""
        cached: list[str] | None = getattr(self, "_models_cache", None)
        if cached is not None:
            return cached
        import asyncio

        async def _fetch() -> list[str]:
            client = await self._get_client()
            try:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=self._timeout,
                )
                if resp.status_code >= 400:
                    return [self.default_model]
                data = resp.json()
                ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
                return ids or [self.default_model]
            except Exception:
                return [self.default_model]

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return [self.default_model]
            result: list[str] = loop.run_until_complete(_fetch())
        except RuntimeError:
            result = [self.default_model]
        self._models_cache = result
        return result

    async def aclose(self) -> None:
        """Release the underlying httpx client if we own it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _build_payload(self, request: LLMRequest, model: str) -> dict[str, Any]:
        """Build a native Messages API payload from a provider-agnostic request.

        The ``system`` role is hoisted to the top-level ``system`` field and
        ``max_tokens`` is defaulted (the API requires it).
        """
        system_parts, messages = self._convert_messages(request.messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        # Sampling parameters are removed on the current Claude generation and
        # return a 400, so request.temperature is not forwarded here. Pass
        # sampling options through request.extra when a target model still
        # takes them.
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if request.stop:
            payload["stop_sequences"] = request.stop
        # Pass through provider-specific extras untouched.
        if request.extra:
            payload.update(request.extra)
        return payload

    def _convert_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Convert provider-agnostic messages to the native Messages format.

        Returns ``(system_parts, messages)`` where ``system_parts`` are the
        hoisted system prompts and ``messages`` are the native message dicts.
        """
        system_parts: list[str] = []
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                if m.content:
                    system_parts.append(m.content)
                continue
            if m.role == "tool":
                # Tool results are fed back as a ``user`` message with a
                # ``tool_result`` content block.
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id or "",
                                "content": m.content,
                            }
                        ],
                    }
                )
                continue
            if m.role == "assistant" and m.tool_calls:
                # Assistant message carrying tool_use blocks.
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": content})
                continue
            out.append({"role": m.role.value, "content": m.content})
        return system_parts, out

    def _tool_to_anthropic(self, tool: ToolDefinition) -> dict[str, Any]:
        """Convert a :class:`ToolDefinition` to the native ``tools`` entry."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _parse_response(self, data: dict[str, Any], fallback_model: str) -> LLMResponse:
        content = ""
        for block in data.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                content += block.get("text") or ""
        usage_raw = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_raw.get("input_tokens", 0)),
            completion_tokens=int(usage_raw.get("output_tokens", 0)),
            total_tokens=int(usage_raw.get("input_tokens", 0))
            + int(usage_raw.get("output_tokens", 0)),
        )
        return LLMResponse(
            content=content,
            model=data.get("model") or fallback_model,
            provider=self.name,
            usage=usage,
            finish_reason=data.get("stop_reason"),
            raw=data,
        )

    def _parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract ``tool_use`` content blocks into :class:`ToolCall` objects.

        Returns ``None`` when the model made no tool calls.
        """
        calls: list[ToolCall] = []
        for block in data.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name") or ""
            if not name:
                continue
            calls.append(
                ToolCall(
                    id=block.get("id") or f"call_{len(calls)}",
                    name=name,
                    arguments=block.get("input") or {},
                )
            )
        return calls or None

    def _parse_stream_event(self, line: str) -> str | None:
        """Extract a text delta from a single Anthropic SSE stream line.

        The Messages API streams ``event:``/``data:`` pairs. Only
        ``content_block_delta`` events whose ``delta.type == 'text_delta'``
        carry visible text; all other events (``message_start``,
        ``content_block_start``, ``content_block_stop``, ``message_delta``,
        ``message_stop``, ping) yield ``None``.
        """
        if not line:
            return None
        if line.startswith("event:"):
            return None
        if not line.startswith("data:"):
            return None
        data = line[len("data:") :].strip()
        if not data:
            return None
        import json

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        if obj.get("type") != "content_block_delta":
            return None
        delta = obj.get("delta") or {}
        if not isinstance(delta, dict) or delta.get("type") != "text_delta":
            return None
        return delta.get("text")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client
