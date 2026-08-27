"""OpenAI-compatible LLM provider.

Talks to any OpenAI-compatible Chat Completions endpoint
(``POST {base_url}/v1/chat/completions``) over httpx — the official OpenAI
API (GPT-4o, o1, etc.) or any self-hosted gateway that mirrors the OpenAI
wire format. Connection settings are sourced from constructor arguments
with environment-variable fallbacks:

================ ==================== ==============================
Setting          Env var              Default
================ ==================== ==============================
``base_url``      ``OPENAI_BASE_URL``      ``https://api.openai.com``
``api_key``       ``OPENAI_API_KEY``       (none)
``default_model`` ``OPENAI_MODEL`` / ``OPENAI_DEFAULT_MODEL``  ``\"gpt-4o\"``
``timeout``       ``OPENAI_TIMEOUT``       ``60``
================ ==================== ==============================

The provider never imports the ``openai`` SDK: it speaks plain
OpenAI-style JSON, keeping the dependency surface minimal and making it
trivially testable with a mock httpx transport.
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


class OpenAIProvider(LLMProvider):
    """LLM provider backed by an OpenAI-compatible Chat Completions API."""

    name = "openai"

    DEFAULT_BASE_URL = "https://api.openai.com"

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
            or os.environ.get("OPENAI_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.default_model = (
            default_model
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENAI_DEFAULT_MODEL")
            or "gpt-4o"
        )
        timeout_s = timeout or float(os.environ.get("OPENAI_TIMEOUT", "60"))
        self._timeout = timeout_s
        # Allow callers (e.g. tests) to inject a mock client.
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a chat completion via the OpenAI-compatible endpoint."""
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
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI request failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI returned HTTP {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"OpenAI returned non-JSON body: {resp.text}",
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

        Sends the OpenAI-compatible ``tools`` array and parses any
        ``tool_calls`` the model emits into :class:`ToolCall` objects on the
        returned :class:`LLMResponse`.
        """
        if not await self.validate(request):
            raise ProviderError(
                "Invalid request: messages must be non-empty with content",
                provider=self.name,
            )
        model = request.model or self.default_model
        payload = self._build_payload(request, model)
        payload["tools"] = [t.to_openai() for t in tools]

        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI request failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI returned HTTP {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"OpenAI returned non-JSON body: {resp.text}",
                provider=self.name,
                cause=exc,
            ) from exc

        response = self._parse_response(data, model)
        response.tool_calls = self._parse_tool_calls(data)
        return response


    async def stream(self, request: LLMRequest) -> Any:
        """Stream completion chunks from the OpenAI-compatible endpoint.

        Yields ``str`` delta-content chunks. Falls back to a single
        non-streamed call when the server does not support streaming.
        """
        if request.stream is False:
            response = await self.complete(request.model_copy(update={"stream": False}))
            yield response.content
            return

        model = request.model or self.default_model
        payload = self._build_payload(request, model, stream=True)
        client = await self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"OpenAI returned HTTP {resp.status_code}: {body.decode(errors='replace')}",
                        provider=self.name,
                    )
                async for line in resp.aiter_lines():
                    chunk = self._parse_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"OpenAI stream failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

    async def health(self) -> bool:
        """Probe liveness by listing models via ``GET /v1/models``.

        Returns ``True`` when the endpoint responds with HTTP 2xx; any
        transport error or non-2xx status is treated as unhealthy.
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
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, request: LLMRequest, model: str, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._wire_message(m) for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        payload["stream"] = stream or request.stream
        # Pass through provider-specific extras untouched.
        if request.extra:
            payload.update(request.extra)
        return payload

    @staticmethod
    def _wire_message(message: LLMMessage) -> dict[str, Any]:
        """Serialize one message into OpenAI wire shape.

        ``tool_calls`` on assistant messages must become the OpenAI
        ``{"type": "function", "function": {"name", "arguments"}}`` form
        with JSON-string arguments; sending pydantic's dict form back in
        multi-turn tool conversations is rejected as invalid by the API.
        """
        wire: dict[str, Any] = {
            "role": str(message.role.value),
            "content": message.content,
        }
        if message.name:
            wire["name"] = message.name
        if message.tool_call_id:
            wire["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            import json

            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return wire

    def _parse_response(self, data: dict[str, Any], fallback_model: str) -> LLMResponse:
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        content = message.get("content") or ""
        usage_raw = data.get("usage") or {}
        usage = LLMUsage(
            prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
            completion_tokens=int(usage_raw.get("completion_tokens", 0)),
            total_tokens=int(usage_raw.get("total_tokens", 0)),
        )
        return LLMResponse(
            content=content,
            model=data.get("model") or fallback_model,
            provider=self.name,
            usage=usage,
            finish_reason=first.get("finish_reason"),
            raw=data,
        )

    def _parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract tool calls from an OpenAI-style chat-completion payload.

        Returns ``None`` when the model made no tool calls. Each call's
        ``arguments`` is parsed from its JSON string; unparseable arguments
        are preserved as an empty dict rather than raising.
        """
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        raw_calls = message.get("tool_calls") or []
        if not raw_calls:
            return None

        import json

        calls: list[ToolCall] = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            fn = raw.get("function") or {}
            name = fn.get("name") or ""
            if not name:
                continue
            arguments: dict[str, Any] = {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        arguments = parsed
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{len(calls)}",
                    name=name,
                    arguments=arguments,
                )
            )
        return calls or None

    def _parse_stream_line(self, line: str) -> str | None:
        if not line:
            return None
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            return None
        if not line:
            return None
        import json

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client
