"""Ollama Cloud LLM provider.

Talks to Ollama Cloud's OpenAI-compatible Chat Completions endpoint
(``POST {base_url}/v1/chat/completions``) over httpx. Connection settings
are sourced from constructor arguments with environment-variable fallbacks:

================ ============ =========================================
Setting          Env var      Default
================ ============ =========================================
``base_url``      ``OLLAMA_BASE_URL``      ``https://api.olama.cloud``
``api_key``       ``OLLAMA_API_KEY``       (none)
``default_model`` ``OLLAMA_MODEL`` / ``OLLAMA_DEFAULT_MODEL``  ``"llama3"``
``timeout``       ``OLLAMA_TIMEOUT``       ``60``
================ ============ =========================================

The provider never imports ``llama_index`` or any vendor SDK: it speaks
plain OpenAI-style JSON, keeping the dependency surface minimal.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
)


class OllamaCloudProvider(LLMProvider):
    """LLM provider backed by Ollama Cloud's OpenAI-compatible API."""

    name = "ollama"

    DEFAULT_BASE_URL = "https://api.olama.cloud"

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
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY") or ""
        self.default_model = (
            default_model
            or os.environ.get("OLLAMA_MODEL")
            or os.environ.get("OLLAMA_DEFAULT_MODEL")
            or "llama3"
        )
        timeout_s = timeout or float(os.environ.get("OLLAMA_TIMEOUT", "60"))
        self._timeout = timeout_s
        # Allow callers (e.g. tests) to inject a mock client.
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a chat completion via Ollama Cloud."""
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
                f"Ollama Cloud request failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama Cloud returned HTTP {resp.status_code}: {resp.text}",
                provider=self.name,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Ollama Cloud returned non-JSON body: {resp.text}",
                provider=self.name,
                cause=exc,
            ) from exc

        return self._parse_response(data, model)

    async def stream(self, request: LLMRequest) -> Any:
        """Stream completion chunks from Ollama Cloud.

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
                        f"Ollama Cloud returned HTTP {resp.status_code}: {body.decode(errors='replace')}",
                        provider=self.name,
                    )
                async for line in resp.aiter_lines():
                    chunk = self._parse_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Ollama Cloud stream failed: {exc}",
                provider=self.name,
                cause=exc,
            ) from exc

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

    def _build_payload(self, request: LLMRequest, model: str, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
