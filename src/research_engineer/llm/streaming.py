"""Streaming-first LLM completions.

The provider ``stream()`` methods yield raw ``str`` delta chunks. This module
adds the higher-level pieces that make streaming a first-class, production
path for agents:

- :class:`StreamChunk` — a single streamed delta with optional accumulated
  metadata (model, provider, finish reason, usage) so callers can render
  tokens as they arrive *and* inspect the final response.
- :func:`collect_stream` — consume a provider's ``stream()`` and assemble a
  complete :class:`~research_engineer.llm.base.LLMResponse` (content, model,
  provider, usage, finish_reason) from the chunks.
- :func:`stream_with_retry` — resilient streaming with the same retry/backoff
  semantics as :func:`~research_engineer.llm.resilience.complete_with_retry`,
  plus an optional ``on_complete`` hook for cost/observability stamping.

The router's :class:`~research_engineer.llm.router._BoundProvider` uses these
helpers so that streamed calls get the same cost accounting (D1) and
observability (D2) treatment as non-streamed completions.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
)
from research_engineer.llm.resilience import is_permanent_provider_error

logger = logging.getLogger(__name__)

#: Optional callback invoked with the assembled (final) response before it is
#: returned to the caller. Mirrors the ``on_complete`` hook in
#: :mod:`research_engineer.llm.resilience` so the router can stamp cost and
#: emit observability events for streamed calls too. Best-effort: exceptions
#: are logged and never propagated.
OnStreamComplete = Callable[[LLMRequest, LLMResponse, float], None]


class StreamChunk(BaseModel):
    """A single streamed delta plus optional accumulated response metadata.

    ``text`` is the raw delta for this chunk. The remaining fields are
    populated only when the provider surfaces them (typically on the final
    chunk or via a trailing metadata event); callers that only need to render
    tokens can ignore them.
    """

    text: str = Field(..., description="Delta content for this chunk")
    model: str | None = Field(default=None, description="Model id (when known)")
    provider: str | None = Field(default=None, description="Provider name (when known)")
    finish_reason: str | None = Field(
        default=None, description="Why generation stopped (when known)"
    )
    usage: LLMUsage | None = Field(
        default=None, description="Token usage (when surfaced by the provider)"
    )

async def collect_stream(
    provider: LLMProvider,
    request: LLMRequest,
) -> LLMResponse:
    """Consume ``provider.stream(request)`` and assemble a full response.

    Iterates the provider's ``stream()`` async generator, concatenating the
    ``str`` deltas into ``content`` and filling in ``model``, ``provider``,
    ``usage``, and ``finish_reason`` from the request/provider defaults when
    the stream does not carry them.

    Raises :class:`ProviderError` when the provider does not support
    streaming (its ``stream()`` raises ``NotImplementedError``).
    """
    model = request.model or provider.default_model
    parts: list[str] = []
    stream = provider.stream(request)
    # The base ``LLMProvider.stream`` is a plain coroutine that raises
    # ``NotImplementedError``; concrete providers override it as an async
    # generator. Await the coroutine case so unsupported providers surface a
    # clear error instead of a confusing ``TypeError``.
    try:
        if inspect.isawaitable(stream):
            await stream  # raises NotImplementedError for unsupported providers
            return LLMResponse(
                content="",
                model=model,
                provider=provider.name,
                usage=LLMUsage(),
                finish_reason=None,
            )
        async for chunk in stream:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, StreamChunk):
                parts.append(chunk.text)
                if chunk.model:
                    model = chunk.model
            else:
                parts.append(str(chunk))
    except NotImplementedError as exc:
        raise ProviderError(
            f"{provider.name} does not support streaming",
            provider=provider.name,
            cause=exc,
        ) from exc

    return LLMResponse(
        content="".join(parts),
        model=model,
        provider=provider.name,
        usage=LLMUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
        finish_reason=None,
    )


async def stream_with_retry(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    agent_name: str | None = None,
    on_complete: OnStreamComplete | None = None,
) -> LLMResponse:
    """Stream a completion with retry/backoff, returning the assembled response.

    Mirrors :func:`~research_engineer.llm.resilience.complete_with_retry`:
    transient transport/API errors are retried with exponential backoff,
    permanent errors fail fast, and the final response is passed through the
    optional ``on_complete`` hook (best-effort) before being returned.

    Raises :class:`ProviderError` when all attempts are exhausted.
    """
    import time

    start = time.monotonic()
    last_error: BaseException | None = None
    label = agent_name or type(provider).__name__

    for attempt in range(1, max_attempts + 1):
        try:
            response = await collect_stream(provider, request)
        except Exception as e:  # noqa: BLE001 - deliberate broad retry gate
            if is_permanent_provider_error(e):
                raise
            last_error = e
            logger.warning(
                "LLM stream attempt %d/%d failed for %s (%s: %s)",
                attempt,
                max_attempts,
                label,
                type(e).__name__,
                e,
            )
        else:
            _invoke_on_complete(
                on_complete, request, response, time.monotonic() - start, label
            )
            return response

        if attempt < max_attempts:
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    raise ProviderError(
        f"LLM streaming failed after {max_attempts} attempts for {label}: {last_error}",
        cause=last_error if isinstance(last_error, Exception) else None,
    )


def _invoke_on_complete(
    on_complete: OnStreamComplete | None,
    request: LLMRequest,
    response: LLMResponse,
    latency_seconds: float,
    agent_name: str,
) -> None:
    """Best-effort invocation of the ``on_complete`` callback."""
    if on_complete is None:
        return
    try:
        on_complete(request, response, latency_seconds)
    except Exception:  # noqa: BLE001 - observability hooks must not break callers
        logger.warning(
            "on_complete callback failed for %s",
            agent_name or "unknown",
            exc_info=True,
        )


__all__ = [
    "StreamChunk",
    "collect_stream",
    "stream_with_retry",
    "OnStreamComplete",
]

