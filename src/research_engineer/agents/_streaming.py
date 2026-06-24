"""Streaming helper for interactive agents.

Wraps :meth:`research_engineer.llm.base.LLMProvider.stream` so agents
can stream LLM tokens to stdout (or a caller-supplied sink) during
interactive phases. Non-streaming callers are unaffected: the helper is
purely additive and only invoked when ``stream=True``.
"""

from __future__ import annotations

import sys
from typing import TextIO

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
)


async def stream_complete(
    provider: LLMProvider | None,
    request: LLMRequest,
    *,
    sink: TextIO | None = None,
) -> LLMResponse | None:
    """Stream a completion to ``sink`` and return a best-effort response.

    Args:
        provider: LLM provider to call. If ``None`` or streaming is
            unsupported, returns ``None`` so the caller can fall back to
            a non-streaming :meth:`complete` call.
        request: Completion request. ``request.stream`` is forced to True.
        sink: Where to write streamed tokens (defaults to ``sys.stdout``).

    Returns:
        An :class:`LLMResponse` with concatenated content if streaming
        succeeded, or ``None`` if the provider does not support streaming
        or raised an error (caller should fall back to ``complete``).
    """
    if provider is None:
        return None
    out = sink or sys.stdout
    request = request.model_copy(update={"stream": True})
    parts: list[str] = []
    model = request.model or getattr(provider, "default_model", "stream")
    try:
        async for chunk in provider.stream(request):  # type: ignore[misc]
            if chunk:
                out.write(chunk)
                out.flush()
                parts.append(chunk)
    except (NotImplementedError, ProviderError, Exception):
        # Provider does not support streaming or failed mid-stream.
        if not parts:
            return None
        # Return partial content if we got anything before the error.
    content = "".join(parts)
    if not content:
        return None
    return LLMResponse(
        content=content,
        model=model,
        provider=getattr(provider, "name", "stream"),
    )


__all__ = ["stream_complete"]
