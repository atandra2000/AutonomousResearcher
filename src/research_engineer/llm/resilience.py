"""Resilient LLM completions: retry with backoff plus an empty-content guard.

Production failure modes covered:
- Transient transport/API errors (network, timeouts, HTTP 429/5xx) are
  retried with exponential backoff.
- Permanent API errors (HTTP 4xx other than 429, e.g. retired models)
  fail fast — retrying cannot succeed.
- Empty responses caused by token-budget exhaustion (``finish_reason ==
  "length"``) are retried once with a doubled ``max_tokens`` budget;
  reasoning-style models routinely spend the entire budget before
  emitting visible content.
- Non-empty responses that are still truncated (``finish_reason ==
  "length"``) are also retried once with a doubled ``max_tokens`` budget,
  so answers cut off mid-generation get a chance to complete.
- Exhausting all attempts raises :class:`ProviderError` so callers'
  existing fallback paths engage instead of silently degrading.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

#: Hard ceiling for max_tokens escalation on truncated responses.
_MAX_TOKENS_CAP = 16_384

_HTTP_CODE_RE = re.compile(r"HTTP (\d{3})")

#: Optional callback invoked with the final (non-retried) response before it
#: is returned to the caller. Receives the original request (for prompt
#: hashing), the response, and the measured wall-clock latency in seconds.
#: Used by the router to stamp cost + record usage (D1) and to emit a
#: structured observability event (D2). Best-effort: exceptions are logged
#: and never propagated so hooks cannot break the completion path.
OnComplete = Callable[[LLMRequest, LLMResponse, float], None]


def is_permanent_provider_error(error: BaseException) -> bool:
    """True when retrying this error cannot possibly succeed."""
    if isinstance(error, ProviderError):
        match = _HTTP_CODE_RE.search(str(error))
        if match:
            code = int(match.group(1))
            return not (code == 429 or code >= 500)
    return False


def _handle_truncation(
    response: LLMResponse,
    request: LLMRequest,
    *,
    escalated_tokens: bool,
    agent_name: str,
    attempt: int,
    max_attempts: int,
) -> tuple[LLMRequest, bool]:
    """Handle a truncated or empty response, escalating the budget once.

    Truncated responses (``finish_reason == 'length'``, empty or non-empty)
    trigger a one-time ``max_tokens`` escalation. Non-truncated empty
    responses are logged but never escalated.

    Returns the (possibly updated) request and the new ``escalated_tokens``
    flag.
    """
    if response.truncated and not escalated_tokens and request.max_tokens:
        doubled = min(request.max_tokens * 2, _MAX_TOKENS_CAP)
        if doubled > request.max_tokens:
            logger.warning(
                "LLM returned truncated content (finish_reason='length', "
                "content_len=%d) for %s; retrying with max_tokens=%d",
                len(response.content),
                agent_name,
                doubled,
            )
            return request.model_copy(update={"max_tokens": doubled}), True
        logger.warning(
            "LLM returned truncated content for %s but max_tokens is "
            "already at the cap (%d); not escalating further",
            agent_name,
            _MAX_TOKENS_CAP,
        )
    elif response.truncated:
        logger.warning(
            "LLM returned truncated content for %s (attempt %d/%d, "
            "content_len=%d); escalation already applied",
            agent_name,
            attempt,
            max_attempts,
            len(response.content),
        )
    else:
        logger.warning(
            "LLM returned empty content for %s (attempt %d/%d, "
            "finish_reason=%s)",
            agent_name,
            attempt,
            max_attempts,
            response.finish_reason,
        )
    return request, escalated_tokens


async def complete_with_retry(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    agent_name: str = "",
    tools: list[ToolDefinition] | None = None,
    on_complete: OnComplete | None = None,
) -> LLMResponse:
    """Call ``provider.complete`` with retries and truncation handling.

    When ``tools`` is provided, the call is routed through
    ``provider.complete_with_tools`` so the model may request tool calls.

    Responses whose ``finish_reason == 'length'`` (whether empty or
    non-empty) are retried once with a doubled ``max_tokens`` budget, up
    to ``_MAX_TOKENS_CAP``. Non-truncated, non-empty responses are
    returned immediately.

    When ``on_complete`` is provided it is invoked exactly once with the
    final response (the one returned to the caller) before that response
    is returned. The callback is best-effort: exceptions it raises are
    logged and never propagated so observability/cost hooks cannot break
    the completion path.

    Raises the last observed error when all attempts are exhausted.
    """
    import time

    start = time.monotonic()
    last_error: BaseException | None = None
    last_response: LLMResponse | None = None
    escalated_tokens = False

    for attempt in range(1, max_attempts + 1):
        try:
            response = await _call_once(provider, request, tools)
        except Exception as e:  # noqa: BLE001 - deliberate broad retry gate
            if is_permanent_provider_error(e):
                raise
            last_error = e
            logger.warning(
                "LLM attempt %d/%d failed for %s (%s: %s)",
                attempt,
                max_attempts,
                agent_name or type(provider).__name__,
                type(e).__name__,
                e,
            )
        else:
            # Normal completion: non-empty content and not truncated.
            if response.content.strip() and not response.truncated:
                _invoke_on_complete(
                    on_complete, request, response, time.monotonic() - start, agent_name
                )
                return response
            last_response = response
            # Truncated (empty or non-empty) or empty: escalate the budget
            # once, then keep retrying until max_attempts is exhausted.
            request, escalated_tokens = _handle_truncation(
                response,
                request,
                escalated_tokens=escalated_tokens,
                agent_name=agent_name or type(provider).__name__,
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if attempt < max_attempts:
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    if last_response is not None:
        # Return the final (empty) response rather than fabricating one;
        # callers already handle empty content defensively.
        _invoke_on_complete(
            on_complete, request, last_response, time.monotonic() - start, agent_name
        )
        return last_response
    raise ProviderError(
        f"LLM completion failed after {max_attempts} attempts for "
        f"{agent_name or type(provider).__name__}: {last_error}",
        cause=last_error if isinstance(last_error, Exception) else None,
    )


def _invoke_on_complete(
    on_complete: OnComplete | None,
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


async def _call_once(
    provider: LLMProvider,
    request: LLMRequest,
    tools: list[ToolDefinition] | None,
) -> LLMResponse:
    """Perform a single completion, routing to tool calling when requested."""
    if tools:
        return await provider.complete_with_tools(request, tools)
    return await provider.complete(request)
