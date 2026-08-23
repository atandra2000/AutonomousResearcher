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
- Exhausting all attempts raises :class:`ProviderError` so callers'
  existing fallback paths engage instead of silently degrading.
"""

from __future__ import annotations

import asyncio
import logging
import re

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
)

logger = logging.getLogger(__name__)

#: Hard ceiling for max_tokens escalation on truncated responses.
_MAX_TOKENS_CAP = 16_384

_HTTP_CODE_RE = re.compile(r"HTTP (\d{3})")


def is_permanent_provider_error(error: BaseException) -> bool:
    """True when retrying this error cannot possibly succeed."""
    if isinstance(error, ProviderError):
        match = _HTTP_CODE_RE.search(str(error))
        if match:
            code = int(match.group(1))
            return not (code == 429 or code >= 500)
    return False


async def complete_with_retry(
    provider: LLMProvider,
    request: LLMRequest,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    agent_name: str = "",
) -> LLMResponse:
    """Call ``provider.complete`` with retries and empty-content handling.

    Raises the last observed error when all attempts are exhausted.
    """
    last_error: BaseException | None = None
    last_response: LLMResponse | None = None
    escalated_tokens = False

    for attempt in range(1, max_attempts + 1):
        try:
            response = await provider.complete(request)
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
            if response.content.strip():
                return response
            last_response = response
            # Truncated before any content: reasoning models spend the
            # whole budget invisibly. Escalate once, then give up.
            if (
                response.finish_reason == "length"
                and not escalated_tokens
                and request.max_tokens
            ):
                doubled = min(request.max_tokens * 2, _MAX_TOKENS_CAP)
                if doubled > request.max_tokens:
                    request = request.model_copy(
                        update={"max_tokens": doubled}
                    )
                    escalated_tokens = True
                    logger.warning(
                        "LLM returned empty content with finish_reason="
                        "'length' for %s; retrying with max_tokens=%d",
                        agent_name or type(provider).__name__,
                        doubled,
                    )
            else:
                logger.warning(
                    "LLM returned empty content for %s "
                    "(attempt %d/%d, finish_reason=%s)",
                    agent_name or type(provider).__name__,
                    attempt,
                    max_attempts,
                    response.finish_reason,
                )

        if attempt < max_attempts:
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    if last_response is not None:
        # Return the final (empty) response rather than fabricating one;
        # callers already handle empty content defensively.
        return last_response
    raise ProviderError(
        f"LLM completion failed after {max_attempts} attempts for "
        f"{agent_name or type(provider).__name__}: {last_error}",
        cause=last_error if isinstance(last_error, Exception) else None,
    )
