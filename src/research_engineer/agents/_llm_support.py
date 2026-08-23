"""Shared LLM integration helpers for agents.

This module provides the glue that lets agents accept an externally
configured :class:`~research_engineer.llm.base.LLMProvider` *or* lazily
resolve one from the :class:`~research_engineer.llm.router.ModelRouter`
based on the agent's declared ``agent_name``.

Agents MUST obtain providers through :func:`resolve_llm` rather than
instantiating a concrete provider themselves, satisfying the rule that no
agent calls a model directly.
"""

from __future__ import annotations

import logging

from research_engineer.llm.base import LLMProvider
from research_engineer.llm.router import ModelRouter, get_router

logger = logging.getLogger(__name__)

#: Sentinel used by agents that genuinely have no LLM requirement.
LLM_DISABLED = "disabled"


def resolve_llm(
    agent_name: str,
    explicit: LLMProvider | None,
    llm_enabled: bool = True,
    router: ModelRouter | None = None,
) -> LLMProvider | None:
    """Return the provider an agent should use.

    Resolution order:
      1. An explicit provider passed to the agent constructor wins.
      2. If ``llm_enabled`` is False, return ``None`` (LLM not requested).
      3. Otherwise resolve via the (process-wide) ``ModelRouter``.
    """
    if explicit is not None:
        return explicit
    if not llm_enabled:
        return None
    try:
        r = router or get_router()
    except Exception as e:
        logger.warning(
            "LLM router unavailable for %s (%s: %s); "
            "falling back to rule-based mode",
            agent_name,
            type(e).__name__,
            e,
        )
        return None
    try:
        return r.for_agent(agent_name)
    except Exception as e:
        logger.warning(
            "No LLM provider resolved for %s (%s: %s); "
            "falling back to rule-based mode",
            agent_name,
            type(e).__name__,
            e,
        )
        return None


__all__ = ["resolve_llm", "LLM_DISABLED"]
