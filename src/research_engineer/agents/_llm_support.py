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

from research_engineer.llm.base import LLMProvider
from research_engineer.llm.router import ModelRouter, get_router

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
    except Exception:
        return None
    try:
        return r.for_agent(agent_name)
    except Exception:
        return None


__all__ = ["resolve_llm", "LLM_DISABLED"]
