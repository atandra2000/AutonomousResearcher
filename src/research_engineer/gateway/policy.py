"""E3 - Tool policy registry.

The :class:`ToolPolicyRegistry` is the single source of truth for which
tools are registered and what they are allowed to do. It implements the
"default deny for unknown tools" principle: a tool that is not registered
is refused by the gateway.

Policies are keyed by stable tool name and are *not* hardcoded into
individual tools. Operators register a :class:`ToolPolicy` per tool; the
gateway consults the registry on every invocation.
"""

from __future__ import annotations

from typing import Any

from research_engineer.gateway.models import (
    RiskLevel,
    ToolPermission,
    ToolPolicy,
)


class ToolPolicyRegistry:
    """Registry of per-tool policies with default-deny semantics.

    Args:
        default_deny: When True (default), an unregistered tool is refused.
            When False, an unregistered tool is allowed with a permissive
            default policy (used only for backward-compatible adapters that
            opt in to permissive behaviour).
    """

    def __init__(self, default_deny: bool = True) -> None:
        self._policies: dict[str, ToolPolicy] = {}
        self._default_deny = default_deny

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        tool_name: str,
        *,
        risk_level: RiskLevel = RiskLevel.LOW,
        permission: ToolPermission | None = None,
        requires_approval: bool = False,
        description: str = "",
        budget: Any = None,
    ) -> ToolPolicy:
        """Register (or replace) the policy for ``tool_name``.

        Returns the created :class:`ToolPolicy` so callers can inspect or
        mutate it.
        """
        policy = ToolPolicy(
            tool_name=tool_name,
            risk_level=risk_level,
            permission=permission or ToolPermission(),
            requires_approval=requires_approval,
            description=description,
        )
        if budget is not None:
            policy.budget = budget
        self._policies[tool_name] = policy
        return policy

    def register_policy(self, policy: ToolPolicy) -> ToolPolicy:
        """Register a fully-formed :class:`ToolPolicy`."""
        self._policies[policy.tool_name] = policy
        return policy

    def unregister(self, tool_name: str) -> bool:
        """Remove a tool's policy; return True if it was present."""
        return self._policies.pop(tool_name, None) is not None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, tool_name: str) -> ToolPolicy | None:
        """Return the policy for ``tool_name``, or None if unregistered."""
        return self._policies.get(tool_name)

    def is_registered(self, tool_name: str) -> bool:
        """True when ``tool_name`` has an explicit policy."""
        return tool_name in self._policies

    def is_allowed(self, tool_name: str) -> bool:
        """True when the tool may be invoked.

        An unregistered tool is allowed only when ``default_deny`` is False.
        A registered tool is allowed only when its policy permission
        ``allow`` is True.
        """
        policy = self._policies.get(tool_name)
        if policy is None:
            return not self._default_deny
        return policy.permission.allow

    def names(self) -> list[str]:
        """Return the sorted list of registered tool names."""
        return sorted(self._policies)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._policies


__all__ = ["ToolPolicyRegistry"]
