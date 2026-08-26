"""E3 - Human approval integration for high-risk tools.

The gateway consults an :class:`ApprovalHandler` before invoking any tool
whose policy marks ``requires_approval=True``. The handler is an async
callable that receives a human-readable request and returns a boolean
decision.

The default :class:`CallbackApprovalHandler` wraps an existing approval
callback (e.g. the loop agent's ``ApprovalCallback``) so the gateway can
reuse the platform's existing approval machinery rather than rebuilding it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from research_engineer.gateway.models import (
    RiskLevel,
    ToolExecutionContext,
    ToolPolicy,
)


class ApprovalRequest:
    """A human-approval request for a high-risk tool invocation.

    This is a lightweight, gateway-specific request object. It carries the
    tool name, risk level, a human-readable summary, and the execution
    context so a reviewer can decide before the tool runs.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        risk_level: RiskLevel,
        summary: str,
        context: ToolExecutionContext,
        policy: ToolPolicy,
    ) -> None:
        self.tool_name = tool_name
        self.risk_level = risk_level
        self.summary = summary
        self.context = context
        self.policy = policy

    def __repr__(self) -> str:
        return (
            f"<ApprovalRequest tool={self.tool_name} "
            f"risk={self.risk_level.value} call={self.context.call_id}>"
        )


class ApprovalHandler(Protocol):
    """Async protocol for deciding a high-risk tool invocation.

    Implementations return True to approve (allow the tool to run) or False
    to deny (the invocation fails with ``APPROVAL_DENIED``).
    """

    async def approve(self, request: ApprovalRequest) -> bool: ...


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[bool]]


class CallbackApprovalHandler:
    """Wrap an existing async approval callback as an :class:`ApprovalHandler`.

    This lets the gateway reuse the platform's existing approval callbacks
    (e.g. the loop agent's ``ApprovalCallback``) unchanged. When no callback
    is provided, the handler auto-approves in autonomous mode (matching the
    loop agent's behaviour) unless ``enforce`` is True, in which case it
    denies.
    """

    def __init__(
        self,
        callback: ApprovalCallback | None = None,
        *,
        enforce: bool = False,
    ) -> None:
        self._callback = callback
        self._enforce = enforce

    async def approve(self, request: ApprovalRequest) -> bool:
        if self._callback is not None:
            try:
                return await self._callback(request)
            except Exception:
                return False
        # No callback: auto-approve in autonomous mode, deny when enforcing.
        return not self._enforce


__all__ = [
    "ApprovalRequest",
    "ApprovalHandler",
    "ApprovalCallback",
    "CallbackApprovalHandler",
]
