"""E5 - Human approval for policy-defined pause situations.

When a safety decision resolves to ``PAUSE_FOR_APPROVAL``, the
:class:`SafetyController` consults a :class:`PauseApprovalGate`. This is a
thin, E5-specific gate that deliberately mirrors the E3 gateway's
:class:`~research_engineer.gateway.approval.ApprovalHandler` pattern; it
does *not* replace the gateway approval chain (which still runs first on
every tool invocation).

If no gate is configured the controller fails closed: the run terminates
with ``APPROVAL_REQUIRED`` rather than resuming autonomously.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field

from research_engineer.safety.models import ControlTrigger


class PauseRequest(BaseModel):
    """A human-approval request raised by the safety layer."""

    run_id: str = Field(default="", description="Execution id of the paused run")
    step: int = Field(default=0, ge=0)
    trigger: ControlTrigger = ControlTrigger.NONE
    reason_code: str = ""
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary."""
        return f"[{self.trigger.value}] {self.reason} (code={self.reason_code})"


class PauseApprovalGate(Protocol):
    """Async protocol deciding whether an autonomous pause may continue."""

    async def approve_pause(self, request: PauseRequest) -> bool: ...


PauseApprovalCallback = Callable[[PauseRequest], Awaitable[bool]]


class CallbackPauseGate:
    """Wrap an existing async callback as a :class:`PauseApprovalGate`.

    Mirrors :class:`~research_engineer.gateway.approval.CallbackApprovalHandler`
    so platform approval callbacks can be reused unchanged.
    """

    def __init__(self, callback: PauseApprovalCallback | None = None) -> None:
        self._callback = callback

    async def approve_pause(self, request: PauseRequest) -> bool:
        if self._callback is None:
            # Fail closed: no reviewer wired means no autonomous resume.
            return False
        try:
            return await self._callback(request)
        except Exception:  # noqa: BLE001 - approval errors deny, never crash
            return False


__all__ = [
    "PauseRequest",
    "PauseApprovalGate",
    "PauseApprovalCallback",
    "CallbackPauseGate",
]
