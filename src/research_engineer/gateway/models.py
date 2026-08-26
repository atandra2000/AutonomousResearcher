"""E3 - Tool Gateway models.

Typed Pydantic models for the centralized tool gateway: risk levels,
per-tool policy/permission metadata, budgets, execution context, and
execution results. These models describe the security boundary every
autonomous tool invocation passes through.

The gateway enforces, in order:

    policy -> permission -> budget -> approval -> sandbox -> tool
        -> result validation

and classifies failures as either *recoverable* (transient: timeout,
resource contention) or *policy/security* (terminal for that invocation:
denied, unknown, sandbox violation, approval denied).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLevel(StrEnum):
    """Risk classification for a tool invocation.

    These are *policy metadata*, not hardcoded into individual tools. The
    operator assigns a risk level to each registered tool via
    :class:`ToolPolicy`; the gateway uses it to decide whether human
    approval is required and how strictly sandbox rules apply.

    LOW      -> read-only / internal operations
    MEDIUM   -> code execution / filesystem mutation
    HIGH     -> network / package installation / external side effects
    CRITICAL -> infrastructure or production-changing operations
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolCallStatus(StrEnum):
    """Outcome of a tool invocation through the gateway."""

    ALLOWED = "allowed"
    DENIED = "denied"
    UNKNOWN = "unknown"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DENIED = "approval_denied"
    SANDBOX_VIOLATION = "sandbox_violation"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    ERROR = "error"
    SUCCESS = "success"


class ToolFailureKind(StrEnum):
    """Classification of a failed tool invocation.

    ``RECOVERABLE`` failures are transient (timeout, resource contention)
    and may be retried by the caller. ``POLICY`` failures are terminal for
    that invocation: the tool was denied, unknown, violated a sandbox rule,
    or was rejected at an approval gate. Policy failures must never be
    silently retried. ``INTERNAL`` failures are unexpected tool/runtime
    errors that should be surfaced to an operator.
    """

    RECOVERABLE = "recoverable"
    POLICY = "policy"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Policy / permission
# ---------------------------------------------------------------------------


class ToolPermission(BaseModel):
    """What a tool is allowed to do.

    ``allow`` gates whether the tool may run at all. ``filesystem`` and
    ``network`` describe the resource access the tool is granted; the
    sandbox enforces these against the configured workspace.
    """

    allow: bool = Field(
        default=False, description="Whether the tool may be invoked at all"
    )
    filesystem: bool = Field(
        default=False, description="Whether the tool may touch the filesystem"
    )
    network: bool = Field(
        default=False, description="Whether the tool may make network calls"
    )
    workspace_only: bool = Field(
        default=True,
        description=(
            "When True, filesystem access is confined to the configured "
            "workspace root(s)."
        ),
    )
    read_only: bool = Field(
        default=True,
        description="When True, filesystem access is read-only (no mutation).",
    )


class ToolBudget(BaseModel):
    """Per-tool and per-run resource budgets.

    All fields are optional; ``None`` means "unbounded". The gateway checks
    these before and after each invocation and fails the call with
    ``BUDGET_EXCEEDED`` when exceeded.
    """

    max_calls: int | None = Field(
        default=None, ge=1, description="Maximum invocations of this tool"
    )
    max_runtime_seconds: float | None = Field(
        default=None, gt=0.0, description="Maximum wall-clock per invocation"
    )
    max_memory_mb: float | None = Field(
        default=None, gt=0.0, description="Maximum memory per invocation (MB)"
    )
    max_output_bytes: int | None = Field(
        default=None, ge=1, description="Maximum result payload size (bytes)"
    )


class ToolPolicy(BaseModel):
    """Per-tool policy/risk metadata.

    This is the single source of truth the gateway consults for a tool:
    its risk level, permissions, budgets, and whether human approval is
    required. Policies are registered by name and are *not* hardcoded into
    individual tools.
    """

    tool_name: str = Field(..., description="Stable tool identifier")
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Risk classification"
    )
    permission: ToolPermission = Field(
        default_factory=ToolPermission, description="Allowed operations"
    )
    budget: ToolBudget = Field(
        default_factory=ToolBudget, description="Per-tool resource budgets"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether human approval is required before invocation",
    )
    description: str = Field(
        default="", description="Human-readable description of the tool"
    )


# ---------------------------------------------------------------------------
# Execution context / result
# ---------------------------------------------------------------------------


class ToolExecutionContext(BaseModel):
    """Metadata for a single tool invocation through the gateway.

    ``call_id`` is a stable, unique identifier for the invocation, used for
    correlation across observability events and logs.
    """

    call_id: str = Field(
        default_factory=lambda: f"tool_{uuid4().hex[:12]}",
        description="Unique tool-call identifier",
    )
    tool_name: str = Field(..., description="Tool being invoked")
    agent_name: str = Field(
        default="", description="Agent that requested the invocation"
    )
    run_id: str = Field(
        default="", description="Parent run/execution identifier, if any"
    )
    started_at: datetime = Field(
        default_factory=datetime.now, description="Invocation start time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary caller metadata"
    )


class ToolExecutionResult(BaseModel):
    """The outcome of a tool invocation through the gateway.

    ``status`` and ``failure_kind`` let callers distinguish a successful
    call from a recoverable failure (retryable) and a policy/security
    failure (terminal for that invocation). ``output`` is the sanitized
    tool result; ``error`` is a sanitized error message.
    """

    call_id: str = Field(..., description="Matching tool-call identifier")
    tool_name: str = Field(..., description="Tool that was invoked")
    status: ToolCallStatus = Field(
        default=ToolCallStatus.SUCCESS, description="Outcome of the call"
    )
    failure_kind: ToolFailureKind | None = Field(
        default=None, description="Failure classification, when failed"
    )
    output: Any = Field(default=None, description="Sanitized tool output")
    error: str = Field(default="", description="Sanitized error message")
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Wall-clock duration of the call"
    )
    started_at: datetime = Field(
        default_factory=datetime.now, description="Invocation start time"
    )
    finished_at: datetime | None = Field(default=None)

    @property
    def ok(self) -> bool:
        """True when the invocation succeeded."""
        return self.status == ToolCallStatus.SUCCESS

    @property
    def is_policy_failure(self) -> bool:
        """True when the failure is a terminal policy/security violation."""
        return self.failure_kind == ToolFailureKind.POLICY

    @property
    def is_recoverable(self) -> bool:
        """True when the failure is transient and may be retried."""
        return self.failure_kind == ToolFailureKind.RECOVERABLE


# ---------------------------------------------------------------------------
# Gateway configuration
# ---------------------------------------------------------------------------


class ToolGatewayConfig(BaseModel):
    """Configuration for the :class:`ToolGateway`.

    ``default_deny`` implements the "default deny for unknown tools"
    principle: any tool not explicitly registered is refused. ``workspace``
    is the set of allowed filesystem roots; when empty, filesystem access
    is denied entirely (least privilege).
    """

    default_deny: bool = Field(
        default=True,
        description="Refuse unknown/unregistered tools by default",
    )
    workspace: list[str] = Field(
        default_factory=list,
        description="Allowed filesystem roots (absolute paths)",
    )
    allow_network: bool = Field(
        default=False,
        description="Global network gate; per-tool policy must also allow it",
    )
    default_timeout_seconds: float = Field(
        default=60.0, gt=0.0, description="Default per-invocation timeout"
    )
    max_output_bytes: int = Field(
        default=1_000_000, ge=1, description="Default result size cap"
    )
    enforce_approval: bool = Field(
        default=False,
        description="When True, approval-required tools are gated even if no handler is set",
    )


__all__ = [
    "RiskLevel",
    "ToolCallStatus",
    "ToolFailureKind",
    "ToolPermission",
    "ToolBudget",
    "ToolPolicy",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolGatewayConfig",
]
