"""E3 - Production Tool Gateway & Security Boundary.

A centralized, policy-enforcing gateway that every autonomous tool
invocation passes through. It enforces, in order:

    policy -> permission -> budget -> approval -> sandbox -> tool
        -> result validation

and emits structured observability events for every call. Existing tools
are wrapped with :class:`ToolGatewayAdapter` so they run through the
gateway unchanged.

Public surface::

    from research_engineer.gateway import (
        ToolGateway,
        ToolGatewayConfig,
        ToolPolicy,
        ToolPolicyRegistry,
        ToolPermission,
        ToolBudget,
        RiskLevel,
        ToolCallStatus,
        ToolFailureKind,
        ToolExecutionContext,
        ToolExecutionResult,
        ToolGatewayAdapter,
        ApprovalHandler,
        ApprovalRequest,
        CallbackApprovalHandler,
        Sandbox,
        SandboxError,
    )
"""

from research_engineer.gateway.adapter import ToolGatewayAdapter
from research_engineer.gateway.approval import (
    ApprovalHandler,
    ApprovalRequest,
    CallbackApprovalHandler,
)
from research_engineer.gateway.gateway import APPROVAL_RISK_THRESHOLD, ToolGateway
from research_engineer.gateway.models import (
    RiskLevel,
    ToolBudget,
    ToolCallStatus,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolFailureKind,
    ToolGatewayConfig,
    ToolPermission,
    ToolPolicy,
)
from research_engineer.gateway.policy import ToolPolicyRegistry
from research_engineer.gateway.sandbox import Sandbox, SandboxError

__all__ = [
    "ToolGateway",
    "ToolGatewayConfig",
    "ToolPolicy",
    "ToolPolicyRegistry",
    "ToolPermission",
    "ToolBudget",
    "RiskLevel",
    "ToolCallStatus",
    "ToolFailureKind",
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolGatewayAdapter",
    "ApprovalHandler",
    "ApprovalRequest",
    "CallbackApprovalHandler",
    "Sandbox",
    "SandboxError",
    "APPROVAL_RISK_THRESHOLD",
]
