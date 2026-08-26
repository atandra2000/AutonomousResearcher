"""E3 - Centralized Tool Gateway.

The :class:`ToolGateway` is the single, mandatory entry point for every
autonomous tool invocation. It enforces, in order:

    policy -> permission -> budget -> approval -> sandbox -> tool
        -> result validation

and emits structured observability events for every call. No tool can
bypass the gateway: agents receive a gateway (or a gateway-backed runtime)
and invoke tools only through it.

Security principles enforced here:

* default deny for unknown/unregistered tools
* least privilege by default (permissions default to deny)
* no unrestricted filesystem access (workspace-confined)
* no unrestricted network access (global + per-tool gate)
* human approval enforceable for configured high-risk tools
* policy violations are terminal for that invocation and clearly observable
* no claim of OS/container isolation (this is a policy boundary)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from research_engineer.gateway.adapter import ToolGatewayAdapter
from research_engineer.gateway.approval import (
    ApprovalHandler,
    ApprovalRequest,
    CallbackApprovalHandler,
)
from research_engineer.gateway.models import (
    RiskLevel,
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
from research_engineer.tools.base import Tool, ToolError

#: Risk levels that require human approval by default when the policy does
#: not explicitly set ``requires_approval``. Operators can override per tool.
APPROVAL_RISK_THRESHOLD = RiskLevel.HIGH


class ToolGateway:
    """Centralized, policy-enforcing tool dispatch.

    Args:
        config: Gateway configuration (workspace, network gate, timeouts).
        registry: Optional policy registry; a fresh one is created when
            omitted.
        approval_handler: Optional :class:`ApprovalHandler` for high-risk
            tools. When omitted, a :class:`CallbackApprovalHandler` with no
            callback is used (auto-approve in autonomous mode).
        event_bus: Optional observability event bus. When omitted, the
            process-wide bus from
            :func:`research_engineer.observability.get_event_bus` is used.
    """

    def __init__(
        self,
        config: ToolGatewayConfig | None = None,
        registry: ToolPolicyRegistry | None = None,
        approval_handler: ApprovalHandler | None = None,
        event_bus: Any | None = None,
    ) -> None:
        self.config = config or ToolGatewayConfig()
        self.registry = registry or ToolPolicyRegistry(
            default_deny=self.config.default_deny
        )
        self.sandbox = Sandbox(
            self.config.workspace, allow_network=self.config.allow_network
        )
        self.approval_handler = approval_handler or CallbackApprovalHandler(
            enforce=self.config.enforce_approval
        )
        self._event_bus = event_bus
        self._tools: dict[str, ToolGatewayAdapter] = {}
        self._call_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()


    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        tool: Tool[Any, Any] | ToolGatewayAdapter,
        *,
        name: str | None = None,
        policy: ToolPolicy | None = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        requires_approval: bool = False,
        description: str = "",
    ) -> ToolGatewayAdapter:
        """Register a tool for gateway dispatch.

        A tool must be registered before it can be invoked (default deny).
        If no policy is provided, one is created from the given risk level
        and approval flag. Returns the adapter so callers can inspect it.
        """
        adapter = (
            tool if isinstance(tool, ToolGatewayAdapter) else ToolGatewayAdapter(tool, name)
        )
        self._tools[adapter.name] = adapter
        if policy is not None:
            self.registry.register_policy(policy)
        else:
            # Explicit registration is an explicit grant: the tool is allowed
            # to run (subject to risk/approval/sandbox gates). The risk level
            # and approval flag still govern how strictly it is gated.
            self.registry.register(
                adapter.name,
                risk_level=risk_level,
                permission=ToolPermission(allow=True),
                requires_approval=requires_approval,
                description=description,
            )
        return adapter

    def register_policy(self, policy: ToolPolicy) -> ToolPolicy:
        """Register a fully-formed policy (tool need not be registered yet)."""
        return self.registry.register_policy(policy)

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool and its policy; return True if it was present."""
        removed = self._tools.pop(name, None) is not None
        self.registry.unregister(name)
        return removed

    def registered_tools(self) -> list[str]:
        """Return the sorted list of registered tool names.

        This reflects the policy registry (the source of truth for what is
        registered), which includes tools registered via :meth:`register_tool`
        and policies registered via :meth:`register_policy`.
        """
        return self.registry.names()

    def get_policy(self, tool_name: str) -> ToolPolicy | None:
        """Return the policy for ``tool_name``, if registered."""
        return self.registry.get(tool_name)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        input: Any,
        *,
        agent_name: str = "",
        run_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """Invoke ``tool_name`` with ``input`` through the full policy chain.

        Returns a :class:`ToolExecutionResult` describing the outcome. This
        method never raises for policy/security failures; it returns a
        result with the appropriate status and ``failure_kind`` so callers
        can distinguish recoverable from terminal failures.
        """
        ctx = ToolExecutionContext(
            tool_name=tool_name,
            agent_name=agent_name,
            run_id=run_id,
            metadata=metadata or {},
        )
        t0 = time.monotonic()
        self._emit("tool_call_start", ctx)

        # 1. Policy: is the tool registered and allowed?
        policy = self.registry.get(tool_name)
        if policy is None:
            return self._finish(
                ctx, t0, ToolCallStatus.UNKNOWN, ToolFailureKind.POLICY,
                error=f"Unknown tool: {tool_name}",
            )
        if not policy.permission.allow:
            return self._finish(
                ctx, t0, ToolCallStatus.DENIED, ToolFailureKind.POLICY,
                error=f"Tool denied by policy: {tool_name}",
            )

        # 2. Budget: per-tool call count.
        if not await self._check_call_budget(ctx, policy):
            return self._finish(
                ctx, t0, ToolCallStatus.BUDGET_EXCEEDED, ToolFailureKind.POLICY,
                error=f"Tool budget exceeded for {tool_name}",
            )

        # 3. Approval: high-risk tools require human approval.
        if not await self._check_approval(ctx, policy):
            return self._finish(
                ctx, t0, ToolCallStatus.APPROVAL_DENIED, ToolFailureKind.POLICY,
                error=f"Approval denied for {tool_name}",
            )

        # 4. Sandbox: filesystem/network restrictions.
        sandbox_error = self._check_sandbox(ctx, policy, input)
        if sandbox_error is not None:
            return self._finish(
                ctx, t0, ToolCallStatus.SANDBOX_VIOLATION, ToolFailureKind.POLICY,
                error=sandbox_error,
            )

        # 5. Tool: execute with timeout.
        adapter = self._tools.get(tool_name)
        if adapter is None:
            return self._finish(
                ctx, t0, ToolCallStatus.UNKNOWN, ToolFailureKind.POLICY,
                error=f"Tool not registered for dispatch: {tool_name}",
            )

        return await self._run_tool(ctx, adapter, input, policy, t0)


    # ------------------------------------------------------------------
    # Pipeline internals
    # ------------------------------------------------------------------

    async def _check_call_budget(
        self, ctx: ToolExecutionContext, policy: ToolPolicy
    ) -> bool:
        """Enforce the per-tool call-count budget."""
        if policy.budget.max_calls is None:
            return True
        async with self._lock:
            count = self._call_counts.get(ctx.tool_name, 0)
            if count >= policy.budget.max_calls:
                return False
            self._call_counts[ctx.tool_name] = count + 1
        return True

    async def _check_approval(
        self, ctx: ToolExecutionContext, policy: ToolPolicy
    ) -> bool:
        """Enforce human approval for high-risk tools.

        A tool requires approval when its policy sets ``requires_approval``
        or when its risk level is at or above the approval threshold.
        """
        requires = policy.requires_approval or (
            policy.risk_level.value
            >= APPROVAL_RISK_THRESHOLD.value
        )
        if not requires:
            return True
        request = ApprovalRequest(
            tool_name=ctx.tool_name,
            risk_level=policy.risk_level,
            summary=policy.description or f"Invoke {ctx.tool_name}",
            context=ctx,
            policy=policy,
        )
        self._emit("tool_approval_requested", ctx, risk=policy.risk_level.value)
        approved = await self.approval_handler.approve(request)
        self._emit(
            "tool_approval_resolved", ctx, approved=approved,
            risk=policy.risk_level.value,
        )
        return approved

    def _check_sandbox(
        self, ctx: ToolExecutionContext, policy: ToolPolicy, input: Any
    ) -> str | None:
        """Validate filesystem/network access; return an error string or None.

        The sandbox enforces *declared* access: if the tool's input declares
        filesystem paths, those paths must be within the granted workspace
        and the permission must grant filesystem access. Network access is
        enforced only when the permission grants it (a tool that declares no
        network intent is not penalized for lacking network permission).
        """
        try:
            paths = self._extract_paths(input)
            if paths:
                self.sandbox.check_filesystem(policy.permission, paths)
            if policy.permission.network:
                self.sandbox.check_network(policy.permission)
            return None
        except SandboxError as exc:
            return str(exc)

    async def _run_tool(
        self,
        ctx: ToolExecutionContext,
        adapter: ToolGatewayAdapter,
        input: Any,
        policy: ToolPolicy,
        t0: float,
    ) -> ToolExecutionResult:
        """Execute the tool with timeout and result validation."""
        timeout = policy.budget.max_runtime_seconds or self.config.default_timeout_seconds
        try:
            output = await asyncio.wait_for(adapter.execute(input), timeout=timeout)
        except TimeoutError:
            return self._finish(
                ctx, t0, ToolCallStatus.TIMEOUT, ToolFailureKind.RECOVERABLE,
                error=f"Tool {ctx.tool_name} timed out after {timeout:.1f}s",
            )
        except ToolError as exc:
            return self._finish(
                ctx, t0, ToolCallStatus.ERROR, ToolFailureKind.RECOVERABLE,
                error=self._sanitize(str(exc)),
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            return self._finish(
                ctx, t0, ToolCallStatus.ERROR, ToolFailureKind.INTERNAL,
                error=self._sanitize(str(exc)),
            )

        # Result validation: cap output size.
        output = self._cap_output(output, policy)
        return self._finish(ctx, t0, ToolCallStatus.SUCCESS, None, output=output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_paths(input: Any) -> list[Any]:
        """Best-effort extraction of filesystem paths from a tool input.

        Supports pydantic models with a ``repo_path``/``working_dir`` field
        and plain dicts with those keys. This is intentionally conservative:
        it only inspects well-known path fields.
        """
        paths: list[Any] = []
        if isinstance(input, dict):
            for key in ("repo_path", "working_dir", "path", "file_path"):
                if key in input and input[key]:
                    paths.append(input[key])
            return paths
        for attr in ("repo_path", "working_dir", "path", "file_path"):
            if hasattr(input, attr):
                value = getattr(input, attr)
                if value:
                    paths.append(value)
        return paths

    def _cap_output(self, output: Any, policy: ToolPolicy) -> Any:
        """Cap the result payload size to bound memory."""
        limit = policy.budget.max_output_bytes or self.config.max_output_bytes
        if isinstance(output, str) and len(output) > limit:
            return output[:limit] + "\n...[truncated]"
        return output

    @staticmethod
    def _sanitize(message: str) -> str:
        """Sanitize an error message for safe logging/return.

        Strips control characters and caps length to avoid leaking huge or
        binary payloads through the error channel.
        """
        cleaned = "".join(ch for ch in message if ch.isprintable() or ch in "\n\t")
        return cleaned[:2000]

    def _finish(
        self,
        ctx: ToolExecutionContext,
        t0: float,
        status: ToolCallStatus,
        failure_kind: ToolFailureKind | None,
        *,
        output: Any = None,
        error: str = "",
    ) -> ToolExecutionResult:
        """Build the result, stamp timing, and emit observability events."""
        duration = round(time.monotonic() - t0, 6)
        result = ToolExecutionResult(
            call_id=ctx.call_id,
            tool_name=ctx.tool_name,
            status=status,
            failure_kind=failure_kind,
            output=output,
            error=self._sanitize(error),
            duration_seconds=duration,
            started_at=ctx.started_at,
            finished_at=datetime.now(),
        )
        self._emit(
            "tool_call_end",
            ctx,
            status=status.value,
            failure_kind=failure_kind.value if failure_kind else None,
            duration_seconds=duration,
            error=result.error or None,
        )
        return result

    def _emit(self, kind: str, ctx: ToolExecutionContext, **extra: Any) -> None:
        """Emit a structured ``tool_gateway`` event (best-effort)."""
        try:
            bus = self._event_bus
            if bus is None:
                from research_engineer.observability import get_event_bus

                bus = get_event_bus()
            event: dict[str, Any] = {
                "kind": "tool_gateway",
                "event": kind,
                "call_id": ctx.call_id,
                "tool_name": ctx.tool_name,
                "agent_name": ctx.agent_name,
                "run_id": ctx.run_id,
            }
            event.update(extra)
            bus.emit(event)
        except Exception:  # noqa: BLE001 - observability must not break dispatch
            pass


__all__ = ["ToolGateway", "APPROVAL_RISK_THRESHOLD"]
