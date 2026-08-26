"""Tests for E3 - Centralized Tool Gateway.

Covers the full dispatch pipeline enforced by :class:`ToolGateway`:

    policy -> permission -> budget -> approval -> sandbox -> tool
        -> result validation

and the runtime integration (:meth:`AgentRuntime.call_tool`).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from research_engineer.gateway import (
    APPROVAL_RISK_THRESHOLD,
    ApprovalRequest,
    CallbackApprovalHandler,
    RiskLevel,
    Sandbox,
    SandboxError,
    ToolBudget,
    ToolCallStatus,
    ToolFailureKind,
    ToolGateway,
    ToolGatewayAdapter,
    ToolGatewayConfig,
    ToolPermission,
    ToolPolicy,
    ToolPolicyRegistry,
)
from research_engineer.observability import NullSink, get_event_bus, reset_event_bus
from research_engineer.runtime import AgentRuntime
from research_engineer.tools.base import Tool, ToolError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EchoTool(Tool[Any, Any]):
    """A trivial tool that echoes its input back."""

    async def execute(self, input: Any) -> Any:
        return {"echo": input}

    async def validate(self, input: Any) -> bool:
        return True


class StringTool(Tool[Any, Any]):
    """A tool that returns a plain string (for output-capping tests)."""

    async def execute(self, input: Any) -> Any:
        return "x" * 100

    async def validate(self, input: Any) -> bool:
        return True


class FailingTool(Tool[Any, Any]):
    """A tool that raises a recoverable :class:`ToolError`."""

    async def execute(self, input: Any) -> Any:
        raise ToolError("boom")

    async def validate(self, input: Any) -> bool:
        return True


class ExplodingTool(Tool[Any, Any]):
    """A tool that raises an unexpected (internal) exception."""

    async def execute(self, input: Any) -> Any:
        raise ValueError("unexpected")

    async def validate(self, input: Any) -> bool:
        return True


class SlowTool(Tool[Any, Any]):
    """A tool that sleeps longer than the configured timeout."""

    async def execute(self, input: Any) -> Any:
        await asyncio.sleep(5)
        return {"done": True}

    async def validate(self, input: Any) -> bool:
        return True

# ---------------------------------------------------------------------------
# Policy / registration
# ---------------------------------------------------------------------------


def test_default_deny_unknown_tool() -> None:
    """An unregistered tool is refused (default deny)."""
    gw = _gateway()
    result = asyncio.run(gw.execute("nope", {}))
    assert result.status == ToolCallStatus.UNKNOWN
    assert result.failure_kind == ToolFailureKind.POLICY
    assert result.is_policy_failure
    assert not result.ok


def test_registered_tool_is_allowed() -> None:
    """Explicit registration grants the tool permission to run."""
    gw = _gateway()
    result = asyncio.run(gw.execute("echo", {"msg": "hi"}))
    assert result.status == ToolCallStatus.SUCCESS
    assert result.ok
    assert result.output == {"echo": {"msg": "hi"}}


def test_denied_tool_is_refused() -> None:
    """A tool whose permission ``allow`` is False is refused."""
    gw = _gateway()
    gw.get_policy("echo").permission.allow = False
    result = asyncio.run(gw.execute("echo", {}))
    assert result.status == ToolCallStatus.DENIED
    assert result.failure_kind == ToolFailureKind.POLICY


def test_registered_tools_listing() -> None:
    """registered_tools returns the sorted tool names."""
    gw = _gateway()
    gw.register_tool(EchoTool(), name="zeta", risk_level=RiskLevel.LOW)
    assert gw.registered_tools() == ["echo", "zeta"]


def test_unregister_tool() -> None:
    """unregister_tool removes the tool and its policy."""
    gw = _gateway()
    assert gw.unregister_tool("echo") is True
    assert gw.unregister_tool("echo") is False
    assert "echo" not in gw.registered_tools()
    result = asyncio.run(gw.execute("echo", {}))
    assert result.status == ToolCallStatus.UNKNOWN


def test_register_policy_directly() -> None:
    """A fully-formed policy can be registered without a tool instance."""
    gw = _gateway()
    policy = ToolPolicy(
        tool_name="custom",
        risk_level=RiskLevel.MEDIUM,
        permission=ToolPermission(allow=True),
    )
    gw.register_policy(policy)
    assert gw.get_policy("custom") is policy
    assert "custom" in gw.registered_tools()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_max_calls() -> None:
    """The per-tool call-count budget is enforced."""
    gw = _gateway()
    _policy(gw, "echo").budget = ToolBudget(max_calls=1)
    first = asyncio.run(gw.execute("echo", {}))
    second = asyncio.run(gw.execute("echo", {}))
    assert first.status == ToolCallStatus.SUCCESS
    assert second.status == ToolCallStatus.BUDGET_EXCEEDED
    assert second.failure_kind == ToolFailureKind.POLICY



# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


def test_high_risk_requires_approval() -> None:
    """A high-risk tool is gated behind human approval."""
    async def deny(request: ApprovalRequest) -> bool:
        return False

    gw = ToolGateway(
        ToolGatewayConfig(workspace=["/tmp"]),
        approval_handler=CallbackApprovalHandler(deny),
    )
    gw.register_tool(EchoTool(), name="risky", risk_level=RiskLevel.HIGH)
    result = asyncio.run(gw.execute("risky", {}))
    assert result.status == ToolCallStatus.APPROVAL_DENIED
    assert result.failure_kind == ToolFailureKind.POLICY


def test_high_risk_approved_runs() -> None:
    """A high-risk tool approved by the handler runs normally."""
    async def approve(request: ApprovalRequest) -> bool:
        return True

    gw = ToolGateway(
        ToolGatewayConfig(workspace=["/tmp"]),
        approval_handler=CallbackApprovalHandler(approve),
    )
    gw.register_tool(EchoTool(), name="risky", risk_level=RiskLevel.HIGH)
    result = asyncio.run(gw.execute("risky", {}))
    assert result.status == ToolCallStatus.SUCCESS
    assert result.ok


def test_low_risk_skips_approval() -> None:
    """A low-risk tool does not require approval."""
    gw = _gateway()
    result = asyncio.run(gw.execute("echo", {}))
    assert result.status == ToolCallStatus.SUCCESS


def test_approval_threshold_constant() -> None:
    """The approval threshold is HIGH."""
    assert APPROVAL_RISK_THRESHOLD == RiskLevel.HIGH


def test_requires_approval_flag() -> None:
    """A tool can require approval via its policy flag regardless of risk."""
    async def deny(request: ApprovalRequest) -> bool:
        return False

    gw = ToolGateway(
        ToolGatewayConfig(workspace=["/tmp"]),
        approval_handler=CallbackApprovalHandler(deny),
    )
    gw.register_tool(
        EchoTool(), name="gated", risk_level=RiskLevel.LOW,
        requires_approval=True,
    )
    result = asyncio.run(gw.execute("gated", {}))
    assert result.status == ToolCallStatus.APPROVAL_DENIED


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_filesystem_path_outside_workspace() -> None:
    """A filesystem tool with a path outside the workspace is refused."""
    gw = _gateway()
    _policy(gw, "echo").permission.filesystem = True
    result = asyncio.run(gw.execute("echo", {"repo_path": "/etc/passwd"}))
    assert result.status == ToolCallStatus.SANDBOX_VIOLATION
    assert result.failure_kind == ToolFailureKind.POLICY


def test_filesystem_path_inside_workspace() -> None:
    """A filesystem tool with a path inside the workspace runs."""
    gw = _gateway()
    _policy(gw, "echo").permission.filesystem = True
    result = asyncio.run(gw.execute("echo", {"repo_path": "/tmp/x"}))
    assert result.status == ToolCallStatus.SUCCESS


def test_network_denied_when_gate_closed() -> None:
    """Network access is refused when the global gate is closed."""
    gw = _gateway()
    _policy(gw, "echo").permission.network = True
    result = asyncio.run(gw.execute("echo", {}))
    assert result.status == ToolCallStatus.SANDBOX_VIOLATION


def test_network_allowed_when_gate_open() -> None:
    """Network access is allowed when the global gate is open."""
    gw = _gateway(allow_network=True)
    _policy(gw, "echo").permission.network = True
    result = asyncio.run(gw.execute("echo", {}))
    assert result.status == ToolCallStatus.SUCCESS


def test_sandbox_class_direct() -> None:
    """The Sandbox class enforces workspace confinement directly."""
    sb = Sandbox(["/tmp"], allow_network=False)
    assert sb.is_within_workspace("/tmp/x")
    assert not sb.is_within_workspace("/etc/passwd")
    with pytest.raises(SandboxError):
        sb.check_filesystem(ToolPermission(filesystem=True), ["/etc/passwd"])
    with pytest.raises(SandboxError):
        sb.check_network(ToolPermission(network=True))

def test_budget_unbounded_by_default() -> None:
    """Without a budget, a tool may be called repeatedly."""
    gw = _gateway()
    for _ in range(5):
        result = asyncio.run(gw.execute("echo", {}))
        assert result.status == ToolCallStatus.SUCCESS


def _gateway(**config: Any) -> ToolGateway:
    """Build a gateway with a workspace and a registered echo tool."""
    cfg = ToolGatewayConfig(workspace=["/tmp"], **config)
    gw = ToolGateway(cfg)
    gw.register_tool(EchoTool(), name="echo", risk_level=RiskLevel.LOW)
    return gw


def _policy(gw: ToolGateway, name: str) -> ToolPolicy:
    """Return the policy for ``name``, asserting it is registered."""
    policy = gw.get_policy(name)
    assert policy is not None, f"policy for {name!r} not registered"
    return policy


async def _noop_planner(ctx: Any) -> Any:
    return None


async def _noop_actor(ctx: Any, action: Any) -> Any:
    return None


async def _noop_observer(ctx: Any, event: Any) -> Any:
    return None


async def _done_evaluator(ctx: Any) -> Any:
    return None


def _runtime(gw: ToolGateway | None = None) -> AgentRuntime:
    """Build an AgentRuntime with noop callables and an optional gateway."""
    return AgentRuntime(
        planner=_noop_planner,
        actor=_noop_actor,
        observer=_noop_observer,
        evaluator=_done_evaluator,
        tool_gateway=gw,
    )

# ---------------------------------------------------------------------------
# Tool execution / result validation
# ---------------------------------------------------------------------------


def test_recoverable_tool_error() -> None:
    """A ToolError is classified as a recoverable failure."""
    gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
    gw.register_tool(FailingTool(), name="fail", risk_level=RiskLevel.LOW)
    result = asyncio.run(gw.execute("fail", {}))
    assert result.status == ToolCallStatus.ERROR
    assert result.failure_kind == ToolFailureKind.RECOVERABLE
    assert result.is_recoverable
    assert "boom" in result.error


def test_internal_tool_error() -> None:
    """An unexpected exception is classified as an internal failure."""
    gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
    gw.register_tool(ExplodingTool(), name="explode", risk_level=RiskLevel.LOW)
    result = asyncio.run(gw.execute("explode", {}))
    assert result.status == ToolCallStatus.ERROR
    assert result.failure_kind == ToolFailureKind.INTERNAL


def test_timeout() -> None:
    """A tool that exceeds its timeout is classified as recoverable."""
    gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"], default_timeout_seconds=0.1))
    gw.register_tool(SlowTool(), name="slow", risk_level=RiskLevel.LOW)
    result = asyncio.run(gw.execute("slow", {}))
    assert result.status == ToolCallStatus.TIMEOUT
    assert result.failure_kind == ToolFailureKind.RECOVERABLE


def test_output_capped() -> None:
    """Oversized string output is truncated to the configured cap."""
    gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"], max_output_bytes=10))
    gw.register_tool(StringTool(), name="str", risk_level=RiskLevel.LOW)
    result = asyncio.run(gw.execute("str", {}))
    assert result.status == ToolCallStatus.SUCCESS
    assert len(result.output) <= 10 + len("\n...[truncated]")


def test_result_timing_stamped() -> None:
    """Results carry start/finish timestamps and duration."""
    gw = _gateway()
    result = asyncio.run(gw.execute("echo", {}))
    assert result.duration_seconds >= 0.0
    assert result.started_at is not None
    assert result.finished_at is not None


def test_error_sanitized() -> None:
    """Error messages are sanitized (control chars stripped, length capped)."""
    gw = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
    gw.register_tool(FailingTool(), name="fail", risk_level=RiskLevel.LOW)
    result = asyncio.run(gw.execute("fail", {}))
    assert "\x00" not in result.error
    assert len(result.error) <= 2000


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_emits_gateway_events() -> None:
    """The gateway emits structured tool_gateway events to the event bus."""
    reset_event_bus()
    bus = get_event_bus()
    bus.add_sink(NullSink())
    captured: list[dict[str, Any]] = []
    original_emit = bus.emit

    def _wrapped_emit(event: dict[str, Any]) -> None:
        captured.append(event)
        original_emit(event)

    bus.emit = _wrapped_emit  # type: ignore[method-assign]

    gw = _gateway()
    asyncio.run(gw.execute("echo", {}))
    kinds = [e.get("event") for e in captured if e.get("kind") == "tool_gateway"]
    assert "tool_call_start" in kinds
    assert "tool_call_end" in kinds
    reset_event_bus()


# ---------------------------------------------------------------------------
# Runtime integration
# ---------------------------------------------------------------------------


def test_runtime_call_tool_through_gateway() -> None:
    """AgentRuntime.call_tool routes through the configured gateway."""
    gw = _gateway()
    rt = _runtime(gw)
    result = asyncio.run(rt.call_tool("echo", {"msg": "via runtime"}, agent_name="t"))
    assert result.status == ToolCallStatus.SUCCESS
    assert result.ok


def test_runtime_without_gateway_raises() -> None:
    """AgentRuntime.call_tool raises when no gateway is configured."""
    rt = _runtime()
    with pytest.raises(RuntimeError):
        asyncio.run(rt.call_tool("echo", {}))


def test_runtime_exposes_gateway() -> None:
    """AgentRuntime.tool_gateway returns the configured gateway."""
    gw = _gateway()
    rt = _runtime(gw)
    assert rt.tool_gateway is gw


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def test_adapter_wraps_tool() -> None:
    """ToolGatewayAdapter wraps an existing tool and exposes its name."""
    tool = EchoTool()
    adapter = ToolGatewayAdapter(tool, name="echo")
    assert adapter.name == "echo"
    assert adapter.tool is tool
    assert asyncio.run(adapter.validate({})) is True
    assert asyncio.run(adapter.execute({"a": 1})) == {"echo": {"a": 1}}


def test_adapter_default_name_from_class() -> None:
    """The adapter derives a name from the tool class when none is given."""
    adapter = ToolGatewayAdapter(EchoTool())
    assert adapter.name == "EchoTool"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_default_deny() -> None:
    """The registry refuses unregistered tools by default."""
    reg = ToolPolicyRegistry(default_deny=True)
    assert not reg.is_allowed("nope")
    assert not reg.is_registered("nope")


def test_registry_permissive_mode() -> None:
    """With default_deny=False, unregistered tools are allowed."""
    reg = ToolPolicyRegistry(default_deny=False)
    assert reg.is_allowed("nope")


def test_registry_register_and_lookup() -> None:
    """Registering a policy makes it discoverable and allowed."""
    reg = ToolPolicyRegistry()
    policy = reg.register(
        "echo", risk_level=RiskLevel.LOW, permission=ToolPermission(allow=True)
    )
    assert reg.is_registered("echo")
    assert reg.get("echo") is policy
    assert "echo" in reg.names()
    assert reg.is_allowed("echo")

