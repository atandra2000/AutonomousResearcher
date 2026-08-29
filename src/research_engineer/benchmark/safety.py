"""Benchmark safety chain — E3 gateway + E5 safety controller assembly.

Every benchmark case executes behind the same enforcement a real
autonomous run gets: tool calls are dispatched through the E3
:class:`~research_engineer.gateway.gateway.ToolGateway` (workspace-confined,
approval-enforced) and evaluated by the E5
:class:`~research_engineer.safety.controller.SafetyController`. The
deterministic sandbox tool set (notes, probes) is registered on the
gateway, rooted at the case's sandbox directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_default_safety_chain(
    workspace: Path | str,
) -> tuple[Any, Any]:
    """Construct the production E3 gateway + E5 safety chain (P1 §1).

    The gateway sandbox is rooted at ``workspace`` (the only writable root
    agents get), approvals are enforced — autonomous runs cannot grant
    themselves permission for HIGH-risk tools — and the deterministic
    sandbox tool set is registered.
    """
    from research_engineer.benchmark.bench_agents import (
        register_sandbox_tools,
    )
    from research_engineer.gateway.gateway import ToolGateway
    from research_engineer.gateway.models import ToolGatewayConfig
    from research_engineer.safety.controller import SafetyController

    root = Path(workspace)
    gateway = ToolGateway(
        ToolGatewayConfig(workspace=[str(root.resolve())],
                          enforce_approval=True)
    )
    register_sandbox_tools(gateway, workspace=root / "sandbox")
    return gateway, SafetyController()


__all__ = ["build_default_safety_chain"]
