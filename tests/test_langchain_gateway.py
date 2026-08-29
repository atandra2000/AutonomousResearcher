"""Tests for the LangChain wrapper around the platform tool gateway."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from research_engineer.gateway import (
    RiskLevel,
    ToolGateway,
    ToolGatewayConfig,
    as_langchain_tool,
)
from research_engineer.tools.base import Tool


class _EchoInput(BaseModel):
    message: str


class _EchoTool(Tool[_EchoInput, dict[str, str]]):
    async def execute(self, input: _EchoInput) -> dict[str, str]:
        return {"echo": input.message}

    async def validate(self, input: _EchoInput) -> bool:
        return bool(input.message)


@pytest.mark.asyncio
async def test_langchain_tool_routes_arguments_through_gateway() -> None:
    gateway = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
    gateway.register_tool(_EchoTool(), name="echo", risk_level=RiskLevel.LOW)
    tool = as_langchain_tool(
        gateway,
        name="echo",
        description="Echo a safe string.",
        args_schema=_EchoInput,
        agent_name="ResearchGraph",
        run_id="run_123",
    )

    result = await tool.ainvoke({"message": "evidence"})

    assert result == {
        "ok": True,
        "status": "success",
        "failure_kind": None,
        "output": {"echo": "evidence"},
        "error": "",
    }


@pytest.mark.asyncio
async def test_langchain_tool_reports_gateway_policy_failures() -> None:
    gateway = ToolGateway(ToolGatewayConfig(workspace=["/tmp"]))
    tool = as_langchain_tool(
        gateway,
        name="not_registered",
        description="A deliberately unavailable tool.",
        args_schema=_EchoInput,
    )

    result: dict[str, Any] = await tool.ainvoke({"message": "blocked"})

    assert result["ok"] is False
    assert result["status"] == "unknown"
    assert result["failure_kind"] == "policy"
