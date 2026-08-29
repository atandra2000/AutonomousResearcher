"""LangChain tools that preserve the platform gateway as execution authority."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from research_engineer.gateway.gateway import ToolGateway


def as_langchain_tool(
    gateway: ToolGateway,
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    agent_name: str = "",
    run_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> StructuredTool:
    """Return a LangChain tool whose only execution path is ``gateway``.

    ``args_schema`` is intentionally explicit: a model may supply only
    fields accepted by the existing typed tool input.  The gateway then
    applies its policy, budget, approval, sandbox, timeout, and output cap.
    """
    invocation_metadata = {"framework": "langchain", **(metadata or {})}

    async def invoke(**arguments: Any) -> dict[str, Any]:
        tool_input = args_schema.model_validate(arguments)
        result = await gateway.execute(
            name,
            tool_input,
            agent_name=agent_name,
            run_id=run_id,
            metadata=invocation_metadata,
        )
        return {
            "ok": result.ok,
            "status": result.status.value,
            "failure_kind": (
                result.failure_kind.value if result.failure_kind is not None else None
            ),
            "output": result.output,
            "error": result.error,
        }

    return StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=description,
        args_schema=args_schema,
    )


__all__ = ["as_langchain_tool"]
