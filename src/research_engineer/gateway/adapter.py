"""E3 - Backward-compatible adapter for existing tools.

The :class:`ToolGatewayAdapter` wraps an existing
:class:`~research_engineer.tools.base.Tool` instance so it can be invoked
through the gateway without modification. It exposes the tool's ``execute``
and ``validate`` methods and a stable ``name`` derived from the tool's
class name.

This lets the gateway dispatch to the platform's existing 56+ tools
unchanged, satisfying the "backward-compatible adapter for existing tools"
requirement.
"""

from __future__ import annotations

from typing import Any

from research_engineer.tools.base import Tool


class ToolGatewayAdapter:
    """Adapt an existing :class:`Tool` for gateway dispatch.

    Args:
        tool: The existing tool instance to wrap.
        name: Optional stable name; defaults to the tool's class name.
    """

    def __init__(self, tool: Tool[Any, Any], name: str | None = None) -> None:
        self._tool = tool
        self.name = name or tool.__class__.__name__

    @property
    def tool(self) -> Tool[Any, Any]:
        """The wrapped tool instance."""
        return self._tool

    async def validate(self, input: Any) -> bool:
        """Delegate to the wrapped tool's ``validate``."""
        return await self._tool.validate(input)

    async def execute(self, input: Any) -> Any:
        """Delegate to the wrapped tool's ``execute``."""
        return await self._tool.execute(input)

    def __repr__(self) -> str:
        return f"<ToolGatewayAdapter name={self.name} tool={self._tool!r}>"


__all__ = ["ToolGatewayAdapter"]
