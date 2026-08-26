"""E3 - Sandbox enforcement for filesystem and network access.

The :class:`Sandbox` enforces the workspace/filesystem and network
restrictions declared by a tool's :class:`ToolPermission`. It is a
*policy-level* sandbox: it validates the paths and network intent a tool
declares in its input before the tool runs, and it refuses access that the
policy does not grant.

Security note: this is a policy boundary, not OS/container isolation. It
does not claim to sandbox a subprocess or prevent a malicious tool from
escaping. It enforces the *declared* access against the configured
workspace so that a tool cannot silently reach outside its granted scope.
"""

from __future__ import annotations

from pathlib import Path

from research_engineer.gateway.models import ToolPermission


class SandboxError(Exception):
    """Raised when a tool invocation violates a sandbox rule."""


class Sandbox:
    """Enforce filesystem/workspace and network restrictions.

    Args:
        workspace: Allowed filesystem roots (absolute paths). When empty,
            filesystem access is denied entirely.
        allow_network: Global network gate. Per-tool permission must also
            allow network access.
    """

    def __init__(
        self,
        workspace: list[str] | None = None,
        *,
        allow_network: bool = False,
    ) -> None:
        self._workspace = [Path(p).resolve() for p in (workspace or [])]
        self._allow_network = allow_network

    # ------------------------------------------------------------------
    # Workspace / filesystem
    # ------------------------------------------------------------------

    def workspace_roots(self) -> list[Path]:
        """Return the resolved workspace roots."""
        return list(self._workspace)

    def is_within_workspace(self, path: str | Path) -> bool:
        """True when ``path`` resolves inside a configured workspace root."""
        if not self._workspace:
            return False
        resolved = Path(path).resolve()
        return any(
            resolved == root or root in resolved.parents for root in self._workspace
        )

    def check_filesystem(
        self,
        permission: ToolPermission,
        paths: list[str | Path] | None = None,
    ) -> None:
        """Validate filesystem access against ``permission``.

        Raises :class:`SandboxError` when the permission does not grant
        filesystem access, when the tool is not workspace-confined and the
        workspace is empty, or when any provided path falls outside the
        workspace.
        """
        if not permission.filesystem:
            raise SandboxError("Tool is not permitted filesystem access")
        if permission.workspace_only and not self._workspace:
            raise SandboxError(
                "Tool requires workspace confinement but no workspace is configured"
            )
        if paths:
            for p in paths:
                if permission.workspace_only and not self.is_within_workspace(p):
                    raise SandboxError(
                        f"Path outside workspace: {p}"
                    )

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def check_network(self, permission: ToolPermission) -> None:
        """Validate network access against ``permission``.

        Raises :class:`SandboxError` when the permission does not grant
        network access or the global network gate is closed.
        """
        if not permission.network:
            raise SandboxError("Tool is not permitted network access")
        if not self._allow_network:
            raise SandboxError("Network access is disabled globally")


__all__ = ["Sandbox", "SandboxError"]
