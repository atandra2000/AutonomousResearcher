"""E7 - Artifact storage for agent runs.

Reuses the project's artifact conventions (SHA-256 checksums + manifest,
mirroring :mod:`research_engineer.tools.artifact_collector`) over a plain
filesystem volume instead of inventing a new storage layer. In the Docker
Compose stack the artifact root is a bind-mounted volume shared between API
and worker containers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from research_engineer.service.models import ArtifactReference


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore:
    """Filesystem artifact store rooted at a single directory."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _run_dir(self, run_id: str) -> Path:
        path = self._root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_text(
        self, run_id: str, name: str, content: str, content_type: str = "text/plain"
    ) -> ArtifactReference:
        data = content.encode("utf-8")
        ref = self.save_bytes(run_id, name, data, content_type)
        return ref

    def save_bytes(
        self,
        run_id: str,
        name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> ArtifactReference:
        directory = self._run_dir(run_id)
        target = directory / Path(name).name  # never escape the run dir
        target.write_bytes(data)
        return ArtifactReference(
            name=target.name,
            path=f"{run_id}/{target.name}",
            sha256=sha256_of(data),
            size_bytes=len(data),
            content_type=content_type,
        )

    def open(self, reference: ArtifactReference) -> bytes:
        """Read an artifact back; validates its checksum when present."""
        target = self._root / reference.path
        data = target.read_bytes()
        if reference.sha256 and sha256_of(data) != reference.sha256:
            raise ValueError(f"Artifact checksum mismatch for {reference.path}")
        return data

    def list_artifacts(self, run_id: str) -> list[ArtifactReference]:
        directory = self._root / run_id
        if not directory.is_dir():
            return []
        refs: list[ArtifactReference] = []
        for f in sorted(directory.iterdir()):
            if not f.is_file():
                continue
            data = f.read_bytes()
            refs.append(
                ArtifactReference(
                    name=f.name,
                    path=f"{run_id}/{f.name}",
                    sha256=sha256_of(data),
                    size_bytes=len(data),
                )
            )
        return refs

    def persist_result(
        self, run_id: str, result: dict[str, Any]
    ) -> ArtifactReference:
        """Persist the structured result JSON as an artifact."""
        payload = json.dumps(result, default=str, indent=2)
        return self.save_text(run_id, "result.json", payload,
                              content_type="application/json")

    def new_run_id(self) -> str:
        return uuid.uuid4().hex


def copy_into_store(
    store: ArtifactStore, run_id: str, source_dir: Path, patterns: list[str]
) -> list[ArtifactReference]:
    """Copy files matching glob patterns into the store (collector-style)."""
    refs: list[ArtifactReference] = []
    for pattern in patterns:
        for src in sorted(source_dir.glob(pattern)):
            if not src.is_file():
                continue
            target_name = src.name
            shutil.copy2(src, store.root / run_id / target_name)
            data = src.read_bytes()
            refs.append(
                ArtifactReference(
                    name=src.name,
                    path=f"{run_id}/{src.name}",
                    sha256=sha256_of(data),
                    size_bytes=len(data),
                )
            )
    return refs


__all__ = ["ArtifactStore", "copy_into_store", "sha256_of"]
