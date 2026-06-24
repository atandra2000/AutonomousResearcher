"""Artifact Collector Tool for Phase 7.

Discovers and catalogs artifacts produced by an experiment via glob
patterns. Optionally copies artifacts to an output directory with
SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from research_engineer.models.experiment import (
    ArtifactCollectorInput,
    ArtifactCollectorOutput,
    ArtifactPattern,
    ArtifactType,
    ExperimentArtifact,
)
from research_engineer.tools.base import Tool, ToolError

DEFAULT_ARTIFACT_PATTERNS: list[ArtifactPattern] = [
    ArtifactPattern(
        name="checkpoints",
        glob_pattern="**/*.pt",
        artifact_type=ArtifactType.CHECKPOINT,
        max_files=10,
    ),
    ArtifactPattern(
        name="checkpoints_ckpt",
        glob_pattern="**/*.ckpt",
        artifact_type=ArtifactType.CHECKPOINT,
        max_files=10,
    ),
    ArtifactPattern(
        name="checkpoints_safetensors",
        glob_pattern="**/*.safetensors",
        artifact_type=ArtifactType.CHECKPOINT,
        max_files=10,
    ),
    ArtifactPattern(
        name="logs",
        glob_pattern="**/*.log",
        artifact_type=ArtifactType.LOG,
        max_files=20,
    ),
    ArtifactPattern(
        name="tensorboard",
        glob_pattern="**/events.out.tfevents.*",
        artifact_type=ArtifactType.METRIC_FILE,
        max_files=5,
    ),
    ArtifactPattern(
        name="metrics_json",
        glob_pattern="**/metrics*.json",
        artifact_type=ArtifactType.METRIC_FILE,
        max_files=10,
    ),
    ArtifactPattern(
        name="metrics_csv",
        glob_pattern="**/metrics*.csv",
        artifact_type=ArtifactType.METRIC_FILE,
        max_files=10,
    ),
    ArtifactPattern(
        name="plots_png",
        glob_pattern="**/*.png",
        artifact_type=ArtifactType.PLOT,
        max_files=20,
    ),
    ArtifactPattern(
        name="plots_jpg",
        glob_pattern="**/*.jpg",
        artifact_type=ArtifactType.PLOT,
        max_files=20,
    ),
    ArtifactPattern(
        name="plots_svg",
        glob_pattern="**/*.svg",
        artifact_type=ArtifactType.PLOT,
        max_files=20,
    ),
    ArtifactPattern(
        name="plots_pdf",
        glob_pattern="**/*.pdf",
        artifact_type=ArtifactType.PLOT,
        max_files=20,
    ),
    ArtifactPattern(
        name="configs_yaml",
        glob_pattern="**/*.yaml",
        artifact_type=ArtifactType.CONFIG,
        max_files=10,
    ),
    ArtifactPattern(
        name="configs_yml",
        glob_pattern="**/*.yml",
        artifact_type=ArtifactType.CONFIG,
        max_files=10,
    ),
    ArtifactPattern(
        name="configs_json",
        glob_pattern="**/*.json",
        artifact_type=ArtifactType.CONFIG,
        max_files=10,
    ),
    ArtifactPattern(
        name="configs_toml",
        glob_pattern="**/*.toml",
        artifact_type=ArtifactType.CONFIG,
        max_files=10,
    ),
]


class ArtifactCollectorTool(Tool[ArtifactCollectorInput, ArtifactCollectorOutput]):
    """Collect and catalog experiment artifacts."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ArtifactCollectorInput) -> bool:
        return bool(input.experiment_id) and bool(input.output_dir)

    async def execute(self, input: ArtifactCollectorInput) -> ArtifactCollectorOutput:
        try:
            patterns = input.artifact_patterns or DEFAULT_ARTIFACT_PATTERNS
            base = Path(input.working_dir)
            out_base = Path(input.output_dir) / "artifacts"
            artifacts: list[ExperimentArtifact] = []
            errors: list[str] = []
            total_bytes = 0

            for pattern in patterns:
                matched = self._match_pattern(base, pattern)
                for path in matched[: pattern.max_files]:
                    try:
                        size = path.stat().st_size
                    except OSError as e:
                        errors.append(f"Stat error for {path}: {e}")
                        continue
                    size_mb = size / (1024 * 1024)
                    if size_mb > input.max_artifact_size_mb:
                        errors.append(
                            f"Skipping {path}: {size_mb:.1f}MB exceeds "
                            f"limit {input.max_artifact_size_mb}MB"
                        )
                        continue

                    stored_path: str | None = None
                    checksum: str | None = None
                    if input.copy_artifacts:
                        stored_path, checksum, copy_err = self._copy_artifact(
                            path, out_base, pattern
                        )
                        if copy_err:
                            errors.append(copy_err)
                            continue

                    total_bytes += size
                    artifacts.append(
                        ExperimentArtifact(
                            name=path.name,
                            original_path=str(path),
                            stored_path=stored_path,
                            artifact_type=pattern.artifact_type,
                            size_bytes=size,
                            checksum=checksum,
                            metadata={"pattern": pattern.name},
                        )
                    )

            total_mb = total_bytes / (1024 * 1024)
            return ArtifactCollectorOutput(
                artifacts=artifacts,
                total_size_mb=round(total_mb, 2),
                output_dir=input.output_dir,
                collection_errors=errors,
            )
        except Exception as e:
            raise ToolError(f"Artifact collection failed: {e}", input, e)

    def _match_pattern(
        self, base: Path, pattern: ArtifactPattern
    ) -> list[Path]:
        """Match files in base directory using a glob pattern."""
        if not base.exists():
            return []
        try:
            return sorted(base.glob(pattern.glob_pattern))
        except Exception:
            return []

    def _copy_artifact(
        self, src: Path, out_base: Path, pattern: ArtifactPattern
    ) -> tuple[str | None, str | None, str | None]:
        """Copy an artifact to the output directory. Returns (path, checksum, error)."""
        try:
            dest_dir = out_base / pattern.artifact_type.value
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            # Avoid name collisions
            if dest.exists():
                dest = dest.with_name(
                    f"{src.stem}_{src.stat().st_mtime_ns}{src.suffix}"
                )
            shutil.copy2(src, dest)
            checksum = self._checksum(dest)
            return str(dest), checksum, None
        except OSError as e:
            return None, None, f"Copy error for {src}: {e}"

    @staticmethod
    def _checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            pass
        return h.hexdigest()
