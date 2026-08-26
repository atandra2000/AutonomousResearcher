"""Artifact Collector Tool for Phase 7.

Discovers and catalogs artifacts produced by an experiment via glob
patterns. Optionally copies artifacts to an output directory with
SHA-256 checksums.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from research_engineer.models.experiment import (
    ArtifactCollectorInput,
    ArtifactCollectorOutput,
    ArtifactManifest,
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
            base = Path(input.working_dir)
            out_base = Path(input.output_dir)
            patterns = input.artifact_patterns or DEFAULT_ARTIFACT_PATTERNS
            artifacts: list[ExperimentArtifact] = []
            errors: list[str] = []
            total_bytes = 0
            dedup_count = 0
            seen_paths: set[str] = set()

            for pattern in patterns:
                arts, errs, b_added, d_added = self._collect_pattern(
                    base, out_base, pattern, input, seen_paths
                )
                artifacts.extend(arts)
                errors.extend(errs)
                total_bytes += b_added
                dedup_count += d_added

            total_mb = total_bytes / (1024 * 1024)
            manifest = ArtifactManifest(
                experiment_id=input.experiment_id,
                artifacts=artifacts,
                total_size_bytes=total_bytes,
                total_size_mb=round(total_mb, 2),
                dedup_count=dedup_count,
                version=1,
                output_dir=input.output_dir,
            )

            # Persist manifest sidecar
            manifest_file = out_base / "artifact_manifest.json"
            try:
                manifest_file.parent.mkdir(parents=True, exist_ok=True)
                manifest_file.write_text(
                    json.dumps(manifest.model_dump(), indent=2), encoding="utf-8"
                )
            except OSError as e:
                errors.append(f"Manifest write error: {e}")

            return ArtifactCollectorOutput(
                artifacts=artifacts,
                total_size_mb=round(total_mb, 2),
                output_dir=input.output_dir,
                collection_errors=errors,
                manifest=manifest,
                dedup_count=dedup_count,
            )
        except Exception as e:
            raise ToolError(f"Artifact collection failed: {e}", input, e)

    def _collect_pattern(
        self,
        base: Path,
        out_base: Path,
        pattern: ArtifactPattern,
        input: ArtifactCollectorInput,
        seen_paths: set[str],
    ) -> tuple[list[ExperimentArtifact], list[str], int, int]:
        artifacts: list[ExperimentArtifact] = []
        errors: list[str] = []
        total_bytes = 0
        dedup_count = 0

        matched = self._match_pattern(base, pattern)
        for path in matched[: pattern.max_files]:
            path_str = str(path.resolve())
            if path_str in seen_paths:
                continue

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

            seen_paths.add(path_str)
            stored_path: str | None = None
            checksum: str | None = None
            extra_meta: dict[str, str] = {"pattern": pattern.name}

            if input.copy_artifacts:
                stored_path, checksum, is_dedup, copy_err = self._copy_artifact(
                    path, out_base, pattern
                )
                if copy_err:
                    errors.append(copy_err)
                    continue
                if is_dedup:
                    dedup_count += 1
                    extra_meta["deduplicated"] = "true"
            else:
                checksum = self._checksum(path)

            total_bytes += size
            artifacts.append(
                ExperimentArtifact(
                    name=path.name,
                    original_path=str(path),
                    stored_path=stored_path,
                    artifact_type=pattern.artifact_type,
                    size_bytes=size,
                    checksum=checksum,
                    metadata=extra_meta,
                )
            )

        return artifacts, errors, total_bytes, dedup_count

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
    ) -> tuple[str | None, str | None, bool, str | None]:
        """Copy an artifact with CAS hash-based dedup and sidecar metadata generation.

        Returns (stored_path, checksum, is_deduplicated, error).
        """
        try:
            checksum = self._checksum(src)
            cas_dir = out_base / ".cas" / checksum[:2]
            cas_file = cas_dir / checksum
            is_dedup = False

            if cas_file.exists():
                is_dedup = True
            else:
                cas_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, cas_file)

            dest_dir = out_base / pattern.artifact_type.value
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if dest.exists() and dest != cas_file:
                dest = dest.with_name(
                    f"{src.stem}_{src.stat().st_mtime_ns}{src.suffix}"
                )

            shutil.copy2(cas_file, dest)

            # Generate sidecar metadata
            sidecar_file = dest.parent / f"{dest.name}.meta.json"
            sidecar_data = {
                "artifact_id": str(uuid4()),
                "name": src.name,
                "original_path": str(src),
                "stored_path": str(dest),
                "artifact_type": pattern.artifact_type.value,
                "size_bytes": dest.stat().st_size,
                "checksum": checksum,
                "cas_path": str(cas_file),
                "deduplicated": is_dedup,
                "created_at": datetime.datetime.now().isoformat(),
                "version": 1,
                "metadata": {"pattern": pattern.name},
            }
            sidecar_file.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")

            return str(dest), checksum, is_dedup, None
        except OSError as e:
            return None, None, False, f"Copy error for {src}: {e}"

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

    @staticmethod
    def retrieve_manifest(output_dir: str | Path) -> ArtifactManifest | None:
        """Retrieve artifact manifest from output directory."""
        p = Path(output_dir) / "artifact_manifest.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ArtifactManifest.model_validate(data)
        except Exception:
            return None

    @staticmethod
    def get_artifact(
        manifest: ArtifactManifest,
        artifact_id: str | None = None,
        name: str | None = None,
        checksum: str | None = None,
    ) -> ExperimentArtifact | None:
        """Get a specific artifact from a manifest by ID, name, or checksum."""
        for a in manifest.artifacts:
            if artifact_id and a.artifact_id == artifact_id:
                return a
            if name and a.name == name:
                return a
            if checksum and a.checksum == checksum:
                return a
        return None

    @staticmethod
    def list_artifacts(
        output_dir: str | Path,
        artifact_type: ArtifactType | str | None = None,
    ) -> list[ExperimentArtifact]:
        """List artifacts from stored manifest in output_dir with optional type filtering."""
        manifest = ArtifactCollectorTool.retrieve_manifest(output_dir)
        if not manifest:
            return []
        if not artifact_type:
            return manifest.artifacts
        type_val = (
            artifact_type.value
            if isinstance(artifact_type, ArtifactType)
            else str(artifact_type)
        )
        return [
            a
            for a in manifest.artifacts
            if a.artifact_type.value == type_val or a.artifact_type == artifact_type
        ]
