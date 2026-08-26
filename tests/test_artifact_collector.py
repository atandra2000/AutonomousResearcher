"""Tests for Artifact Collection, CAS Deduplication, Sidecars, and Retrieval API (Workstream C3)."""

import json
from pathlib import Path

import pytest

from research_engineer.models.experiment import (
    ArtifactCollectorInput,
    ArtifactManifest,
    ArtifactType,
    ExperimentArtifact,
    ExperimentRecord,
    ExperimentStatus,
    ExperimentStorageInput,
    ExperimentType,
)
from research_engineer.tools.artifact_collector import ArtifactCollectorTool
from research_engineer.tools.experiment_storage import ExperimentStorageTool


class TestArtifactCollectorC3:
    @pytest.mark.asyncio
    async def test_cas_hash_deduplication(self, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        out_dir = tmp_path / "out"

        # Create two separate files with identical content
        content = b"identical_model_weights_bytes_12345"
        (work_dir / "model_v1.pt").write_bytes(content)
        (work_dir / "model_v2.pt").write_bytes(content)

        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp_dedup_1",
            output_dir=str(out_dir),
            working_dir=str(work_dir),
            copy_artifacts=True,
        )

        out = await tool.execute(inp)
        assert len(out.artifacts) == 2
        assert out.dedup_count == 1
        assert out.manifest is not None
        assert out.manifest.dedup_count == 1

        # Check CAS directory contains only 1 deduplicated hash file
        cas_dir = out_dir / ".cas"
        cas_files = list(cas_dir.rglob("*"))
        cas_files = [f for f in cas_files if f.is_file()]
        assert len(cas_files) == 1

    @pytest.mark.asyncio
    async def test_sidecar_metadata_generation(self, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        out_dir = tmp_path / "out"

        (work_dir / "train.log").write_text("Epoch 1: loss 0.5\nEpoch 2: loss 0.3")

        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp_sidecar_1",
            output_dir=str(out_dir),
            working_dir=str(work_dir),
            copy_artifacts=True,
        )

        out = await tool.execute(inp)
        assert len(out.artifacts) >= 1
        log_art = [a for a in out.artifacts if a.name == "train.log"][0]
        assert log_art.stored_path is not None

        # Check sidecar meta file
        meta_file = Path(log_art.stored_path).parent / "train.log.meta.json"
        assert meta_file.exists()
        meta_data = json.loads(meta_file.read_text())
        assert meta_data["name"] == "train.log"
        assert meta_data["artifact_type"] == "log"
        assert meta_data["checksum"] == log_art.checksum
        assert "cas_path" in meta_data
        assert meta_data["version"] == 1

    @pytest.mark.asyncio
    async def test_manifest_creation_and_retrieval_api(self, tmp_path):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        out_dir = tmp_path / "out"

        (work_dir / "ckpt.pt").write_bytes(b"checkpoint_data")
        (work_dir / "metrics.json").write_text('{"loss": 0.1}')

        tool = ArtifactCollectorTool()
        inp = ArtifactCollectorInput(
            experiment_id="exp_manifest_1",
            output_dir=str(out_dir),
            working_dir=str(work_dir),
            copy_artifacts=True,
        )

        out = await tool.execute(inp)
        assert out.manifest is not None

        manifest_file = out_dir / "artifact_manifest.json"
        assert manifest_file.exists()

        # Test retrieval API
        retrieved_manifest = ArtifactCollectorTool.retrieve_manifest(out_dir)
        assert retrieved_manifest is not None
        assert retrieved_manifest.experiment_id == "exp_manifest_1"
        assert len(retrieved_manifest.artifacts) == 2

        # Test get_artifact
        ckpt_art = ArtifactCollectorTool.get_artifact(retrieved_manifest, name="ckpt.pt")
        assert ckpt_art is not None
        assert ckpt_art.name == "ckpt.pt"

        # Test list_artifacts by type
        ckpts = ArtifactCollectorTool.list_artifacts(out_dir, artifact_type=ArtifactType.CHECKPOINT)
        assert len(ckpts) == 1
        assert ckpts[0].name == "ckpt.pt"

    @pytest.mark.asyncio
    async def test_sqlite_manifest_persistence(self, tmp_path):
        db_file = tmp_path / "test_storage.db"
        storage = ExperimentStorageTool(db_path=str(db_file))

        manifest = ArtifactManifest(
            experiment_id="exp_sqlite_1",
            artifacts=[
                ExperimentArtifact(
                    name="model.pt",
                    original_path="/path/model.pt",
                    artifact_type=ArtifactType.CHECKPOINT,
                    size_bytes=100,
                    checksum="abc123hash",
                )
            ],
            total_size_bytes=100,
            total_size_mb=0.0001,
            dedup_count=0,
            version=1,
        )

        from datetime import datetime
        record = ExperimentRecord(
            experiment_id="exp_sqlite_1",
            repo_path="/repo",
            command=["python", "train.py"],
            experiment_type=ExperimentType.TRAINING,
            status=ExperimentStatus.COMPLETED,
            start_time=datetime.now(),
            artifacts=manifest.artifacts,
            artifact_manifest=manifest,
        )

        store_out = await storage.execute(ExperimentStorageInput(experiment=record))
        assert store_out.success

        fetched = await storage.get_by_id("exp_sqlite_1")
        assert fetched is not None
        assert fetched.artifact_manifest is not None
        assert fetched.artifact_manifest.experiment_id == "exp_sqlite_1"
        assert len(fetched.artifact_manifest.artifacts) == 1
        assert fetched.artifact_manifest.artifacts[0].name == "model.pt"
