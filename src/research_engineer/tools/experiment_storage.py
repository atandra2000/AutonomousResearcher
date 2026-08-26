"""Experiment Storage Tool for Phase 7.

Persists experiment records to SQLite and supports querying by ID,
paper, repo, status, type, and text search.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from research_engineer.models.experiment import (
    ArtifactManifest,
    ExperimentArtifact,
    ExperimentQueryInput,
    ExperimentQueryOutput,
    ExperimentRecord,
    ExperimentStorageInput,
    ExperimentStorageOutput,
)
from research_engineer.tools.base import Tool, ToolError


class ExperimentStorageTool(
    Tool[ExperimentStorageInput | ExperimentQueryInput, ExperimentStorageOutput | ExperimentQueryOutput]
):
    """SQLite storage for experiment records."""

    def __init__(self, db_path: str = "data/research_engineer.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                paper_id TEXT,
                plan_id TEXT,
                patch_id TEXT,
                implementation_id TEXT,
                repo_path TEXT NOT NULL,
                command_json TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration_seconds REAL,
                exit_code INTEGER,
                metrics_json TEXT,
                failure_mode TEXT,
                failure_severity TEXT,
                root_cause TEXT,
                output_dir TEXT,
                memory_id TEXT,
                tags TEXT,
                notes TEXT,
                artifact_manifest_json TEXT,
                artifacts_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        cursor.execute("PRAGMA table_info(experiments)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if "artifact_manifest_json" not in existing_cols:
            cursor.execute(
                "ALTER TABLE experiments ADD COLUMN artifact_manifest_json TEXT"
            )
        if "artifacts_json" not in existing_cols:
            cursor.execute(
                "ALTER TABLE experiments ADD COLUMN artifacts_json TEXT"
            )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_paper ON experiments(paper_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_repo ON experiments(repo_path)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments(experiment_type)"
        )
        conn.commit()
        conn.close()

    async def validate(self, input: ExperimentStorageInput | ExperimentQueryInput) -> bool:
        if isinstance(input, ExperimentStorageInput):
            return input.experiment is not None
        return True

    async def execute(
        self, input: ExperimentStorageInput | ExperimentQueryInput
    ) -> ExperimentStorageOutput | ExperimentQueryOutput:
        try:
            if isinstance(input, ExperimentStorageInput):
                return await self._store(input)
            return await self._query(input)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Experiment storage failed: {e}", input, e)

    async def _store(self, input: ExperimentStorageInput) -> ExperimentStorageOutput:
        record = input.experiment
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            manifest_json = (
                json.dumps(record.artifact_manifest.model_dump())
                if record.artifact_manifest
                else None
            )
            artifacts_json = json.dumps(
                [a.model_dump() for a in record.artifacts]
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, paper_id, plan_id, patch_id,
                    implementation_id, repo_path, command_json,
                    experiment_type, status, start_time, end_time,
                    duration_seconds, exit_code, metrics_json,
                    failure_mode, failure_severity, root_cause,
                    output_dir, memory_id, tags, notes, artifact_manifest_json,
                    artifacts_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.paper_id,
                    record.plan_id,
                    record.patch_id,
                    record.implementation_id,
                    record.repo_path,
                    json.dumps(record.command),
                    record.experiment_type.value
                    if hasattr(record.experiment_type, "value")
                    else str(record.experiment_type),
                    record.status.value
                    if hasattr(record.status, "value")
                    else str(record.status),
                    record.start_time.isoformat(),
                    record.end_time.isoformat() if record.end_time else None,
                    record.duration_seconds,
                    record.exit_code,
                    json.dumps(record.metrics),
                    record.failure_mode,
                    record.failure_severity.value
                    if hasattr(record.failure_severity, "value")
                    else str(record.failure_severity),
                    record.root_cause,
                    record.output_dir,
                    record.memory_id,
                    json.dumps(record.tags),
                    record.notes,
                    manifest_json,
                    artifacts_json,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat() if record.updated_at else None,
                ),
            )
            conn.commit()
            return ExperimentStorageOutput(
                experiment_id=record.experiment_id,
                success=True,
                message=f"Experiment {input.operation}d successfully",
            )
        except sqlite3.Error as e:
            return ExperimentStorageOutput(
                experiment_id=record.experiment_id,
                success=False,
                message=f"Storage error: {e}",
            )
        finally:
            conn.close()

    async def _query(self, input: ExperimentQueryInput) -> ExperimentQueryOutput:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            query, params = self._build_query(input)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            experiments = [self._row_to_record(row) for row in rows]

            count_query, count_params = self._build_count_query(input)
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            return ExperimentQueryOutput(experiments=experiments, total=total)
        except sqlite3.Error as e:
            raise ToolError(f"Query failed: {e}", input, e)
        finally:
            conn.close()

    def _build_query(self, input: ExperimentQueryInput) -> tuple[str, list[Any]]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params: list[Any] = []
        query, params = self._apply_filters(query, params, input)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([input.limit, input.offset])
        return query, params

    def _build_count_query(
        self, input: ExperimentQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT COUNT(*) FROM experiments WHERE 1=1"
        params: list[Any] = []
        return self._apply_filters(query, params, input)

    def _apply_filters(
        self, query: str, params: list[Any], input: ExperimentQueryInput
    ) -> tuple[str, list[Any]]:
        if input.experiment_id:
            query += " AND experiment_id = ?"
            params.append(input.experiment_id)
        if input.paper_id:
            query += " AND paper_id = ?"
            params.append(input.paper_id)
        if input.repo_path:
            query += " AND repo_path = ?"
            params.append(input.repo_path)
        if input.status:
            query += " AND status = ?"
            params.append(input.status.value)
        if input.experiment_type:
            query += " AND experiment_type = ?"
            params.append(input.experiment_type.value)
        if input.search_text:
            query += (
                " AND (command_json LIKE ? OR notes LIKE ? OR tags LIKE ?)"
            )
            like = f"%{input.search_text}%"
            params.extend([like, like, like])
        return query, params

    def _row_to_record(self, row: sqlite3.Row | tuple) -> ExperimentRecord:
        """Convert a database row to an ExperimentRecord."""
        if isinstance(row, sqlite3.Row):
            d = dict(row)
        else:
            keys = [
                "experiment_id",
                "paper_id",
                "plan_id",
                "patch_id",
                "implementation_id",
                "repo_path",
                "command_json",
                "experiment_type",
                "status",
                "start_time",
                "end_time",
                "duration_seconds",
                "exit_code",
                "metrics_json",
                "failure_mode",
                "failure_severity",
                "root_cause",
                "output_dir",
                "memory_id",
                "tags",
                "notes",
                "artifact_manifest_json",
                "artifacts_json",
                "created_at",
                "updated_at",
            ]
            d = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}

        manifest = None
        manifest_raw = d.get("artifact_manifest_json")
        if manifest_raw:
            try:
                manifest = ArtifactManifest.model_validate(json.loads(manifest_raw))
            except Exception:
                manifest = None

        artifacts = []
        artifacts_raw = d.get("artifacts_json")
        if artifacts_raw:
            try:
                artifacts = [
                    ExperimentArtifact.model_validate(a)
                    for a in json.loads(artifacts_raw)
                ]
            except Exception:
                pass
        elif manifest and manifest.artifacts:
            artifacts = manifest.artifacts

        return ExperimentRecord(
            experiment_id=d["experiment_id"],
            paper_id=d.get("paper_id"),
            plan_id=d.get("plan_id"),
            patch_id=d.get("patch_id"),
            implementation_id=d.get("implementation_id"),
            repo_path=d["repo_path"],
            command=json.loads(d["command_json"]) if d.get("command_json") else [],
            experiment_type=d["experiment_type"],
            status=d["status"],
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            duration_seconds=d.get("duration_seconds") or 0.0,
            exit_code=d.get("exit_code"),
            metrics=json.loads(d["metrics_json"]) if d.get("metrics_json") else {},
            failure_mode=d.get("failure_mode"),
            failure_severity=d.get("failure_severity") or "none",
            root_cause=d.get("root_cause"),
            output_dir=d.get("output_dir"),
            memory_id=d.get("memory_id"),
            tags=json.loads(d["tags"]) if d.get("tags") else [],
            notes=d.get("notes") or "",
            artifact_manifest=manifest,
            artifacts=artifacts,
            created_at=d["created_at"],
            updated_at=d.get("updated_at"),
        )

    async def get_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve a single experiment by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except sqlite3.Error as e:
            raise ToolError(f"Get by ID failed: {e}", None, e)
        finally:
            conn.close()
