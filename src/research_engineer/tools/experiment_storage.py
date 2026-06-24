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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
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
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, paper_id, plan_id, patch_id,
                    implementation_id, repo_path, command_json,
                    experiment_type, status, start_time, end_time,
                    duration_seconds, exit_code, metrics_json,
                    failure_mode, failure_severity, root_cause,
                    output_dir, memory_id, tags, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.paper_id,
                    record.plan_id,
                    record.patch_id,
                    record.implementation_id,
                    record.repo_path,
                    json.dumps(record.command),
                    record.experiment_type.value,
                    record.status.value,
                    record.start_time.isoformat(),
                    record.end_time.isoformat() if record.end_time else None,
                    record.duration_seconds,
                    record.exit_code,
                    json.dumps(record.metrics),
                    record.failure_mode,
                    record.failure_severity.value,
                    record.root_cause,
                    record.output_dir,
                    record.memory_id,
                    json.dumps(record.tags),
                    record.notes,
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

    def _row_to_record(self, row: tuple) -> ExperimentRecord:
        """Convert a database row to an ExperimentRecord."""
        return ExperimentRecord(
            experiment_id=row[0],
            paper_id=row[1],
            plan_id=row[2],
            patch_id=row[3],
            implementation_id=row[4],
            repo_path=row[5],
            command=json.loads(row[6]) if row[6] else [],
            experiment_type=row[7],
            status=row[8],
            start_time=row[9],
            end_time=row[10],
            duration_seconds=row[11] or 0.0,
            exit_code=row[12],
            metrics=json.loads(row[13]) if row[13] else {},
            failure_mode=row[14],
            failure_severity=row[15] or "none",
            root_cause=row[16],
            output_dir=row[17],
            memory_id=row[18],
            tags=json.loads(row[19]) if row[19] else [],
            notes=row[20] or "",
            created_at=row[21],
            updated_at=row[22],
        )

    async def get_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve a single experiment by ID."""
        conn = sqlite3.connect(self.db_path)
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
