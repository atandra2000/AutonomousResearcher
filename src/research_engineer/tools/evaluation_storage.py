"""Evaluation Storage Tool for Phase 8.

Persists evaluation records to SQLite and supports querying by
evaluation_id, experiment_id (contained), paper_id, repo_path, and
text search. Mirrors the ExperimentStorageTool pattern.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from research_engineer.models.evaluation import (
    EvaluationQueryInput,
    EvaluationQueryOutput,
    EvaluationRecord,
    EvaluationStorageInput,
    EvaluationStorageOutput,
)
from research_engineer.tools.base import Tool, ToolError


class EvaluationStorageTool(
    Tool[EvaluationStorageInput | EvaluationQueryInput, EvaluationStorageOutput | EvaluationQueryOutput]
):
    """SQLite storage for evaluation records."""

    def __init__(self, db_path: str = "data/research_engineer.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                experiment_ids_json TEXT NOT NULL,
                paper_id TEXT,
                repo_path TEXT,
                comparison_json TEXT,
                dynamics_json TEXT,
                significance_json TEXT,
                next_experiments_json TEXT,
                summary TEXT,
                conclusions_json TEXT,
                memory_ids_json TEXT,
                tags_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_paper ON evaluations(paper_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_repo ON evaluations(repo_path)"
        )
        conn.commit()
        conn.close()

    async def validate(
        self, input: EvaluationStorageInput | EvaluationQueryInput
    ) -> bool:
        if isinstance(input, EvaluationStorageInput):
            return input.evaluation is not None
        return True

    async def execute(
        self, input: EvaluationStorageInput | EvaluationQueryInput
    ) -> EvaluationStorageOutput | EvaluationQueryOutput:
        try:
            if isinstance(input, EvaluationStorageInput):
                return await self._store(input)
            return await self._query(input)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Evaluation storage failed: {e}", input, e)

    async def _store(
        self, input: EvaluationStorageInput
    ) -> EvaluationStorageOutput:
        rec = input.evaluation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluations (
                    evaluation_id, experiment_ids_json, paper_id, repo_path,
                    comparison_json, dynamics_json, significance_json,
                    next_experiments_json, summary, conclusions_json,
                    memory_ids_json, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.evaluation_id,
                    json.dumps(rec.experiment_ids),
                    rec.paper_id,
                    rec.repo_path,
                    rec.comparison.model_dump_json()
                    if rec.comparison
                    else None,
                    json.dumps(
                        [d.model_dump(mode="json") for d in rec.dynamics]
                    ),
                    rec.significance.model_dump_json()
                    if rec.significance
                    else None,
                    rec.next_experiments.model_dump_json()
                    if rec.next_experiments
                    else None,
                    rec.summary,
                    json.dumps(rec.conclusions),
                    json.dumps(rec.memory_ids),
                    json.dumps(rec.tags),
                    rec.created_at.isoformat(),
                    rec.updated_at.isoformat()
                    if rec.updated_at
                    else None,
                ),
            )
            conn.commit()
            return EvaluationStorageOutput(
                evaluation_id=rec.evaluation_id,
                success=True,
                message=f"Evaluation {input.operation}d successfully",
            )
        except sqlite3.Error as e:
            return EvaluationStorageOutput(
                evaluation_id=rec.evaluation_id,
                success=False,
                message=f"Storage error: {e}",
            )
        finally:
            conn.close()

    async def _query(
        self, input: EvaluationQueryInput
    ) -> EvaluationQueryOutput:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            query, params = self._build_query(input)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            evaluations = [self._row_to_record(row) for row in rows]

            count_query, count_params = self._build_count_query(input)
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            return EvaluationQueryOutput(
                evaluations=evaluations, total=total
            )
        except sqlite3.Error as e:
            raise ToolError(f"Query failed: {e}", input, e)
        finally:
            conn.close()

    def _build_query(
        self, input: EvaluationQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT * FROM evaluations WHERE 1=1"
        params: list[Any] = []
        query, params = self._apply_filters(query, params, input)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([input.limit, input.offset])
        return query, params

    def _build_count_query(
        self, input: EvaluationQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT COUNT(*) FROM evaluations WHERE 1=1"
        params: list[Any] = []
        return self._apply_filters(query, params, input)

    def _apply_filters(
        self,
        query: str,
        params: list[Any],
        input: EvaluationQueryInput,
    ) -> tuple[str, list[Any]]:
        if input.evaluation_id:
            query += " AND evaluation_id = ?"
            params.append(input.evaluation_id)
        if input.paper_id:
            query += " AND paper_id = ?"
            params.append(input.paper_id)
        if input.repo_path:
            query += " AND repo_path = ?"
            params.append(input.repo_path)
        if input.experiment_id:
            query += " AND experiment_ids_json LIKE ?"
            params.append(f'%"{input.experiment_id}"%')
        if input.search_text:
            query += (
                " AND (summary LIKE ? OR conclusions_json LIKE ? "
                "OR tags_json LIKE ?)"
            )
            like = f"%{input.search_text}%"
            params.extend([like, like, like])
        return query, params

    def _row_to_record(self, row: tuple) -> EvaluationRecord:
        from research_engineer.models.evaluation import (
            ExperimentComparisonOutput,
            NextExperimentOutput,
            StatisticalSignificanceOutput,
            TrainingDynamicsOutput,
        )

        comparison = None
        if row[4]:
            comparison = ExperimentComparisonOutput.model_validate_json(
                row[4]
            )
        dynamics: list[TrainingDynamicsOutput] = []
        if row[5]:
            for item in json.loads(row[5]):
                dynamics.append(
                    TrainingDynamicsOutput.model_validate(item)
                )
        significance = None
        if row[6]:
            significance = StatisticalSignificanceOutput.model_validate_json(
                row[6]
            )
        next_experiments = None
        if row[7]:
            next_experiments = NextExperimentOutput.model_validate_json(
                row[7]
            )
        return EvaluationRecord(
            evaluation_id=row[0],
            experiment_ids=json.loads(row[1]) if row[1] else [],
            paper_id=row[2],
            repo_path=row[3],
            comparison=comparison,
            dynamics=dynamics,
            significance=significance,
            next_experiments=next_experiments,
            summary=row[8] or "",
            conclusions=json.loads(row[9]) if row[9] else [],
            memory_ids=json.loads(row[10]) if row[10] else [],
            tags=json.loads(row[11]) if row[11] else [],
            created_at=row[12],
            updated_at=row[13],
        )

    async def get_by_id(self, evaluation_id: str) -> EvaluationRecord | None:
        """Retrieve a single evaluation by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except sqlite3.Error as e:
            raise ToolError(f"Get by ID failed: {e}", None, e)
        finally:
            conn.close()
