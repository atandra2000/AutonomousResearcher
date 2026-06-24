"""Loop Storage Tool for Phase 9.

Persists research loops and iterations to SQLite. Mirrors the
EvaluationStorageTool pattern: a single Tool subclass that dispatches on
input type to store/query loops or iterations.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from research_engineer.models.loop import (
    IterationQueryInput,
    IterationQueryOutput,
    IterationStorageInput,
    IterationStorageOutput,
    LoopIteration,
    LoopQueryInput,
    LoopQueryOutput,
    LoopRecord,
    LoopStorageInput,
    LoopStorageOutput,
)
from research_engineer.tools.base import Tool, ToolError


class LoopStorageTool(
    Tool[
        LoopStorageInput
        | LoopQueryInput
        | IterationStorageInput
        | IterationQueryInput,
        LoopStorageOutput
        | LoopQueryOutput
        | IterationStorageOutput
        | IterationQueryOutput,
    ]
):
    """SQLite storage for research loops and iterations."""

    def __init__(self, db_path: str = "data/research_engineer.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS research_loops (
                loop_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                iteration_count INTEGER DEFAULT 0,
                best_metric_value REAL,
                primary_metric_name TEXT,
                stopping_condition TEXT,
                stopping_reason TEXT,
                memory_ids_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_loop_status "
            "ON research_loops(status)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS loop_iterations (
                iteration_id TEXT PRIMARY KEY,
                loop_id TEXT NOT NULL,
                iteration_number INTEGER NOT NULL,
                phase TEXT NOT NULL,
                paper_id TEXT,
                paper_title TEXT,
                plan_id TEXT,
                implementation_id TEXT,
                experiment_id TEXT,
                evaluation_id TEXT,
                metrics_json TEXT,
                primary_metric_name TEXT,
                primary_metric_value REAL,
                best_metric_value REAL,
                improvement REAL,
                decision TEXT,
                memory_ids_json TEXT,
                error TEXT,
                status TEXT NOT NULL,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_iter_loop "
            "ON loop_iterations(loop_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_iter_paper "
            "ON loop_iterations(paper_id)"
        )
        conn.commit()
        conn.close()

    async def validate(
        self,
        input: (
            LoopStorageInput
            | LoopQueryInput
            | IterationStorageInput
            | IterationQueryInput
        ),
    ) -> bool:
        if isinstance(input, LoopStorageInput):
            return input.loop is not None
        if isinstance(input, IterationStorageInput):
            return input.iteration is not None
        return True

    async def execute(
        self,
        input: (
            LoopStorageInput
            | LoopQueryInput
            | IterationStorageInput
            | IterationQueryInput
        ),
    ) -> (
        LoopStorageOutput
        | LoopQueryOutput
        | IterationStorageOutput
        | IterationQueryOutput
    ):
        try:
            if isinstance(input, LoopStorageInput):
                return await self._store_loop(input)
            if isinstance(input, IterationStorageInput):
                return await self._store_iteration(input)
            if isinstance(input, LoopQueryInput):
                return await self._query_loops(input)
            return await self._query_iterations(input)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Loop storage failed: {e}", input, e)

    # --- Loop storage ---

    async def _store_loop(
        self, input: LoopStorageInput
    ) -> LoopStorageOutput:
        rec = input.loop
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO research_loops (
                    loop_id, goal, config_json, status, iteration_count,
                    best_metric_value, primary_metric_name,
                    stopping_condition, stopping_reason, memory_ids_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.loop_id,
                    rec.goal,
                    rec.config_json,
                    rec.status.value,
                    rec.iteration_count,
                    rec.best_metric_value,
                    rec.primary_metric_name,
                    rec.stopping_condition.value
                    if rec.stopping_condition
                    else None,
                    rec.stopping_reason,
                    json.dumps(rec.memory_ids),
                    rec.created_at.isoformat(),
                    rec.updated_at.isoformat()
                    if rec.updated_at
                    else None,
                ),
            )
            conn.commit()
            return LoopStorageOutput(
                loop_id=rec.loop_id,
                success=True,
                message=f"Loop {input.operation}d successfully",
            )
        except sqlite3.Error as e:
            return LoopStorageOutput(
                loop_id=rec.loop_id,
                success=False,
                message=f"Storage error: {e}",
            )
        finally:
            conn.close()

    async def _query_loops(
        self, input: LoopQueryInput
    ) -> LoopQueryOutput:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            query, params = self._build_loop_query(input)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            loops = [self._row_to_loop(row) for row in rows]

            count_query, count_params = self._build_loop_count_query(input)
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            return LoopQueryOutput(loops=loops, total=total)
        except sqlite3.Error as e:
            raise ToolError(f"Query failed: {e}", input, e)
        finally:
            conn.close()

    def _build_loop_query(
        self, input: LoopQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT * FROM research_loops WHERE 1=1"
        params: list[Any] = []
        query, params = self._apply_loop_filters(query, params, input)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([input.limit, input.offset])
        return query, params

    def _build_loop_count_query(
        self, input: LoopQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT COUNT(*) FROM research_loops WHERE 1=1"
        params: list[Any] = []
        return self._apply_loop_filters(query, params, input)

    def _apply_loop_filters(
        self,
        query: str,
        params: list[Any],
        input: LoopQueryInput,
    ) -> tuple[str, list[Any]]:
        if input.loop_id:
            query += " AND loop_id = ?"
            params.append(input.loop_id)
        if input.status:
            query += " AND status = ?"
            params.append(input.status.value)
        if input.search_text:
            query += " AND (goal LIKE ? OR stopping_reason LIKE ?)"
            like = f"%{input.search_text}%"
            params.extend([like, like])
        return query, params

    def _row_to_loop(self, row: tuple) -> LoopRecord:
        return LoopRecord(
            loop_id=row[0],
            goal=row[1],
            config_json=row[2],
            status=row[3],
            iteration_count=row[4] or 0,
            best_metric_value=row[5],
            primary_metric_name=row[6],
            stopping_condition=row[7] if row[7] else None,
            stopping_reason=row[8] or "",
            memory_ids=json.loads(row[9]) if row[9] else [],
            created_at=row[10],
            updated_at=row[11],
        )

    # --- Iteration storage ---

    async def _store_iteration(
        self, input: IterationStorageInput
    ) -> IterationStorageOutput:
        rec = input.iteration
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO loop_iterations (
                    iteration_id, loop_id, iteration_number, phase,
                    paper_id, paper_title, plan_id, implementation_id,
                    experiment_id, evaluation_id, metrics_json,
                    primary_metric_name, primary_metric_value,
                    best_metric_value, improvement, decision,
                    memory_ids_json, error, status, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.iteration_id,
                    rec.loop_id,
                    rec.iteration_number,
                    rec.phase.value,
                    rec.paper_id,
                    rec.paper_title,
                    rec.plan_id,
                    rec.implementation_id,
                    rec.experiment_id,
                    rec.evaluation_id,
                    json.dumps(rec.metrics),
                    rec.primary_metric_name,
                    rec.primary_metric_value,
                    rec.best_metric_value,
                    rec.improvement,
                    rec.decision.value if rec.decision else None,
                    json.dumps(rec.memory_ids),
                    rec.error,
                    rec.status.value,
                    rec.timestamp.isoformat()
                    if isinstance(rec.timestamp, datetime)
                    else rec.timestamp,
                ),
            )
            conn.commit()
            return IterationStorageOutput(
                iteration_id=rec.iteration_id,
                success=True,
                message=f"Iteration {input.operation}d successfully",
            )
        except sqlite3.Error as e:
            return IterationStorageOutput(
                iteration_id=rec.iteration_id,
                success=False,
                message=f"Storage error: {e}",
            )
        finally:
            conn.close()

    async def _query_iterations(
        self, input: IterationQueryInput
    ) -> IterationQueryOutput:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            query, params = self._build_iteration_query(input)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            iterations = [self._row_to_iteration(row) for row in rows]

            count_query, count_params = (
                self._build_iteration_count_query(input)
            )
            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

            return IterationQueryOutput(
                iterations=iterations, total=total
            )
        except sqlite3.Error as e:
            raise ToolError(f"Query failed: {e}", input, e)
        finally:
            conn.close()

    def _build_iteration_query(
        self, input: IterationQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT * FROM loop_iterations WHERE 1=1"
        params: list[Any] = []
        query, params = self._apply_iteration_filters(query, params, input)
        query += " ORDER BY iteration_number ASC LIMIT ? OFFSET ?"
        params.extend([input.limit, input.offset])
        return query, params

    def _build_iteration_count_query(
        self, input: IterationQueryInput
    ) -> tuple[str, list[Any]]:
        query = "SELECT COUNT(*) FROM loop_iterations WHERE 1=1"
        params: list[Any] = []
        return self._apply_iteration_filters(query, params, input)

    def _apply_iteration_filters(
        self,
        query: str,
        params: list[Any],
        input: IterationQueryInput,
    ) -> tuple[str, list[Any]]:
        if input.iteration_id:
            query += " AND iteration_id = ?"
            params.append(input.iteration_id)
        if input.loop_id:
            query += " AND loop_id = ?"
            params.append(input.loop_id)
        if input.paper_id:
            query += " AND paper_id = ?"
            params.append(input.paper_id)
        if input.status:
            query += " AND status = ?"
            params.append(input.status.value)
        if input.search_text:
            query += (
                " AND (error LIKE ? OR paper_title LIKE ? "
                "OR metrics_json LIKE ?)"
            )
            like = f"%{input.search_text}%"
            params.extend([like, like, like])
        return query, params

    def _row_to_iteration(self, row: tuple) -> LoopIteration:
        return LoopIteration(
            iteration_id=row[0],
            loop_id=row[1],
            iteration_number=row[2],
            phase=row[3],
            paper_id=row[4],
            paper_title=row[5],
            plan_id=row[6],
            implementation_id=row[7],
            experiment_id=row[8],
            evaluation_id=row[9],
            metrics=json.loads(row[10]) if row[10] else {},
            primary_metric_name=row[11],
            primary_metric_value=row[12],
            best_metric_value=row[13],
            improvement=row[14],
            decision=row[15] if row[15] else None,
            memory_ids=json.loads(row[16]) if row[16] else [],
            error=row[17],
            status=row[18],
            timestamp=row[19],
        )

    # --- Convenience getters ---

    async def get_loop_by_id(self, loop_id: str) -> LoopRecord | None:
        """Retrieve a single loop by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM research_loops WHERE loop_id = ?",
                (loop_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_loop(row)
        except sqlite3.Error as e:
            raise ToolError(f"Get loop by ID failed: {e}", None, e)
        finally:
            conn.close()

    async def get_iteration_by_id(
        self, iteration_id: str
    ) -> LoopIteration | None:
        """Retrieve a single iteration by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM loop_iterations WHERE iteration_id = ?",
                (iteration_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_iteration(row)
        except sqlite3.Error as e:
            raise ToolError(f"Get iteration by ID failed: {e}", None, e)
        finally:
            conn.close()
