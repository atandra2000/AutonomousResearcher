"""Storage tool for persistent data storage."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.paper import Paper
from research_engineer.models.plan import EngineeringReport
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class StorageInput(BaseModel):
    """Input for storage tool."""

    paper: Paper = Field(..., description="Paper to store")
    summary: ResearchSummary = Field(..., description="Research summary")
    plan: EngineeringReport = Field(..., description="Engineering report")


class StorageOutput(BaseModel):
    """Output from storage tool."""

    record_id: int = Field(..., description="Database record ID")
    paper_id: str = Field(..., description="Stored paper ID")
    stored_at: datetime = Field(..., description="Storage timestamp")
    message: str = Field(..., description="Status message")


class StorageTool(Tool[StorageInput, StorageOutput]):
    """SQLite storage for analyzed papers."""

    def __init__(self, db_path: str = "data/research_engineer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON papers(paper_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON papers(created_at)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT UNIQUE NOT NULL,
                paper_id TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                compatibility_json TEXT NOT NULL,
                implementation_plan_json TEXT NOT NULL,
                impact_json TEXT NOT NULL,
                experiment_matrix_json TEXT NOT NULL,
                validation_plan_json TEXT NOT NULL,
                risk_assessment_json TEXT NOT NULL,
                compute_estimate_json TEXT NOT NULL,
                result_prediction_json TEXT NOT NULL,
                engineering_report_md TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_id ON plans(plan_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_plan_paper_id ON plans(paper_id)")

        conn.commit()
        conn.close()

    async def validate(self, input: StorageInput) -> bool:
        """Validate storage input."""
        return bool(
            input.paper and
            input.paper.paper_id and
            input.summary and
            input.plan
        )

    async def execute(self, input: StorageInput) -> StorageOutput:
        """Store paper analysis results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Convert models to JSON with datetime serialization
            authors_json = json.dumps(
                [a.model_dump() for a in input.paper.authors],
                default=str
            )
            summary_data = input.summary.model_dump(mode="json")
            summary_json = json.dumps(summary_data, default=str)
            plan_data = input.plan.model_dump(mode="json")
            plan_json = json.dumps(plan_data, default=str)

            # Insert or update
            cursor.execute("""
                INSERT INTO papers (
                    paper_id, title, authors_json, summary_json, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    summary_json = excluded.summary_json,
                    plan_json = excluded.plan_json,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                input.paper.paper_id,
                input.paper.title,
                authors_json,
                summary_json,
                plan_json,
                datetime.now()
            ))

            record_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()

            return StorageOutput(
                record_id=record_id,
                paper_id=input.paper.paper_id,
                stored_at=datetime.now(),
                message="Paper stored successfully"
            )

        except Exception as e:
            raise ToolError(f"Failed to store paper: {e}", input, e)

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Retrieve a stored paper by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "id": row[0],
                    "paper_id": row[1],
                    "title": row[2],
                    "authors_json": json.loads(row[3]),
                    "summary_json": json.loads(row[4]),
                    "plan_json": json.loads(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7]
                }
            return None

        except Exception as e:
            raise ToolError(f"Failed to retrieve paper: {e}", None, e)

    async def list_papers(self, limit: int = 100, offset: int = 0) -> list:
        """List stored papers with pagination."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM papers ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = cursor.fetchall()
            conn.close()

            papers = []
            for row in rows:
                papers.append({
                    "id": row[0],
                    "paper_id": row[1],
                    "title": row[2],
                    "authors_json": json.loads(row[3]),
                    "summary_json": json.loads(row[4]),
                    "plan_json": json.loads(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7]
                })
            return papers

        except Exception as e:
            raise ToolError(f"Failed to list papers: {e}", None, e)

    async def search_papers(self, query: str) -> list:
        """Search papers by title or author."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            search_pattern = f"%{query}%"
            cursor.execute("""
                SELECT * FROM papers 
                WHERE title LIKE ? OR authors_json LIKE ?
                ORDER BY created_at DESC
            """, (search_pattern, search_pattern))

            rows = cursor.fetchall()
            conn.close()

            papers = []
            for row in rows:
                papers.append({
                    "id": row[0],
                    "paper_id": row[1],
                    "title": row[2],
                    "authors_json": json.loads(row[3]),
                    "summary_json": json.loads(row[4]),
                    "plan_json": json.loads(row[5]),
                    "created_at": row[6],
                    "updated_at": row[7]
                })
            return papers

        except Exception as e:
            raise ToolError(f"Failed to search papers: {e}", None, e)

    async def delete_paper(self, paper_id: str) -> bool:
        """Delete a paper by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()

            return deleted

        except Exception as e:
            raise ToolError(f"Failed to delete paper: {e}", None, e)

    async def close(self):
        """Close database connection."""
        pass


# Compatibility wrapper
StorageToolAlias = StorageTool


class SQLiteStorage(Tool[StorageInput, StorageOutput]):
    """SQLite storage for analyzed papers (compatible name)."""

    def __init__(self, db_path: str = "data/research_engineer.db"):
        self.storage = StorageTool(db_path=db_path)

    async def validate(self, input: StorageInput) -> bool:
        """Validate storage input."""
        return await self.storage.validate(input)

    async def execute(self, input: StorageInput) -> StorageOutput:
        """Execute storage operation."""
        return await self.storage.execute(input)

    async def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """Get paper by ID."""
        return await self.storage.get_paper(paper_id)

    async def list_papers(self, limit: int = 100, offset: int = 0) -> list:
        """List papers."""
        return await self.storage.list_papers(limit, offset)

    async def search_papers(self, query: str) -> list:
        """Search papers."""
        return await self.storage.search_papers(query)

    async def delete_paper(self, paper_id: str) -> bool:
        """Delete paper."""
        return await self.storage.delete_paper(paper_id)

    async def close(self):
        """Close resources."""
        await self.storage.close()
