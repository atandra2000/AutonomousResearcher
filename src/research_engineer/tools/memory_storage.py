"""Memory storage tool for persistent knowledge storage."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.memory import (
    MemoryBase,
    MemoryFilters,
    MemoryRelationship,
    MemoryStats,
    MemoryType,
    MemoryVersion,
)
from research_engineer.tools.base import Tool, ToolError


class MemoryStorageInput(BaseModel):
    """Input for memory storage operations."""

    memory: MemoryBase = Field(..., description="Memory to store")
    operation: str = Field("store", description="Operation type (store, update, delete)")
    version_comment: str = Field("", description="Comment for versioning")


class MemoryStorageOutput(BaseModel):
    """Output from memory storage operations."""

    memory_id: str = Field(..., description="Memory ID")
    operation: str = Field(..., description="Operation performed")
    success: bool = Field(..., description="Whether operation succeeded")
    version: int | None = Field(None, description="Version number if applicable")
    message: str = Field(..., description="Status message")


class MemoryRelationshipInput(BaseModel):
    """Input for relationship operations."""

    relationship: MemoryRelationship = Field(..., description="Relationship to store")


class MemoryRelationshipOutput(BaseModel):
    """Output from relationship operations."""

    relationship_id: str = Field(..., description="Relationship ID")
    success: bool = Field(..., description="Whether operation succeeded")
    message: str = Field(..., description="Status message")


class MemoryQueryInput(BaseModel):
    """Input for memory queries."""

    memory_id: str | None = Field(None, description="Specific memory ID")
    memory_type: MemoryType | None = Field(None, description="Filter by type")
    filters: MemoryFilters | None = Field(None, description="Additional filters")
    limit: int = Field(100, description="Maximum results to return")
    offset: int = Field(0, description="Pagination offset")


class MemoryQueryOutput(BaseModel):
    """Output from memory queries."""

    memories: list[dict] = Field(default_factory=list, description="Retrieved memories")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")


class MemoryStorageTool(Tool[MemoryStorageInput | MemoryQueryInput, MemoryStorageOutput | MemoryQueryOutput]):
    """SQLite storage for memory system."""

    def __init__(self, db_path: str = "data/research_engineer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create memory tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                embedding_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                accessed_count INTEGER DEFAULT 0,
                last_accessed_at TIMESTAMP,
                tags TEXT,
                confidence_score REAL,
                is_archived BOOLEAN DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validated BOOLEAN DEFAULT 0,
                FOREIGN KEY (source_memory_id) REFERENCES memories(memory_id),
                FOREIGN KEY (target_memory_id) REFERENCES memories(memory_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                access_type TEXT NOT NULL,
                accessed_by TEXT,
                context TEXT,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_versions (
                version_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                content_json TEXT NOT NULL,
                change_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories(tags)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON memory_relationships(source_memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON memory_relationships(target_memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON memory_relationships(relationship_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_access_log_memory ON memory_access_log(memory_id)")

        conn.commit()
        conn.close()

    async def validate(self, input: MemoryStorageInput | MemoryQueryInput) -> bool:
        """Validate input."""
        if isinstance(input, MemoryStorageInput):
            return bool(input.memory and input.memory.memory_id)
        elif isinstance(input, MemoryQueryInput):
            return True
        return False

    async def execute(self, input: MemoryStorageInput | MemoryQueryInput) -> MemoryStorageOutput | MemoryQueryOutput:
        """Execute storage operation."""
        if isinstance(input, MemoryStorageInput):
            return await self._execute_storage(input)
        elif isinstance(input, MemoryQueryInput):
            return await self._execute_query(input)
        else:
            raise ToolError(f"Unknown input type: {type(input)}", input, None)

    async def _execute_storage(self, input: MemoryStorageInput) -> MemoryStorageOutput:
        """Execute storage operation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if input.operation == "store":
                result = await self._store_memory(cursor, input)
            elif input.operation == "update":
                result = await self._update_memory(cursor, input)
            elif input.operation == "delete":
                result = await self._delete_memory(cursor, input)
            else:
                raise ToolError(f"Unknown operation: {input.operation}", input, None)

            conn.commit()
            conn.close()
            return result

        except Exception as e:
            raise ToolError(f"Failed to execute storage operation: {e}", input, e)

    async def _store_memory(self, cursor: sqlite3.Cursor, input: MemoryStorageInput) -> MemoryStorageOutput:
        """Store a new memory."""
        memory = input.memory
        content_json = json.dumps(memory.to_dict(), default=str)
        tags_json = json.dumps(memory.tags)

        cursor.execute("""
            INSERT INTO memories (
                memory_id, memory_type, content_json, embedding_key,
                tags, confidence_score, accessed_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id,
            memory.memory_type.value,
            content_json,
            memory.embedding_key,
            tags_json,
            memory.confidence_score,
            memory.accessed_count,
            memory.created_at
        ))

        if input.version_comment:
            await self._create_version(cursor, memory, input.version_comment)

        return MemoryStorageOutput(
            memory_id=memory.memory_id,
            operation="store",
            success=True,
            version=1 if input.version_comment else None,
            message="Memory stored successfully"
        )

    async def _update_memory(self, cursor: sqlite3.Cursor, input: MemoryStorageInput) -> MemoryStorageOutput:
        """Update an existing memory with versioning."""
        memory = input.memory

        content_json = json.dumps(memory.to_dict(), default=str)
        tags_json = json.dumps(memory.tags)

        cursor.execute("""
            UPDATE memories SET
                content_json = ?,
                tags = ?,
                confidence_score = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE memory_id = ?
        """, (content_json, tags_json, memory.confidence_score, memory.memory_id))

        if cursor.rowcount == 0:
            return MemoryStorageOutput(
                memory_id=memory.memory_id,
                operation="update",
                success=False,
                message="Memory not found"
            )

        if input.version_comment:
            version = await self._get_latest_version(cursor, memory.memory_id)
            await self._create_version(cursor, memory, input.version_comment, version + 1)

        return MemoryStorageOutput(
            memory_id=memory.memory_id,
            operation="update",
            success=True,
            version=version + 1 if input.version_comment else None,
            message="Memory updated successfully"
        )

    async def _delete_memory(self, cursor: sqlite3.Cursor, input: MemoryStorageInput) -> MemoryStorageOutput:
        """Soft delete a memory."""
        memory = input.memory

        cursor.execute("""
            UPDATE memories SET
                is_archived = 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE memory_id = ?
        """, (memory.memory_id,))

        if cursor.rowcount == 0:
            return MemoryStorageOutput(
                memory_id=memory.memory_id,
                operation="delete",
                success=False,
                message="Memory not found"
            )

        return MemoryStorageOutput(
            memory_id=memory.memory_id,
            operation="delete",
            success=True,
            message="Memory archived successfully"
        )

    async def _execute_query(self, input: MemoryQueryInput) -> MemoryQueryOutput:
        """Execute query operation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            memories = await self._query_memories(cursor, input)
            total = await self._count_memories(cursor, input)

            conn.close()

            return MemoryQueryOutput(
                memories=memories,
                total=total,
                limit=input.limit,
                offset=input.offset
            )

        except Exception as e:
            raise ToolError(f"Failed to execute query: {e}", input, e)

    async def _query_memories(self, cursor: sqlite3.Cursor, input: MemoryQueryInput) -> list[dict]:
        """Query memories with filters."""
        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if input.memory_type:
            query += " AND memory_type = ?"
            params.append(input.memory_type.value)

        if input.filters:
            if input.filters.memory_types:
                type_values = [t.value for t in input.filters.memory_types]
                placeholders = ",".join(["?"] * len(type_values))
                query += f" AND memory_type IN ({placeholders})"
                params.extend(type_values)

            if input.filters.tags:
                for tag in input.filters.tags:
                    query += " AND tags LIKE ?"
                    params.append(f"%{tag}%")

            if input.filters.min_confidence > 0:
                query += " AND confidence_score >= ?"
                params.append(input.filters.min_confidence)

            if input.filters.exclude_archived:
                query += " AND is_archived = 0"

            if input.filters.ids:
                id_placeholders = ",".join(["?"] * len(input.filters.ids))
                query += f" AND memory_id IN ({id_placeholders})"
                params.extend(input.filters.ids)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([input.limit, input.offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        memories = []
        for row in rows:
            memories.append(self._row_to_memory_dict(row))

        return memories

    async def _count_memories(self, cursor: sqlite3.Cursor, input: MemoryQueryInput) -> int:
        """Count memories matching filters."""
        query = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params = []

        if input.memory_type:
            query += " AND memory_type = ?"
            params.append(input.memory_type.value)

        if input.filters:
            if input.filters.memory_types:
                type_values = [t.value for t in input.filters.memory_types]
                placeholders = ",".join(["?"] * len(type_values))
                query += f" AND memory_type IN ({placeholders})"
                params.extend(type_values)

            if input.filters.exclude_archived:
                query += " AND is_archived = 0"

        cursor.execute(query, params)
        return cursor.fetchone()[0]

    def _row_to_memory_dict(self, row: tuple) -> dict:
        """Convert database row to memory dictionary."""
        return {
            "memory_id": row[0],
            "memory_type": row[1],
            "content_json": json.loads(row[2]),
            "embedding_key": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "accessed_count": row[6],
            "last_accessed_at": row[7],
            "tags": json.loads(row[8]) if row[8] else [],
            "confidence_score": row[9],
            "is_archived": bool(row[10])
        }

    async def _create_version(self, cursor: sqlite3.Cursor, memory: MemoryBase, comment: str, version_number: int = None):
        """Create a version entry for a memory."""
        if version_number is None:
            version_number = await self._get_latest_version(cursor, memory.memory_id) + 1

        content_json = json.dumps(memory.to_dict(), default=str)

        cursor.execute("""
            INSERT INTO memory_versions (
                version_id, memory_id, version_number, content_json, change_summary
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            str(Path(memory.memory_id).name),
            memory.memory_id,
            version_number,
            content_json,
            comment
        ))

    async def _get_latest_version(self, cursor: sqlite3.Cursor, memory_id: str) -> int:
        """Get the latest version number for a memory."""
        cursor.execute("""
            SELECT MAX(version_number) FROM memory_versions WHERE memory_id = ?
        """, (memory_id,))
        result = cursor.fetchone()[0]
        return result if result else 0

    async def store_relationship(self, relationship: MemoryRelationship) -> MemoryRelationshipOutput:
        """Store a relationship between memories."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metadata_json = json.dumps(relationship.metadata, default=str)

            cursor.execute("""
                INSERT INTO memory_relationships (
                    relationship_id, source_memory_id, target_memory_id,
                    relationship_type, confidence, metadata_json, validated
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.relationship_id,
                relationship.source_memory_id,
                relationship.target_memory_id,
                relationship.relationship_type.value,
                relationship.confidence,
                metadata_json,
                relationship.validated
            ))

            conn.commit()
            conn.close()

            return MemoryRelationshipOutput(
                relationship_id=relationship.relationship_id,
                success=True,
                message="Relationship stored successfully"
            )

        except Exception as e:
            raise ToolError(f"Failed to store relationship: {e}", relationship, e)

    async def get_relationships(self, memory_id: str, relationship_type: str | None = None) -> list[dict]:
        """Get relationships for a memory."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM memory_relationships WHERE source_memory_id = ? OR target_memory_id = ?"
            params = [memory_id, memory_id]

            if relationship_type:
                query += " AND relationship_type = ?"
                params.append(relationship_type)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            relationships = []
            for row in rows:
                relationships.append({
                    "relationship_id": row[0],
                    "source_memory_id": row[1],
                    "target_memory_id": row[2],
                    "relationship_type": row[3],
                    "confidence": row[4],
                    "metadata_json": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6],
                    "validated": bool(row[7])
                })

            return relationships

        except Exception as e:
            raise ToolError(f"Failed to get relationships: {e}", None, e)

    async def log_access(self, memory_id: str, access_type: str, accessed_by: str | None = None, context: str | None = None):
        """Log memory access."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO memory_access_log (
                    memory_id, access_type, accessed_by, context
                ) VALUES (?, ?, ?, ?)
            """, (memory_id, access_type, accessed_by, context))

            cursor.execute("""
                UPDATE memories SET
                    accessed_count = accessed_count + 1,
                    last_accessed_at = CURRENT_TIMESTAMP
                WHERE memory_id = ?
            """, (memory_id,))

            conn.commit()
            conn.close()

        except Exception as e:
            raise ToolError(f"Failed to log access: {e}", None, e)

    async def get_stats(self) -> MemoryStats:
        """Get memory storage statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM memories")
            total = cursor.fetchone()[0]

            cursor.execute("""
                SELECT memory_type, COUNT(*) 
                FROM memories 
                GROUP BY memory_type
            """)
            by_type = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT COUNT(*) FROM memory_relationships")
            relationships = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM memories WHERE is_archived = 1")
            archived = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(confidence_score) FROM memories")
            avg_confidence = cursor.fetchone()[0] or 0.0

            cursor.execute("""
                SELECT memory_id FROM memories 
                ORDER BY accessed_count DESC 
                LIMIT 10
            """)
            most_accessed = [row[0] for row in cursor.fetchall()]

            cursor.execute("""
                SELECT memory_id FROM memories 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            recent = [row[0] for row in cursor.fetchall()]

            cursor.execute("SELECT page_count * page_size / 1048576.0 FROM pragma_page_count(), pragma_page_size()")
            size_mb = cursor.fetchone()[0] or 0.0

            conn.close()

            return MemoryStats(
                total_memories=total,
                memories_by_type=by_type,
                total_relationships=relationships,
                archived_count=archived,
                avg_confidence=avg_confidence,
                most_accessed=most_accessed,
                recent_memories=recent,
                storage_size_mb=size_mb
            )

        except Exception as e:
            raise ToolError(f"Failed to get stats: {e}", None, e)

    async def get_memory_by_id(self, memory_id: str) -> dict | None:
        """Retrieve a specific memory by ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return self._row_to_memory_dict(row)
            return None

        except Exception as e:
            raise ToolError(f"Failed to retrieve memory: {e}", None, e)

    async def close(self):
        """Close database connection."""
        pass
