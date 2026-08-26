"""E2 - Durable checkpointing and crash recovery.

Provides a persistence abstraction that lets an
:class:`~research_engineer.runtime.runtime.AgentRuntime` execution be
checkpointed and later resumed after a process/machine failure.

The public surface is:

* :class:`CheckpointMetadata` - a lightweight, human-readable description of
  a checkpoint (run id, state, step, budget usage, termination, timestamps,
  schema version).
* :class:`Checkpoint` - the full serializable runtime state: the metadata
  plus the complete :class:`~research_engineer.runtime.models.AgentContext`.
* :class:`CheckpointStore` - the abstract persistence interface. Storage is
  kept behind this interface so the backend can be swapped later without
  touching the runtime.
* A set of typed exceptions for the failure modes the runtime must handle
  (missing, corrupted, version-mismatched, concurrently-resumed).

Concrete stores live in :mod:`research_engineer.runtime.checkpoint_stores`:

* :class:`~research_engineer.runtime.checkpoint_stores.InMemoryCheckpointStore`
  - for tests and ephemeral use.
* :class:`~research_engineer.runtime.checkpoint_stores.SQLiteCheckpointStore`
  - the default production store, using the project's stdlib ``sqlite3``
  convention.
* :class:`~research_engineer.runtime.checkpoint_stores.PostgresCheckpointStore`
  - a production-ready PostgreSQL store that lazily imports ``psycopg``
  (optional dependency, mirroring the project's ChromaDB pattern) and reads
  connection settings from ``RE_``-prefixed environment variables.

Checkpoint versioning: :attr:`CheckpointMetadata.schema_version` records the
schema version a checkpoint was written with. On load, a version that is not
supported by the current code raises :class:`CheckpointVersionError` so the
schema can evolve safely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.runtime.models import (
    AgentContext,
    AgentState,
    AgentTermination,
)

#: The current checkpoint schema version. Bump this whenever the on-disk
#: representation changes in a way that older readers cannot understand.
CHECKPOINT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base class for all checkpoint-related errors."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when no checkpoint exists for a requested run id."""


class CheckpointCorruptedError(CheckpointError):
    """Raised when a stored checkpoint cannot be parsed/validated."""


class CheckpointVersionError(CheckpointError):
    """Raised when a checkpoint's schema version is not supported."""


class CheckpointLockError(CheckpointError):
    """Raised when a resume lock cannot be acquired (concurrent resume)."""


class CheckpointWriteError(CheckpointError):
    """Raised when persisting a checkpoint fails."""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CheckpointMetadata(BaseModel):
    """Lightweight description of a checkpoint.

    This is the "index" view of a checkpoint: enough to list, filter, and
    reason about checkpoints without loading the full runtime state. The full
    serializable state lives in :class:`Checkpoint.context`.
    """

    schema_version: int = Field(
        default=CHECKPOINT_SCHEMA_VERSION,
        description="Checkpoint schema version (for safe evolution)",
    )
    run_id: str = Field(..., description="Execution id of the run")
    goal: str = Field(..., description="High-level goal of the run")
    state: AgentState = Field(
        default=AgentState.CREATED, description="Lifecycle state at checkpoint"
    )
    step: int = Field(default=0, ge=0, description="Steps completed")
    tool_calls: int = Field(default=0, ge=0, description="Total tool calls made")
    tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Total USD cost")
    recoverable_errors: int = Field(
        default=0, ge=0, description="Recoverable errors tolerated so far"
    )
    stagnation_count: int = Field(
        default=0, ge=0, description="Consecutive steps without progress"
    )
    best_score: float | None = Field(
        default=None, description="Best evaluation score observed"
    )
    termination: AgentTermination | None = Field(
        default=None, description="Why the run terminated (when terminal)"
    )
    termination_reason: str = Field(
        default="", description="Human-readable termination reason"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the checkpoint was written"
    )


class Checkpoint(BaseModel):
    """A full, serializable snapshot of a runtime execution.

    ``metadata`` is the lightweight index view; ``context`` is the complete
    :class:`~research_engineer.runtime.models.AgentContext` that can be fed
    back into :meth:`AgentRuntime.run` to resume the run without repeating
    already-completed work.
    """

    metadata: CheckpointMetadata = Field(..., description="Checkpoint metadata")
    context: AgentContext = Field(..., description="Full serializable runtime state")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the checkpoint was written"
    )

    @classmethod
    def from_context(cls, context: AgentContext) -> Checkpoint:
        """Build a checkpoint from a runtime context.

        The metadata is derived from the context's current progress so the
        index view stays in sync with the full state.
        """
        return cls(
            metadata=CheckpointMetadata(
                run_id=context.execution_id,
                goal=context.goal,
                state=context.state,
                step=context.current_step,
                tool_calls=context.tool_calls,
                tokens=context.tokens,
                cost_usd=context.cost_usd,
                recoverable_errors=context.recoverable_errors,
                stagnation_count=context.stagnation_count,
                best_score=context.best_score,
                termination=context.termination,
                termination_reason=context.termination_reason,
            ),
            context=context,
        )


# ---------------------------------------------------------------------------
# Store interface
# ---------------------------------------------------------------------------


class CheckpointStore(ABC):
    """Abstract persistence interface for checkpoints.

    All operations are async so the interface can back both synchronous
    (SQLite) and asynchronous (PostgreSQL) drivers. Implementations must be
    safe to call from a single event loop.

    Concurrency: :meth:`acquire_lock` / :meth:`release_lock` provide
    optimistic mutual exclusion for resume. A store that cannot guarantee
    cross-process exclusion (e.g. the in-memory store) may implement a
    best-effort in-process lock.
    """

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist ``checkpoint``, keyed by ``checkpoint.metadata.run_id``.

        Saving an existing run id overwrites the previous checkpoint
        (idempotent upsert).
        """

    @abstractmethod
    async def load(self, run_id: str) -> Checkpoint:
        """Load the checkpoint for ``run_id``.

        Raises :class:`CheckpointNotFoundError` when absent,
        :class:`CheckpointCorruptedError` when unparseable, and
        :class:`CheckpointVersionError` on an unsupported schema version.
        """

    @abstractmethod
    async def delete(self, run_id: str) -> None:
        """Delete the checkpoint for ``run_id`` (no-op when absent)."""

    @abstractmethod
    async def exists(self, run_id: str) -> bool:
        """Return True when a checkpoint exists for ``run_id``."""

    @abstractmethod
    async def list(self) -> list[CheckpointMetadata]:
        """Return metadata for all stored checkpoints (unsorted)."""

    @abstractmethod
    async def acquire_lock(self, run_id: str) -> bool:
        """Try to acquire the resume lock for ``run_id``.

        Returns True on success, False when another resumer holds the lock.
        """

    @abstractmethod
    async def release_lock(self, run_id: str) -> None:
        """Release the resume lock for ``run_id`` (no-op when not held)."""

    async def close(self) -> None:
        """Release any held resources (default no-op)."""


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptedError",
    "CheckpointVersionError",
    "CheckpointLockError",
    "CheckpointWriteError",
    "CheckpointMetadata",
    "Checkpoint",
    "CheckpointStore",
]

