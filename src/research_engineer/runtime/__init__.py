"""E1 - Production Agent Runtime.

A generic, async-first orchestration layer for autonomous agents. The
:class:`~research_engineer.runtime.runtime.AgentRuntime` drives a
``plan -> act -> observe -> evaluate`` loop with budgets, termination,
cancellation, error recovery, and observability — while remaining
compatible with existing agents via
:class:`~research_engineer.runtime.adapters.AgentAdapter`.

Public surface::

    from research_engineer.runtime import (
        AgentRuntime,
        AgentAdapter,
        AgentState,
        AgentPhase,
        AgentTermination,
        AgentBudget,
        AgentPolicy,
        AgentError,
        AgentStep,
        AgentContext,
        AgentExecution,
        classify_error,
        Checkpoint,
        CheckpointMetadata,
        CheckpointStore,
        InMemoryCheckpointStore,
        SQLiteCheckpointStore,
        PostgresCheckpointStore,
        CheckpointError,
        CheckpointNotFoundError,
        CheckpointCorruptedError,
        CheckpointVersionError,
        CheckpointLockError,
        CheckpointWriteError,
    )
"""

from research_engineer.runtime.adapters import AgentAdapter
from research_engineer.runtime.checkpoint import (
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointLockError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointStore,
    CheckpointVersionError,
    CheckpointWriteError,
)
from research_engineer.runtime.checkpoint_stores import (
    InMemoryCheckpointStore,
    PostgresCheckpointStore,
    SQLiteCheckpointStore,
)
from research_engineer.runtime.models import (
    AgentBudget,
    AgentContext,
    AgentError,
    AgentExecution,
    AgentPhase,
    AgentPolicy,
    AgentState,
    AgentStep,
    AgentTermination,
)
from research_engineer.runtime.runtime import AgentRuntime, classify_error

__all__ = [
    "AgentRuntime",
    "AgentAdapter",
    "AgentState",
    "AgentPhase",
    "AgentTermination",
    "AgentBudget",
    "AgentPolicy",
    "AgentError",
    "AgentStep",
    "AgentContext",
    "AgentExecution",
    "classify_error",
    "Checkpoint",
    "CheckpointMetadata",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "SQLiteCheckpointStore",
    "PostgresCheckpointStore",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointCorruptedError",
    "CheckpointVersionError",
    "CheckpointLockError",
    "CheckpointWriteError",
]
