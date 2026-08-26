"""Tests for E2 - Durable checkpointing and crash recovery.

Covers the :class:`Checkpoint` model, the :class:`CheckpointStore` interface
and its concrete backends, and the crash-recovery integration in
:class:`AgentRuntime` (checkpoint-after-steps, ``resume``, resume locking,
and observability events).
"""

from __future__ import annotations

from typing import Any

import pytest

from research_engineer.observability import get_event_bus, reset_event_bus
from research_engineer.runtime import (
    AgentBudget,
    AgentContext,
    AgentPolicy,
    AgentRuntime,
    AgentState,
    AgentStep,
    AgentTermination,
    Checkpoint,
    CheckpointCorruptedError,
    CheckpointError,
    CheckpointLockError,
    CheckpointMetadata,
    CheckpointNotFoundError,
    CheckpointStore,
    CheckpointVersionError,
    CheckpointWriteError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from research_engineer.runtime.checkpoint import CHECKPOINT_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingSink:
    """Test sink that captures events into a list."""

    def __init__(self, store: list[dict]) -> None:
        self._store = store

    def emit(self, event: dict) -> None:
        self._store.append(event)

    def close(self) -> None:
        return None


async def _noop_planner(ctx: AgentContext) -> Any:
    return {"goal": ctx.goal}


async def _noop_actor(ctx: AgentContext, plan: Any) -> Any:
    return {"action": "done"}


async def _noop_observer(ctx: AgentContext, action: Any) -> Any:
    return action


async def _done_evaluator(ctx: AgentContext, observation: Any) -> Any:
    return {"done": True, "output": observation}


async def _never_done_evaluator(ctx: AgentContext, observation: Any) -> Any:
    return {"done": False}


def _runtime(
    *,
    store: CheckpointStore | None = None,
    planner: Any = None,
    actor: Any = None,
    observer: Any = None,
    evaluator: Any = None,
    policy: AgentPolicy | None = None,
    **kwargs: Any,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner or _noop_planner,
        actor=actor or _noop_actor,
        observer=observer or _noop_observer,
        evaluator=evaluator or _done_evaluator,
        policy=policy,
        checkpoint_store=store,
        **kwargs,
    )


def _context(goal: str = "goal", *, steps: int = 0) -> AgentContext:
    ctx = AgentContext(goal=goal)
    for i in range(steps):
        ctx.steps.append(AgentStep(step=i + 1))
        ctx.current_step += 1
    return ctx

# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------


class TestCheckpointModel:
    def test_from_context_derives_metadata(self) -> None:
        ctx = _context("my goal", steps=3)
        ctx.state = AgentState.RUNNING
        ctx.tool_calls = 7
        ctx.tokens = 100
        ctx.cost_usd = 0.5
        ctx.best_score = 0.9
        ctx.stagnation_count = 2
        ctx.recoverable_errors = 1

        cp = Checkpoint.from_context(ctx)

        assert cp.metadata.run_id == ctx.execution_id
        assert cp.metadata.goal == "my goal"
        assert cp.metadata.state == AgentState.RUNNING
        assert cp.metadata.step == 3
        assert cp.metadata.tool_calls == 7
        assert cp.metadata.tokens == 100
        assert cp.metadata.cost_usd == 0.5
        assert cp.metadata.best_score == 0.9
        assert cp.metadata.stagnation_count == 2
        assert cp.metadata.recoverable_errors == 1
        assert cp.metadata.schema_version == CHECKPOINT_SCHEMA_VERSION
        assert cp.context is ctx

    def test_serialization_round_trip(self) -> None:
        ctx = _context("round trip", steps=2)
        ctx.state = AgentState.TERMINATED
        ctx.termination = AgentTermination.SUCCESS
        ctx.termination_reason = "done"
        ctx.output = {"result": "ok"}
        cp = Checkpoint.from_context(ctx)

        data = cp.model_dump_json()
        restored = Checkpoint.model_validate_json(data)

        assert restored.metadata.run_id == ctx.execution_id
        assert restored.metadata.step == 2
        assert restored.metadata.state == AgentState.TERMINATED
        assert restored.metadata.termination == AgentTermination.SUCCESS
        assert restored.context.goal == "round trip"
        assert restored.context.current_step == 2
        assert len(restored.context.steps) == 2
        assert restored.context.output == {"result": "ok"}

    def test_metadata_serialization_round_trip(self) -> None:
        meta = CheckpointMetadata(
            run_id="exec_abc", goal="g", state=AgentState.RUNNING, step=4
        )
        restored = CheckpointMetadata.model_validate_json(meta.model_dump_json())
        assert restored.run_id == "exec_abc"
        assert restored.step == 4
        assert restored.state == AgentState.RUNNING


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


class TestInMemoryCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_load_round_trip(self) -> None:
        store = InMemoryCheckpointStore()
        cp = Checkpoint.from_context(_context("g", steps=1))

        await store.save(cp)
        loaded = await store.load(cp.metadata.run_id)

        assert loaded.metadata.run_id == cp.metadata.run_id
        assert loaded.context.current_step == 1
        assert loaded.context.goal == "g"

    @pytest.mark.asyncio
    async def test_exists_and_list(self) -> None:
        store = InMemoryCheckpointStore()
        cp = Checkpoint.from_context(_context("g", steps=1))
        assert not await store.exists(cp.metadata.run_id)

        await store.save(cp)
        assert await store.exists(cp.metadata.run_id)

        metas = await store.list()
        assert len(metas) == 1
        assert metas[0].run_id == cp.metadata.run_id
        assert metas[0].step == 1

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        store = InMemoryCheckpointStore()
        cp = Checkpoint.from_context(_context("g"))
        await store.save(cp)
        await store.delete(cp.metadata.run_id)
        assert not await store.exists(cp.metadata.run_id)

    @pytest.mark.asyncio
    async def test_load_missing_raises(self) -> None:
        store = InMemoryCheckpointStore()
        with pytest.raises(CheckpointNotFoundError):
            await store.load("exec_missing")

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self) -> None:
        store = InMemoryCheckpointStore()
        run_id = "exec_upsert"
        cp1 = Checkpoint.from_context(_context("g", steps=1))
        cp1.metadata.run_id = run_id
        cp1.context.execution_id = run_id
        cp2 = Checkpoint.from_context(_context("g", steps=2))
        cp2.metadata.run_id = run_id
        cp2.context.execution_id = run_id

        await store.save(cp1)
        await store.save(cp2)
        loaded = await store.load(run_id)
        assert loaded.context.current_step == 2

    @pytest.mark.asyncio
    async def test_lock_acquire_release(self) -> None:
        store = InMemoryCheckpointStore()
        assert await store.acquire_lock("exec_lock")
        assert not await store.acquire_lock("exec_lock")
        await store.release_lock("exec_lock")
        assert await store.acquire_lock("exec_lock")


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


class TestSQLiteCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_load_round_trip(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        cp = Checkpoint.from_context(_context("g", steps=2))

        await store.save(cp)
        loaded = await store.load(cp.metadata.run_id)

        assert loaded.metadata.run_id == cp.metadata.run_id
        assert loaded.context.current_step == 2
        assert loaded.context.goal == "g"
        await store.close()

    @pytest.mark.asyncio
    async def test_persists_across_instances(self, tmp_path) -> None:
        db = str(tmp_path / "ckpt.db")
        cp = Checkpoint.from_context(_context("g", steps=3))
        store1 = SQLiteCheckpointStore(db)
        await store1.save(cp)
        await store1.close()

        store2 = SQLiteCheckpointStore(db)
        loaded = await store2.load(cp.metadata.run_id)
        assert loaded.context.current_step == 3
        await store2.close()

    @pytest.mark.asyncio
    async def test_load_missing_raises(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        with pytest.raises(CheckpointNotFoundError):
            await store.load("exec_missing")
        await store.close()

    @pytest.mark.asyncio
    async def test_lock_acquire_release(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        assert await store.acquire_lock("exec_lock")
        assert not await store.acquire_lock("exec_lock")
        await store.release_lock("exec_lock")
        assert await store.acquire_lock("exec_lock")
        await store.close()

    @pytest.mark.asyncio
    async def test_corrupted_payload_raises(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        cp = Checkpoint.from_context(_context("g"))
        await store.save(cp)
        # Corrupt the stored payload directly.
        conn = store._ensure()
        conn.execute(
            "UPDATE checkpoints SET payload_json = ? WHERE run_id = ?",
            ("{not valid json", cp.metadata.run_id),
        )
        conn.commit()
        with pytest.raises(CheckpointCorruptedError):
            await store.load(cp.metadata.run_id)
        await store.close()

    @pytest.mark.asyncio
    async def test_version_mismatch_raises(self, tmp_path) -> None:
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        cp = Checkpoint.from_context(_context("g"))
        await store.save(cp)
        # Bump the schema version inside the stored JSON payload.
        cp.metadata.schema_version = CHECKPOINT_SCHEMA_VERSION + 1
        conn = store._ensure()
        conn.execute(
            "UPDATE checkpoints SET payload_json = ? WHERE run_id = ?",
            (cp.model_dump_json(), cp.metadata.run_id),
        )
        conn.commit()
        with pytest.raises(CheckpointVersionError):
            await store.load(cp.metadata.run_id)
        await store.close()


# ---------------------------------------------------------------------------
# Runtime integration: checkpointing
# ---------------------------------------------------------------------------


class TestRuntimeCheckpointing:
    @pytest.mark.asyncio
    async def test_checkpoint_after_each_step(self) -> None:
        store = InMemoryCheckpointStore()
        steps_run: list[int] = []

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            steps_run.append(ctx.current_step)
            return {"action": "done"}

        # Run 3 steps then stop via budget.
        runtime = _runtime(
            store=store,
            actor=actor,
            evaluator=_never_done_evaluator,
            policy=AgentPolicy(budget=AgentBudget(max_steps=3)),
        )
        result = await runtime.run("g")
        assert result.termination == AgentTermination.BUDGET_EXCEEDED
        assert result.context.current_step == 3
        assert steps_run == [0, 1, 2]

        # A checkpoint should exist reflecting the final state.
        assert await store.exists(result.context.execution_id)
        loaded = await store.load(result.context.execution_id)
        assert loaded.context.current_step == 3
        assert loaded.metadata.state == AgentState.TERMINATED
        assert loaded.metadata.termination == AgentTermination.BUDGET_EXCEEDED

    @pytest.mark.asyncio
    async def test_checkpoint_on_success(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        assert result.termination == AgentTermination.SUCCESS
        loaded = await store.load(result.context.execution_id)
        assert loaded.metadata.state == AgentState.TERMINATED
        assert loaded.metadata.termination == AgentTermination.SUCCESS

    @pytest.mark.asyncio
    async def test_no_checkpoint_without_store(self) -> None:
        runtime = _runtime(store=None)
        result = await runtime.run("g")
        assert result.termination == AgentTermination.SUCCESS
        assert result.context.current_step == 1

    @pytest.mark.asyncio
    async def test_checkpoint_failure_is_best_effort(self) -> None:
        class _FailingStore(InMemoryCheckpointStore):
            async def save(self, checkpoint: Checkpoint) -> None:
                raise CheckpointWriteError("disk full")

        store = _FailingStore()
        runtime = _runtime(store=store)
        # Checkpoint failures must not break the run.
        result = await runtime.run("g")
        assert result.termination == AgentTermination.SUCCESS


# ---------------------------------------------------------------------------
# Runtime integration: resume / crash recovery
# ---------------------------------------------------------------------------


class TestRuntimeResume:
    @pytest.mark.asyncio
    async def test_resume_returns_restored_context(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        run_id = result.context.execution_id

        ctx = await runtime.resume(run_id)
        assert ctx.execution_id == run_id
        assert ctx.goal == "g"
        assert ctx.current_step == result.context.current_step

    @pytest.mark.asyncio
    async def test_resume_missing_raises(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        with pytest.raises(CheckpointNotFoundError):
            await runtime.resume("exec_missing")

    @pytest.mark.asyncio
    async def test_resume_without_store_raises(self) -> None:
        runtime = _runtime(store=None)
        with pytest.raises(CheckpointError):
            await runtime.resume("exec_any")

    @pytest.mark.asyncio
    async def test_resume_after_crash_no_duplicate_steps(self) -> None:
        """Simulate a crash mid-run and verify continuation without rework."""
        store = InMemoryCheckpointStore()
        steps_run: list[int] = []

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            steps_run.append(ctx.current_step)
            return {"action": "done"}

        # First run: crash after 2 steps by raising a fatal error.
        async def crash_evaluator(ctx: AgentContext, observation: Any) -> Any:
            if ctx.current_step >= 2:
                raise RuntimeError("simulated crash")
            return {"done": False}

        runtime = _runtime(
            store=store,
            actor=actor,
            evaluator=crash_evaluator,
            policy=AgentPolicy(budget=AgentBudget(max_steps=5)),
        )
        result = await runtime.run("g")
        assert result.termination == AgentTermination.ERROR
        # The crash happened during step 3 (current_step 2), so the step
        # counter advanced to 3 but the step was not completed.
        assert result.context.current_step == 3
        assert steps_run == [0, 1, 2]

        # Resume: continue from the last *completed* checkpoint (step 2).
        ctx = await runtime.resume(result.context.execution_id)
        assert ctx.current_step == 2

        resumed = _runtime(
            store=store,
            actor=actor,
            evaluator=_done_evaluator,
            policy=AgentPolicy(budget=AgentBudget(max_steps=5)),
        )
        final = await resumed.run("g", context=ctx)
        assert final.termination == AgentTermination.SUCCESS
        # Completed steps 0 and 1 were not repeated; the crashed step 2 was
        # re-run once on resume and completed the run.
        assert steps_run == [0, 1, 2, 2]
        assert final.context.current_step == 3

    @pytest.mark.asyncio
    async def test_resume_releases_lock_after_run(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        run_id = result.context.execution_id

        ctx = await runtime.resume(run_id)
        await runtime.run("g", context=ctx)
        # Lock should be released by the finally block.
        assert await store.acquire_lock(run_id)

    @pytest.mark.asyncio
    async def test_concurrent_resume_raises_lock_error(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        run_id = result.context.execution_id

        # First resumer acquires the lock.
        await runtime.resume(run_id)
        # Second resumer must fail.
        with pytest.raises(CheckpointLockError):
            await runtime.resume(run_id)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class TestCheckpointObservability:
    @pytest.mark.asyncio
    async def test_checkpoint_events_emitted(self) -> None:
        reset_event_bus()
        bus = get_event_bus()
        events: list[dict] = []
        bus.add_sink(_CapturingSink(events))

        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")

        kinds = [e["event"] for e in events if e.get("kind") == "agent_runtime"]
        assert "checkpoint" in kinds
        assert "terminate" in kinds
        assert "start" in kinds
        assert "end" in kinds
        assert result.termination == AgentTermination.SUCCESS
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_resume_event_emitted(self) -> None:
        reset_event_bus()
        bus = get_event_bus()
        events: list[dict] = []
        bus.add_sink(_CapturingSink(events))

        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        await runtime.resume(result.context.execution_id)

        kinds = [e["event"] for e in events if e.get("kind") == "agent_runtime"]
        assert "resume" in kinds
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_checkpoint_failed_event_emitted(self) -> None:
        reset_event_bus()
        bus = get_event_bus()
        events: list[dict] = []
        bus.add_sink(_CapturingSink(events))

        class _FailingStore(InMemoryCheckpointStore):
            async def save(self, checkpoint: Checkpoint) -> None:
                raise CheckpointWriteError("boom")

        runtime = _runtime(store=_FailingStore())
        await runtime.run("g")

        kinds = [e["event"] for e in events if e.get("kind") == "agent_runtime"]
        assert "checkpoint_failed" in kinds
        reset_event_bus()


# ---------------------------------------------------------------------------
# End-to-end: run -> checkpoint -> crash -> resume -> continue
# ---------------------------------------------------------------------------


class TestEndToEndCrashRecovery:
    @pytest.mark.asyncio
    async def test_full_crash_recovery_flow(self, tmp_path) -> None:
        """Run, checkpoint, simulate a crash, resume, and verify continuation."""
        store = SQLiteCheckpointStore(str(tmp_path / "ckpt.db"))
        steps_run: list[int] = []

        async def actor(ctx: AgentContext, plan: Any) -> Any:
            steps_run.append(ctx.current_step)
            return {"action": "done"}

        # Phase 1: run 2 steps then "crash" (fatal error).
        async def crash_evaluator(ctx: AgentContext, observation: Any) -> Any:
            if ctx.current_step >= 2:
                raise RuntimeError("process died")
            return {"done": False}

        runtime = _runtime(
            store=store,
            actor=actor,
            evaluator=crash_evaluator,
            policy=AgentPolicy(budget=AgentBudget(max_steps=6)),
        )
        result = await runtime.run("g")
        assert result.termination == AgentTermination.ERROR
        assert result.context.current_step == 3

        # The checkpoint is durable in SQLite, reflecting the last completed
        # step (2), not the crashed step.
        assert await store.exists(result.context.execution_id)
        loaded = await store.load(result.context.execution_id)
        assert loaded.context.current_step == 2

        # Phase 2: a fresh runtime resumes from the durable checkpoint.
        fresh = _runtime(
            store=store,
            actor=actor,
            evaluator=_done_evaluator,
            policy=AgentPolicy(budget=AgentBudget(max_steps=6)),
        )
        ctx = await fresh.resume(result.context.execution_id)
        assert ctx.current_step == 2

        final = await fresh.run("g", context=ctx)
        assert final.termination == AgentTermination.SUCCESS
        assert final.context.current_step == 3
        # Completed steps 0 and 1 were not repeated; the crashed step 2 was
        # re-run once on resume, then step 3 completed the run.
        assert steps_run == [0, 1, 2, 2]
        await store.close()

    @pytest.mark.asyncio
    async def test_resume_version_mismatch_raises(self) -> None:
        store = InMemoryCheckpointStore()
        runtime = _runtime(store=store)
        result = await runtime.run("g")
        run_id = result.context.execution_id

        # Corrupt the stored schema version.
        cp = await store.load(run_id)
        cp.metadata.schema_version = CHECKPOINT_SCHEMA_VERSION + 1
        await store.save(cp)

        with pytest.raises(CheckpointVersionError):
            await runtime.resume(run_id)

