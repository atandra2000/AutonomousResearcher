"""E7 tests — worker execution, persistence, recovery, concurrency, config."""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path

import pytest

from research_engineer.runtime.checkpoint_stores import InMemoryCheckpointStore
from research_engineer.service.agents import AgentFactoryRegistry
from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.config import load_service_config
from research_engineer.service.manager import ResumeNotAvailableError, RunManager
from research_engineer.service.models import CreateRunRequest, RunRecord, RunStatus
from research_engineer.service.queue import InMemoryQueue
from research_engineer.service.store import InvalidTransitionError, SQLiteRunStore
from research_engineer.service.telemetry import ServiceTelemetry
from research_engineer.service.worker import AgentWorker


@pytest.fixture()
def tmp_store(tmp_path: Path) -> SQLiteRunStore:
    return SQLiteRunStore(tmp_path / "runs.db")


@pytest.fixture()
def env(tmp_path: Path):
    queue = InMemoryQueue()
    store = SQLiteRunStore(tmp_path / "runs.db")
    manager = RunManager(store, queue, ServiceTelemetry(),
                         stale_run_timeout_seconds=0.3)
    worker = AgentWorker(
        store=store,
        queue=queue,
        checkpoint_store=InMemoryCheckpointStore(),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        factories=AgentFactoryRegistry(config_max_steps=5),
        telemetry=ServiceTelemetry(),
        poll_interval_seconds=0.02,
        stale_run_timeout_seconds=0.3,
    )
    return store, queue, manager, worker


async def _drive(worker: AgentWorker, cycles: int = 8) -> None:
    for _ in range(cycles):
        await worker.try_claim_and_execute()
        await asyncio.sleep(0.02)
    await worker.drain(10)


class TestPersistence:
    @pytest.mark.asyncio
    async def test_run_record_round_trip(self, tmp_store) -> None:
        rec = RunRecord(run_id="run_x", goal="g", status=RunStatus.QUEUED)
        await tmp_store.create(rec)
        loaded = await tmp_store.get_required("run_x")
        assert loaded.goal == "g"
        assert loaded.status == RunStatus.QUEUED

    @pytest.mark.asyncio
    async def test_terminal_transition_guard(self, tmp_store) -> None:
        rec = RunRecord(run_id="run_y", goal="g", status=RunStatus.CANCELLED)
        await tmp_store.create(rec)
        with pytest.raises(InvalidTransitionError):
            rec.status = RunStatus.RUNNING
            await tmp_store.update(rec)

    @pytest.mark.asyncio
    async def test_status_counts(self, tmp_store) -> None:
        await tmp_store.create(
            RunRecord(run_id="a", goal="g", status=RunStatus.QUEUED)
        )
        counts = await tmp_store.count_by_status()
        assert counts["queued"] == 1


class TestWorkerExecution:
    @pytest.mark.asyncio
    async def test_full_lifecycle_completion(self, env) -> None:
        store, _queue, manager, worker = env
        created = await manager.submit(
            CreateRunRequest(goal="Step one. Step two.")
        )
        await _drive(worker)
        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.COMPLETED
        assert final.result is not None
        assert final.result["termination"] == "success"
        assert len(final.artifacts) == 1

    @pytest.mark.asyncio
    async def test_failing_factory_marks_failed(self, env) -> None:
        store, _queue, manager, worker = env

        async def boom(_overrides: dict) -> object:
            raise RuntimeError("factory exploded")

        worker.factories.register("boom", boom)
        created = await manager.submit(CreateRunRequest(goal="doomed."))
        record = await store.get_required(created.run_id)
        record.metadata["agent_kind"] = "boom"
        await store.update(record)
        await _drive(worker)
        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.FAILED
        assert final.error is not None
        assert "RuntimeError" in (final.error or "")

    @pytest.mark.asyncio
    async def test_cancellation_of_running_run(self, env) -> None:
        store, _queue, manager, worker = env
        long_goal = ". ".join(f"step {i}" for i in range(40)) + "."
        created = await manager.submit(CreateRunRequest(goal=long_goal))
        claimed = await worker.try_claim_and_execute()
        assert claimed == created.run_id
        cancelled, _rec = await manager.cancel(created.run_id)
        assert cancelled is True
        await worker.drain(30)
        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.CANCELLED


class TestRecovery:
    @pytest.mark.asyncio
    async def test_crash_then_recover_and_resume(self, env) -> None:
        store, _queue, manager, worker = env
        created = await manager.submit(CreateRunRequest(goal="Crash me."))
        record = await store.get_required(created.run_id)
        # Simulate a lost worker that claimed but never heartbeated.
        record.status = RunStatus.RUNNING
        await store.update(record)
        await asyncio.sleep(0.35)
        recovered = await manager.recover_stale_runs()
        assert recovered == [created.run_id]
        after = await store.get_required(created.run_id)
        assert after.status == RunStatus.RESUMABLE
        resumed = await manager.resume(created.run_id)
        assert resumed.status == RunStatus.QUEUED
        await _drive(worker)
        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_heartbeat_prevents_false_recovery(self, env) -> None:
        store, _queue, _manager, _worker = env
        live = RunRecord(
            run_id="live",
            goal="g",
            status=RunStatus.RUNNING,
            updated_at=datetime.datetime.now(),
        )
        await store.create(live)
        await asyncio.sleep(0.05)
        stale = await store.find_stale_running(0.2)
        assert all(r.run_id != "live" for r in stale)


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_runs_all_complete(self, env) -> None:
        store, _queue, manager, worker = env
        created_runs = [
            await manager.submit(
                CreateRunRequest(goal=f"Goal {i}. Then done.")
            )
            for i in range(4)
        ]
        await _drive(worker, cycles=16)
        statuses = [
            (await store.get_required(r.run_id)).status for r in created_runs
        ]
        assert all(s == RunStatus.COMPLETED for s in statuses)

    @pytest.mark.asyncio
    async def test_duplicate_claim_is_suppressed(self, env) -> None:
        store, queue, manager, worker = env
        created = await manager.submit(CreateRunRequest(goal="once only."))
        record = await store.get_required(created.run_id)
        record.status = RunStatus.RUNNING
        await store.update(record)  # fresh heartbeat → suppressed on claim
        await queue.release(created.run_id)
        run_id = await worker.try_claim_and_execute()
        assert run_id == created.run_id
        await asyncio.sleep(0.05)
        still_running = await store.get_required(created.run_id)
        assert still_running.status == RunStatus.RUNNING


class TestConfiguration:
    def test_defaults_are_dev_safe(self) -> None:
        cfg = load_service_config(env={})
        assert cfg.api_token == ""
        assert not cfg.use_postgres

    def test_env_overrides(self) -> None:
        env = {
            "RE_SERVICE_API_TOKEN": "t0k3n",
            "RE_POSTGRES_DSN": "postgresql://u:p@db:5432/re",
            "RE_SERVICE_CORS_ORIGINS": "https://a.example, https://b.example",
            "RE_WORKER_CONCURRENCY": "8",
            "RE_SERVICE_MAX_BODY_BYTES": "2048",
            "RE_DEFAULT_MAX_STEPS": "40",
            "RE_SERVICE_ARTIFACT_DIR": "/tmp/arts",
        }
        cfg = load_service_config(env=env)
        assert cfg.api_token == "t0k3n"
        assert cfg.use_postgres
        assert cfg.cors_origins == ["https://a.example", "https://b.example"]
        assert cfg.worker_concurrency == 8
        assert cfg.max_request_body_bytes == 2048
        assert cfg.default_max_steps == 40
        assert str(cfg.artifact_dir) == "/tmp/arts"

    def test_invalid_values_fail_fast(self) -> None:
        with pytest.raises(Exception):
            load_service_config(env={"RE_WORKER_CONCURRENCY": "not-a-number"})

    @pytest.mark.asyncio
    async def test_resume_of_running_run_raises(self, env) -> None:
        store, _queue, manager, _worker = env
        rec = RunRecord(run_id="busy", goal="g", status=RunStatus.RUNNING)
        await store.create(rec)
        with pytest.raises(ResumeNotAvailableError):
            await manager.resume("busy")
