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


class TestResumeLockHygiene:
    @pytest.mark.asyncio
    async def test_maybe_resume_does_not_leak_resume_lock(self, env) -> None:
        from research_engineer.runtime.checkpoint import Checkpoint
        from research_engineer.runtime.models import AgentContext, AgentState

        store, _queue, manager, worker = env
        created = await manager.submit(
            CreateRunRequest(goal="Step one. Step two.")
        )
        record = await store.get_required(created.run_id)
        # Crash simulation: recovery marked the run RESUMABLE and re-queued
        # it; this claim is the second execution (claim_count 2), so the
        # worker probes for a resumable checkpoint.
        record.status = RunStatus.RUNNING
        record.claim_count = 2
        await store.update(record)
        ctx = AgentContext(goal=record.goal, state=AgentState.RUNNING)
        ctx.execution_id = created.run_id
        await worker.checkpoint_store.save(Checkpoint.from_context(ctx))

        first = await worker._maybe_resume(record)
        assert first is not None
        # The probe's resume lock must not outlive _maybe_resume: a second
        # recovery of the same run (another crash cycle) used to wedge here
        # with CheckpointLockError, making the run permanently unresumable.
        second = await worker._maybe_resume(record)
        assert second is not None


class TestOutcomeSettlementRaces:
    @pytest.mark.asyncio
    async def test_superseded_outcome_does_not_resurrect_terminal(
        self, tmp_path: Path
    ) -> None:
        """A cancel landing while the run finishes must win: no crash, no
        resurrection to COMPLETED, and the queue entry still gets acked."""

        class AckRecorder(InMemoryQueue):
            def __init__(self) -> None:
                super().__init__()
                self.acked: list[str] = []

            async def ack(self, run_id: str) -> None:
                self.acked.append(run_id)
                await super().ack(run_id)

        from research_engineer.runtime.models import AgentContext

        queue = AckRecorder()
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
        )
        created = await manager.submit(
            CreateRunRequest(goal="Step one. Step two. Done.")
        )
        # Simulate a live claim (status running).
        record = await store.get_required(created.run_id)
        record.status = RunStatus.RUNNING
        await store.update(record)

        # The stale snapshot the worker has been mutating in memory while
        # executing; its work completed successfully...
        stale_record = await store.get_required(created.run_id)
        stale_record.status = RunStatus.COMPLETED
        stale_record.finished_at = datetime.datetime.now()

        # ...but before it could settle, another writer terminalized the
        # store row as cancelled.
        cancelled = await store.get_required(created.run_id)
        cancelled.status = RunStatus.CANCELLED
        cancelled.finished_at = datetime.datetime.now()
        await store.update(cancelled)

        ctx = AgentContext(goal=created.goal, execution_id=created.run_id)
        ctx.output = {"done": True}
        ctx.duration_seconds = 0.01

        # Pre-fix: InvalidTransitionError escaped ``_persist_outcome`` as an
        # orphan task exception and the queue entry was never acked.
        await worker._persist_outcome(stale_record, ctx)

        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.CANCELLED
        assert queue.acked == [created.run_id]


# ---------------------------------------------------------------------------
# P1 §1: production safety wiring (fail-closed + full chain execution)
# ---------------------------------------------------------------------------


class TestProductionSafetyChain:
    """Production autonomous execution requires E3 gateway + E5 safety."""

    @staticmethod
    def _enforcing_worker(store, queue, tmp_path, **kwargs):
        return AgentWorker(
            store=store,
            queue=queue,
            checkpoint_store=InMemoryCheckpointStore(),
            artifacts=ArtifactStore(tmp_path / "artifacts"),
            factories=AgentFactoryRegistry(config_max_steps=8),
            telemetry=ServiceTelemetry(),
            poll_interval_seconds=0.02,
            require_safety_chain=True,
            **kwargs,
        )

    async def _submit(self, store, queue, tmp_path, goal="Step one. Step two.",
                      metadata=None):
        manager = RunManager(store, queue, ServiceTelemetry())
        return await manager.submit(
            CreateRunRequest(goal=goal, metadata=metadata or {})
        )

    @pytest.mark.asyncio
    async def test_enforce_mode_fails_closed_without_chain(
        self, tmp_store, tmp_path
    ) -> None:
        """No gateway/controller wired -> run refuses to execute agent code."""
        from research_engineer.service.queue import InMemoryQueue

        queue = InMemoryQueue()
        worker = self._enforcing_worker(tmp_store, queue, tmp_path)
        created = await self._submit(tmp_store, queue, tmp_path)
        await _drive(worker)
        final = await tmp_store.get_required(created.run_id)
        assert final.status == RunStatus.FAILED
        assert final.error is not None
        assert "fail-closed" in final.error
        assert "ToolGateway" in final.error
        assert "SafetyController" in final.error
        # The refusal acked the queue entry (no re-delivery loop).
        assert final.claim_count <= 1
        assert not final.artifacts  # nothing executed, nothing persisted

    @pytest.mark.asyncio
    async def test_enforce_mode_fails_closed_gateway_only(
        self, tmp_store, tmp_path
    ) -> None:
        """Gateway alone is insufficient; the pair is REQUIRED."""
        from research_engineer.gateway.gateway import ToolGateway
        from research_engineer.gateway.models import ToolGatewayConfig
        from research_engineer.service.queue import InMemoryQueue

        queue = InMemoryQueue()
        ws = tmp_path / "sandbox"
        ws.mkdir(parents=True)
        gateway = ToolGateway(ToolGatewayConfig(workspace=[str(ws)]))
        worker = self._enforcing_worker(
            tmp_store, queue, tmp_path, tool_gateway=gateway
        )
        created = await self._submit(tmp_store, queue, tmp_path)
        await _drive(worker)
        final = await tmp_store.get_required(created.run_id)
        assert final.status == RunStatus.FAILED
        assert "SafetyController" in (final.error or "")

    @pytest.mark.asyncio
    async def test_full_chain_executes_tool_calls_through_gateway(
        self, tmp_store, tmp_path
    ) -> None:
        """With both components wired, a bench_tool run completes and its
        tool calls visibly traverse the gateway+safety chain."""
        from research_engineer.gateway.gateway import ToolGateway
        from research_engineer.gateway.models import ToolGatewayConfig
        from research_engineer.safety.controller import SafetyController
        from research_engineer.service.bench_agents import (
            KIND_BENCH_TOOL,
            register_benchmark_kinds,
            register_sandbox_tools,
        )
        from research_engineer.service.queue import InMemoryQueue

        queue = InMemoryQueue()
        factories = AgentFactoryRegistry(config_max_steps=10)
        register_benchmark_kinds(factories)
        workspace = tmp_path / "artifacts"
        gateway = ToolGateway(
            ToolGatewayConfig(workspace=[str(workspace.resolve())],
                              enforce_approval=True)
        )
        register_sandbox_tools(gateway, workspace=workspace / "sandbox")
        controller = SafetyController()

        worker = AgentWorker(
            store=tmp_store,
            queue=queue,
            checkpoint_store=InMemoryCheckpointStore(),
            artifacts=ArtifactStore(workspace),
            factories=factories,
            telemetry=ServiceTelemetry(),
            poll_interval_seconds=0.02,
            tool_gateway=gateway,
            safety_controller=controller,
            require_safety_chain=True,
        )
        created = await self._submit(
            tmp_store,
            queue,
            tmp_path,
            goal=(
                "Survey gradient checkpointing tradeoffs for MoE models. "
                "Benchmark attention kernel variants under bf16."
            ),
            metadata={"agent_kind": KIND_BENCH_TOOL},
        )
        await _drive(worker, cycles=12)
        final = await tmp_store.get_required(created.run_id)
        assert final.status == RunStatus.COMPLETED
        assert final.result is not None

        summary = final.result["context"]
        # P1 §2 payload enrichment: analysis-grade context travels with the
        # terminal record.
        for field in (
            "current_step", "tool_calls", "tokens", "cost_usd",
            "recoverable_errors", "fatal_errors", "human_interventions",
            "tool_call_log", "safety_state", "steps",
        ):
            assert field in summary, f"missing context summary field {field}"
        # Real gateway-dispatched calls are recorded in order.
        statuses = [
            e["status"] for e in summary["tool_call_log"]
            if e["tool"] == "research_note_write"
        ]
        assert statuses and all(s == "success" for s in statuses)
        assert summary["tool_calls"] >= 4
        # Files really landed inside the approved sandbox root (the goal
        # decomposes into exactly two checklist chunks >8 chars).
        notes = sorted((workspace / "sandbox" / "notes").glob("*.txt"))
        assert len(notes) == 2

    @pytest.mark.asyncio
    async def test_loop_kind_is_stopped_by_safety_controller(
        self, tmp_store, tmp_path
    ) -> None:
        """A looping benchmark run is terminated by the E5 chain."""
        from research_engineer.gateway.gateway import ToolGateway
        from research_engineer.gateway.models import ToolGatewayConfig
        from research_engineer.safety.controller import SafetyController
        from research_engineer.service.bench_agents import (
            KIND_BENCH_LOOP,
            register_benchmark_kinds,
            register_sandbox_tools,
        )
        from research_engineer.service.queue import InMemoryQueue

        queue = InMemoryQueue()
        factories = AgentFactoryRegistry(config_max_steps=15)
        register_benchmark_kinds(factories)
        workspace = tmp_path / "artifacts"
        gateway = ToolGateway(
            ToolGatewayConfig(workspace=[str(workspace.resolve())])
        )
        register_sandbox_tools(gateway, workspace=workspace / "sandbox")

        worker = AgentWorker(
            store=tmp_store,
            queue=queue,
            checkpoint_store=InMemoryCheckpointStore(),
            artifacts=ArtifactStore(workspace),
            factories=factories,
            telemetry=ServiceTelemetry(),
            poll_interval_seconds=0.02,
            tool_gateway=gateway,
            safety_controller=SafetyController(),
            require_safety_chain=True,
        )
        created = await self._submit(
            tmp_store,
            queue,
            tmp_path,
            goal="Probe the same source repeatedly.",
            metadata={"agent_kind": KIND_BENCH_LOOP},
        )
        await _drive(worker, cycles=16)
        final = await tmp_store.get_required(created.run_id)
        assert final.status == RunStatus.FAILED
        summary = final.result["context"] if final.result else {}
        decisions = ((summary.get("safety_state") or {}).get("decisions")) or []
        # At least one non-CONTINUE deterministic control fired.
        assert any(
            str(d.get("action", "")) not in ("", "continue", "None")
            for d in decisions if isinstance(d, dict)
        ), f"expected safety intervention, decisions={decisions}"

    @pytest.mark.asyncio
    async def test_risky_kind_denied_under_enforced_approvals(
        self, tmp_store, tmp_path
    ) -> None:
        """HIGH-risk approval-gated tools cannot self-approve in production."""
        from research_engineer.gateway.gateway import ToolGateway
        from research_engineer.gateway.models import ToolGatewayConfig
        from research_engineer.safety.controller import SafetyController
        from research_engineer.service.bench_agents import (
            KIND_BENCH_RISKY,
            register_benchmark_kinds,
            register_sandbox_tools,
        )
        from research_engineer.service.queue import InMemoryQueue

        queue = InMemoryQueue()
        factories = AgentFactoryRegistry(config_max_steps=10)
        register_benchmark_kinds(factories)
        workspace = tmp_path / "artifacts"
        gateway = ToolGateway(
            ToolGatewayConfig(workspace=[str(workspace.resolve())],
                              enforce_approval=True)
        )
        register_sandbox_tools(gateway, workspace=workspace / "sandbox")

        worker = AgentWorker(
            store=tmp_store,
            queue=queue,
            checkpoint_store=InMemoryCheckpointStore(),
            artifacts=ArtifactStore(workspace),
            factories=factories,
            telemetry=ServiceTelemetry(),
            poll_interval_seconds=0.02,
            tool_gateway=gateway,
            safety_controller=SafetyController(),
            require_safety_chain=True,
        )
        created = await self._submit(
            tmp_store,
            queue,
            tmp_path,
            goal="Fetch external benchmark corpora.",
            metadata={"agent_kind": KIND_BENCH_RISKY},
        )
        await _drive(worker, cycles=14)
        final = await tmp_store.get_required(created.run_id)
        assert final.status.is_terminal()
        summary = final.result["context"] if final.result else {}
        denials = [
            e for e in summary.get("tool_call_log", [])
            if e.get("tool") == "external_probe"
            and "denied" in str(e.get("status", ""))
        ]
        assert denials, (
            "expected approval_denied entries; got "
            f"{summary.get('tool_call_log')}"
        )

    def test_config_flag_parses(self) -> None:
        cfg = load_service_config({"RE_SERVICE_ENFORCE_SAFETY": "1"})
        assert cfg.enforce_safety_chain is True
        dev = load_service_config({})
        assert dev.enforce_safety_chain is False
