"""P4 - Integration regression: production safety-chain invariant (E3+E5).

The production service must NEVER execute an autonomous run without both
the E3 ToolGateway and the E5 SafetyController. These tests prove the
invariant at the *service assembly* level (the same wiring
``research_engineer.service.serve._worker_main`` performs), not just at
the unit level:

1. The production assembly (``enforce_safety_chain=True``) wires both
   components and a run executed through it demonstrably flows through
   gateway policy and the safety controller.
2. A misconfigured enforcing worker (components missing) refuses to run
   agent code at all - fail-closed - regardless of what the run requests.
3. Development flexibility is preserved: ``enforce_safety_chain=False``
   (the default) still permits chain-less workers for tests.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from research_engineer.runtime.checkpoint_stores import InMemoryCheckpointStore
from research_engineer.service.agents import AgentFactoryRegistry
from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.bench_agents import (
    KIND_BENCH_TOOL,
    register_benchmark_kinds,
)
from research_engineer.service.config import load_service_config
from research_engineer.service.manager import RunManager
from research_engineer.service.models import CreateRunRequest, RunStatus
from research_engineer.service.queue import InMemoryQueue
from research_engineer.service.serve import build_default_safety_chain
from research_engineer.service.store import SQLiteRunStore
from research_engineer.service.telemetry import ServiceTelemetry
from research_engineer.service.worker import AgentWorker


async def _drive(worker: AgentWorker, cycles: int = 12) -> None:
    for _ in range(cycles):
        await worker.try_claim_and_execute()
        await asyncio.sleep(0.02)
    await worker.drain(10)


class _ProdConfigShim:
    """Mimics the production ServiceConfig (safety chain enforced)."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = str(artifact_dir)
        self.enforce_safety_chain = True


class TestProductionSafetyInvariant:
    @pytest.mark.asyncio
    async def test_production_assembly_wires_full_chain_and_routes_through_it(
        self, tmp_path: Path
    ) -> None:
        """The serve.py assembly yields a worker whose runs go through
        ToolGateway + SafetyController (gateway log + safety state prove
        the routing)."""
        config = _ProdConfigShim(tmp_path / "artifacts")
        gateway, controller = build_default_safety_chain(config)
        assert gateway is not None
        assert controller is not None

        store = SQLiteRunStore(tmp_path / "runs.db")
        queue = InMemoryQueue()
        manager = RunManager(store, queue, ServiceTelemetry())
        factories = AgentFactoryRegistry(config_max_steps=8)
        register_benchmark_kinds(factories)
        worker = AgentWorker(  # identical to serve._worker_main wiring
            store=store,
            queue=queue,
            checkpoint_store=InMemoryCheckpointStore(),
            artifacts=ArtifactStore(config.artifact_dir),
            factories=factories,
            telemetry=ServiceTelemetry(),
            poll_interval_seconds=0.02,
            tool_gateway=gateway,
            safety_controller=controller,
            require_safety_chain=config.enforce_safety_chain,
        )
        created = await manager.submit(CreateRunRequest(
            goal="Write the file.",
            metadata={"agent_kind": KIND_BENCH_TOOL},
        ))
        await _drive(worker)
        final = await store.get_required(created.run_id)
        assert final.status == RunStatus.COMPLETED
        ctx = (final.result or {}).get("context", {})
        # Gateway routing: every tool call was dispatched through the E3
        # gateway (recorded in the persisted tool_call_log).
        log = [e for e in ctx.get("tool_call_log", []) if isinstance(e, dict)]
        assert log, "expected gateway-routed tool calls"
        assert all("status" in e and "tool" in e for e in log), (
            f"unexpected log entries: {log}"
        )
        # E5 safety state is tracked on the run context.
        assert "safety_state" in ctx, (
            "expected safety controller state on the run context"
        )

    @pytest.mark.asyncio
    async def test_enforcing_service_cannot_execute_without_both_components(
        self, tmp_path: Path
    ) -> None:
        """Fail-closed: with ``require_safety_chain=True`` (production),
        a worker missing EITHER component must never execute agent code -
        the run is refused with an explicit fail-closed error naming the
        missing pieces."""
        from research_engineer.gateway.gateway import ToolGateway
        from research_engineer.gateway.models import ToolGatewayConfig

        variants: list[dict] = [
            {},  # neither component
            {"tool_gateway": ToolGateway(
                ToolGatewayConfig(workspace=[str(tmp_path)])
            )},  # gateway only
        ]
        for i, kwargs in enumerate(variants):
            store = SQLiteRunStore(tmp_path / f"runs_{i}.db")
            queue = InMemoryQueue()
            manager = RunManager(store, queue, ServiceTelemetry())
            worker = AgentWorker(
                store=store,
                queue=queue,
                checkpoint_store=InMemoryCheckpointStore(),
                artifacts=ArtifactStore(tmp_path / f"artifacts_{i}"),
                factories=AgentFactoryRegistry(config_max_steps=8),
                telemetry=ServiceTelemetry(),
                poll_interval_seconds=0.02,
                require_safety_chain=True,  # production discipline
                **kwargs,
            )
            created = await manager.submit(CreateRunRequest(
                goal="Autonomous run that must never execute.",
            ))
            await _drive(worker, cycles=6)
            final = await store.get_required(created.run_id)
            assert final.status == RunStatus.FAILED
            assert final.error is not None
            assert "fail-closed" in final.error
            assert "ToolGateway" in final.error or (
                "SafetyController" in final.error
            )
            assert not final.artifacts, "agent code must not have executed"

    def test_dev_mode_keeps_chain_optional(self) -> None:
        """Development/test flexibility is preserved: the default config
        does not enforce the chain."""
        cfg = load_service_config({})
        assert cfg.enforce_safety_chain is False
        prod = load_service_config({"RE_SERVICE_ENFORCE_SAFETY": "1"})
        assert prod.enforce_safety_chain is True

    def test_improvement_store_factory_selects_backend(self) -> None:
        """E8 ImprovementStore: PostgreSQL in production (postgres_dsn),
        JSON filesystem store for development/tests."""
        from research_engineer.improve.pg_store import (
            PostgresImprovementStore,
            build_improvement_store,
        )
        from research_engineer.improve.pipeline import ImprovementStore

        class _Cfg:
            postgres_dsn = "postgresql://user:pw@localhost:5432/prod"

        class _DevCfg:
            postgres_dsn = ""

        assert isinstance(
            build_improvement_store(_Cfg()), PostgresImprovementStore
        )
        assert isinstance(
            build_improvement_store(_DevCfg(), default_root="output/x"),
            ImprovementStore,
        )
