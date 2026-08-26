"""E7 - Worker entry point.

Run with::

    uv run python -m research_engineer.service.serve api   # API process
    uv run python -m research_engineer.service.serve worker  # worker process

Both read configuration exclusively from environment variables.
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from research_engineer.runtime.checkpoint_stores import (
    InMemoryCheckpointStore,
    PostgresCheckpointStore,
    SQLiteCheckpointStore,
)
from research_engineer.service.agents import AgentFactoryRegistry
from research_engineer.service.api import create_app
from research_engineer.service.config import load_service_config
from research_engineer.service.telemetry import ServiceTelemetry
from research_engineer.service.worker import AgentWorker


def _build_checkpoint_store(config: object) -> object:
    """Select the E2 checkpoint store matching the service backend."""
    dsn = getattr(config, "postgres_dsn", "")
    if dsn:
        return PostgresCheckpointStore(dsn)
    db_path = getattr(config, "checkpoint_db_path", None)
    if db_path is None:
        return InMemoryCheckpointStore()
    return SQLiteCheckpointStore(str(db_path))


def serve_api() -> None:
    config = load_service_config()
    app = create_app(config)
    # Attach the checkpoint store so /runs/{id} can report resumability.
    app.state.checkpoint_store = _build_checkpoint_store(config)
    uvicorn.run(app, host=config.api_host, port=config.api_port,
                log_level="info")


async def _worker_main() -> None:
    config = load_service_config()
    telemetry = ServiceTelemetry()
    from research_engineer.service.artifacts import ArtifactStore
    from research_engineer.service.queue import build_run_queue
    from research_engineer.service.store import build_run_store

    worker = AgentWorker(
        store=build_run_store(config),
        queue=build_run_queue(config),
        checkpoint_store=_build_checkpoint_store(config),
        artifacts=ArtifactStore(config.artifact_dir),
        factories=AgentFactoryRegistry(
            config_max_steps=config.default_max_steps,
            config_max_runtime_seconds=config.default_max_runtime_seconds,
            config_step_delay_seconds=config.step_delay_seconds,
        ),
        telemetry=telemetry,
        concurrency=config.worker_concurrency,
        poll_interval_seconds=config.queue_poll_interval_seconds,
        stale_run_timeout_seconds=config.stale_run_timeout_seconds,
    )
    await worker.serve_forever()


def serve_worker() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_worker_main())


def main() -> None:
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "api":
        serve_api()
    elif mode == "worker":
        serve_worker()
    else:
        raise SystemExit("usage: python -m research_engineer.service.serve {api|worker}")


if __name__ == "__main__":
    main()

__all__ = ["main", "serve_api", "serve_worker"]
