"""E7 - Production deployment & service architecture.

Public surface::

    from research_engineer.service import (
        ServiceConfig, load_service_config,   # configuration
        RunManager, RunStatus,                # lifecycle
        AgentWorker,                          # execution
        create_app,                           # FastAPI application
    )

The service reuses every existing layer unchanged: E1 ``AgentRuntime`` +
adapters, E2 ``CheckpointStore`` backends (PostgreSQL when
``RE_POSTGRES_DSN`` is set), E3 gateway / E5 safety via the runtime hooks,
and E6 observability for events/metrics.
"""

from research_engineer.service.agents import (
    DEFAULT_AGENT_KIND,
    AgentFactoryRegistry,
)
from research_engineer.service.api import create_app
from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.config import ServiceConfig, load_service_config
from research_engineer.service.manager import RunManager
from research_engineer.service.models import CreateRunRequest, RunStatus
from research_engineer.service.queue import build_run_queue
from research_engineer.service.store import build_run_store
from research_engineer.service.telemetry import ServiceTelemetry
from research_engineer.service.worker import AgentWorker

__all__ = [
    "DEFAULT_AGENT_KIND",
    "AgentFactoryRegistry",
    "AgentWorker",
    "ArtifactStore",
    "CreateRunRequest",
    "RunManager",
    "RunStatus",
    "ServiceConfig",
    "ServiceTelemetry",
    "build_run_queue",
    "build_run_store",
    "create_app",
    "load_service_config",
]
