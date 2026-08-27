"""E7 - Service configuration from environment variables.

All configuration is read from ``RE_SERVICE_*``/``RE_`` prefixed environment
variables so containers need no config files or rebuilds to change settings.
Pydantic validates every value; invalid configuration fails fast at startup.

Development defaults are safe (no auth required against localhost, SQLite
backends); production-sensitive capabilities (API auth tokens, Postgres DSNs)
must be provided explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """Configuration for the E7 API + worker services."""

    #: Bearer token required by the API. Empty disables authentication
    #: (development only; always set in production).
    api_token: str = ""
    #: CORS origins allowed by the API. Empty disables CORS entirely.
    cors_origins: list[str] = Field(default_factory=list)
    #: Maximum accepted JSON request body size in bytes.
    max_request_body_bytes: int = Field(
        default=1024 * 1024, gt=0, le=64 * 1024 * 1024
    )
    #: SQLite database path for run records (default backend).
    db_path: Path = Path("data/run_service.db")
    #: PostgreSQL DSN. When set, runs/checkpoints/queue all use PostgreSQL.
    postgres_dsn: str = ""
    #: Root directory of the artifact volume (bind mount in containers).
    artifact_dir: Path = Path("data/artifacts")
    #: Checkpoint database path used by SQLiteCheckpointStore (SQLite mode).
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    #: Worker concurrency: number of concurrent runs per worker process.
    worker_concurrency: int = Field(default=2, ge=1, le=32)
    #: How long (seconds) a claimed-but-not-updated run is considered lost
    #: and becomes resumable after worker failure.
    stale_run_timeout_seconds: float = Field(default=120.0, gt=0.0)
    #: Poll interval for the worker queue loop (seconds).
    queue_poll_interval_seconds: float = Field(default=0.5, gt=0.0)
    #: Default runtime budgets applied to runs without overrides.
    default_max_steps: int = Field(default=25, ge=1, le=10_000)
    default_max_runtime_seconds: float = Field(default=600.0, gt=0.0)
    #: Optional artificial pause between agent steps (testing/demos).
    step_delay_seconds: float = Field(default=0.0, ge=0.0)
    #: Host/port for the API process.
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    #: Optional OpenTelemetry OTLP endpoint for telemetry export.
    otlp_endpoint: str = ""
    #: Production discipline: when True, autonomous execution fails closed
    #: unless a ToolGateway *and* a SafetyController are wired into the
    #: worker (P1 §1). Development/test deployments may leave this off.
    enforce_safety_chain: bool = False

    @property
    def use_postgres(self) -> bool:
        return bool(self.postgres_dsn)


def _parse_scalar_map(env: Mapping[str, str]) -> dict[str, object]:
    """Extract integer/float/list/path overrides from the environment."""
    kwargs: dict[str, object] = {}
    if env.get("RE_SERVICE_API_TOKEN") is not None:
        kwargs["api_token"] = env["RE_SERVICE_API_TOKEN"]
    if env.get("RE_SERVICE_CORS_ORIGINS"):
        kwargs["cors_origins"] = [
            o.strip() for o in env["RE_SERVICE_CORS_ORIGINS"].split(",")
            if o.strip()
        ]
    for key, field_name in [
        ("RE_SERVICE_MAX_BODY_BYTES", "max_request_body_bytes"),
        ("RE_WORKER_CONCURRENCY", "worker_concurrency"),
        ("RE_DEFAULT_MAX_STEPS", "default_max_steps"),
    ]:
        raw = env.get(key)
        if raw:
            kwargs[field_name] = int(raw)
    for key, field_name in [
        ("RE_STALE_RUN_TIMEOUT_SECONDS", "stale_run_timeout_seconds"),
        ("RE_QUEUE_POLL_SECONDS", "queue_poll_interval_seconds"),
        ("RE_DEFAULT_MAX_RUNTIME_SECONDS", "default_max_runtime_seconds"),
        ("RE_STEP_DELAY_SECONDS", "step_delay_seconds"),
    ]:
        raw = env.get(key)
        if raw:
            kwargs[field_name] = float(raw)
    return kwargs


def _parse_paths_and_dsn(
    env: Mapping[str, str], kwargs: dict[str, object]
) -> None:
    for key, field_name in [
        ("RE_SERVICE_DB_PATH", "db_path"),
        ("RE_CHECKPOINT_DB", "checkpoint_db_path"),
        ("RE_SERVICE_ARTIFACT_DIR", "artifact_dir"),
    ]:
        raw = env.get(key)
        if raw:
            kwargs[field_name] = Path(raw)
    if env.get("RE_POSTGRES_DSN"):
        kwargs["postgres_dsn"] = env["RE_POSTGRES_DSN"]
    if env.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        kwargs["otlp_endpoint"] = env["OTEL_EXPORTER_OTLP_ENDPOINT"]
    raw_enforce = env.get("RE_SERVICE_ENFORCE_SAFETY", "").strip().lower()
    kwargs["enforce_safety_chain"] = raw_enforce in ("1", "true", "yes", "on")


def load_service_config(env: dict[str, str] | None = None) -> ServiceConfig:
    """Build a :class:`ServiceConfig` from environment variables.

    Recognized variables (all optional): ``RE_SERVICE_API_TOKEN``,
    ``RE_SERVICE_CORS_ORIGINS``, ``RE_SERVICE_MAX_BODY_BYTES``,
    ``RE_SERVICE_DB_PATH``, ``RE_POSTGRES_DSN`` (enables PostgreSQL backends
    for runs+queue+checkpoints), ``RE_SERVICE_ARTIFACT_DIR``,
    ``RE_CHECKPOINT_DB``, ``RE_WORKER_CONCURRENCY``,
    ``RE_STALE_RUN_TIMEOUT_SECONDS``, ``RE_QUEUE_POLL_SECONDS``,
    ``RE_DEFAULT_MAX_STEPS``, ``RE_DEFAULT_MAX_RUNTIME_SECONDS``,
    ``RE_SERVICE_ENFORCE_SAFETY`` (fail closed without gateway+safety),
    ``OTEL_EXPORTER_OTLP_ENDPOINT``.
    """
    e = env if env is not None else os.environ
    kwargs: dict[str, object] = {}
    kwargs.update(_parse_scalar_map(e))
    _parse_paths_and_dsn(e, kwargs)
    return ServiceConfig(**kwargs)


def build_postgres_dsn(config: ServiceConfig) -> str:
    """Return the effective PostgreSQL DSN (same var as the E2 store)."""
    return config.postgres_dsn


__all__ = ["ServiceConfig", "build_postgres_dsn", "load_service_config"]
