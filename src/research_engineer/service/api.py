"""E7 - Production API.

A minimal FastAPI application exposing the run lifecycle:

    POST   /runs                    submit a new agent run (non-blocking)
    GET    /runs/{run_id}           query status
    POST   /runs/{run_id}/cancel    request cancellation
    POST   /runs/{run_id}/resume    resume a resumable/failed run
    GET    /runs/{run_id}/result    fetch the result + artifact references
    GET    /health                  liveness probe (process is up)
    GET    /ready                   readiness probe (dependencies reachable)

Security posture: optional bearer-token authentication
(``RE_SERVICE_API_TOKEN``; health endpoints stay unauthenticated so probes
work), restricted CORS (empty by default = no CORS headers), bounded request
bodies, and typed models only - no prompts, secrets, credentials, or tool
interfaces are exposed over the wire.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.config import ServiceConfig, load_service_config
from research_engineer.service.manager import (
    ResumeNotAvailableError,
    RunManager,
)
from research_engineer.service.models import (
    CancelResponse,
    CreateRunRequest,
    HealthResponse,
    ReadinessResponse,
    ResultResponse,
    ResumeResponse,
    RunCreatedResponse,
    RunRecord,
)
from research_engineer.service.queue import build_run_queue
from research_engineer.service.store import build_run_store
from research_engineer.service.telemetry import ServiceTelemetry


def _status_payload(record: RunRecord, has_checkpoint: bool) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "termination_reason": record.termination_reason,
        "has_checkpoint": has_checkpoint,
        "claim_count": record.claim_count,
        "artifacts": record.artifacts,
    }


def _result_payload(record: RunRecord) -> ResultResponse:
    if not record.status.is_terminal():
        return ResultResponse(run_id=record.run_id, status=record.status)
    result = record.result or {}
    ctx = result.get("context") or {}
    if not isinstance(ctx, dict):
        ctx = {}
    duration = None
    if record.started_at is not None and record.finished_at is not None:
        duration = (record.finished_at - record.started_at).total_seconds()
    return ResultResponse(
        run_id=record.run_id,
        status=record.status,
        output=result.get("output"),
        termination_reason=result.get("reason") or record.termination_reason,
        termination=result.get("termination"),
        duration_seconds=duration,
        steps=int(result.get("steps", 0)),
        tokens=int(ctx.get("tokens", 0)),
        tool_calls=int(ctx.get("tool_calls", 0)),
        recoverable_errors=int(ctx.get("recoverable_errors", 0)),
        fatal_errors=int(ctx.get("fatal_errors", 0)),
        artifacts=record.artifacts,
        available=True,
    )


def _build_state(
    cfg: ServiceConfig, telemetry: ServiceTelemetry | None
) -> tuple[FastAPI, RunManager]:
    """Construct the app object with its shared state wired."""
    store = build_run_store(cfg)
    queue = build_run_queue(cfg)
    manager = RunManager(
        store,
        queue,
        telemetry or ServiceTelemetry(),
        stale_run_timeout_seconds=cfg.stale_run_timeout_seconds,
    )
    app = FastAPI(
        title="Autonomous ML Research Engineer - Run Service",
        version="0.9.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.config = cfg
    app.state.store = store
    app.state.queue = queue
    app.state.manager = manager
    app.state.artifacts = ArtifactStore(cfg.artifact_dir)
    return app, manager


def _install_cors(app: FastAPI, cfg: ServiceConfig) -> None:
    if not cfg.cors_origins:
        return
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


def _install_body_limit_middleware(app: FastAPI, cfg: ServiceConfig) -> None:
    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: Any) -> Any:
        if request.method == "POST":
            length_header = request.headers.get("content-length")
            if length_header is not None:
                try:
                    if int(length_header) > cfg.max_request_body_bytes:
                        from fastapi.responses import JSONResponse

                        return JSONResponse(
                            status_code=413,
                            content={"detail": "request body too large"},
                        )
                except ValueError:
                    pass
        return await call_next(request)


def _make_auth(cfg: ServiceConfig) -> Any:
    async def require_auth(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not cfg.api_token:
            return
        if authorization != f"Bearer {cfg.api_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    return Depends(require_auth)


def _make_shutdown(
    app: FastAPI, manager: RunManager
) -> Callable[[], Awaitable[None]]:
    async def _shutdown() -> None:
        await manager.close()
        close_store = getattr(app.state.store, "close", None)
        if close_store is not None:
            await close_store()

    return _shutdown


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    shutdown = getattr(app.state, "shutdown_hook", None)
    if shutdown is not None:
        await shutdown()


def create_app(
    config: ServiceConfig | None = None,
    telemetry: ServiceTelemetry | None = None,
) -> FastAPI:
    """Build the E7 API application (also used directly by tests)."""
    cfg = config or load_service_config()
    app, manager = _build_state(cfg, telemetry)
    _install_cors(app, cfg)
    _install_body_limit_middleware(app, cfg)
    auth_dependency = _make_auth(cfg)
    _register_health_routes(app)
    _register_run_routes(app, auth_dependency)
    app.state.shutdown_hook = _make_shutdown(app, manager)
    return app


def _register_health_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> ReadinessResponse:
        checks: dict[str, bool] = {}
        for name, probe in (
            ("store", app.state.store.count_by_status),
            ("queue", app.state.queue.depth),
        ):
            try:
                await probe()
                checks[name] = True
            except Exception:  # noqa: BLE001 - probes must not raise
                checks[name] = False
        checks["artifacts"] = app.state.artifacts.root.is_dir()
        return ReadinessResponse(ready=all(checks.values()), checks=checks)


def _register_run_routes(app: FastAPI, auth_dependency: Any) -> None:
    _register_submit_status_routes(app, auth_dependency)
    _register_action_result_routes(app, auth_dependency)


def _register_submit_status_routes(
    app: FastAPI, auth_dependency: Any
) -> None:
    manager: RunManager = app.state.manager
    checkpoint_store = getattr(app.state, "checkpoint_store", None)

    @app.post("/runs", response_model=RunCreatedResponse, status_code=202,
              dependencies=[auth_dependency])
    async def submit_run(request: CreateRunRequest) -> RunCreatedResponse:
        record = await manager.submit(request)
        return RunCreatedResponse(
            run_id=record.run_id,
            status=record.status,
            created_at=record.created_at,
        )

    @app.get("/runs/{run_id}", dependencies=[auth_dependency])
    async def run_status(run_id: str) -> dict[str, Any]:
        try:
            record = await manager.get(run_id)
        except Exception:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from None
        has_checkpoint = False
        if checkpoint_store is not None:
            try:
                has_checkpoint = bool(await checkpoint_store.exists(run_id))
            except Exception:  # noqa: BLE001 - probes must not 500
                has_checkpoint = False
        return _status_payload(record, has_checkpoint)


def _register_action_result_routes(
    app: FastAPI, auth_dependency: Any
) -> None:
    manager: RunManager = app.state.manager

    @app.post("/runs/{run_id}/cancel", response_model=CancelResponse,
              dependencies=[auth_dependency])
    async def cancel_run(run_id: str) -> CancelResponse:
        try:
            cancelled, record = await manager.cancel(run_id)
        except Exception:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from None
        return CancelResponse(
            run_id=record.run_id, status=record.status, cancelled=cancelled
        )

    @app.post("/runs/{run_id}/resume", response_model=ResumeResponse,
              dependencies=[auth_dependency])
    async def resume_run(run_id: str) -> ResumeResponse:
        try:
            record = await manager.resume(run_id)
        except ResumeNotAvailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except Exception:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from None
        return ResumeResponse(
            run_id=record.run_id, status=record.status, resumed=True
        )

    @app.get("/runs/{run_id}/result", response_model=ResultResponse,
             dependencies=[auth_dependency])
    async def run_result(run_id: str) -> ResultResponse:
        try:
            record = await manager.get(run_id)
        except Exception:
            raise HTTPException(
                status_code=404, detail="run not found"
            ) from None
        return _result_payload(record)


__all__ = ["create_app"]
