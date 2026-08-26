"""E7 tests — API validation, auth, health/readiness, run lifecycle.

Runs entirely against in-process fakes (SQLite store, in-memory queue) via
FastAPI's TestClient; container smoke tests live in scripts/smoke_test.sh.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_engineer.service.api import create_app
from research_engineer.service.config import ServiceConfig


@pytest.fixture()
def config(tmp_path) -> ServiceConfig:
    return ServiceConfig(
        db_path=tmp_path / "runs.db",
        artifact_dir=tmp_path / "artifacts",
        checkpoint_db_path=str(tmp_path / "checkpoints.db"),
    )


@pytest.fixture()
def client(config: ServiceConfig):
    app = create_app(config)
    with TestClient(app) as c:
        yield c


def _submit(client: TestClient, goal: str = "Test goal. Then finish it.",
            **kwargs: object) -> dict:
    response = client.post("/runs", json={"goal": goal, **kwargs})
    assert response.status_code == 202
    return response.json()


class TestValidationAndSecurity:
    def test_empty_goal_rejected(self, client: TestClient) -> None:
        assert client.post("/runs", json={"goal": ""}).status_code == 422

    def test_oversized_goal_rejected(self, client: TestClient) -> None:
        response = client.post("/runs", json={"goal": "x" * 20_000})
        assert response.status_code == 422

    def test_invalid_payload_type_rejected(self, client: TestClient) -> None:
        assert client.post("/runs", json={"goal": 12345}).status_code == 422

    def test_request_body_size_bounded(self, tmp_path) -> None:
        cfg = ServiceConfig(
            db_path=tmp_path / "runs.db",
            artifact_dir=tmp_path / "artifacts",
            max_request_body_bytes=10,
        )
        with TestClient(create_app(cfg)) as small_client:
            response = small_client.post(
                "/runs", json={"goal": "a much bigger body than ten bytes"}
            )
            assert response.status_code == 413

    def test_auth_required_when_configured(self, tmp_path) -> None:
        cfg = ServiceConfig(
            db_path=tmp_path / "runs.db",
            artifact_dir=tmp_path / "artifacts",
            api_token="secret-token-1",
        )
        app = create_app(cfg)
        headers = {"Authorization": "Bearer secret-token-1"}
        with TestClient(app) as guarded:
            assert guarded.post("/runs", json={"goal": "g"}).status_code == 401
            assert guarded.get("/runs/run_x").status_code == 401
            ok = guarded.post("/runs", json={"goal": "g"}, headers=headers)
            assert ok.status_code == 202
        # Health probes stay unauthenticated.
        with TestClient(create_app(cfg)) as fresh:
            assert fresh.get("/health").status_code == 200
            assert fresh.get("/ready").status_code == 200

    def test_no_openapi_docs_exposed(self, client: TestClient) -> None:
        assert client.get("/openapi.json").status_code == 404

    def test_cors_off_by_default(self, client: TestClient) -> None:
        response = client.options(
            "/health",
            headers={"Origin": "https://evil.example",
                     "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in response.headers


class TestHealthReadiness:
    def test_health(self, client: TestClient) -> None:
        assert client.get("/health").json()["status"] == "ok"

    def test_ready_lists_dependency_checks(self, client: TestClient) -> None:
        body = client.get("/ready").json()
        assert body["ready"] is True
        assert set(body["checks"]) >= {"store", "queue", "artifacts"}


class TestRunLifecycle:
    def test_submit_then_query(self, client: TestClient) -> None:
        created = _submit(client)
        status = client.get(f"/runs/{created['run_id']}")
        assert status.status_code == 200
        assert status.json()["run_id"] == created["run_id"]
        assert status.json()["has_checkpoint"] is False

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get("/runs/run_missing").status_code == 404
        assert client.post("/runs/run_missing/cancel").status_code == 404
        assert client.get("/runs/run_missing/result").status_code == 404
        assert client.post("/runs/run_missing/resume").status_code == 404

    def test_result_unavailable_before_terminal(self, client: TestClient) -> None:
        created = _submit(client)
        body = client.get(f"/runs/{created['run_id']}/result").json()
        assert body["available"] is False
        assert body["output"] is None

    def test_cancel_queued_run(self, client: TestClient) -> None:
        created = _submit(client)
        cancelled = client.post(f"/runs/{created['run_id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json() == {
            "run_id": created["run_id"],
            "status": "cancelled",
            "cancelled": True,
        }
        again = client.post(f"/runs/{created['run_id']}/cancel").json()
        assert again["cancelled"] is False

    def test_resume_of_completed_run_conflict(self, client: TestClient) -> None:
        created = _submit(client)
        client.post(f"/runs/{created['run_id']}/cancel")
        response = client.post(f"/runs/{created['run_id']}/resume")
        assert response.status_code == 409


class TestTelemetryPropagation:
    def test_service_events_carry_correlation_ids(
        self, config: ServiceConfig
    ) -> None:
        from research_engineer.observability import get_event_bus
        from research_engineer.observability.context import (
            CorrelationContext,
            reset_correlation,
            set_correlation,
        )

        captured: list[dict] = []
        bus = get_event_bus()
        sink = type("S", (), {"emit": staticmethod(captured.append)})()
        bus.add_sink(sink)
        token = set_correlation(
            CorrelationContext(trace_id="trc_1", span_id="spn_1")
        )
        try:
            with TestClient(create_app(config)) as tc:
                tc.post("/runs", json={"goal": "hello."})
        finally:
            reset_correlation(token)
        events = [e for e in captured if e.get("kind") == "service"]
        assert events, "expected a service lifecycle event"
        submitted = [e for e in events if e["event"] == "run_submitted"]
        assert submitted
        assert submitted[0].get("trace_id") == "trc_1"
        assert "run_id" in submitted[0]

    def test_metrics_registry_records_lifecycle(self) -> None:
        from research_engineer.service.telemetry import ServiceTelemetry

        telemetry = ServiceTelemetry()
        telemetry.run_submitted("run_m")
        telemetry.run_completed("run_m", duration_seconds=1.5)
        telemetry.queue_latency("run_m", 0.25)
        snapshot = telemetry.registry.snapshot()
        assert any("service_runs_submitted" in key for key in snapshot)
        assert any("service_run_duration_seconds" in key for key in snapshot)


