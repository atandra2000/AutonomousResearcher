# E7 — Production Deployment

Docker Compose stack for running the autonomous agent system as a
long-running service.

## Services

| Service        | Purpose                                             |
|----------------|-----------------------------------------------------|
| `api`          | FastAPI service for run submission/status/result    |
| `worker`       | Long-running executor driving E1 `AgentRuntime`     |
| `postgres`     | Runs table, queue (`SKIP LOCKED`), E2 checkpoints   |
| `otel-collector` | OpenTelemetry OTLP receiver (E6 telemetry)        |
| `web`          | Next.js console proxying the run API (Auth.js OIDC gate) |
| artifacts      | Named Docker volume mounted into api+worker         |

No Redis is used: PostgreSQL provides both durable persistence *and* the
queue with `FOR UPDATE SKIP LOCKED`, matching the backend already required
by the E2 checkpoint store. Only services actually needed by the current
implementation are included.

## Start

```bash
cd deploy
cp .env.example .env          # set POSTGRES_PASSWORD + RE_SERVICE_API_TOKEN
docker compose up -d --build
curl -s localhost:8000/health
```

## Endpoints (auth: `Authorization: Bearer $RE_SERVICE_API_TOKEN`)

```
POST /runs                      {"goal": "...", "metadata": {...}}
GET  /runs/{run_id}
POST /runs/{run_id}/cancel
POST /runs/{run_id}/resume
GET  /runs/{run_id}/result
GET  /health                    liveness (no auth)
GET  /ready                     readiness incl. dependency checks (no auth)
```

## Recovery semantics

* Workers heartbeat run records; the recovery scan marks runs whose worker
  disappeared as `resumable`, re-queues them, and execution resumes from the
  last E2 checkpoint.
* Duplicate delivery while an execution is live is suppressed using
  heartbeats; after true worker loss the takeover restarts from the last
  checkpoint (never earlier).
* At-least-once boundary: side effects performed by the agent between the
  last checkpoint and terminal-state commit may be redone if the worker dies
  exactly then. Checkpoints themselves cannot be corrupted (full-payload
  versioned upserts).

## Security notes (actual scope)

* Bearer-token auth on all run endpoints; probes unauthenticated.
* Restricted CORS via `RE_SERVICE_CORS_ORIGINS`.
* Request bodies bounded (`RE_SERVICE_MAX_BODY_BYTES`, default 1 MiB).
* Non-root container user, no privileged mode or added capabilities.
* No secrets in logs: the service never logs goals/metadata payloads.
* This is process-level hardening, **not** sandbox-grade workload isolation;
  tool sandboxing remains E3 gateway policy.

## LangGraph & web console (framework-stack integration)

* `RE_LANGGRAPH_CHECKPOINT_DSN` is pre-wired to the in-stack Postgres for
  the api/worker env block: `research-engineer research --engine langgraph`
  then persists durable per-node snapshots via `AsyncPostgresSaver`
  (requires the `[service]` extra, already in the image). Without the DSN
  the LangGraph engine simply runs without snapshots.
* `LANGSMITH_TRACING` (default `false`), `LANGSMITH_PROJECT`,
  `LANGSMITH_ENDPOINT`, and `LANGSMITH_API_KEY` enable LangSmith tracing
  for LangChain-backed LLM runs; they are pass-throughs and stay off by
  default.
* The `web` service builds `apps/web/Dockerfile`, waits for `api` to be
  healthy, and proxies the run API with the same bearer token. It is gated
  by Auth.js OIDC — set `AUTH_SECRET`, `AUTH_OIDC_ISSUER`, and
  `AUTH_OIDC_CLIENT_ID` in `.env` (compose fails fast without them).

See [Framework-Stack Migration](framework_stack_migration.md) for the
full component map.
