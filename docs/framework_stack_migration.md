# Framework-Stack Migration — LangGraph & LangChain

The platform's execution core (agents, tools, typed models, gateway, service
API) is framework-agnostic. This migration adds an **opt-in** LangGraph /
LangChain runtime alongside the native one without changing any default
behavior, public contract, or existing tool.

## Design invariants

- **Native engine remains the default.** `ResearchConfig.engine` is
  `"native"` unless explicitly set to `"langgraph"`; the CLI flag mirrors
  this (`--engine native`).
- **The gateway stays the execution authority.** LangChain tools are thin
  adapters whose only execution path is `ToolGateway.execute` — policy,
  approval, sandboxing, timeouts, and output caps all still apply.
- **Tools and models are unchanged.** The existing `Tool[Input, Output]`
  ABC and Pydantic v2 models sit at every boundary; nothing was rewritten.
- **Everything is optional.** Durable graph snapshots need the `[service]`
  extra; LangSmith tracing is off unless env vars enable it.

## Components

### LangGraph workflow engine — `graphs/research.py`

`ResearchGraph` compiles the seven `ResearchStageType` stages into a
LangGraph `StateGraph`:

```python
graph = ResearchGraph(workflow, checkpointer=checkpointer)   # checkpointer optional
result = await graph.run(goal, repo_path, config=cfg, thread_id="t1")
```

- One `initialize` node plus one node per stage; conditional edges jump to
  `END` when a stage fails (workflow status becomes `PARTIAL`) or after
  the last stage.
- Stage execution is delegated to the existing
  `ResearchWorkflowFramework._run_stage` — the graph owns ordering and
  persistence, the framework stays the stage authority.
- `skip_stages` from `ResearchConfig` is honored (skipped stages produce
  `SKIPPED` records exactly like the native engine).
- `thread_id` (generated `research_<hex>` when omitted) namespaces
  checkpoint snapshots; the final state is translated back into the
  legacy `ResearchResult`, so callers keep one public result type.

### Checkpointing — `graphs/checkpoints.py`

`checkpoint_from_environment()` is an async context manager:

| Condition | Yields |
|-----------|--------|
| `RE_LANGGRAPH_CHECKPOINT_DSN` set | `AsyncPostgresSaver` (`setup()` runs on open; requires the `[service]` extra — `langgraph-checkpoint-postgres`) |
| DSN set but extra missing | `RuntimeError` (fail fast, no silent downgrade) |
| No DSN (default) | `None` → graph runs without snapshots |

### LangChain LLM provider — `llm/langchain_provider.py`

`LangChainChatProvider` (registered as provider type `langchain` in
`llm/factory.py`) adapts any LangChain chat model to the platform's
`LLMProvider` contract (`complete`, `complete_with_tools`, `stream`).
The default model is `ChatOpenAI`, which also speaks OpenAI-compatible
gateways via `base_url`. Model resolution order: `default_model`
argument → `LANGCHAIN_MODEL` → `OPENAI_MODEL` → `gpt-4o`.

```yaml
# llm_config.yaml
providers:
  langchain:
    type: langchain
    base_url: https://api.openai.com
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4o
    timeout: 60
```

Tool schemas are bound with `bind_tools`, but **the provider never
executes tools** — returned `tool_calls` are dispatched through the
gateway by the calling agent. LangSmith tracing comes free with any
LangChain run when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are
set.

### Gateway adapter — `gateway/langchain.py`

`as_langchain_tool(gateway, name=..., description=..., args_schema=...)`
wraps the E3 `ToolGateway` as a LangChain `StructuredTool`:

- `args_schema` is explicit — the model can only pass fields accepted by
  the existing typed tool input (validated with `model_validate`).
- Results return as `{ok, status, failure_kind, output, error}` dicts, so
  failed gateway executions surface as data instead of exceptions.
- Invocation metadata records `framework: "langchain"` for audit trails.

### Orchestrator + CLI wiring

- `ResearchConfig` gained `engine` (validated against
  `{"native", "langgraph"}`) and `thread_id`.
- `ResearchOrchestrator.run` imports `research_engineer.graphs` lazily
  and, for `engine == "langgraph"`, opens `checkpoint_from_environment()`
  around the run; otherwise the native framework path is unchanged.

```bash
research-engineer research "Design a more efficient diffusion transformer" --engine langgraph
research-engineer research "Novel loss function" --engine langgraph --thread-id migration-001
```

### Web console — `apps/web/`

A minimal Next.js (App Router) console that proxies the E7 run API:

- `proxy.ts` (Next 16 proxy/middleware) gates the UI; Auth.js v5
  (`next-auth@beta`) authenticates users via OIDC.
- Server routes forward run submission/status/cancel/resume/result to the
  FastAPI service with the bearer token — service credentials never reach
  the browser.
- Local dev: `pnpm install && pnpm dev` inside `apps/web`; CI runs
  `pnpm typecheck` + `pnpm build` (the `web` job in
  `.github/workflows/ci.yml`).

## Deployment wiring

`deploy/docker-compose.yml` adds:

| Piece | Detail |
|-------|--------|
| `web` service | Builds `apps/web/Dockerfile`, depends on `api` being healthy, gets `RE_SERVICE_API_URL` + bearer token + `AUTH_SECRET`/`AUTH_OIDC_ISSUER`/`AUTH_OIDC_CLIENT_ID` |
| API/worker env | `RE_LANGGRAPH_CHECKPOINT_DSN` (in-stack Postgres), `LANGSMITH_TRACING`/`PROJECT`/`ENDPOINT`/`API_KEY`, `LANGCHAIN_CALLBACKS_BACKGROUND` |

`deploy/.env.example` documents all of these (LangSmith defaults off).

## Dependencies

| Package | Extra | Purpose |
|---------|-------|---------|
| `langchain>=1.0.0`, `langchain-openai>=1.0.0`, `langgraph>=1.0.0`, `langsmith>=0.4.0` | core | engine + provider + adapter |
| `langgraph-checkpoint-postgres>=3.0.0` | `[service]` | durable graph snapshots |

## Testing

| Suite | Covers |
|-------|--------|
| `tests/test_research_graph.py` | graph compiles, honors skip stages, returns a legacy `ResearchResult` |
| `tests/test_langgraph_checkpoints.py` | no DSN → `None`; DSN without the extra → `RuntimeError` |
| `tests/test_langchain_provider.py` | complete / tools / stream mapping onto the local contract |
| `tests/test_langchain_gateway.py` | adapter routes through the gateway; gateway failures surface as data |
| `tests/test_research_workflow.py` | engine validation + orchestrator dispatch (13 new tests total) |
