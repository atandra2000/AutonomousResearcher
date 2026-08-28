---
type: operations reference
title: Configuration and Runtime Controls
description: Runtime configuration reference for CLI workflows, LLM and memory backends, execution budgets, safety gates, service deployment, and test controls. Distinguishes built-in defaults from file and environment overrides.
tags: [configuration, operations, runtime, safety, llm, service]
verified:
  - by: openwiki/0.4.3
    at: 2026-08-28T12:25:37.430Z
sources:
  - id: openwiki-source-e245b9400c79da09d079565e
    resource: repo://llm_config.yaml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-0cd1dc265699c357afc4b69c
    resource: repo://src/research_engineer/agents/task_agent.py
  - id: openwiki-source-614771b0d5f5df521ba66521
    resource: repo://src/research_engineer/cli/__init__.py
  - id: openwiki-source-e972704d5a35e6521dab0f88
    resource: repo://src/research_engineer/gateway/models.py
  - id: openwiki-source-7fc3b05352792d239a1e19c2
    resource: repo://src/research_engineer/llm/base.py
  - id: openwiki-source-d72872cf029e5f0226404a76
    resource: repo://src/research_engineer/llm/factory.py
  - id: openwiki-source-af19d6833a13931533a3c2d2
    resource: repo://src/research_engineer/memory/factory.py
  - id: openwiki-source-2feed5672d9ef2dadd5b7d02
    resource: repo://src/research_engineer/models/experiment.py
  - id: openwiki-source-61704f20cac171be06e61737
    resource: repo://src/research_engineer/models/loop.py
  - id: openwiki-source-74f4e0955003658a54266998
    resource: repo://src/research_engineer/models/task.py
  - id: openwiki-source-e823f3b95d1ff96a2250704f
    resource: repo://src/research_engineer/runtime/models.py
  - id: openwiki-source-8bcfead868fac10d38887260
    resource: repo://src/research_engineer/safety/models.py
  - id: openwiki-source-c94e63458afc9a95e7ab6ccb
    resource: repo://src/research_engineer/safety/policies.py
  - id: openwiki-source-7f58e946a79986835476a96c
    resource: repo://src/research_engineer/service/agents.py
  - id: openwiki-source-ecec878837dc19e36a7a15e1
    resource: repo://src/research_engineer/service/config.py
  - id: openwiki-source-bbf1aa88296dbf52024a5ea2
    resource: repo://src/research_engineer/service/models.py
  - id: openwiki-source-ebebf65ceee173773a151fd4
    resource: repo://src/research_engineer/tools/experiment_runner.py
  - id: openwiki-source-9918f8e38d2becf743be7dfd
    resource: repo://src/research_engineer/tools/terminal.py
  - id: openwiki-source-f0a6e7dc03522b2682f88655
    resource: repo://tests/conftest.py
  - id: openwiki-source-c734ef322ed9337292c384f5
    resource: repo://tests/test_loop_models.py
  - id: openwiki-source-59d4dd4e48e1c857959b39ae
    resource: repo://tests/test_service_api.py
generated: { by: "openwiki/0.4.3", at: "2026-08-28T12:25:37.430Z" }
---

# Configuration and Runtime Controls

This page is the operator reference for controls that affect execution. Most local workflows are configured by CLI flags and Pydantic models; LLM and repository-memory selection comes from `llm_config.yaml`; the run service is configured exclusively through environment variables. See [Runtime behavior](../runtime/runtime-behavior.md), [LLM layer](../concepts/llm-layer.md), [Tools](../concepts/tools.md), and [Safety](safety.md) for the associated mechanisms.

## Precedence and safe starting point

| Concern | Configuration source | Effective default / consequence |
| --- | --- | --- |
| LLM providers and routing | `llm_config.yaml`; `RE_LLM_CONFIG` can select another file; `${VAR}` is expanded from process environment | Repository config selects Ollama and `glm-5.3-flash`; provider requests time out after 60 seconds. Unset secrets expand to an empty string. |
| Repository memory | Optional `embedding` and `vector_store` sections in the resolved LLM config | Offline `HashingEmbedder` plus in-memory vectors; state is lost on exit. Requested unavailable semantic/persistent backends fall back rather than preventing construction. |
| CLI task and loop runs | Typer options converted to `TaskConfig` / `LoopConfig` | Both start dry-run by default. Testing, delegation, and loop approvals are opt-in. |
| Generic runtime / service run | Per-request overrides, otherwise `ServiceConfig` environment settings | Service applies 25 steps and 600 seconds unless overridden. The generic runtime only constrains a dimension when its budget is set. |
| Service deployment | `RE_SERVICE_*`, `RE_*`, and `OTEL_EXPORTER_OTLP_ENDPOINT` | Local SQLite/artifact paths, no API token, CORS disabled, two workers, and safety-chain enforcement off. Set production controls explicitly. |

`research-engineer llm status` displays resolved provider/model routing, while `research-engineer llm config --config PATH` prints a selected YAML config. The process-wide provider factory is lazy and cached: a changed environment or config file is not reflected by an already-created factory until it is reset or the process is restarted.

## Orchestration controls

### Autonomous research loop

`research-engineer loop run GOAL --repo PATH` builds a `LoopConfig`. The loop model has bounds and stop criteria beyond the flags exposed by this command.

| Knob | Default and valid range | Effect |
| --- | --- | --- |
| `--max-iterations` / `max_iterations` | `5`; 1–100 | Absolute iteration cap. |
| `--target-metric`, `--target-value`, `--higher-is-better` | unset, unset, `False` | Defines target completion and metric direction; lower is better unless changed. |
| `--budget-hours`, `--budget-cost` | unset | Optional GPU-hour and USD stop budgets. |
| `approval_mode` / `--approval` | `False` | Enables human gates at plan, implementation, and next-iteration decisions. Approval requests offer `approve`, `modify`, and `stop`, and may include diff, cost, risk, and metric context. |
| `--dry-run/--no-dry-run` | `True` | Experiments are planned rather than executed unless explicitly disabled. |
| `skip_literature_after_first` / `--skip-literature` | `True` | Reuses initial discovery instead of repeating it each iteration. |
| `stagnation_window` | `3`, minimum 2 | Number of non-improving iterations before no-improvement stopping logic. |
| `improvement_threshold` | `1e-4`, >0 | Minimum metric change treated as improvement. |
| `stop_on_error` | `False` | Determines whether an iteration error ends the loop. |
| `cost_per_gpu_hour` | `2.0`, >0 | Converts tracked GPU-hours to USD for loop budget tracking. |
| `output_dir` | `output/loops` | Loop artifacts and reports destination. |

A loop can be created, running, iterating, awaiting approval, evaluated, stopped, or failed. Terminal stopping reasons are target achieved, max iterations, budget exceeded, or no improvement; records preserve configured JSON, accumulated cost, iterations, metrics, pending approval, and next command.

### Generic runtime and safety policy

The async runtime's `AgentBudget` can independently bound steps, tool calls, wall-clock seconds, USD cost, and tokens. `AgentPolicy` defaults to three recoverable errors, a three-step stagnation window, and a zero score-delta threshold; fatal errors terminate immediately. These are model defaults—not global environment settings. The service maps request `budget_overrides` into these fields for worker-created agents.

The deterministic `AutonomyPolicy` is a separate control layer. Its defaults replan after loop/duplicate detection or three consecutive failures; terminate after five stagnant steps or diminishing returns; allow two replans; warn at 80% of any budget; pause for approval on high/critical risk; and terminate on gateway policy failure. Setting `enabled=False` makes these controls no-ops. Mandatory hard-limit, deny-policy, and approval decisions cannot be changed by an LLM advisor.

## Task runs

`research-engineer task GOAL` is the terminal-first coding entrypoint. It analyzes the target repository, plans, generates a patch, produces a diff, and optionally tests; `--delegate` changes this to specialized research, architecture, coding, review, test, and repair stages.

| Knob | Default | Effect / operational warning |
| --- | --- | --- |
| `--repo` | `.` | Repository operated on. |
| `--paper` | unset | Adds paper ID, URL, or PDF context. |
| `--dry-run/--no-dry-run` | dry run | Patches are generated without applying by default. Use a deliberate `--no-dry-run` for mutation. |
| `--run-tests` | `False` | Enables the test step. |
| `--test-command` | `uv run pytest` | Command used when testing is enabled. |
| `timeout_seconds` | `600`, >=1 | Test-command timeout in `TaskConfig`; it is not exposed by the current `task` CLI. |
| `--stream/--no-stream` | streaming enabled | Controls planning-token output to stdout. |
| `--output-dir` | `output/tasks` | Task artifact destination. |
| `--delegate` | `False` | Enables the multi-agent pipeline. |
| `--max-repairs` / `max_repair_iterations` | `2`; 0–10 | Caps review/test repair cycles in delegated mode. |

## Tools and experiment execution

The experiment runner accepts only an allowlisted command prefix, defaults to dry-run, and uses a 3,600-second command timeout. `memory_limit_mb` is optional; on POSIX it is applied as an address-space limit. A real run is killed and recorded as `timeout` if it exceeds the configured timeout; stdout and stderr are truncated to `max_output_bytes` (10,000,000 by default). Batch execution defaults to at most four concurrent inputs.

The task `TerminalTool` is independently constrained: command prefixes are allowlisted, commands run in `repo_path`, command timeout defaults to 300 seconds, returned command/file output is capped at 1,000,000 bytes, and command dry-run is available. These tool-level defaults are distinct from the task model's 600-second test timeout.

At the central gateway, tools are default-denied unless registered. A `ToolPolicy` carries per-tool risk, permission, resource limits, and `requires_approval`; an empty workspace means filesystem access is denied. Gateway ordering is policy → permission → budget → approval → sandbox → tool → result validation, so policy/security failures must not be silently retried.

## Memory

The shared `llm_config.yaml` may set `embedding.backend` to `hashing` or `sentence_transformer` and `vector_store.backend` to `inmemory` or `chromadb`. The code default is dependency-free hashing plus in-memory vectors. Semantic embeddings default to `sentence-transformers/all-MiniLM-L6-v2` when selected without a model; Chroma defaults to collection `repository_memory` and uses `persist_path` when supplied. Unlike the older generic `VectorStoreConfig` defaults, repository-memory construction uses this YAML-driven factory.

## LLM configuration

The YAML config defines a default provider/model, named provider connection options, token prices, and per-agent assignments. Supported provider types are `ollama`, `openai`, `anthropic`, and `local_ollama`; additional types can be registered programmatically. Agent routing falls back from an agent assignment to configured defaults, then the first provider; if no provider is configured, a stock environment-configured Ollama provider is used.

The repository's checked-in configuration uses `ollama` at `https://ollama.com`, `glm-5.3-flash`, and a 60-second provider timeout. `${OLLAMA_API_KEY}` keeps the key out of the file. The `pricing` section overrides or extends built-in price data and is used to compute completion cost. Per-request `LLMRequest` defaults are temperature `0.2`, `top_p` `1.0`, non-streaming output, and no generation-token cap; callers must set `max_tokens` when a response-length bound is required.

## Service deployment

All service values are environment driven and Pydantic-validated at startup.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `RE_SERVICE_API_TOKEN` | empty | Empty disables API authentication; set a bearer token in production. Health and readiness remain unauthenticated. |
| `RE_SERVICE_CORS_ORIGINS` | empty | CORS is disabled unless a comma-separated allowlist is supplied. |
| `RE_SERVICE_MAX_BODY_BYTES` | 1 MiB; max 64 MiB | Maximum accepted JSON request size. |
| `RE_SERVICE_DB_PATH`, `RE_CHECKPOINT_DB`, `RE_SERVICE_ARTIFACT_DIR` | `data/run_service.db`, `data/checkpoints.db`, `data/artifacts` | SQLite run/checkpoint locations and artifact root. |
| `RE_POSTGRES_DSN` | empty | Switches runs, checkpoints, and queue to PostgreSQL when set. |
| `RE_WORKER_CONCURRENCY` | 2; 1–32 | Concurrent runs per worker. |
| `RE_STALE_RUN_TIMEOUT_SECONDS`, `RE_QUEUE_POLL_SECONDS` | 120, 0.5 | Recovery threshold for lost claims and worker queue polling interval. |
| `RE_DEFAULT_MAX_STEPS`, `RE_DEFAULT_MAX_RUNTIME_SECONDS` | 25, 600 | Default run budgets, superseded by request overrides. |
| `RE_STEP_DELAY_SECONDS` | 0 | Optional per-step delay for tests/demos. |
| `RE_SERVICE_ENFORCE_SAFETY` | false | When true, worker execution fails closed unless both gateway and safety controller are wired. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | Optional telemetry export endpoint. |

The service listens on `0.0.0.0:8000` by default. Requests may override `max_steps` (1–1000) and `max_runtime_seconds`, and pass factory-specific scalar `budget_overrides`; they cannot expose internal prompts, credentials, or tool interfaces through the public response models.

## Tests and CI controls

`pyproject.toml` sets pytest discovery under `tests`, verbose short-trace output, and `src` on the test path. `pytest-timeout` applies a 300-second timeout to every test using the thread method so one hung test cannot indefinitely stall CI. Tests marked `network` are skipped when a session-level three-second TCP probe cannot reach either `arxiv.org:443` or `export.arxiv.org:443`; this keeps offline CI deterministic. Focused CLI, loop-model, provider, service API, safety, task-agent, and tool tests cover the defaults and boundaries summarized here.
