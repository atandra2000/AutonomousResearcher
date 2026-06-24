# Provider-Agnostic LLM Layer (Phase 10)

The Autonomous ML Research Engineer platform now talks to language models
through a **provider-agnostic abstraction**. Agents never call a vendor SDK
directly — they ask the router for a provider and call `complete()`. The
concrete provider (Ollama Cloud by default) is selected and configured via
`llm_config.yaml`, so **switching models or providers is a config-only
change**.

## Files

| Path | Purpose |
|------|---------|
| `src/research_engineer/llm/base.py` | `LLMProvider` ABC + `LLMRequest`/`LLMResponse`/`LLMMessage`/`LLMRole`/`LLMUsage`/`ProviderError` |
| `src/research_engineer/llm/ollama_provider.py` | `OllamaCloudProvider` (OpenAI-compatible Chat Completions over httpx) |
| `src/research_engineer/llm/factory.py` | `ProviderFactory`, config loading, env expansion, provider registry |
| `src/research_engineer/llm/router.py` | `ModelRouter` — resolves provider+model per agent, binds model on every request |
| `src/research_engineer/llm/__init__.py` | Public exports |
| `src/research_engineer/agents/_llm_support.py` | `resolve_llm()` helper used by every agent |
| `llm_config.yaml` | Default provider + per-agent model assignments |
| `tests/test_llm.py` | 29 unit tests (models, Ollama provider w/ mock transport, factory, router, agent wiring) |
| `docs/llm_integration.md` | This document |

## Architecture

```
                    llm_config.yaml
                          │
                          ▼
                   ProviderFactory ──builds──► OllamaCloudProvider (cached)
                          │                         │
                          ▼                         │
                    ModelRouter ───for_agent()──────►│ (bound to model)
                          │                         ▼
   agents/__init__  resolve_llm(agent_name, llm) ──► _BoundProvider
                          │
                          ▼
              agent.llm_provider  ◄── every agent exposes this
                          │
                          ▼
              await provider.complete(LLMRequest) ──► LLMResponse
```

### Resolution rules (`resolve_llm`)

1. An explicit `LLMProvider` passed to the agent constructor **wins**.
2. If the agent is constructed with `llm_enabled=False` (e.g.
   `RepositoryAgent` default), no provider is attached.
3. Otherwise the process-wide `ModelRouter` resolves a provider for the
   agent's `agent_name` from `llm_config.yaml`.

### Per-agent model binding

`ModelRouter.for_agent(name)` returns a `_BoundProvider` that pins the
configured `model` onto every `LLMRequest` whose `model` is `None`. This
means agents call `provider.complete(request)` without naming a model and
the router applies the configured one — **no agent calls a model directly**.

## Configuration

`llm_config.yaml` (root of the repo; override path with `$RE_LLM_CONFIG`):

```yaml
default_provider: ollama
default_model: llama3

providers:
  ollama:
    type: ollama
    base_url: https://api.olama.cloud
    api_key: ${OLLAMA_API_KEY}      # expanded from the environment
    default_model: llama3
    timeout: 60

agents:
  ResearchAgent:          {provider: ollama, model: llama3}
  RepositoryAgent:        {provider: ollama, model: llama3}
  ExperimentPlannerAgent: {provider: ollama, model: llama3}
  CodingAgent:            {provider: ollama, model: llama3}
  MemoryAgent:            {provider: ollama, model: llama3}
  LiteratureAgent:        {provider: ollama, model: llama3}
  ExperimentAgent:        {provider: ollama, model: llama3}
  EvaluationAgent:        {provider: ollama, model: llama3}
  ResearchLoopAgent:      {provider: ollama, model: llama3}
```

### Environment variables

| Variable | Used by | Default |
|----------|---------|---------|
| `RE_LLM_CONFIG` | factory | path to `llm_config.yaml` |
| `OLLAMA_BASE_URL` | OllamaCloudProvider | `https://api.olama.cloud` |
| `OLLAMA_API_KEY` | OllamaCloudProvider | (none) |
| `OLLAMA_MODEL` / `OLLAMA_DEFAULT_MODEL` | OllamaCloudProvider | `llama3` |
| `OLLAMA_TIMEOUT` | OllamaCloudProvider | `60` |

`${VAR}` placeholders in the YAML are expanded against the process
environment, so secrets never need to be committed.

## Running on Ollama Cloud

1. `export OLLAMA_API_KEY=<your-key>`
2. (optional) `export OLLAMA_BASE_URL=https://api.olama.cloud`
3. `research-engineer llm status` — verify routing per agent
4. Any agent now resolves an Ollama Cloud provider automatically.

Example call site (see `repository_agent.py` Step 13):

```python
from research_engineer.llm import LLMMessage, LLMRequest, LLMRole
resp = await self.llm_provider.complete(LLMRequest(
    messages=[
        LLMMessage(role=LLMRole.SYSTEM, content="You are a senior ML engineer..."),
        LLMMessage(role=LLMRole.USER, content=f"Analyse:\n{code}"),
    ],
    temperature=0.2,
))
print(resp.content, resp.usage.total_tokens)
```

## Using different models per agent

Edit `llm_config.yaml` only — no source changes. For example, give the
CodingAgent a coder-tuned model:

```yaml
agents:
  CodingAgent: {provider: ollama, model: qwen2.5-coder}
```

Then `research-engineer llm status` shows:

```
CodingAgent              -> ollama / qwen2.5-coder
```

All other agents keep their configured models. Reload is automatic on
process restart (the factory is a lazy singleton; tests call
`reset_factory()`/`reset_router()`).

## Adding a new provider

1. Implement `LLMProvider` (override `complete`, optionally `stream`).
2. Register it: `register_provider_type("myprov", MyProvider)` (e.g. in a
   plugin's import, or `llm/__init__.py`).
3. Add it to `llm_config.yaml`:

   ```yaml
   providers:
     myprov: {type: myprov, api_key: ${MYP_KEY}, default_model: x}
   agents:
     ResearchAgent: {provider: myprov, model: x}
   ```

No agent code changes required.

## Agent integration summary

Every agent constructor gained an optional `llm: LLMProvider | None = None`
keyword argument and exposes:

- `agent_name: str` — canonical name for routing
- `llm_provider: LLMProvider | None` — resolved provider (may be a
  model-bound `_BoundProvider`)

| Agent | `agent_name` | Default routing |
|-------|--------------|-----------------|
| ResearchAgent | `ResearchAgent` | ollama / llama3 |
| RepositoryAgent | `RepositoryAgent` | *(none unless `llm_enabled=True`)* |
| ExperimentPlannerAgent | `ExperimentPlannerAgent` | ollama / llama3 |
| CodingAgent | `CodingAgent` | ollama / llama3 |
| MemoryAgent | `MemoryAgent` | ollama / llama3 |
| LiteratureAgent | `LiteratureAgent` | ollama / llama3 |
| ExperimentAgent | `ExperimentAgent` | ollama / llama3 |
| EvaluationAgent | `EvaluationAgent` | ollama / llama3 |
| ResearchLoopAgent | `ResearchLoopAgent` | ollama / llama3 |

### Backwards compatibility

- All existing constructor signatures are preserved (new params are
  keyword-only with `None` defaults).
- `RepositoryAgent.llm` (legacy LlamaIndex tool) is retained for callers
  that still pass `llm_enabled=True` without a provider; when a provider is
  available the new path is used preferentially.
- `RepositoryAgent._llm_enabled` and the `llm_enabled`/`llm_model` CLI flags
  behave exactly as before.
- The full existing test suite (658 tests) passes unchanged; 29 new tests
  cover the LLM layer.

## CLI

```bash
research-engineer llm status            # show per-agent provider/model
research-engineer llm status --format json
research-engineer llm config             # dump resolved YAML config
research-engineer llm config --config path/to/llm_config.yaml
```

## Verification

```bash
uv run pytest -q            # 687 passed (658 existing + 29 new)
uv run mypy src/research_engineer/llm   # clean
uv run ruff check src/research_engineer/llm src/research_engineer/agents/_llm_support.py tests/test_llm.py   # clean
```