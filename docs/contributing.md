# Contributing

Contributions are welcome — especially new LLM providers, new tools, new retrieval strategies, and new agents. This guide covers setup, conventions, and how to extend each layer.

> See [Architecture](architecture.md) for the big picture and [System Design](system_design.md) for contracts.

---

## 1. Development setup

```bash
# Clone
git clone https://github.com/<your-org>/AutonomousMLResearchEngineer.git
cd AutonomousMLResearchEngineer

# Install
uv sync

# Verify
research-engineer --help
research-engineer llm status
```

### Run the checks before every PR

```bash
uv run pytest -q                              # 878 tests
uv run ruff check .                            # lint
uv run mypy src/research_engineer/llm          # type-check the LLM layer
```

> **Target:** all tests passing, ruff clean, mypy clean on changed files, > 90% coverage on new code.

---

## 2. Conventions

These are enforced by the existing codebase and reviewed in PRs:

| Convention | Detail |
|------------|--------|
| **Python** | ≥ 3.12 |
| **Async-first** | All tools and agents use `async`/`await`. |
| **Pydantic v2** | All I/O models are Pydantic v2 `BaseModel`. |
| **Enums** | All enums use `StrEnum` (not `str, Enum`). |
| **Typed tools** | Subclass `Tool[InputType, OutputType]`; implement `async execute()`. |
| **Line length** | 88 chars (ruff). |
| **Type checking** | `disallow_untyped_defs = true`, `strict_optional = true`. |
| **No direct model calls** | Agents obtain LLM providers via `resolve_llm()`; model switching is config-only. |
| **Patch-first** | Never mutate user code directly; produce reviewable unified diffs. |
| **Repository-agnostic** | Never hardcode assumptions about specific repos. |
| **Paper-agnostic** | Works for any ML paper (attention, MoE, diffusion, etc.). |
| **Tests** | Must pass before PR (target > 90% coverage). |

---

## 3. Project layout

```
src/research_engineer/
├── agents/      # 23 agents + delegation + self-repair + research workflow + _llm_support.py
├── llm/         # Phase 10: base, ollama_provider, factory, router
├── memory/      # Phase 12: indexer, symbol_graph, retriever, storage
├── models/      # 186 Pydantic models across 18 modules
├── tools/       # 61 typed tools
└── cli/         # 56 Typer commands
tests/           # 45+ test files, 878 tests
llm_config.yaml  # provider + per-agent model config
docs/            # 13 documentation files
```

---

## 4. How to add a new LLM provider

1. Implement `LLMProvider`:

```python
# src/research_engineer/llm/my_provider.py
from research_engineer.llm.base import LLMProvider, LLMRequest, LLMResponse, ProviderError

class MyProvider(LLMProvider):
    name = "myprov"
    default_model = "my-model"

    def __init__(self, *, api_key: str | None = None, default_model: str | None = None) -> None:
        self.api_key = api_key or ""
        if default_model:
            self.default_model = default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Call your vendor's API and return an LLMResponse
        ...
```

2. Register it (in `llm/__init__.py` or a plugin):

```python
from research_engineer.llm import register_provider_type
register_provider_type("myprov", MyProvider)
```

3. Add it to `llm_config.yaml`:

```yaml
providers:
  myprov:
    type: myprov
    api_key: ${MYP_API_KEY}
    default_model: my-model
agents:
  ResearchAgent: {provider: myprov, model: my-model}
```

4. Add tests in `tests/test_llm.py` (use an injectable HTTP client or mock).

**No agent code changes required.**

---

## 5. How to add a new tool

1. Define typed I/O models (Pydantic v2):

```python
class MyInput(BaseModel):
    query: str

class MyOutput(BaseModel):
    result: str
```

2. Subclass `Tool[Input, Output]`:

```python
from research_engineer.tools.base import Tool, ToolError

class MyTool(Tool[MyInput, MyOutput]):
    async def execute(self, input: MyInput) -> MyOutput:
        if not input.query:
            raise ToolError("query required", input)
        return MyOutput(result=f"processed: {input.query}")

    async def validate(self, input: MyInput) -> bool:
        return bool(input.query)
```

3. Export from `tools/__init__.py`.

4. Wire into the relevant agent constructor.

5. Add tests under `tests/`.

---

## 6. How to add a new agent

1. Create `agents/my_agent.py`:

```python
from research_engineer.llm import LLMProvider
from research_engineer.agents._llm_support import resolve_llm

class MyAgent:
    def __init__(self, ..., llm: LLMProvider | None = None) -> None:
        self.agent_name = "MyAgent"
        ...
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def do_work(self, ...) -> MyResult:
        ...
```

2. Export from `agents/__init__.py`.

3. Add a `_get_my_agent()` helper and CLI command(s) in `cli/__init__.py`.

4. Add the agent name to `ModelRouter.KNOWN_AGENTS` in `llm/router.py`.

5. Add a `agents:` entry in `llm_config.yaml`.

6. Add tests: `tests/test_my_agent.py`, `tests/test_my_agent_cli.py`.

---

## 7. How to add a retrieval strategy

1. Subclass `RetrievalStrategy` in `tools/retrieval_strategies.py`:

```python
class MyStrategy(RetrievalStrategy):
    async def retrieve(self, query: RetrievalQuery) -> list[MemoryResult]:
        ...
```

2. Register in `STRATEGY_REGISTRY`.

3. Add tests in `tests/test_retrieval_strategies.py`.

---

## 8. PR checklist

Before opening a PR:

- [ ] `uv run pytest -q` passes (all 690 + your new tests).
- [ ] `uv run ruff check .` is clean on changed files.
- [ ] `uv run mypy <changed files>` is clean.
- [ ] New code has tests (target > 90% coverage).
- [ ] No direct model calls — agents use `resolve_llm()`.
- [ ] No hardcoded repo/paper assumptions.
- [ ] New tools subclass `Tool[Input, Output]` with typed Pydantic I/O.
- [ ] New enums use `StrEnum`.
- [ ] Documentation updated (if touching public APIs).
- [ ] Commit message follows the repo style.

---

## 9. Reporting issues

When filing an issue, include:

- Platform (macOS / Linux / Windows).
- Python version (`python --version`).
- The exact command that failed.
- The full error traceback.
- `research-engineer llm status` output (if LLM-related).

---

## 10. Code of conduct

Be kind. Be specific. Assume good intent. Critique code, not people.

---

*Version: 2.0 · Contributions welcome*