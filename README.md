<div align="center">

# Autonomous ML Research Engineer

**A codex-style CLI for autonomous ML research.** Reads papers, understands
codebases, plans experiments, writes patches, runs training, evaluates
results, and conducts end-to-end research workflows — entirely in your
terminal, orchestrated by LangGraph.

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-8A2BE2)](https://www.langchain.com/langgraph)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Why

Existing ML-research tooling is either notebooks (not reproducible), SaaS
(your data leaves), or scaffolding (you still write the loop).
`research-engineer` is a single binary that closes the loop: it is an
agent you talk to like codex — one prompt becomes literature review,
hypotheses, an experiment plan, patches, runs, evaluation, and a report —
with every stage typed, checkpointed, and approval-gated.

**Stack:** Python 3.12 · typer CLI · **LangGraph** (the only research
engine) · LangChain providers · pydantic v2 · rich + prompt_toolkit
terminal UX. No server, no frontend, no Docker — a pure CLI product.

## Install

```bash
git clone <this-repo> && cd AutonomousMLResearchEngineer
uv sync
uv run research-engineer --help
```

## Quick start

```bash
# 1. The interactive session (codex-style REPL)
uv run research-engineer chat --repo ./my-repo
# Later: uv run research-engineer chat --resume session_ab12cd34ef56

#    research-engineer> /research "Design a more efficient diffusion transformer"
#    research-engineer> /analyze 2503.12345
#    research-engineer> What loss schedules work best for MoE routing?
#    research-engineer> /exit

# 2. One-shot paper analysis (rule-based, no LLM cost)
uv run research-engineer analyze 2503.12345

# 3. End-to-end research workflow (LangGraph engine)
export OLLAMA_API_KEY="..."
uv run research-engineer research "Improve attention efficiency" \
  --max-papers 30 --dry-run
# -> output/research/<workflow_id>/research_report.md

# 4. Autonomous loop with approval gates
uv run research-engineer loop run "Reduce val loss below 0.1" \
  --repo ./my-repo --max-iterations 5 --approval
```

## The chat session

`research-engineer chat` opens a terminal REPL over the same agents the
one-shot commands use:

| Command | What it does |
|---------|--------------|
| `/task [--apply] [--tests] [--delegate] <goal>` | Run the terminal-first coding agent in the active repository; dry-run by default |
| `/repo [path]` | Show or change the active repository workspace |
| `/research <goal>` | Full autonomous workflow: literature → synthesis → hypotheses → planning → execution → analysis → report |
| `/analyze <paper>` | Analyze an arXiv ID, arXiv URL, or local PDF |
| `/llm <prompt>` | Ask the configured LLM directly |
| `/clear` | Clear the bounded LLM conversation context and turn history |
| `/status` | Session + provider status |
| `/help`, `/exit` | The usual |

Free-form lines continue a bounded LLM conversation. Coding turns reuse the
same `TaskAgent` as the one-shot `task` command, including repository memory,
delegation, review/repair, and patch-first safety. Output renders as rich
panels/tables; input history uses prompt_toolkit. Workspace, turn history, and
conversation context are saved atomically under `output/sessions/`; the banner
shows the session ID accepted by `chat --resume`.

## What's inside

**15 phases, one CLI.** Paper analysis (1), repository analysis (2),
experiment planning (3), patch-first implementation (4), research memory
(5), literature intelligence (6), experiment execution (7), evaluation
(8), the autonomous loop (9), a provider-agnostic LLM layer (10),
terminal-first coding (11), repository memory with hybrid retrieval (12),
multi-agent delegation (13), self-repair (14), and the end-to-end
research workflow (15) — 23 agents, 61 typed tools, 250+ pydantic models.

**LangGraph is the research engine.** The seven research stages run as a
LangGraph `StateGraph` (`graphs/research.py`) with checkpointed,
resumable threads (`--thread-id`; PostgreSQL snapshots via
`RE_LANGGRAPH_CHECKPOINT_DSN` + the `postgres` extra). The workflow
framework remains the stage-executor authority — the graph owns
ordering, state, and persistence.

**Benchmarks over the production path.** The P1/P2 benchmark suites
(`research-engineer benchmark`) execute every case directly through the
same `AgentRuntime → ToolGateway → SafetyController → checkpointing`
stack an autonomous run uses, then grade the persisted payloads — no
separate serving tier, no difference between what is measured and what
is shipped.

## Configuration

**LLM routing** — `llm_config.yaml` maps agents to models
(`RE_LLM_CONFIG` overrides the path):

```yaml
routing:
  ResearchAgent: glm-5.3-flash
  CodingAgent: kimi-k2.7-code
default_model: glm-5.3-flash
```

Providers: Ollama Cloud (default), OpenAI, Anthropic, local Ollama, and
any LangChain chat model — all behind one `LLMProvider` ABC with health
checks, failover, cost tracking, and streaming. Phases 1–3 stay
rule-based (zero API cost); LLM is spent where it adds value.

**Key env vars:** `OLLAMA_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_TIMEOUT`,
`OPENAI_API_KEY`, `RE_LLM_CONFIG`, `RE_LANGGRAPH_CHECKPOINT_DSN`.

## Development

```bash
uv run python -m pytest -q   # full suite (offline-safe; network tests skip)
uv run ruff check .          # lint
scripts/ci_mypy.sh           # mypy — fails only on NEW errors vs baseline
```

## Docs

See [`docs/`](docs/) — architecture, CLI reference, LLM integration,
benchmark methodology, and the LangGraph framework-stack notes.

## License

MIT
