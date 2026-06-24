# CLI Reference

Complete reference for all **49 CLI commands** in the Autonomous ML Research Engineer, organized by sub-app.

> Run `research-engineer <command> --help` for the full flag list.

---

## Core commands

### `analyze`

Analyze an ML paper and generate an implementation plan.

```bash
research-engineer analyze <paper> [--output-format json|markdown|console] [--output-dir DIR]
```

**`<paper>`** — arXiv ID (`2503.12345`), arXiv URL (`https://arxiv.org/abs/2503.12345`), or PDF path (`./paper.pdf`).

```bash
research-engineer analyze 2503.12345
research-engineer analyze https://arxiv.org/abs/2503.12345
research-engineer analyze ./papers/attention.pdf --output-format markdown
```

### `analyze-repo`

Analyze a repository and generate documentation.

```bash
research-engineer analyze-repo <repo_path> [--output-dir DIR] [--output-format markdown|json|console] [--enable-caching] [--rate-limit] [--llm-enabled] [--llm-model MODEL] [--cache-path PATH]
```

```bash
research-engineer analyze-repo ./my_repo --output-format markdown
research-engineer analyze-repo ./my_repo --llm-enabled
```

### `plan`

Generate a 9-file experiment plan from a paper + repository.

```bash
research-engineer plan <paper> <repo_path> [--output-format markdown|json] [--output-dir DIR]
```

```bash
research-engineer plan 2503.12345 ./my_repo
research-engineer plan paper.pdf ./DeepSeek --output-format json
research-engineer plan https://arxiv.org/abs/2503.12345 ./repo --output-dir output/
```

### `implement`

Generate code changes as patches + tests + reports.

```bash
research-engineer implement --repo <path> [--paper PAPER] [--task TASK] [--plan PLAN_FILE] [--output-dir DIR]
```

```bash
research-engineer implement --paper 2503.12345 --repo ./repo
research-engineer implement --task "Add Grouped Query Attention" --repo ./repo
research-engineer implement --plan output/plans/.../implementation_plan.md --repo ./repo
```

### `get`

Retrieve a previously analyzed paper.

```bash
research-engineer get <paper_id>
```

### `search`

Search stored papers.

```bash
research-engineer search <query>
```

### `history`

Show analysis history.

```bash
research-engineer history [--limit N]
```

### `cache-status`

Show cache statistics.

```bash
research-engineer cache-status
```

---

## `memory` sub-app

Research memory management (Phase 5).

```bash
research-engineer memory search <query>           # Search memories
research-engineer memory list [--type T] [--limit N] [--tag TAG]
research-engineer memory --stats                    # Memory statistics
research-engineer memory --related <memory_id>      # Related memories
research-engineer memory graph                       # Knowledge graph stats
research-engineer memory export [--output FILE]      # Export to JSON
research-engineer memory import [--input FILE] [--skip-existing]
research-engineer memory archive <memory_id>         # Archive a memory
```

---

## `literature` sub-app

Literature intelligence (Phase 6).

```bash
research-engineer literature search <query> [--sources local,arxiv,semantic_scholar] [--max-papers N]
research-engineer literature compare <paper_ids>            # comma-separated
research-engineer literature review <topic> [--depth brief|standard|comprehensive]
research-engineer literature relationships <paper_ids>
research-engineer literature trends <topic>
research-engineer literature recommend <topic> [--repo PATH]
research-engineer literature relevance <paper_id> --repo <path>
research-engineer literature discover <topic> [--repo PATH] [--max-papers N]  # full workflow
```

---

## `experiment` sub-app

Experiment execution (Phase 7).

```bash
research-engineer experiment run --command <cmd> --repo <path> [--paper ID] [--plan ID] [--patch ID] [--implementation ID] [--type training|evaluation|inference|ablation|benchmark] [--timeout SECONDS] [--no-dry-run] [--output-dir DIR]
research-engineer experiment monitor <experiment_id>
research-engineer experiment list [--status STATUS] [--limit N]
research-engineer experiment get <experiment_id>
research-engineer experiment search <query>
research-engineer experiment cancel <experiment_id>
research-engineer experiment history [--paper ID] [--limit N]
```

> **Dry-run is the default.** Pass `--no-dry-run` to actually execute. Commands must be in the allowlist: `python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`.

---

## `evaluate` sub-app

Evaluation (Phase 8).

```bash
research-engineer evaluate run <experiment_id>            # Evaluate one experiment
research-engineer evaluate compare <id1> <id2> [...]      # Compare experiments
research-engineer evaluate analyze <ids>                  # Full evaluation
research-engineer evaluate dynamics <id>                   # Training dynamics
research-engineer evaluate significance <ids>              # Statistical significance
research-engineer evaluate next <ids>                      # Recommend next experiments
research-engineer evaluate list [--paper ID] [--repo PATH] [--experiment ID]
research-engineer evaluate get <evaluation_id>
research-engineer evaluate search <query>
```

---

## `loop` sub-app

Autonomous research loop (Phase 9).

```bash
research-engineer loop run <goal> --repo <path> [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo` | required | Repository path. |
| `--max-iterations` | 5 | Maximum iterations (1–100). |
| `--target-metric` | None | Metric name to optimize. |
| `--target-value` | None | Target metric value. |
| `--higher-is-better` | False | Higher metric is better. |
| `--budget-hours` | None | GPU-hour budget. |
| `--budget-cost` | None | USD budget. |
| `--approval` | False | Enable human-approval gates. |
| `--dry-run` | True | Dry-run experiments. |

```bash
research-engineer loop run "Improve training stability" --repo ./repo --max-iterations 3 --dry-run
research-engineer loop run "Reduce loss" --repo ./repo --target-metric loss --target-value 0.01
research-engineer loop run "Boost accuracy" --repo ./repo --approval --max-iterations 5
```

**Query commands:**

```bash
research-engineer loop list [--status STATUS]
research-engineer loop get <loop_id>
research-engineer loop iterations <loop_id>
research-engineer loop iteration <iteration_id>
research-engineer loop search <query>
research-engineer loop report <loop_id> [--output-dir DIR] [--format console|json]
```

---

## `llm` sub-app

LLM layer inspection (Phase 10).

```bash
research-engineer llm status [--format console|json]
research-engineer llm config [--config PATH]
```

**`llm status`** shows the per-agent provider/model routing:

```
LLM layer status
  default provider: ollama
  default model:    llama3
  providers:        ollama

Per-agent routing:
  ResearchAgent            -> ollama / llama3
  CodingAgent              -> ollama / qwen2.5-coder
  ...
```

**`llm config`** dumps the resolved YAML config as JSON.

---

## Global options

```bash
research-engineer --help     # Top-level help
research-engineer --version  # Version
```

---

## Programmatic equivalents

Every CLI command has a programmatic equivalent via the agents:

```python
import asyncio
from research_engineer.agents import ResearchAgent, ResearchLoopAgent

async def main():
    # analyze
    r = await ResearchAgent().analyze("2503.12345")
    print(r["title"])

    # loop run
    res = await ResearchLoopAgent().run("Improve stability", repo_path="./repo", )
    print(res.status, res.iteration_count)

asyncio.run(main())
```

See [Agents](agents.md) for the full agent API.

---

*Version: 1.0 · 49 commands · 6 sub-apps*