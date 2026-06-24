# Quick Start Guide

Get from zero to a running autonomous research loop in under 10 minutes.

---

## 1. Installation

### Prerequisites

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`

### Steps

```bash
# 1. Clone
git clone https://github.com/<your-org>/AutonomousMLResearchEngineer.git
cd AutonomousMLResearchEngineer

# 2. Install dependencies (uv recommended)
uv sync

# …or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Verify
research-engineer --help
```

### Optional — enable Ollama Cloud (Phase 10)

The platform runs without an LLM for Phases 1–3 (rule-based extraction). To enable LLM-powered analysis, code generation, and the autonomous loop:

```bash
export OLLAMA_API_KEY="your-key"
# Optional overrides
export OLLAMA_BASE_URL="https://api.olama.cloud"
export OLLAMA_MODEL="llama3"

# Verify routing
research-engineer llm status
```

See [LLM Integration](llm_integration.md) for per-agent model configuration.

---

## 2. Your first paper analysis (Phase 1)

```bash
# arXiv ID
research-engineer analyze 2503.12345

# arXiv URL
research-engineer analyze https://arxiv.org/abs/2503.12345

# Local PDF
research-engineer analyze ./papers/attention.pdf --output-dir output/
```

**Output** (`output/`):

```
output/
├── 2503.12345_summary.json   # 14-field ResearchSummary
└── 2503.12345_plan.json       # EngineeringReport (complexity, files, effort)
```

**Output formats:** `console` (default), `json`, `markdown`.

```bash
research-engineer analyze 2503.12345 --format json
```

---

## 3. Analyze a repository (Phase 2)

```bash
research-engineer analyze-repo ./my_model_repo --output-format markdown
```

Produces architecture overview, dependency graph, training-pipeline extraction, config analysis, knowledge graph, and generated documentation under `output/<repo_name>/`.

---

## 4. Plan an experiment (Phase 3)

Combine a paper and a repository into a 9-file experiment plan:

```bash
research-engineer plan 2503.12345 ./my_model_repo
```

**Output** (`output/plans/<paper_id>_<repo>/`):

| File | Content |
|------|---------|
| `compatibility_analysis.md` | 7-dimension compatibility report |
| `implementation_plan.md` | Ordered implementation steps |
| `experiment_matrix.md` | 5 experiment groups (baseline, MVE, ablation, stress, scaling) |
| `validation_strategy.md` | 6 test suites |
| `risk_assessment.md` | 7 risk categories + mitigations |
| `cost_estimation.md` | GPU hours, memory, storage, cloud cost |
| `expected_results.md` | Best/likely/worst scenarios + failure modes |
| `engineering_report.md` | Consolidated engineering report |
| `plan_result.json` | Full `PlanResult` serialized |

---

## 5. Implement changes as patches (Phase 4)

```bash
# From a paper + repo
research-engineer implement --paper 2503.12345 --repo ./my_model_repo

# From a free-text task
research-engineer implement --task "Add rotary positional embeddings" --repo ./my_model_repo

# From an implementation plan file
research-engineer implement --plan output/plans/.../implementation_plan.md --repo ./my_model_repo
```

Produces reviewable **unified-diff patches**, a self-review, test suites, migration + rollback plans, and an implementation report under `output/<implementation_id>/`. **Patches are never applied automatically** — application is a separate, explicit, approval-gated step.

---

## 6. Search the literature (Phase 6)

```bash
# Search across local store, arXiv, and Semantic Scholar
research-engineer literature search "mixture of experts"

# Generate a structured review
research-engineer literature review "sparse mixture of experts" --depth comprehensive

# Compare papers
research-engineer literature compare 2401.04088,2302.09913

# Full discovery workflow (search → compare → review → trends → recommend → relevance)
research-engineer literature discover "mixture of experts routing" --repo ./my_moe_repo
```

---

## 7. Run an experiment (Phase 7)

```bash
# Dry-run by default — the command is echoed, not executed
research-engineer experiment run \
  --command "python train.py --config configs/exp.yaml" \
  --repo ./my_model_repo

# Actually execute (command must be in the allowlist: python, torchrun, accelerate, ...)
research-engineer experiment run \
  --command "python train.py --config configs/exp.yaml" \
  --repo ./my_model_repo \
  --no-dry-run

# Monitor / list / inspect
research-engineer experiment list --status completed
research-engineer experiment get <experiment_id>
research-engineer experiment monitor <experiment_id>
```

---

## 8. Evaluate results (Phase 8)

```bash
# Full evaluation of one experiment
research-engineer evaluate analyze <experiment_id>

# Compare two runs
research-engineer evaluate compare exp_aaa exp_bbb

# Statistical significance (Welch t-test, Cohen's d, 95% CIs)
research-engineer evaluate significance exp_aaa exp_bbb

# Training dynamics (over/underfit, convergence, instability)
research-engineer evaluate dynamics <experiment_id>

# Recommend next experiments
research-engineer evaluate next exp_aaa exp_bbb
```

---

## 9. Run the autonomous research loop (Phase 9)

The orchestrator that chains Phases 1–8 in iterative cycles:

```bash
# Dry-run loop (default)
research-engineer loop run "Improve training stability" \
  --repo ./my_model_repo \
  --max-iterations 3

# Loop with a target metric + budget
research-engineer loop run "Reduce validation loss below 0.10" \
  --repo ./my_model_repo \
  --target-metric loss \
  --target-value 0.10 \
  --higher-is-better false \
  --budget-hours 8.0 \
  --max-iterations 10

# Loop with human-approval gates
research-engineer loop run "Boost accuracy" \
  --repo ./my_model_repo \
  --approval \
  --max-iterations 5
```

**Stopping conditions:** `target_achieved`, `max_iterations_reached`, `budget_exceeded`, `no_improvement`.

Inspect results:

```bash
research-engineer loop list --status stopped
research-engineer loop iterations <loop_id>
research-engineer loop report <loop_id> --output-dir ./reports
```

The report (`output/loops/<loop_id>/research_report.md`) contains an executive summary, methodology, iteration history, findings, failures, conclusions, and the full config.

---

## 10. Programmatic usage

```python
import asyncio
from research_engineer.agents import ResearchAgent, ResearchLoopAgent

async def main():
    # Phase 1 — analyze a paper
    agent = ResearchAgent()
    result = await agent.analyze("2503.12345")
    print(result["title"])
    print(result["summary"]["executive_summary"])

    # Phase 9 — autonomous loop
    loop = ResearchLoopAgent()
    loop_result = await loop.run(
        goal="Improve training stability",
        repo_path="./my_model_repo",
    )
    print(loop_result.status, loop_result.iteration_count)

asyncio.run(main())
```

### Injecting a custom LLM provider

```python
from research_engineer.agents import CodingAgent
from research_engineer.llm import OllamaCloudProvider

custom = OllamaCloudProvider(default_model="qwen2.5-coder")
agent = CodingAgent(llm=custom)   # explicit provider wins over router
```

---

## 11. Memory & history

```bash
# Paper history
research-engineer history --limit 20
research-engineer get 2503.12345
research-engineer search "transformer"

# Research memory (Phase 5)
research-engineer memory search "rotary embeddings"
research-engineer memory --stats
research-engineer memory --related <memory_id>
research-engineer memory graph
```

---

## 12. Troubleshooting

| Problem | Fix |
|---------|-----|
| PDF not extracting text | Ensure it's not scanned/image-only; re-download from arXiv. |
| arXiv API rate limiting | Wait 1–2 s between requests; cache locally; use arXiv IDs. |
| Database locked | Close other processes using `data/research_engineer.db`; check write permissions. |
| Import errors | Ensure Python 3.12+; run `uv sync --reinstall`. |
| LLM not routing | Run `research-engineer llm status`; check `OLLAMA_API_KEY` and `llm_config.yaml`. |
| Experiment command rejected | The runner allowlists only `python`, `python3`, `torchrun`, `accelerate`, `pytest`, `bash`, `sh`, `make`, `uv`, `pip`. |

---

## 13. Development

```bash
uv run pytest -q          # 690 tests
uv run ruff check .       # lint
uv run mypy src/research_engineer/llm   # type-check LLM layer
```

See [Contributing](contributing.md) for conventions and PR checklist.

---

## Next steps

- **[Architecture](architecture.md)** — understand the 10-phase design.
- **[CLI Reference](cli_reference.md)** — every command and flag.
- **[LLM Integration](llm_integration.md)** — configure per-agent models.
- **[Agents](agents.md)** — deep-dive into each agent.