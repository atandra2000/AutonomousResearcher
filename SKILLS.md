# SKILLS.md — AutonomousMLResearchEngineer

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific workflows.
>
> This is a companion to `AGENTS.md` (in this folder) — that file holds the
> 15-phase architecture diagrams, source tree, and CLI command list. This
> file holds the interactive workflows that aren't obvious from a single
> `AGENTS.md` read.

---

## Skill 1: Run an Autonomous Research Workflow (Phase 15)

End-to-end paper→report pipeline: goal → literature search → repo analysis
→ plan → patches → experiments → evaluation → research report.

```bash
cd AutonomousMLResearchEngineer
uv run research-engineer research "Optimize Stable Diffusion UNet by adding Min-SNR loss weighting" --repo ../Vision/StableDiffusion
```

**Key flags:** `--repo` (target repo), `--goal` (high-level objective),
`--max-iterations`, `--target-metric`, `--target-value`, `--approval`.

## Skill 2: Run a Task-First Autonomous Coding Cycle (Phase 11)

For single-objective engineering tasks (e.g. implement a specific model
modification):

```bash
uv run research-engineer task "Implement Grouped Query Attention in the GPT-2 block" --repo ../LLM/GPT2
```

## Skill 3: Add a New Agent to the Platform

1. **Create the agent class:** `src/research_engineer/agents/my_new_agent.py`
   implementing `BaseAgent`.
2. **Register LLM routing:** Add agent key to
   `src/research_engineer/_llm_support.py` and assign a default model in
   `llm_config.yaml` (model switching is config-only — never edit source to
   change models).
3. **Register CLI command:** Update `src/research_engineer/cli/__init__.py`
   using Typer.
4. **Export in package:** Add to `src/research_engineer/agents/__init__.py`.
5. **Write tests:** `tests/test_my_new_agent.py` (unit + integration).
6. **Verify suite:** `uv run python -m pytest tests/test_my_new_agent.py -v`.

## Skill 4: Add a Custom Tool

1. **Create I/O models:** `src/research_engineer/models/my_new_tool_models.py`
   using Pydantic v2 `BaseModel` with strict types.
2. **Implement tool:** `src/research_engineer/tools/my_new_tool.py`
   extending `Tool[InputType, OutputType]` ABC.
3. **Write tests:** unit tests covering `execute()` and `validate()`.

## Skill 5: Configure / Debug LLM Routing (Phase 10)

If agents encounter API errors or routing failures:

1. Verify routing: `uv run research-engineer llm status`.
2. Modify provider/model assignments in `llm_config.yaml` (never source).
3. Test connectivity:
   ```bash
   uv run research-engineer memory query "GQA implementation details" --repo ../LLM/LLaMA-3-Lite
   ```

## Skill 6: Run the Quality Pipeline

Always run before committing any change:

```bash
# 1. Tests (1427 passing — always via python -m)
uv run python -m pytest

# 2. Type checking (strict)
uv run mypy .

# 3. Lint + format
uv run ruff check .
```

## Skill 7: Run Agent Evaluations & Improvement Loop (E4 / E8)

Grade agent behavior on scripted suites and mine failures into improvements:

```bash
uv run research-engineer eval-harness --suite <name>      # E4: run graded suite
uv run research-engineer improve <report_id>              # E8: propose/approve/promote fixes
```

## Skill 8: Run Research Benchmarks

```bash
uv run research-engineer benchmark p1        # deterministic tier (no LLM cost)
uv run research-engineer benchmark p2        # LLM-backed tier with judge + ReAct agent
```

## Pitfalls

- **Always use `uv run`** — never bare `python` (root rule).
- **No LLM in Phases 1–3** — deterministic parsing + AST tools only. Don't
  add LLM calls here; preserves budget and stays reproducible.
- **Atomic file writes** — all tools must write output files atomically
  (`.tmp` → rename) to prevent corruption.
- **Pydantic v2 compliance** — when editing models, use strict type-hinting
  and Pydantic v2 schemas; never `dict` / `Any` for inter-agent contracts.
- **Output goes to `output/<phase>/<id>/`** — never write to repo root.
