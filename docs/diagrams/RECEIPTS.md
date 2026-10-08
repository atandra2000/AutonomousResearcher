# Archify Delivery Evidence — Autonomous ML Research Engineer

The [interactive visual systems guide](research_engineer_visual_guide.html) links four standalone showcase architecture, workflow, sequence, and dataflow diagrams.

All four: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light/dark. All required viewport measurements passed horizontal/vertical containment, minimum projected text size, and viewer-control clearance.

---

## Artifact Bindings

### 1. Multi-Agent System Architecture
- **Diagram type:** `architecture`
- **Output:** [research-engineer-architecture.html](research-engineer-architecture.html)
- **Specification:** `docs/diagrams/research-engineer-architecture.architecture.json`
- **Artifact SHA-256:** `6b0da82452e5834940a5991dc0c9ba4d8dfd2570a504a74285fbccd90d6d2680` (720,659 bytes)
- **[Browser receipt](research-engineer-architecture.visual-check.json)** · **[Screenshot contact sheet](research-engineer-architecture.visual-check.html)**
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### 2. Autonomous Research & Self-Repair Workflow
- **Diagram type:** `workflow`
- **Output:** [research-engineer-workflow.html](research-engineer-workflow.html)
- **Specification:** `docs/diagrams/research-engineer-workflow.workflow.json`
- **Artifact SHA-256:** `bc34442a3ead07a1e83e71a9414b4bf135e9b0fc4192c4c87f4fc0b64667b4fb` (713,143 bytes)
- **[Browser receipt](research-engineer-workflow.visual-check.json)** · **[Screenshot contact sheet](research-engineer-workflow.visual-check.html)**
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### 3. Inter-Agent Delegation & Execution Trace
- **Diagram type:** `sequence`
- **Output:** [research-engineer-sequence.html](research-engineer-sequence.html)
- **Specification:** `docs/diagrams/research-engineer-sequence.sequence.json`
- **Artifact SHA-256:** `add3c0bfd1fb35df4f2f8d75d7e1ac3d6bcea434a0c2f8cfc3350d49434e42e2` (713,945 bytes)
- **[Browser receipt](research-engineer-sequence.visual-check.json)** · **[Screenshot contact sheet](research-engineer-sequence.visual-check.html)**
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### 4. Telemetry & Artifact Dataflow
- **Diagram type:** `dataflow`
- **Output:** [research-engineer-dataflow.html](research-engineer-dataflow.html)
- **Specification:** `docs/diagrams/research-engineer-dataflow.dataflow.json`
- **Artifact SHA-256:** `2d7306a7bbcaf527279c870eb4a72d9ee144358c4e915e9d943b118c2949385c` (713,950 bytes)
- **[Browser receipt](research-engineer-dataflow.visual-check.json)** · **[Screenshot contact sheet](research-engineer-dataflow.visual-check.html)**
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

---

## Technical Specifications Anchors

- **Framework & Engine:** LangGraph `StateGraph` in `src/research_engineer/graphs/research.py:ResearchGraph`
- **Multi-Agent Roster:** 20 specialized agents in `src/research_engineer/agents/`
- **Typed Tool Gateway:** 61 Pydantic tools inheriting `Tool[Input, Output]` in `src/research_engineer/tools/`
- **Safety Substrate:** Fail-closed patch review and gatekeeper in `src/research_engineer/safety/`
- **Memory & Storage:** AST symbol graph, BM25 indexing, ChromaDB vectors, and SQLite persistence in `src/research_engineer/memory/`
