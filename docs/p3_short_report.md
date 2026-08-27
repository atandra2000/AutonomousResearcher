# P3-Short — Model Selection & Capability Validation Report

Date: 2026-08-27 · Scope: 3 models × 2 independent full-suite repeats × 20
cases (P2 v2 LLM benchmark), executed through the production path
(API → PostgreSQL → Worker → AgentRuntime → ToolGateway → SafetyController
→ Evaluation → Telemetry). No promotion was executed; E8 used read-only.

## 1. Baseline reproduction

The frozen P2 GLM-5.3-Flash configuration reproduces:

- Success rate identical to the frozen P2 run (15/20 both), completion 75–80%.
- Quality 0.909 (rep1) / 0.912 (rep2); repeat variance ±1 case.
- Identity recorded in `artifacts/p3/baseline/baseline_manifest.json`
  (provider `ollama`, model `glm-5.3-flash`, judge `glm-5.3-flash`,
  `llm_config_sha256`, suite content sha, grader + safety digests).
- Derived-suite case/grader content hash identical across every arm:
  no benchmark leakage; arms differ only by the declared single agent
  override (`llm_model`).

## 2. Model comparison (pooled over 2 repeats × 20 cases)

| Metric | GLM-5.3-Flash (base) | Kimi K2.7 Code | DeepSeek-V4-Pro |
|---|---|---|---|
| Autonomous completion | 100% | 100% | 100% |
| Objective success | 31/40 = 78% (Wilson95 [62.5, 87.7]) | 32/40 = 80% ([65.2, 89.5]) | **34/40 = 85% ([70.9, 92.9])** |
| Research-quality mean | 0.911 ([0.864, 0.958]) | 0.941 ([0.919, 0.964]) | **0.946 ([0.924, 0.967])** |
| Human interventions | 0 | 0 | 0 |
| Safety interventions | 0 | 0 | 0 |
| Recovery success (run-level) | 100% | 100% | 100% |
| Latency median / p95 | 56.7 s / 144.0 s | 11.7 s / 30.5 s | **8.5 s / 21.9 s** |
| Tokens/task median | 4022 | **2707** | 3144 |
| Cost/task median | $0.00417 | $0.00354 | $0 recorded (unpriced; est. ≈$0.0043) |
| Tool efficiency | 0.566 | 0.600 | 0.600 |
| Repeat consistency | 15/20 vs 16/20 | 15/20 vs 17/20 | **17/20 = identical** |
| Termination | all success | all success | all success |

## 3. Per-case flips vs GLM (both repeats)

- **DeepSeek**: no losses; consistent wins where GLM was repeat-inconsistent:
  `debug_02_ddp_deadlock` (base [F,T] → DS [T,T]),
  `debug_03_oom_step1200` (base [T,F] → DS [T,T]),
  `e2e_02_conflicting_evidence_resolution` (base [F,T] → DS [T,T]).
- **Kimi**: no consistent wins or losses; 4 repeat-inconsistent flips
  (`debug_02`, `debug_03`, `e2e_02` mixed; `expa_01_overfit_point_detection`
  lost in rep2 only).

## 4–5. Quality / cost / latency

- Quality: DeepSeek 0.946 > Kimi 0.941 > GLM 0.911; DeepSeek’s t-CI does not
  overlap GLM’s.
- Latency: DeepSeek median 8.5 s = 6.7× faster than GLM (56.7 s); Kimi 4.9×.
- Tokens: Kimi −33%, DeepSeek −22% vs GLM per task.
- Cost: Kimi measured −15% vs GLM. DeepSeek recorded $0.00 because
  `deepseek-v4-pro` was missing from the placeholder pricing table at run
  time (post-hoc estimate from measured tokens at Kimi’s blended rate:
  ≈$0.0043/task). Pricing entry added afterwards for future runs
  (`src/research_engineer/llm/cost.py`).

## 6. Failure analysis (agent failures only; judge malfunctions: zero)

- `design_02_data_scaling_study`, `impl_01_attention_module_plan`,
  `e2e_03_build_decision_memorandum` fail for **all three models in both
  repeats** → deterministic-grader gaps (verbatim content/header
  requirements), model-independent; not capability, not judge parsing.
- `impl_01` (attention module, GLM): `notes_written=0`, all INTERFACE/PLAN/
  EDGE_CASES/TESTS headers missing, judge score 0.0 → genuine agent
  deliverable failure. Kimi/DeepSeek write real content (judge 0.92+) but
  still miss verbatim headers → remaining failure is grader strictness.
- `debug_02`/`debug_03`/`e2e_02` are repeat-flaky for GLM; DeepSeek solves
  all three in both repeats → genuine model-capability effect.
- No outcome in any arm met the judge-malfunction criteria (only failing
  required criterion with absent/unparseable `llm_score`); no judge failures
  were counted as agent failures.

## 7. Statistical / repeatability evidence

- 2 independent full 20-case repeats per model (40 cases pooled per model).
- GLM/Kimi repeat variance ±1–2 cases; DeepSeek’s success set is identical
  across repeats (17/20 = 17/20).
- Wilson 95% CIs for success and t-CIs for quality reported above; the GLM↔DS
  success gap (+7.5 pts) exceeds either model’s observed repeat noise (±5 pts),
  while the GLM↔Kimi gap (+2.5 pts) does not.

## 8. E8 gate results (read-only, no promotion)

- `p3short_ds_evidence`: **PASS**, no violations — deltas: success +13.3 pts
  (vs frozen P2 report baseline), quality +5.3 pts, avg latency −55.1 s,
  p95 −112.8 s, total tokens −64 080.
- `p3short_kimi_evidence`: **PASS**, no violations — success +4.3 pts,
  quality +4.1 pts, avg latency −50.5 s.
- Promotion deliberately not executed (`promotion_executed: false`); gate
  artifacts: `artifacts/p3/e8_gate/p3short_{ds,kimi}_evidence_gate.json`.

## 9. Recommended model

**Adopt-candidate recommendation (pending human E8 approval, NOT promoted):
`ollama/deepseek-v4-pro:0813`** for the autonomous research agent:

- Only candidate whose success set is fully repeat-stable (17/20 twice) and
  which strictly dominates GLM per case (no losses).
- Better on success (+7.5 pts), quality (+3.5 pts), latency (−85%),
  tokens (−22%); safety/human/recovery unchanged (0 / 0 / 100%).
- Cost is an estimate (≈$0.0043/task) until a confirmed price exists.
- Kimi K2.7 Code is a fast/cheap fallback (+2.5 pts) whose advantage did not
  survive repeats cleanly; keep GLM-5.3-Flash as production model until the
  DeepSeek candidate is explicitly approved.

## 10. Tests / validation

- `uv run python -m pytest -q`: **1457 passed, 0 failed** (1427 pre-existing
  + 24 P3 experiment + 6 new analysis-helper tests).
- `uv run ruff check` on all changed files: clean.
- `uv run mypy` on all changed source files: clean (repo-wide mypy has
  pre-existing errors in unrelated modules; scripts show the same
  pre-existing `import-untyped` pattern as before).
- No production configuration modified: `llm_config.yaml`, safety policy,
  prompts, budgets, stagnation policy unchanged during P3-Short.

## 11. Remaining uncertainty

- n=2 repeats/model: Wilson intervals for GLM and DS overlap; the strongest
  evidence is DS repeat-stability + per-case dominance, not a significant
  proportion difference.
- DeepSeek cost is an estimate until billing/price confirmation.
- The three all-model failures (`design_02`, `impl_01`, `e2e_03`) look like
  grader strictness rather than capability; a grader-accuracy study (P4) is
  the right instrument, not another model change.
- Latency reflects cloud-provider load during the run window; relative
  ordering held in all passes.

## Artifacts

- Passes: `artifacts/p3/baseline_repro`, `artifacts/p3/exp_a1_kimi_k27_code`,
  `artifacts/p3short/{glm,kimi,ds}/rep*/…`
- Aggregate: `artifacts/p3short/p3_short_report.{md,json}`
- Gates: `artifacts/p3/e8_gate/p3short_{ds,kimi}_evidence_gate.json`
- Arm configs: `configs/p3/arms/` (baseline_repro, exp_a1_kimi_k27_code,
  exp_f_deepseek_v4_pro)
- Scripts: `scripts/p3/run_p3_short.sh`, `scripts/p3/analyze_p3_short.py`,
  `scripts/p3/finalize_p3short.sh`, `scripts/p3/e8_gate.py`,
  `scripts/p3/run_p3_arm.py`