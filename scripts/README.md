# `scripts/` — operator and benchmark scripts

## Active

| Script | Purpose |
|--------|---------|
| `ci_mypy.sh` | mypy gate used by CI: fails only on errors **new** vs `configs/mypy-baseline.txt`. |
| `e8_improvement_demo.py` | End-to-end E8 demo: mine failures → candidate → E4 gate → promote/rollback. |
| `p3/e8_gate.py` | E8 promotion gate evaluation over stored eval reports. |
| `p3/analyze_p3_short.py` | P3-Short analysis helpers (covered by `tests/test_p3_short_analysis.py`). |
| `p3/run_p3_short.sh`, `p3/finalize_p3short.sh` | P3-Short orchestration wrappers. |

## Retired (provenance only — do not run)

These drivers targeted the **serving tier** (`research_engineer.service`: run API,
store, queue, worker), which was removed in `c31c536` when the product became
CLI-dedicated. They now fail immediately with
`ModuleNotFoundError: No module named 'research_engineer.service'`. They are kept
because `docs/p3_report.md`, `docs/p3_short_report.md`, and
`docs/p4_closure_report.md` cite them as the provenance of recorded results:

- `pilot_benchmark.py`
- `p4_rescore.py`
- `p3/run_p3_arm.py`
- `p3/freeze_baseline.py`
- `p3/aggregate_p3.py`
- `p3/crash_recovery_validation.py`
- `p3/run_all_p3.sh`, `p3/finalize_p3.sh` (wrappers over the above)

The benchmark capability these scripts exercised lives on as the `benchmark/`
package and `research-engineer benchmark {p1|p2|list|compare}`, which executes
every case directly through `AgentRuntime → ToolGateway → SafetyController`.

To recover a working historical copy: `git show c31c536^:<path>`.