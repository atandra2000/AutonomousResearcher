#!/bin/bash
# P3-Short: model selection - exactly 2 full-suite repeats per model.
#   glm   rep1 = artifacts/p3/baseline_repro/v2_full_suite.json (existing)
#   kimi  rep1 = artifacts/p3/exp_a1_kimi_k27_code/v2_full_suite.json (existing)
# This script runs ONLY the missing passes, each into its own root so the
# original P3 artifacts remain untouched:
#   glm/rep2, kimi/rep2, ds/rep1, ds/rep2
set -u
cd "$(dirname "$0")/../.."
run() { echo "=== $(date '+%H:%M:%S') START $* ==="; uv run python scripts/p3/run_p3_arm.py "$@" --repeats 1 || echo "!!! FAILED: $*"; }
run --arm baseline_repro        --output-root artifacts/p3short/glm/rep2
run --arm exp_a1_kimi_k27_code  --output-root artifacts/p3short/kimi/rep2
run --arm exp_f_deepseek_v4_pro --output-root artifacts/p3short/ds/rep1
run --arm exp_f_deepseek_v4_pro --output-root artifacts/p3short/ds/rep2
echo "=== $(date '+%H:%M:%S') P3-SHORT RUNS DONE ==="
