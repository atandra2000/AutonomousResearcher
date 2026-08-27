#!/bin/bash
# P3 - execute every declared arm sequentially through the production path.
# Usage:   nohup bash scripts/p3/run_all_p3.sh >/tmp/p3_all.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
for arm in \
    baseline_repro \
    exp_a1_kimi_k27_code \
    exp_a2_minimax_m3_cloud \
    exp_a3_gpt_oss_120b \
    exp_b_plan_first_strategy \
    exp_c_reasoning_budget \
    exp_d_stagnation_window6 \
    exp_e_impl_strategy; do
  echo "=== $(date '+%H:%M:%S') START $arm ==="
  uv run python scripts/p3/run_p3_arm.py --arm "$arm" --repeats 3 \
    || echo "!!! ARM FAILED: $arm"
done
echo "=== $(date '+%H:%M:%S') ALL ARMS DONE ==="
