#!/bin/bash
# P3-Short finalize: wait for all 4 passes -> aggregate -> stage 2-repeat
# evidence -> run E8 gates (read-only, no promotion).
set -u
cd "$(dirname "$0")/../.."
until grep -q 'P3-SHORT RUNS DONE' /tmp/p3short.log 2>/dev/null; do sleep 30; done
uv run python scripts/p3/analyze_p3_short.py || exit 1
stage_evidence() { # $1 = evidence dir, $2 rep1 src, $3 rep2 src
  mkdir -p "artifacts/p3/$1"
  cp "$2" "artifacts/p3/$1/v2_repeat1_full_suite.json"
  cp "$3" "artifacts/p3/$1/v2_repeat2_full_suite.json"
}
stage_evidence p3short_kimi_evidence \\\n  artifacts/p3/exp_a1_kimi_k27_code/v2_full_suite.json \\\n  artifacts/p3short/kimi/rep2/exp_a1_kimi_k27_code/v2_full_suite.json
stage_evidence p3short_ds_evidence \\\n  artifacts/p3short/ds/rep1/exp_f_deepseek_v4_pro/v2_full_suite.json \\\n  artifacts/p3short/ds/rep2/exp_f_deepseek_v4_pro/v2_full_suite.json
uv run python scripts/p3/e8_gate.py --candidate p3short_kimi_evidence \\\n  --component model_provider_config \\\n  --changes '{"model_provider.model": "kimi-k2.7-code"}' \\\n  || echo '!!! kimi gate failed'
uv run python scripts/p3/e8_gate.py --candidate p3short_ds_evidence \\\n  --component model_provider_config \\\n  --changes '{"model_provider.model": "deepseek-v4-pro:0813"}' \\\n  || echo '!!! ds gate failed'
echo "=== $(date '+%H:%M:%S') P3-SHORT FINALIZE DONE ==="
