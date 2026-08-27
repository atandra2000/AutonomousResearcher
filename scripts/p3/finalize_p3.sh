#!/bin/bash
# Waits for all P3 arms to finish, aggregates evidence, runs E8 gates on
# every candidate. Promotion remains manual - the gate only issues a verdict.
set -u
cd "$(dirname "$0")/../.."
until grep -q 'ALL ARMS DONE' /tmp/p3_all.log 2>/dev/null; do sleep 60; done
uv run python scripts/p3/aggregate_p3.py || exit 1
for cfg in configs/p3/arms/exp_*.yaml; do
  arm=$(basename "$cfg" .yaml)
  vals=$(uv run python - "$cfg" <<'PY'
import sys, yaml, json
c = yaml.safe_load(open(sys.argv[1]))
print(c.get('gate_component',''), json.dumps(c.get('gate_changes',{})))
PY
)
  comp=$(echo "$vals" | awk '{print $1}')
  chg=$(echo "$vals" | cut -d' ' -f2-)
  echo "--- E8 gate for $arm ($comp)"
  uv run python scripts/p3/e8_gate.py --candidate "$arm" --component "$comp" --changes "$chg" \
    || echo "!!! gate failed for $arm"
done
echo "=== $(date '+%H:%M:%S') P3 FINALIZE DONE ==="
