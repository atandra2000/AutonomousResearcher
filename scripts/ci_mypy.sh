#!/usr/bin/env bash
# P4 CI stabilization: fail ONLY on mypy errors NEW relative to the baseline.
#
# The codebase carries a large legacy mypy debt (857 errors across 59 files
# at baseline, 2026-08-27). Fixing all of it is out of scope; instead this
# gate guarantees the debt never GROWS:
#
#   * every current error is recorded (normalized: file + message, without
#     line numbers so unrelated edits don't shift the baseline) in
#     configs/mypy-baseline.txt;
#   * this script fails only when mypy reports an error that is NOT in the
#     baseline.
#
# To re-baseline deliberately (e.g. after fixing a batch of errors):
#   uv run mypy . 2>&1 | grep -E '^src/.*error:' \
#     | sed -E 's/:[0-9]+: error:/: error:/' | sort -u \
#     > configs/mypy-baseline.txt
#
# Usage: scripts/ci_mypy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASELINE=configs/mypy-baseline.txt

current=$(uv run mypy . 2>&1 | grep -E '^src/.*error:' \
  | sed -E 's/:[0-9]+: error:/: error:/' | LC_ALL=C sort -u || true)

new=$(LC_ALL=C comm -13 \
  <(LC_ALL=C sort -u "$BASELINE") \
  <(printf '%s\n' "$current" | LC_ALL=C sort -u))

if [ -n "$new" ]; then
  echo "mypy: NEW errors not in the baseline:" >&2
  echo "$new" >&2
  exit 1
fi

echo "mypy: no new errors relative to configs/mypy-baseline.txt"
