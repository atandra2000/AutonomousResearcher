#!/usr/bin/env bash
# E7 deployment smoke test.
#
# Builds the Docker Compose stack from deploy/, then exercises:
#   submit -> queue -> worker -> AgentRuntime -> checkpoint
#          -> completion -> result query
# followed by a simulated worker crash (docker stop/restart) and
# recovery-from-checkpoint on the restarted worker.
#
# Usage:  scripts/smoke_test.sh [--down]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"
API_URL="http://localhost:${API_PORT:-8000}"
TOKEN="smoke-test-token-$(date +%s)"
DOWN=0
[[ "${1:-}" == "--down" ]] && DOWN=1

export RE_SERVICE_API_TOKEN="$TOKEN"
export POSTGRES_PASSWORD="smoke-pg-password"
export RE_STALE_RUN_TIMEOUT_SECONDS="6"
export RE_STEP_DELAY_SECONDS="0.5"
export API_PORT="${API_PORT:-8000}"
AUTH=("-H" "Authorization: Bearer $TOKEN")

echo "== Building stack (may take minutes on first run)"
docker compose -f "$DEPLOY/docker-compose.yml" --project-directory "$DEPLOY" up -d --build
trap '[[ $DOWN -eq 1 ]] && docker compose -f "$DEPLOY/docker-compose.yml" --project-directory "$DEPLOY" down -v || true' EXIT

echo "== Waiting for API health/readiness"
for i in $(seq 1 60); do
  if curl -fsS "$API_URL/health" >/dev/null 2>&1 \
     && curl -fsS "$API_URL/ready" | grep -q '"ready":true'; then
    break
  fi
  [[ $i -eq 60 ]] && { echo "FAIL: API not ready"; exit 1; }
  sleep 2
done
echo "OK: /health + /ready"

# --- Happy path -----------------------------------------------------------
GOAL="Analyse the goal. Draft a plan. Execute step one. Execute step two. Summarise findings."
RUN=$(curl -fsS -X POST "$API_URL/runs" "${AUTH[@]}" \
      -H 'Content-Type: application/json' -d "{\"goal\":\"$GOAL\",\"max_steps\":10}")
RUN_ID=$(echo "$RUN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
echo "Submitted run: $RUN_ID"

STATUS="queued"
for i in $(seq 1 60); do
  BODY=$(curl -fsS "$API_URL/runs/$RUN_ID" "${AUTH[@]}")
  STATUS=$(echo "$BODY" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "cancelled" ]] && break
  sleep 2
done
[[ "$STATUS" == "completed" ]] || { echo "FAIL: run ended as $STATUS"; exit 1; }
RESULT=$(curl -fsS "$API_URL/runs/$RUN_ID/result" "${AUTH[@]}")
echo "OK: happy path completed: $RESULT" | head -c 400; echo

# --- Worker crash + checkpoint recovery -----------------------------------
LONG_GOAL="$(python3 -c 'print(". ".join(f"long phase {i}" for i in range(12)) + ".")')"
RUN2=$(curl -fsS -X POST "$API_URL/runs" "${AUTH[@]}" \
       -H 'Content-Type: application/json' -d "{\"goal\":\"$LONG_GOAL\",\"max_steps\":12}")
RUN2_ID=$(echo "$RUN2" | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
echo "Submitted crash-test run: $RUN2_ID"

# Wait until it is actually running before killing the worker.
R_STATUS="queued"
for i in $(seq 1 30); do
  R_STATUS=$(curl -fsS "$API_URL/runs/$RUN2_ID" "${AUTH[@]}" \
             | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  [[ "$R_STATUS" == "running" ]] && break
  sleep 1
done
echo "Simulating worker crash (state was: $R_STATUS)"
docker compose -f "$DEPLOY/docker-compose.yml" --project-directory "$DEPLOY" stop worker
sleep 8   # exceed RE_STALE_RUN_TIMEOUT_SECONDS
docker compose -f "$DEPLOY/docker-compose.yml" --project-directory "$DEPLOY" start worker

FINAL="running"
for i in $(seq 1 90); do
  FINAL=$(curl -fsS "$API_URL/runs/$RUN2_ID" "${AUTH[@]}" \
          | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  case "$FINAL" in completed|failed|cancelled) break;; esac
  sleep 2
done
[[ "$FINAL" == "completed" ]] || { echo "FAIL: recovered run ended as $FINAL"; exit 1; }
CLAIMS=$(curl -fsS "$API_URL/runs/$RUN2_ID" "${AUTH[@]}" \
         | python3 -c 'import sys,json;print(json.load(sys.stdin).get("claim_count",""))') || true
RESULT2=$(curl -fsS "$API_URL/runs/$RUN2_ID/result" "${AUTH[@]}")
echo "OK: recovered after worker crash: $RESULT2" | head -c 300; echo

echo "PASS: E7 smoke test complete (happy path + crash recovery)"
[[ $DOWN -eq 1 ]] && { docker compose -f "$DEPLOY/docker-compose.yml" --project-directory "$DEPLOY" down -v; echo "Stack torn down."; }
exit 0
