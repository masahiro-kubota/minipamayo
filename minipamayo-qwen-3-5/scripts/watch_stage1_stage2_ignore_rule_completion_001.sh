#!/usr/bin/env bash

set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${PROJECT_ROOT}/artifacts/run_logs/completion_ignore_rule_full_001"
STATUS_FILE="${LOG_ROOT}/run.status.json"
MASTER_LOG="${LOG_ROOT}/master.log"
ALERT_FILE="${LOG_ROOT}/monitor.alert"
INTERVAL_S="${INTERVAL_S:-600}"
TAIL_LINES="${TAIL_LINES:-30}"

print_block() {
  local label="$1"
  printf '\n[%s]\n' "$label"
}

while true; do
  printf '\n=== %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  print_block "run.status.json"
  if [ -f "${STATUS_FILE}" ]; then
    cat "${STATUS_FILE}"
  else
    echo "missing"
  fi

  print_block "monitor.alert"
  if [ -f "${ALERT_FILE}" ]; then
    cat "${ALERT_FILE}"
  else
    echo "none"
  fi

  print_block "master.log tail"
  if [ -f "${MASTER_LOG}" ]; then
    tail -n "${TAIL_LINES}" "${MASTER_LOG}"
  else
    echo "missing"
  fi

  if [ -f "${STATUS_FILE}" ]; then
    state="$(python3 - <<'PY' "${STATUS_FILE}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    print(json.load(fh).get("state", "missing"))
PY
)"
    if [ "${state}" = "completed" ] || [ "${state}" = "failed" ] || [ "${state}" = "interrupted" ]; then
      echo
      echo "watch_exit state=${state}"
      exit 0
    fi
  fi

  sleep "${INTERVAL_S}"
done
