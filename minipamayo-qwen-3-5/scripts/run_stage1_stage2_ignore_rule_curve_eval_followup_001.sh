#!/usr/bin/env bash

set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ENV_SCRIPT="${PROJECT_ROOT}/env/cuda-12.8.sh"
ATTEMPT_NAME="completion_ignore_rule_curve_eval_followup_001"
SESSION_NAME="ignore-rule-curve-followup-001"
MAIN_RUN_STATUS_FILE="${PROJECT_ROOT}/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json"
LOG_ROOT="${PROJECT_ROOT}/artifacts/run_logs/${ATTEMPT_NAME}"
STATE_DIR="${LOG_ROOT}/state"
MASTER_LOG="${LOG_ROOT}/master.log"
EXIT_CODE_FILE="${LOG_ROOT}/run.exitcode"
RUN_STATUS_FILE="${LOG_ROOT}/run.status.json"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

CURRENT_STAGE="bootstrap"

STAGE1A_EVAL_CONFIG="configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE1B_EVAL_CONFIG="configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json"
STAGE2_CURVE_PREPROCESS_CONFIG="configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_EVAL_CONFIG="configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_INFER_CONFIG="configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json"

STAGE1A_BEST="${PROJECT_ROOT}/checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/best.pt"
STAGE1B_BEST="${PROJECT_ROOT}/checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe/best.pt"
STAGE2_BEST="${PROJECT_ROOT}/checkpoints/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/best.pt"

STAGE1A_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE1A_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json"
STAGE1B_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json"
STAGE1B_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.progress.json"
STAGE2_CURVE_PREPROCESS_OUTPUT="${PROJECT_ROOT}/datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001_curve_eval/perimeter_cw_holdout_v1/samples_reasoning_sft.jsonl"
STAGE2_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json"
STAGE2_INFER_OUTPUT="${PROJECT_ROOT}/artifacts/inference/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json"
STAGE2_INFER_PROGRESS="${PROJECT_ROOT}/artifacts/inference/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.progress.json"

if [ -f "${CUDA_ENV_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  . "${CUDA_ENV_SCRIPT}"
fi

timestamp_iso() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

timestamp_tag() {
  date "+%Y%m%d_%H%M%S"
}

log_line() {
  printf '[%s] %s\n' "$(timestamp_iso)" "$*"
}

backup_path_if_exists() {
  local path="$1"
  if [ ! -e "$path" ]; then
    return 1
  fi
  local backup_path="${path}_bak_$(timestamp_tag)"
  mkdir -p "$(dirname "$backup_path")"
  mv "$path" "$backup_path"
  log_line "moved_aside path=${path} backup=${backup_path}"
  return 0
}

write_run_status() {
  local state="$1"
  local rc="$2"
  cat >"${RUN_STATUS_FILE}" <<EOF
{
  "attempt_name": "${ATTEMPT_NAME}",
  "session_name": "${SESSION_NAME}",
  "state": "${state}",
  "current_stage": "${CURRENT_STAGE}",
  "exit_code": ${rc},
  "updated_at": "$(timestamp_iso)",
  "log_root": "${LOG_ROOT}",
  "main_run_status_file": "${MAIN_RUN_STATUS_FILE}"
}
EOF
  printf '%s\n' "${rc}" >"${EXIT_CODE_FILE}"
}

handle_signal() {
  local signal_name="$1"
  log_line "signal_received signal=${signal_name} current_stage=${CURRENT_STAGE}"
  write_run_status "interrupted" 130
  exit 130
}

trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

prepare_log_root() {
  mkdir -p "${PROJECT_ROOT}/artifacts/run_logs"
  backup_path_if_exists "${LOG_ROOT}" >/dev/null 2>&1 || true
  mkdir -p "${LOG_ROOT}" "${STATE_DIR}"
  : >"${MASTER_LOG}"
  exec > >(tee -a "${MASTER_LOG}") 2>&1
  log_line "launcher_ready attempt=${ATTEMPT_NAME} session=${SESSION_NAME} log_root=${LOG_ROOT}"
  write_run_status "running" 999
}

run_stage() {
  local stage_name="$1"
  shift
  CURRENT_STAGE="${stage_name}"
  write_run_status "running" 999
  local stage_log="${LOG_ROOT}/${stage_name}.log"
  local rc_file="${STATE_DIR}/${stage_name}.exitcode"
  local status_file="${STATE_DIR}/${stage_name}.status.json"
  local started_at
  local finished_at
  local rc

  started_at="$(timestamp_iso)"
  log_line "stage_start stage=${stage_name} command=$*"

  (
    cd "${PROJECT_ROOT}" || exit 1
    "$@"
  ) 2>&1 | tee "${stage_log}"
  rc=${PIPESTATUS[0]}

  finished_at="$(timestamp_iso)"
  printf '%s\n' "${rc}" >"${rc_file}"
  cat >"${status_file}" <<EOF
{
  "stage": "${stage_name}",
  "started_at": "${started_at}",
  "finished_at": "${finished_at}",
  "exit_code": ${rc},
  "log_path": "${stage_log}"
}
EOF
  log_line "stage_end stage=${stage_name} exit_code=${rc} log=${stage_log}"
  return "${rc}"
}

require_path() {
  local path="$1"
  if [ ! -e "$path" ]; then
    log_line "missing_required_artifact path=${path}"
    return 1
  fi
  return 0
}

require_line_count() {
  local path="$1"
  local expected="$2"
  if [ ! -e "$path" ]; then
    log_line "missing_required_artifact path=${path}"
    return 1
  fi
  local actual
  actual="$(wc -l <"${path}")"
  actual="${actual//[[:space:]]/}"
  if [ "${actual}" != "${expected}" ]; then
    log_line "unexpected_line_count path=${path} expected=${expected} actual=${actual}"
    return 1
  fi
  log_line "validated_line_count path=${path} expected=${expected}"
  return 0
}

backup_outputs() {
  backup_path_if_exists "${STAGE1A_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE1A_EVAL_PROGRESS}" || true
  backup_path_if_exists "${STAGE1B_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE1B_EVAL_PROGRESS}" || true
  backup_path_if_exists "${STAGE2_CURVE_PREPROCESS_OUTPUT}" || true
  backup_path_if_exists "${STAGE2_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE2_EVAL_PROGRESS}" || true
  backup_path_if_exists "${STAGE2_INFER_OUTPUT}" || true
  backup_path_if_exists "${STAGE2_INFER_PROGRESS}" || true
}

read_main_run_state() {
  python - "${MAIN_RUN_STATUS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("missing")
    raise SystemExit(0)
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.get("state", "unknown"))
PY
}

wait_for_main_run_to_finish() {
  local state=""
  while true; do
    state="$(read_main_run_state)"
    log_line "main_run_state state=${state}"
    case "${state}" in
      completed|failed|interrupted)
        printf '%s\n' "${state}"
        return 0
        ;;
      *)
        sleep "${POLL_INTERVAL_S}"
        ;;
    esac
  done
}

main() {
  prepare_log_root
  backup_outputs

  local main_state
  main_state="$(wait_for_main_run_to_finish)"
  log_line "main_run_finished state=${main_state}"

  require_path "${STAGE1A_BEST}" || {
    write_run_status "failed" 20
    return 20
  }
  run_stage \
    "stage1a_curve_eval" \
    uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
      --config-json "${STAGE1A_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1a_curve_eval"

  require_path "${STAGE1B_BEST}" || {
    write_run_status "failed" 30
    return 30
  }
  run_stage \
    "stage1b_curve_eval" \
    uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
      --config-json "${STAGE1B_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1b_curve_eval"

  run_stage \
    "stage2_curve_eval_preprocess" \
    uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess \
      --config-json "${STAGE2_CURVE_PREPROCESS_CONFIG}" || {
    write_run_status "failed" 40
    return 40
  }
  require_line_count "${STAGE2_CURVE_PREPROCESS_OUTPUT}" 569 || {
    write_run_status "failed" 41
    return 41
  }

  require_path "${STAGE2_BEST}" || {
    write_run_status "failed" 50
    return 50
  }
  run_stage \
    "stage2_curve_eval" \
    uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval \
      --config-json "${STAGE2_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage2_curve_eval"

  run_stage \
    "stage2_curve_sample_inference" \
    uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference \
      --config-json "${STAGE2_INFER_CONFIG}" || log_line "non_blocking_failure stage=stage2_curve_sample_inference"

  write_run_status "completed" 0
  log_line "followup_completed exit_code=0"
  return 0
}

main "$@"
