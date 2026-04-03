#!/usr/bin/env bash

set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ENV_SCRIPT="${PROJECT_ROOT}/env/cuda-12.8.sh"
ATTEMPT_NAME="${ATTEMPT_NAME:-completion_ignore_rule_full_001}"
SESSION_NAME="${SESSION_NAME:-ignore-rule-completion-001}"
LOG_ROOT="${PROJECT_ROOT}/artifacts/run_logs/${ATTEMPT_NAME}"
STATE_DIR="${LOG_ROOT}/state"
MASTER_LOG="${LOG_ROOT}/master.log"
EXIT_CODE_FILE="${LOG_ROOT}/run.exitcode"
RUN_STATUS_FILE="${LOG_ROOT}/run.status.json"
START_STAGE="${START_STAGE:-stage1a}"
MAX_STAGE1A_ATTEMPTS="${MAX_STAGE1A_ATTEMPTS:-0}"
STAGE1A_RETRY_SLEEP_S="${STAGE1A_RETRY_SLEEP_S:-30}"
MAX_STAGE1B_ATTEMPTS="${MAX_STAGE1B_ATTEMPTS:-0}"
STAGE1B_RETRY_SLEEP_S="${STAGE1B_RETRY_SLEEP_S:-30}"
MAX_STAGE2_ATTEMPTS="${MAX_STAGE2_ATTEMPTS:-0}"
STAGE2_RETRY_SLEEP_S="${STAGE2_RETRY_SLEEP_S:-30}"

CURRENT_STAGE="bootstrap"

STAGE1A_TRAIN_CONFIG="configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json"
STAGE1A_EVAL_CONFIG="configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE1B_TRAIN_CONFIG="configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json"
STAGE1B_EVAL_CONFIG="configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json"
STAGE2_PREPROCESS_CONFIG="configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json"
STAGE2_CURVE_PREPROCESS_CONFIG="configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_TRAIN_CONFIG="configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json"
STAGE2_EVAL_CONFIG="configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_INFER_CONFIG="configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json"

STAGE1A_SAVE_DIR="${PROJECT_ROOT}/checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb"
STAGE1B_SAVE_DIR="${PROJECT_ROOT}/checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe"
STAGE2_SAVE_DIR="${PROJECT_ROOT}/checkpoints/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb"

STAGE1A_SUMMARY="${STAGE1A_SAVE_DIR}/summary.json"
STAGE1A_BEST="${STAGE1A_SAVE_DIR}/best.pt"
STAGE1A_FINAL="${STAGE1A_SAVE_DIR}/final.pt"
STAGE1B_SUMMARY="${STAGE1B_SAVE_DIR}/summary.json"
STAGE1B_BEST="${STAGE1B_SAVE_DIR}/best.pt"
STAGE1B_LAST="${STAGE1B_SAVE_DIR}/last.pt"
STAGE2_SUMMARY="${STAGE2_SAVE_DIR}/summary.json"
STAGE2_BEST="${STAGE2_SAVE_DIR}/best.pt"

STAGE1A_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE1A_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json"
STAGE1B_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json"
STAGE1B_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.progress.json"
STAGE2_EVAL_OUTPUT="${PROJECT_ROOT}/artifacts/eval/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json"
STAGE2_EVAL_PROGRESS="${PROJECT_ROOT}/artifacts/eval/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json"
STAGE2_INFER_OUTPUT="${PROJECT_ROOT}/artifacts/inference/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json"
STAGE2_INFER_PROGRESS="${PROJECT_ROOT}/artifacts/inference/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.progress.json"

TRAIN_PREPROCESS_OUTPUTS=(
  "${PROJECT_ROOT}/datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8/samples_reasoning_sft.jsonl"
  "${PROJECT_ROOT}/datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8/samples_reasoning_sft.jsonl"
  "${PROJECT_ROOT}/datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8/samples_reasoning_sft.jsonl"
)

TRAIN_PREPROCESS_EXPECTED_COUNTS=(6423 4676 6167)

CURVE_PREPROCESS_OUTPUTS=(
  "${PROJECT_ROOT}/datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001_curve_eval/perimeter_cw_holdout_v1/samples_reasoning_sft.jsonl"
)

CURVE_PREPROCESS_EXPECTED_COUNTS=(569)

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
  "log_root": "${LOG_ROOT}"
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

require_stage1a_artifacts() {
  require_path "${STAGE1A_SUMMARY}" && require_path "${STAGE1A_BEST}" && require_path "${STAGE1A_FINAL}"
}

require_stage1b_artifacts() {
  require_path "${STAGE1B_SUMMARY}" && require_path "${STAGE1B_BEST}" && require_path "${STAGE1B_LAST}"
}

require_stage2_artifacts() {
  require_path "${STAGE2_SUMMARY}" && require_path "${STAGE2_BEST}"
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

prepare_log_root() {
  mkdir -p "${PROJECT_ROOT}/artifacts/run_logs"
  backup_path_if_exists "${LOG_ROOT}" >/dev/null 2>&1 || true
  mkdir -p "${LOG_ROOT}" "${STATE_DIR}"
  : >"${MASTER_LOG}"
  exec > >(tee -a "${MASTER_LOG}") 2>&1
  log_line "launcher_ready attempt=${ATTEMPT_NAME} session=${SESSION_NAME} log_root=${LOG_ROOT}"
  write_run_status "running" 999
}

prepare_preprocess_outputs() {
  local output_path
  for output_path in "${TRAIN_PREPROCESS_OUTPUTS[@]}"; do
    backup_path_if_exists "${output_path}" || true
  done
  for output_path in "${CURVE_PREPROCESS_OUTPUTS[@]}"; do
    backup_path_if_exists "${output_path}" || true
  done
}

prepare_attempt_scoped_artifacts() {
  if [ "${START_STAGE}" = "stage1a" ]; then
    backup_path_if_exists "${STAGE1A_SAVE_DIR}" || true
  fi
  backup_path_if_exists "${STAGE1A_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE1A_EVAL_PROGRESS}" || true
  if [ "${START_STAGE}" = "stage1a" ] || [ "${START_STAGE}" = "stage1b" ]; then
    backup_path_if_exists "${STAGE1B_SAVE_DIR}" || true
  fi
  backup_path_if_exists "${STAGE1B_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE1B_EVAL_PROGRESS}" || true
  backup_path_if_exists "${STAGE2_SAVE_DIR}" || true
  backup_path_if_exists "${STAGE2_EVAL_OUTPUT}" || true
  backup_path_if_exists "${STAGE2_EVAL_PROGRESS}" || true
  backup_path_if_exists "${STAGE2_INFER_OUTPUT}" || true
  backup_path_if_exists "${STAGE2_INFER_PROGRESS}" || true
}

stage1b_retry_loop() {
  local attempt=0
  local rc=0

  while true; do
    attempt=$((attempt + 1))
    backup_path_if_exists "${STAGE1B_SAVE_DIR}" || true
    backup_path_if_exists "${STAGE1B_EVAL_OUTPUT}" || true
    run_stage \
      "stage1b_train_attempt_$(printf '%03d' "${attempt}")" \
      uv run python -m minipamayo_qwen35.stage1.expert_cfm.train \
        --config-json "${STAGE1B_TRAIN_CONFIG}"
    rc=$?

    if [ "${rc}" -eq 0 ] && require_stage1b_artifacts; then
      log_line "stage1b_success attempt=${attempt}"
      return 0
    fi

    log_line "stage1b_retry_required attempt=${attempt} exit_code=${rc} sleep_s=${STAGE1B_RETRY_SLEEP_S}"
    if [ "${MAX_STAGE1B_ATTEMPTS}" -gt 0 ] && [ "${attempt}" -ge "${MAX_STAGE1B_ATTEMPTS}" ]; then
      log_line "stage1b_retry_exhausted attempts=${attempt}"
      return 1
    fi
    sleep "${STAGE1B_RETRY_SLEEP_S}"
  done
}

stage1a_retry_loop() {
  local attempt=0
  local rc=0

  while true; do
    attempt=$((attempt + 1))
    backup_path_if_exists "${STAGE1A_SAVE_DIR}" || true
    run_stage \
      "stage1a_train_attempt_$(printf '%03d' "${attempt}")" \
      uv run python -m minipamayo_qwen35.stage1.vlm_ce.train \
        --config-json "${STAGE1A_TRAIN_CONFIG}"
    rc=$?

    if [ "${rc}" -eq 0 ] && require_stage1a_artifacts; then
      log_line "stage1a_success attempt=${attempt}"
      return 0
    fi

    log_line "stage1a_retry_required attempt=${attempt} exit_code=${rc} sleep_s=${STAGE1A_RETRY_SLEEP_S}"
    if [ "${MAX_STAGE1A_ATTEMPTS}" -gt 0 ] && [ "${attempt}" -ge "${MAX_STAGE1A_ATTEMPTS}" ]; then
      log_line "stage1a_retry_exhausted attempts=${attempt}"
      return 1
    fi
    sleep "${STAGE1A_RETRY_SLEEP_S}"
  done
}

stage2_retry_loop() {
  local attempt=0
  local rc=0

  while true; do
    attempt=$((attempt + 1))
    backup_path_if_exists "${STAGE2_SAVE_DIR}" || true
    run_stage \
      "stage2_train_attempt_$(printf '%03d' "${attempt}")" \
      uv run python -m minipamayo_qwen35.stage2.reasoning_sft.train \
        --config-json "${STAGE2_TRAIN_CONFIG}"
    rc=$?

    if [ "${rc}" -eq 0 ] && require_stage2_artifacts; then
      log_line "stage2_success attempt=${attempt}"
      return 0
    fi

    log_line "stage2_retry_required attempt=${attempt} exit_code=${rc} sleep_s=${STAGE2_RETRY_SLEEP_S}"
    if [ "${MAX_STAGE2_ATTEMPTS}" -gt 0 ] && [ "${attempt}" -ge "${MAX_STAGE2_ATTEMPTS}" ]; then
      log_line "stage2_retry_exhausted attempts=${attempt}"
      return 1
    fi
    sleep "${STAGE2_RETRY_SLEEP_S}"
  done
}

main() {
  prepare_log_root
  log_line "launcher_config start_stage=${START_STAGE} attempt=${ATTEMPT_NAME} session=${SESSION_NAME}"
  prepare_preprocess_outputs
  prepare_attempt_scoped_artifacts

  run_stage \
    "stage2_preprocess" \
    uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess \
      --config-json "${STAGE2_PREPROCESS_CONFIG}" || {
    write_run_status "failed" 10
    return 10
  }

  local output_path
  local expected_count
  local index
  for index in "${!TRAIN_PREPROCESS_OUTPUTS[@]}"; do
    output_path="${TRAIN_PREPROCESS_OUTPUTS[$index]}"
    expected_count="${TRAIN_PREPROCESS_EXPECTED_COUNTS[$index]}"
    require_line_count "${output_path}" "${expected_count}" || {
      write_run_status "failed" 11
      return 11
    }
  done

  run_stage \
    "stage2_curve_eval_preprocess" \
    uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess \
      --config-json "${STAGE2_CURVE_PREPROCESS_CONFIG}" || {
    write_run_status "failed" 12
    return 12
  }

  for index in "${!CURVE_PREPROCESS_OUTPUTS[@]}"; do
    output_path="${CURVE_PREPROCESS_OUTPUTS[$index]}"
    expected_count="${CURVE_PREPROCESS_EXPECTED_COUNTS[$index]}"
    require_line_count "${output_path}" "${expected_count}" || {
      write_run_status "failed" 13
      return 13
    }
  done

  case "${START_STAGE}" in
    stage1a)
      stage1a_retry_loop || {
        write_run_status "failed" 20
        return 20
      }

      run_stage \
        "stage1a_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
          --config-json "${STAGE1A_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1a_curve_eval"

      stage1b_retry_loop || {
        write_run_status "failed" 30
        return 30
      }

      run_stage \
        "stage1b_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
          --config-json "${STAGE1B_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1b_curve_eval"
      ;;
    stage1b)
      require_stage1a_artifacts || {
        write_run_status "failed" 21
        return 21
      }

      stage1b_retry_loop || {
        write_run_status "failed" 30
        return 30
      }

      run_stage \
        "stage1a_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
          --config-json "${STAGE1A_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1a_curve_eval"

      run_stage \
        "stage1b_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
          --config-json "${STAGE1B_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1b_curve_eval"
      ;;
    stage2)
      require_stage1a_artifacts || {
        write_run_status "failed" 21
        return 21
      }
      require_stage1b_artifacts || {
        write_run_status "failed" 31
        return 31
      }

      run_stage \
        "stage1a_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
          --config-json "${STAGE1A_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1a_curve_eval"

      run_stage \
        "stage1b_curve_eval" \
        uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
          --config-json "${STAGE1B_EVAL_CONFIG}" || log_line "non_blocking_failure stage=stage1b_curve_eval"
      ;;
    *)
      log_line "invalid_start_stage value=${START_STAGE}"
      write_run_status "failed" 2
      return 2
      ;;
  esac

  stage2_retry_loop || {
    write_run_status "failed" 40
    return 40
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
  log_line "run_completed exit_code=0"
  return 0
}

main "$@"
