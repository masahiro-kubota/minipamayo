# Stage1 / Stage2 Ignore Rule Completion Run Results

このファイルは、[stage1-stage2-ignore-rule-completion-run-plan.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-plan.md)
に沿って進める、`ignore_rule_data` 3-run の `Stage1A -> Stage1B -> Stage2` completion run の記録用。fresh-only run を前提にし、resume ではなく rerun として扱う。

## Summary Table

| Attempt | Data | Stage2 Preprocess | Stage1A | Stage1B | Stage2 | Smoke Evals | Sample Inference | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| completion_ignore_rule_full_001 | 3 runs, `k64_dt01`, `17266 samples` | pending | pending | pending | pending | pending | pending | pending |

## Shared Setup

- Plan:
  [stage1-stage2-ignore-rule-completion-run-plan.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-plan.md)
- Raw data:
  [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)
- Stage1 extraction config:
  [ignore_rule_data_k64_dt01.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json)
- Stage2 preprocess config:
  [ignore_rule_data_k64_dt01_completion_001.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json)
- Stage1A train config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Stage1A smoke eval config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Stage1B train config:
  [ignore_rule_data_k64_dt01_completion_001_12gb_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json)
- Stage1B smoke eval config:
  [ignore_rule_data_k64_dt01_completion_001_eval_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_eval_safe.json)
- Stage2 train config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Stage2 smoke eval config:
  [ignore_rule_data_k64_dt01_completion_001_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_eval.json)
- Stage2 sample config:
  [ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json)
- Runner script:
  [run_stage1_stage2_ignore_rule_completion_001.sh](/home/masa/minipamayo/minipamayo-qwen-3-5/scripts/run_stage1_stage2_ignore_rule_completion_001.sh)

補足:

- `Stage2 preprocess` の出力先は `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/<run>/samples_reasoning_sft.jsonl`
- `Stage2 preprocess` は毎回 fresh に再生成する。流用しない
- `Stage2 preprocess` は line count `6423 / 4676 / 6167` を満たしたときだけ成功扱い
- `Stage1A` は hard gate。成功 run が 1 本もない場合は `failed`
- `Stage2 train` は `Stage1A completion_001 best.pt` を参照する
- `Stage2 sample inference` は `Stage2 best.pt` と `Stage1B safe best.pt` の両方を参照する
- `Stage1B` は hard gate。成功 run が 1 本もない場合は `partial progress` ではなく `failed`
- `Stage2 train` も hard gate。成功 run が 1 本もないまま停止した場合は `full success` ではなく `partial progress` にとどめる
- 途中停止した場合でも、`Stage1A` と `Stage1B` が成功済みならその時点の状態を `partial progress` として記録してよい

## Attempt Template

### Attempt: `<name>`

- Date:
- Operator:
- Working directory:
  `/home/masa/minipamayo/minipamayo-qwen-3-5`
- Run duration:
- Goal:
- Save dir policy:
  attempt-scoped fresh-only
- Launcher:
- Session / pid:
- Log root:
- Master log:
- Run status json:
- Run exit code file:
- Notes:

#### Stage2 Preprocess

- Command:
- Config:
- Start time:
- End time:
- Exit status:
- Generation policy:
  fresh only
- Pre-run file action:
  absent / overwritten / moved aside
- Validation:
  line count exact match
- Output files:
  - `weave_ccw samples_reasoning_sft.jsonl`:
  - `weave_cw samples_reasoning_sft.jsonl`:
  - `perimeter_cw samples_reasoning_sft.jsonl`:
- Record counts:
  - `weave_ccw`:
  - `weave_cw`:
  - `perimeter_cw`:
- Note:

#### Stage1A Train

- Attempt id:
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
- Config:
- Save dir:
- Pre-run save dir action:
- Start time:
- End time:
- Exit status:
- Failure signature before this attempt:
- Checkpoints:
  - `best.pt`:
  - `final.pt`:
  - `last.pt`:
- Summary:
  - best epoch:
  - best val loss:
  - final train loss:
  - final val loss:
  - train token accuracy:
  - val token accuracy:
- Artifact links:
  - `run_config.json`:
  - `summary.json`:
- Retry decision:

#### Stage1A Smoke Eval

- Command:
- Config:
- Output json:
- Key metrics:
  - teacher_forced_token_accuracy:
  - autoregressive_token_accuracy:
  - action_mae_kappa:
  - ade_m:
  - fde_m:
- Note:

#### Stage1B Train

- Attempt id:
- Profile:
  safe
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
- Config:
- Stage1 checkpoint used:
- Save dir:
- Pre-run save dir action:
- Start time:
- End time:
- Exit status:
- Failure signature before this attempt:
- Checkpoints:
  - `best.pt`:
  - `last.pt`:
- Summary:
  - best epoch:
  - best val loss:
  - final train loss:
  - final val loss:
  - train metric summary:
  - val metric summary:
- Artifact links:
  - `history.json`:
  - `run_config.json`:
  - `summary.json`:
- Retry decision:
  success / retry / failed

#### Stage1B Smoke Eval

- Command:
- Config:
- Output json:
- Key metrics:
  - ade_m:
  - fde_m:
  - mean_max_lateral_error_m:
  - global_max_lateral_error_m:
- Note:

#### Stage2 Train

- Attempt id:
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
- Config:
- Stage1A checkpoint used:
- Save dir:
- Pre-run save dir action:
- Start time:
- End time:
- Exit status:
- Failure signature before this attempt:
- Checkpoints:
  - `best.pt`:
  - `final.pt`:
  - `last.pt`:
  - `best_handoff.pt`:
- Summary:
  - train_size:
  - val_size:
  - completed_epochs:
  - stop_reason:
  - best_metric_name:
  - best_metric:
  - best_epoch:
  - best_handoff_success_rate:
  - best_handoff_epoch:
  - total_wall_time_s:
- Artifact links:
  - `run_config.json`:
  - `summary.json`:
- Retry decision:

#### Stage2 Smoke Eval

- Command:
- Config:
- Output json:
- Key metrics:
  - loss:
  - token_accuracy:
- Note:

#### Stage2 Sample Inference

- Command:
- Config:
- Output json:
- Stage1B checkpoint used:
- Sample target:
- Key fields:
  - sample_id:
  - reasoning_text:
  - ade_m:
  - fde_m:
- Note:

#### Decision

- Run outcome:
  full success / partial progress / failed
- Final reached stage:
  preprocess / stage1a / stage1b / stage2_train / stage2_eval / stage2_inference
- Next action:
- Reporting note:

## Initial Record

### Attempt: `completion_ignore_rule_full_001`

- Date: pending
- Operator: pending
- Working directory:
  `/home/masa/minipamayo/minipamayo-qwen-3-5`
- Run duration: until full completion
- Goal:
  complete `Stage1A -> Stage1B -> Stage2` on full 3-run `ignore_rule_data_k64_dt01`
- Save dir policy:
  attempt-scoped `completion_001`
- Launcher:
  `tmux new-session -d -s ignore-rule-completion-001 'MAX_STAGE1A_ATTEMPTS=0 STAGE1A_RETRY_SLEEP_S=30 MAX_STAGE1B_ATTEMPTS=0 STAGE1B_RETRY_SLEEP_S=30 MAX_STAGE2_ATTEMPTS=0 STAGE2_RETRY_SLEEP_S=30 bash scripts/run_stage1_stage2_ignore_rule_completion_001.sh'`
- Session / pid:
  `ignore-rule-completion-001`
- Log root:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001`
- Master log:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/master.log`
- Run status json:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json`
- Run exit code file:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.exitcode`
- Notes:
  - `Stage1A` success is mandatory before `Stage1B`
  - `Stage1B` success is mandatory before `Stage2`
  - full completion is the target
  - if execution stops before completion, record the last confirmed state and keep it as `partial progress` only after successful `Stage1A` and `Stage1B` runs exist
  - keep the stage order fixed for bookkeeping
  - smoke eval と sample inference は non-blocking で log に残す

#### Stage2 Preprocess

- Command:
  `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess --config-json configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json)
- Start time: pending
- End time: pending
- Exit status: pending
- Generation policy:
  fresh only
- Pre-run file action:
  pending
- Validation:
  `weave_ccw=6423`, `weave_cw=4676`, `perimeter_cw=6167`
- Output files:
  - `weave_ccw samples_reasoning_sft.jsonl`: pending
  - `weave_cw samples_reasoning_sft.jsonl`: pending
  - `perimeter_cw samples_reasoning_sft.jsonl`: pending
- Record counts:
  - `weave_ccw`: pending
  - `weave_cw`: pending
  - `perimeter_cw`: pending
- Note:
  pending

#### Stage1A Train

- Attempt id:
  `attempt_001`
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
  `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train --config-json configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Save dir:
  `checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb`
- Pre-run save dir action:
  pending
- Start time: pending
- End time: pending
- Exit status: pending
- Failure signature before this attempt:
  none
- Checkpoints:
  - `best.pt`: pending
  - `final.pt`: pending
  - `last.pt`: pending
- Summary:
  - best epoch: pending
  - best val loss: pending
  - final train loss: pending
  - final val loss: pending
  - train token accuracy: pending
  - val token accuracy: pending
- Artifact links:
  - `run_config.json`: pending
  - `summary.json`: pending
- Retry decision:
  pending

#### Stage1A Smoke Eval

- Command:
  `uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval --config-json configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Output json:
  `artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json`
- Key metrics:
  - teacher_forced_token_accuracy: pending
  - autoregressive_token_accuracy: pending
  - action_mae_kappa: pending
  - ade_m: pending
  - fde_m: pending
- Note:
  perimeter-only smoke eval

#### Stage1B Train

- Attempt id:
  `attempt_001`
- Profile:
  safe
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
  `uv run python -m minipamayo_qwen35.stage1.expert_cfm.train --config-json configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_12gb_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json)
- Stage1 checkpoint used:
  `checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/best.pt`
- Save dir:
  `checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe`
- Pre-run save dir action:
  pending
- Start time: pending
- End time: pending
- Exit status: pending
- Failure signature before this attempt:
  none
- Checkpoints:
  - `best.pt`: pending
  - `last.pt`: pending
- Summary:
  - best epoch: pending
  - best val loss: pending
  - final train loss: pending
  - final val loss: pending
  - train metric summary: pending
  - val metric summary: pending
- Artifact links:
  - `history.json`: pending
  - `run_config.json`: pending
  - `summary.json`: pending
- Retry decision:
  pending

#### Stage1B Smoke Eval

- Command:
  `uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval --config-json configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_eval_safe.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_eval_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_eval_safe.json)
- Output json:
  `artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json`
- Key metrics:
  - ade_m: pending
  - fde_m: pending
  - mean_max_lateral_error_m: pending
  - global_max_lateral_error_m: pending
- Note:
  perimeter-only smoke eval

#### Stage2 Train

- Attempt id:
  `attempt_001`
- Repeat rule:
  retry した場合はこの block を複製して attempt ごとに 1 つ残す
- Command:
  `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.train --config-json configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- Stage1A checkpoint used:
  `checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/best.pt`
- Save dir:
  `checkpoints/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb`
- Pre-run save dir action:
  pending
- Start time: pending
- End time: pending
- Exit status: pending
- Failure signature before this attempt:
  none
- Checkpoints:
  - `best.pt`: pending
  - `final.pt`: pending
  - `last.pt`: pending
  - `best_handoff.pt`: pending
- Summary:
  - train_size: pending
  - val_size: pending
  - completed_epochs: pending
  - stop_reason: pending
  - best_metric_name: pending
  - best_metric: pending
  - best_epoch: pending
  - best_handoff_success_rate: pending
  - best_handoff_epoch: pending
  - total_wall_time_s: pending
- Artifact links:
  - `run_config.json`: pending
  - `summary.json`: pending
- Retry decision:
  pending

#### Stage2 Smoke Eval

- Command:
  `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval --config-json configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_eval.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_eval.json)
- Output json:
  `artifacts/eval/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json`
- Key metrics:
  - loss: pending
  - token_accuracy: pending
- Note:
  perimeter-only smoke eval

#### Stage2 Sample Inference

- Command:
  `uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference --config-json configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json`
- Config:
  [ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json)
- Output json:
  `artifacts/inference/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json`
- Stage1B checkpoint used:
  `checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe/best.pt`
- Sample target:
  `perimeter_cw`, `sample_index = 0`
- Key fields:
  - sample_id: pending
  - reasoning_text: pending
  - ade_m: pending
  - fde_m: pending
- Note:
  pending

#### Decision

- Run outcome:
  pending
- Final reached stage:
  pending
- Next action:
  pending
- Reporting note:
  record the actual timestamps, metrics, retry history, and the last confirmed stage before execution stopped

## Run Log

- YYYY-MM-DD HH:MM JST:
  - command:
  - output:
  - note:
