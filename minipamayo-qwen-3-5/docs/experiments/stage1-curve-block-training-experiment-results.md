# Stage1 Curve Block Training Experiment Results

`Stage1B` までの `curve block` 診断実験の記録テンプレート。

## Run Info

- experiment id: `stage1_curve_block_training_v1`
- plan:
  [stage1-curve-block-training-experiment-plan.md](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-curve-block-training-experiment-plan.md)
- status: `running`
- owner: `codex`
- last_updated_at: `2026-04-04`

## Dataset Materialization

- helper:
  [build_curve_block_train_pool.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/build_curve_block_train_pool.py)
- manifest:
  [curve_block_train_pool.manifest.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool.manifest.json)
- included pool:
  [curve_block_train_pool_included.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool_included.jsonl)
- excluded pool:
  [curve_block_train_pool_excluded.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool_excluded.jsonl)

Expected:

- included: `7362`
- excluded: `6793`
- excluded_holdout: `569`

Observed:

- included: `7362`
- excluded: `6793`
- excluded_holdout: `569`
- excluded_holdout sample_id sha256: `96107441a727e7d6333836691b15414f91a2e321dc192dd88e17681b05ac5112`

## Baseline

`A full baseline` は既存 artifact を正本とし、rerun しない。

### Stage1A

- artifact:
  [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)
- `teacher_forced_token_accuracy = 0.5685`
- `autoregressive_token_accuracy = 0.3402`
- `action_mae_kappa = 0.02609`
- `ade_m = 7.7326`
- `fde_m = 21.6674`

### Stage1B

- artifact:
  [ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json)
- `ade_m = 7.5221`
- `fde_m = 21.2545`
- `mean_max_lateral_error_m = 17.5960`
- `global_max_lateral_error_m = 37.5569`
- `action_mae_kappa = 0.02933`
- `pid_override.fde_m = 21.2645`

## B1 Curve-Block-Only + Holdout Included

Purpose: `sanity only`

### Stage1A

- train config:
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json)
- artifact:
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json)
- save_dir:
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb](/home/masa/minipamayo/minipamayo-qwen-3-5/checkpoints/stage1/vlm_ce/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb)
- wandb:
  `train=stage1-vlm-ce-ignore-rule-k64-dt01-curve-block-only-included-12gb`, `eval=stage1a-eval-ignore-rule-k64-dt01-curve-block-only-included-curve`
- status: `completed`
- metrics:
  - `teacher_forced_token_accuracy = 0.5991`
  - `autoregressive_token_accuracy = 0.3184`
  - `action_mae_kappa = 0.02891`
  - `ade_m = 6.8475`
  - `fde_m = 20.4462`

### Stage1B

- train config:
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json)
- artifact:
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/expert_cfm/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json)
- save_dir:
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe](/home/masa/minipamayo/minipamayo-qwen-3-5/checkpoints/stage1/expert_cfm/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe)
- wandb:
  `train=stage1-expert-cfm-ignore-rule-k64-dt01-curve-block-only-included-12gb-safe`, `eval=stage1b-eval-ignore-rule-k64-dt01-curve-block-only-included-curve`
- status: `completed`
- metrics:
  - `ade_m = 7.4529`
  - `fde_m = 20.9618`
  - `mean_max_lateral_error_m = 17.2714`
  - `global_max_lateral_error_m = 38.6750`
  - `action_mae_kappa = 0.03178`
  - `pid_override.fde_m = 20.9802`

## B2 Curve-Block-Only + Holdout Excluded

Purpose: `decision`

### Stage1A

- train config:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json)
- artifact:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json)
- save_dir:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb](/home/masa/minipamayo/minipamayo-qwen-3-5/checkpoints/stage1/vlm_ce/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb)
- wandb:
  `train=stage1-vlm-ce-ignore-rule-k64-dt01-curve-block-only-excluded-12gb`, `eval=stage1a-eval-ignore-rule-k64-dt01-curve-block-only-excluded-curve`
- status: `completed`
- metrics:
  - `teacher_forced_token_accuracy = 0.5617`
  - `autoregressive_token_accuracy = 0.3107`
  - `action_mae_kappa = 0.02698`
  - `ade_m = 6.5422`
  - `fde_m = 19.0951`

### Stage1B

- train config:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json)
- artifact:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/expert_cfm/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json)
- save_dir:
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe](/home/masa/minipamayo/minipamayo-qwen-3-5/checkpoints/stage1/expert_cfm/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe)
- wandb:
  `train=stage1-expert-cfm-ignore-rule-k64-dt01-curve-block-only-excluded-12gb-safe`, `eval=stage1b-eval-ignore-rule-k64-dt01-curve-block-only-excluded-curve`
- status: `completed`
- metrics:
  - `ade_m = 7.4957`
  - `fde_m = 21.0906`
  - `mean_max_lateral_error_m = 17.4149`
  - `global_max_lateral_error_m = 38.5248`
  - `action_mae_kappa = 0.02886`
  - `pid_override.fde_m = 21.1049`

## C Full + Curve Upweight x2

Purpose: `decision`

### Stage1A

- train config:
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval.json)
- save_dir: `pending`
- wandb: `pending`
- status: `running`
- metrics:
  - `teacher_forced_token_accuracy = pending`
  - `autoregressive_token_accuracy = pending`
  - `action_mae_kappa = pending`
  - `ade_m = pending`
  - `fde_m = pending`

### Stage1B

- train config:
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb_safe.json)
- eval config:
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval_safe.json)
- save_dir: `pending`
- wandb: `pending`
- status: `pending`
- metrics:
  - `ade_m = pending`
  - `fde_m = pending`
  - `mean_max_lateral_error_m = pending`
  - `global_max_lateral_error_m = pending`
  - `action_mae_kappa = pending`
  - `pid_override.fde_m = pending`

## Decision

- `B1`: `sanity only`
- formal winner candidate: `B2` or `C`
- success condition:
  - `Stage1B fde_m <= 19.1291`
  - and `global_max_lateral_error_m <= 37.5569`
- final decision: `pending`
- next action: `pending`
