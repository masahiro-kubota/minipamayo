# Stage1 Curve Block Training Experiment Plan

`curve block` が十分に抽出できているのに `perimeter_cw curve_holdout` で `Stage1A/Stage1B` が大きく崩れているため、まずは `Stage1B` までの診断実験を行う。

第1弾の目的は 2 つ。

- `学習成立確認`: curve block だけで本当に学習が進むか
- `正式比較`: holdout を汚染せずに baseline より改善するか

このため、arm は次の 4 本に固定する。

- `A full baseline`
- `B1 curve-block-only + holdout included`
- `B2 curve-block-only + holdout excluded`
- `C full + curve upweight x2`

`Stage2` には進まない。勝ち arm が出た時点で別 plan を切る。

## Baseline

既存 baseline 正本はこれを使う。rerun はしない。

- `Stage1A curve eval`
  [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)
  - `teacher_forced_token_accuracy = 0.5685`
  - `autoregressive_token_accuracy = 0.3402`
  - `action_mae_kappa = 0.02609`
  - `ade_m = 7.7326`
  - `fde_m = 21.6674`
- `Stage1B curve eval`
  [ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json)
  - `ade_m = 7.5221`
  - `fde_m = 21.2545`
  - `mean_max_lateral_error_m = 17.5960`
  - `global_max_lateral_error_m = 37.5569`
  - `action_mae_kappa = 0.02933`

## Data

curve 定義の正本:

- [ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_thresholds/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json)

external eval holdout:

- [curve_holdout.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_splits/canonical/perimeter_cw_holdout_v1/curve_holdout.jsonl)
- [split_manifest.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_splits/canonical/perimeter_cw_holdout_v1/split_manifest.json)

train pool は専用 helper で 2 種類 materialize する。

- `included`: `7362 samples`
- `excluded`: `6793 samples`
- excluded holdout: `569 samples`

出力先は次に固定する。

- [curve_block_train_pool_included.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool_included.jsonl)
- [curve_block_train_pool_excluded.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool_excluded.jsonl)
- [curve_block_train_pool.manifest.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/preprocess/stage1/curve_train_pools/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/curve_block_train_pool.manifest.json)

build コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.preprocess.build_curve_block_train_pool \
  --curve-json artifacts/preprocess/stage1/curve_thresholds/canonical/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json \
  --exclude-split-manifest artifacts/preprocess/stage1/curve_splits/canonical/perimeter_cw_holdout_v1/split_manifest.json
```

## Arms

### A. Full Baseline

既存 completion run の artifact を baseline 正本にする。再学習しない。

### B1. Curve-Block-Only + Holdout Included

目的は `sanity` のみ。正式勝者にはしない。

- `Stage1A train`
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb.json)
- `Stage1A eval`
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval.json)
- `Stage1B train`
  [ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_12gb_safe.json)
- `Stage1B eval`
  [ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_included_curve_eval_safe.json)

### B2. Curve-Block-Only + Holdout Excluded

`curve-block-only` の正式比較 arm。

- `Stage1A train`
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb.json)
- `Stage1A eval`
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval.json)
- `Stage1B train`
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_12gb_safe.json)
- `Stage1B eval`
  [ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_only_excluded_curve_eval_safe.json)

### C. Full + Curve Upweight x2

新 sampler は使わず、`full 3-run + excluded curve pool` を `train_jsonl` に並べて `2x` 相当の oversampling を行う。

- `Stage1A train`
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb.json)
- `Stage1A eval`
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval.json)
- `Stage1B train`
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_12gb_safe.json)
- `Stage1B eval`
  [ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/experiments/curve_block_training/ignore_rule_data_k64_dt01_curve_block_upweight_x2_curve_eval_safe.json)

## Execution Order

1. `B1 Stage1A -> Stage1B`
2. `B2 Stage1A -> Stage1B`
3. `C Stage1A -> Stage1B`

各 arm の `Stage1A` 成功後に直ちに `Stage1A curve eval` を回し、その後 `Stage1B` に進む。  
各 arm の `Stage1B` 成功後に直ちに `Stage1B curve eval` を回す。

## Decision Metrics

全 arm 共通の decision-bearing eval は `perimeter_cw curve_holdout 569 samples` に固定する。

primary:

- `Stage1B fde_m`

secondary:

- `Stage1B ade_m`
- `Stage1B mean_max_lateral_error_m`
- `Stage1B global_max_lateral_error_m`
- `Stage1B action_mae_kappa`
- `Stage1B pid_override.fde_m`
- `Stage1A autoregressive_token_accuracy`
- `Stage1A action_mae_kappa`

採否:

- `B1` は `sanity only`
  - 学習が壊れずに完了し、curve 指標が大きく改善するなら `curve block 学習は成立しうる` と判断する
- `B2/C` が正式比較 arm
  - `Stage1B fde_m` が baseline `21.2545` より `10%` 以上改善
  - かつ `global_max_lateral_error_m` を悪化させない
  - どちらも満たさなければ第1弾は negative result として閉じる

## Notes

- primary external eval は現行の `perimeter_cw curve_holdout` から変えない
- `curve upweight` は sampler 実装ではなく duplicate JSONL で実現する
- training loop 本体は変更しない
- internal validation は early stopping 補助であり、勝敗判定には使わない
- `Stage2` はこの plan には含めない
