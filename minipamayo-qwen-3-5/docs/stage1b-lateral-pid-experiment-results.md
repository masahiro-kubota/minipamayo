# Stage1B Lateral PID Experiment Results

このファイルは、[stage1b-lateral-pid-experiment-plan.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/stage1b-lateral-pid-experiment-plan.md)
に沿って進める `Stage1B lateral + longitudinal PID override` 実験の結果を記録する。

使い方:
- 各 rung の実行後に、このファイルへ追記する
- `Stage1A -> Stage1A gate -> Stage1B -> PID 比較評価` を 1 セットとして 1 セクションにまとめる
- 定量結果だけでなく、「次の rung へ進むか」「何がボトルネックに見えたか」も残す

このファイルの数値の読み方:
- `curve_holdout` に対する数値は、現段階では final test ではなく `dev holdout` の結果として扱う
- `Stage1A` / `Stage1B` の `best.pt` 選択と、結果表に載せる holdout 指標は同じ `curve_holdout` を見ている
- したがって、ここに載せる ADE / FDE / lateral error は `dev-selected result` であり、最終主張は後日の新規 simulation test で確認する

## Summary Table

| Rung | Train Data | Dev Holdout | Stage1A Gate (dev) | Stage1B Canonical (dev) | Stage1B + PID (dev) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| probe16 smoke | n/a | n/a | n/a | canonical failed badly | PID less bad but still failed | move to `perimeter_cw_curve128` |
| perimeter_cw_curve128 | `train_128.jsonl` (128) | `curve_holdout.jsonl` dev holdout, gate run on first 128 samples | fail, token collapse (`unique_bins_used=1`) | pending | pending | continue to `Stage1B` for diagnosis, then likely move to next rung |
| perimeter_cw_curve512 | pending | pending | pending | pending | pending | pending |
| perimeter_cw_curve2048 | pending | pending | pending | pending | pending | pending |
| perimeter_cw_full | pending | pending | pending | pending | pending | pending |
| perimeter_cw+weave_cw | pending | pending | pending | pending | pending | pending |
| perimeter_cw+weave_cw+weave_ccw | pending | pending | pending | pending | pending | pending |
| full_ignore_rule_data | pending | pending | pending | pending | pending | pending |

## Shared Setup

- Raw data:
  [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)
- Canonical extraction config:
  [ignore_rule_data_k64_dt01.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json)
- Curve block definition:
  `mode=or`, `max |kappa_gt| >= 0.08` or `|yaw change| >= 0.5`, `pre=1.0s`, `post=2.0s`
- Curve split manifest:
  [split_manifest.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/split_manifest.json)
- Dev holdout interpretation:
  - `curve_holdout` は現時点では exploratory な `dev holdout`
  - `best.pt` model selection と同じ split を見ているため、final test ではない
- PID initial target speed:
  `24 km/h`

## Result Template

### Rung: `<name>`

- Date:
- Goal:
- Train JSONL:
- Dev Holdout JSONL:
- Stage1A train config:
- Stage1A eval config:
- Stage1B train config:
- Stage1B eval config:
- Stage1B inference config:
- Checkpoints:
  - Stage1A:
  - Stage1B:

#### Stage1A Gate (dev holdout)

- teacher_forced_token_accuracy:
- autoregressive_token_accuracy:
- action_mae_kappa:
- ade_m:
- fde_m:
- Gate decision:

#### Stage1B Canonical (dev holdout)

- ade_m:
- fde_m:
- mean_max_lateral_error_m:
- global_max_lateral_error_m:

#### Stage1B + PID (dev holdout)

- target_speed_kmh:
- pid_gains:
- ade_m:
- fde_m:
- mean_max_lateral_error_m:
- global_max_lateral_error_m:

#### Qualitative Notes

- Example plots / samples:
- Observed behavior:

#### Decision

- Stop here / move to next rung:
- Reason:
- Next action:
- Reporting note:
  - この rung の表中メトリクスは `dev-selected result` として読む

## Run Log

- 2026-03-30:
  - command:
    `uv run python -m minipamayo_qwen35.stage1.vlm_ce.train --config-json configs/stage1/vlm_ce/train/canonical/perimeter_cw_curve128_12gb.json`
  - output:
    `best val_loss = 2.9539488103561538`, `best_epoch = 5`, `train_token_accuracy = 0.4464`, `val_token_accuracy = 0.3101`
  - note:
    `Stage1A curve128` completed. Training itself was stable, but validation token accuracy stayed low.

- 2026-03-30:
  - command:
    `uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval --config-json /tmp/perimeter_cw_curve128_eval_128.json`
  - output:
    `teacher_forced_token_accuracy = 0.400390625`, `autoregressive_token_accuracy = 0.40008544921875`, `action_mae_kappa = 0.02081393636763096`, `ade_m = 5.695497989654541`, `fde_m = 16.480146408081055`, `unique_bins_used = 1`
  - note:
    Full `curve_holdout` eval was too slow for the first pass, so gate was run on the first 128 samples from the same dev holdout. The model collapsed to one bin (`128`), so this rung is a gate fail.

- 2026-03-30:
  - command:
    `probe16` Stage1B PID smoke
  - output:
    canonical `ADE ≈ 60.18`, `FDE ≈ 86.67`, `mean/global lateral ≈ 53.31`; PID override `ADE ≈ 17.28`, `FDE ≈ 38.47`, `mean/global lateral ≈ 35.12`
  - note:
    Technical smoke only. It showed the PID branch worked and was less bad than canonical, but both were far from usable.

- YYYY-MM-DD:
  - command:
  - output:
  - note:

## Result Records

### Rung: `probe16 smoke`

- Date: `2026-03-30`
- Goal: `PID override` の技術確認
- Train JSONL: n/a
- Dev Holdout JSONL: n/a
- Stage1A train config: existing `probe16` checkpoint reuse
- Stage1A eval config: n/a
- Stage1B train config: existing `probe16` checkpoint reuse
- Stage1B eval config:
  technical 1-sample temp eval
- Stage1B inference config:
  [ignore_rule_data_probe16_pid_sample.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/inference/canonical/ignore_rule_data_probe16_pid_sample.json)
- Checkpoints:
  - Stage1A: existing `probe16`
  - Stage1B: existing `probe16`

#### Stage1A Gate (dev holdout)

- teacher_forced_token_accuracy: n/a
- autoregressive_token_accuracy: n/a
- action_mae_kappa: n/a
- ade_m: n/a
- fde_m: n/a
- Gate decision: n/a, 技術確認のみ

#### Stage1B Canonical (dev holdout)

- ade_m: about `60.18`
- fde_m: about `86.67`
- mean_max_lateral_error_m: about `53.31`
- global_max_lateral_error_m: about `53.31`

#### Stage1B + PID (dev holdout)

- target_speed_kmh: `24`
- pid_gains: `kp=1.0, ki=0.05, kd=0.0`
- ade_m: about `17.28`
- fde_m: about `38.47`
- mean_max_lateral_error_m: about `35.12`
- global_max_lateral_error_m: about `35.12`

#### Qualitative Notes

- Example plots / samples:
  [ignore_rule_data_probe16_pid_sample.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/inference/stage1/expert_cfm/canonical/ignore_rule_data_probe16_pid_sample.json)
- Observed behavior:
  PID branch is wired correctly and can change the trajectory, but neither canonical nor PID was usable at this smoke size.

#### Decision

- Stop here / move to next rung:
  move to `perimeter_cw_curve128`
- Reason:
  this was only a technical smoke, not a meaningful learning result
- Next action:
  run `perimeter_cw_curve128`
- Reporting note:
  this is a smoke result, not a final claim

### Rung: `perimeter_cw_curve128`

- Date: `2026-03-30`
- Goal: first real rung for `perimeter_cw` with curve-focused split
- Train JSONL:
  [train_128.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/train_128.jsonl)
- Dev Holdout JSONL:
  [curve_holdout.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/curve_holdout.jsonl)
- Stage1A train config:
  [perimeter_cw_curve128_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/perimeter_cw_curve128_12gb.json)
- Stage1A eval config:
  [perimeter_cw_curve128_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/canonical/perimeter_cw_curve128_eval.json)
- Stage1B train config:
  [perimeter_cw_curve128_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/perimeter_cw_curve128_12gb.json)
- Stage1B eval config:
  [perimeter_cw_curve128_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/perimeter_cw_curve128_eval.json)
- Stage1B inference config:
  [perimeter_cw_curve128_sample.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/inference/canonical/perimeter_cw_curve128_sample.json)
- Checkpoints:
  - Stage1A:
    [best.pt](/home/masa/minipamayo/minipamayo-qwen-3-5/checkpoints/stage1/vlm_ce/canonical/perimeter_cw_curve128_12gb/best.pt)
  - Stage1B:
    pending

#### Stage1A Gate (dev holdout)

- teacher_forced_token_accuracy: `0.400390625`
- autoregressive_token_accuracy: `0.40008544921875`
- action_mae_kappa: `0.02081393636763096`
- ade_m: `5.695497989654541`
- fde_m: `16.480146408081055`
- Gate decision:
  fail

#### Stage1B Canonical (dev holdout)

- ade_m: pending
- fde_m: pending
- mean_max_lateral_error_m: pending
- global_max_lateral_error_m: pending

#### Stage1B + PID (dev holdout)

- target_speed_kmh: `24`
- pid_gains: `kp=1.0, ki=0.05, kd=0.0`
- ade_m: pending
- fde_m: pending
- mean_max_lateral_error_m: pending
- global_max_lateral_error_m: pending

#### Qualitative Notes

- Example plots / samples:
  [perimeter_cw_curve128_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/perimeter_cw_curve128_eval.json)
- Observed behavior:
  `Stage1A` did not learn a healthy discrete action distribution. The autoregressive evaluator reports `unique_bins_used = 1`, so the output collapsed to a single bin despite train loss improvement.

#### Decision

- Stop here / move to next rung:
  continue to `Stage1B` for diagnosis, then likely move to `perimeter_cw_curve512`
- Reason:
  gate failed, but one `Stage1B` run is still useful to see whether KV-cache conditioning rescues lateral behavior at this rung
- Next action:
  train and evaluate `Stage1B curve128`
- Reporting note:
  current numbers are from a 128-sample dev holdout pass for turnaround, not the full dev holdout
