# Stage1B Lateral PID Experiment Results

このファイルは、[stage1b-lateral-pid-experiment-plan.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/stage1b-lateral-pid-experiment-plan.md)
に沿って進める `Stage1B lateral + longitudinal PID override` 実験の結果を記録する。

使い方:
- 各 rung の実行後に、このファイルへ追記する
- `Stage1A -> Stage1A gate -> Stage1B -> PID 比較評価` を 1 セットとして 1 セクションにまとめる
- 定量結果だけでなく、「次の rung へ進むか」「何がボトルネックに見えたか」も残す

## Summary Table

| Rung | Train Data | Holdout | Stage1A Gate | Stage1B Canonical | Stage1B + PID | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| probe16 smoke | pending | pending | pending | pending | pending | pending |
| perimeter_cw_curve128 | pending | pending | pending | pending | pending | pending |
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
- PID initial target speed:
  `24 km/h`

## Result Template

### Rung: `<name>`

- Date:
- Goal:
- Train JSONL:
- Holdout JSONL:
- Stage1A train config:
- Stage1A eval config:
- Stage1B train config:
- Stage1B eval config:
- Stage1B inference config:
- Checkpoints:
  - Stage1A:
  - Stage1B:

#### Stage1A Gate

- teacher_forced_token_accuracy:
- autoregressive_token_accuracy:
- action_mae_kappa:
- ade_m:
- fde_m:
- Gate decision:

#### Stage1B Canonical

- ade_m:
- fde_m:
- mean_max_lateral_error_m:
- global_max_lateral_error_m:

#### Stage1B + PID

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

## Run Log

- YYYY-MM-DD:
  - command:
  - output:
  - note:
