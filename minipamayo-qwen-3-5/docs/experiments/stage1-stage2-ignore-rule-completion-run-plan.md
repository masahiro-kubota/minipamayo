# Stage1 / Stage2 Ignore Rule Completion Run Plan

この計画書は、`ignore_rule_data` の 3 run を使って、`Stage1A -> Stage1B -> Stage2` を完了するまで進める completion run の実行方針を固定する。今回の run では `Stage1A`, `Stage1B`, `Stage2 train` を hard gate とし、各 train が成功するまで fresh rerun を継続する。

結果の記録先:

- [stage1-stage2-ignore-rule-completion-run-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-results.md)

## 目的

今回の目的は次の順で優先する。

1. `Stage1A` canonical VLM CE を full 3-run データで完走させる
2. `Stage1B` expert CFM safe profile を、上の `Stage1A best.pt` から確実に通す
3. `Stage2` reasoning SFT の data preprocess と train を完了させる
4. `Stage1A` / `Stage1B` / `Stage2` の smoke eval と `Stage2` sample inference を最後まで残す

目標は full completion であり、timebox は置かない。

## 今回使うデータ

raw data の正本:

- [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)

今回固定する 3 run:

- `20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8`

Stage1 canonical extraction config:

- [ignore_rule_data_k64_dt01.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json)

Stage1 train JSONL:

- [weave_ccw samples.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8/samples.jsonl)
  - `6423 samples`
- [weave_cw samples.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8/samples.jsonl)
  - `4676 samples`
- [perimeter_cw samples.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8/samples.jsonl)
  - `6167 samples`

Stage1 full train sample 数:

- `17266`

Stage2 は上の `samples.jsonl` から attempt-scoped の `samples_reasoning_sft.jsonl` を生成して使う。
preprocess が成功した場合、record 数は元の `samples.jsonl` と同数の `6423 / 4676 / 6167` になる想定で進める。

## 固定する config

### Stage1A

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)

smoke eval:

- [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)

### Stage1B

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json)

smoke eval:

- [ignore_rule_data_k64_dt01_completion_001_eval_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_eval_safe.json)

sample inference:

- [ignore_rule_data_k64_dt01_completion_001_sample_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_safe.json)

safe profile の意図:

- 学習データ、epoch 数、model 幅は canonical full config と同じ
- `num_workers=0` に固定して dataloader 起因の不安定さを避ける
- `save_dir` と artifact 出力先を `completion_001` 専用 path に分けて、古い run と混ざらないようにする
- 今回の completion run では fast profile より安全性を優先し、この safe profile を正本として使う

### Stage2

reasoning JSONL preprocess:

- [ignore_rule_data_k64_dt01_completion_001.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json)

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)

smoke eval:

- [ignore_rule_data_k64_dt01_completion_001_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_eval.json)

sample inference:

- [ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json)

補足:

- `Stage2 train` は `Stage1A best.pt` を使う
- `Stage2 sample inference` は `Stage2 best.pt` に加えて `Stage1B safe best.pt` も使う
- `Stage2 train` 自体は `Stage1A` 依存だが、今回の bookkeeping では `Stage1A` と `Stage1B` の成功を hard gate として `Stage1A -> Stage1B -> Stage2` に固定する

## Artifact Hygiene

- train / eval / inference の checkpoint と出力 artifact は、すべて `completion_001` 専用 config を使い、共有 canonical save dir には書かない。
- `Stage2 preprocess` は毎回 fresh に再生成する。出力先は `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/<run>/samples_reasoning_sft.jsonl` に固定し、既存 artifact は provenance check で流用しない。
- fresh run を始める前に、対象 `save_dir` がすでに存在する場合は削除ではなく退避する。推奨は `<save_dir>_bak_<YYYYMMDD_HHMMSS>` へ rename。
- `Stage1A`, `Stage1B`, `Stage2 train` は retry 時にも同じ config を使い、毎回 partial save dir を退避してから rerun する。
- この plan は fresh-only。in-place resume は定義しない。再実行は退避後の fresh rerun として扱う。

## Runbook

runner script:

- [run_stage1_stage2_ignore_rule_completion_001.sh](/home/masa/minipamayo/minipamayo-qwen-3-5/scripts/run_stage1_stage2_ignore_rule_completion_001.sh)

runner の責務:

- `Stage2 preprocess -> Stage1A train retry loop -> Stage1A smoke eval -> Stage1B train retry loop -> Stage1B smoke eval -> Stage2 train retry loop -> Stage2 smoke eval -> Stage2 sample inference` を自動で連鎖する
- `Stage2 preprocess` 出力、attempt-scoped `save_dir`、eval / inference 出力を開始前に退避する
- `Stage2 preprocess` の出力 line count を `6423 / 4676 / 6167` で検証し、一致しなければ failed で止める
- 各 stage の stdout / stderr を個別 log に保存しつつ、master log にも集約する
- 各 stage の exit code を `state/<stage>.exitcode` に保存する
- run 全体の状態を `run.status.json` と `run.exitcode` に保存し、`run.status.json` の `current_stage` を stage 開始ごとに更新する
- `Stage1A` は hard gate。`MAX_STAGE1A_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する
- `Stage1B` は hard gate。`MAX_STAGE1B_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する
- `Stage2 train` も hard gate。`MAX_STAGE2_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する

固定 path:

- log root:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001`
- master log:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/master.log`
- run status:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json`
- run exit code:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.exitcode`
- per-stage exit codes:
  `/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/state/*.exitcode`

推奨起動方法:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
tmux new-session -d -s ignore-rule-completion-001 \
  'MAX_STAGE1A_ATTEMPTS=0 STAGE1A_RETRY_SLEEP_S=30 MAX_STAGE1B_ATTEMPTS=0 STAGE1B_RETRY_SLEEP_S=30 MAX_STAGE2_ATTEMPTS=0 STAGE2_RETRY_SLEEP_S=30 bash scripts/run_stage1_stage2_ignore_rule_completion_001.sh'
```

fallback:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
nohup env MAX_STAGE1A_ATTEMPTS=0 STAGE1A_RETRY_SLEEP_S=30 \
  MAX_STAGE1B_ATTEMPTS=0 STAGE1B_RETRY_SLEEP_S=30 \
  MAX_STAGE2_ATTEMPTS=0 STAGE2_RETRY_SLEEP_S=30 \
  bash scripts/run_stage1_stage2_ignore_rule_completion_001.sh >/dev/null 2>&1 &
echo $!
```

監視コマンド:

```bash
tail -f /home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/master.log
```

```bash
cat /home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json
cat /home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.exitcode
```

補足:

- train stage は blocking。`Stage2 preprocess` は失敗時に chain を止め、`Stage1A train`, `Stage1B train`, `Stage2 train` は成功するか retry 上限に達するまで rerun する
- smoke eval と sample inference は non-blocking。失敗しても後続 train は止めず、log と exit code に残す
- `Stage1A` 成功前は `partial progress` にしない
- `Stage1B` retry は transient failure を想定した安全策であり、failure signature は結果 markdown に残す

## 実行順

### 手順0. Stage2 reasoning JSONL を先に生成する

理由:

- `Stage2` だけ dataset 契約が違う
- 前処理は短く終わるので、長い train を始める前に fail-fast できる

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess \
  --config-json configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json
```

期待する出力:

- `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/<run>/samples_reasoning_sft.jsonl`
  が 3 run 分そろうこと
- line count が `weave_ccw=6423`, `weave_cw=4676`, `perimeter_cw=6167` で一致すること

skip 条件:

- なし
- 既存 `samples_reasoning_sft.jsonl` があっても fresh に再生成する

### 手順1. Stage1A を full 3-run で学習する

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.vlm_ce.train \
  --config-json configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json
```

完了条件:

- `checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/summary.json`
  が書かれる
- `best.pt` と `final.pt` が残る

retry ルール:

- `Stage1A` は hard gate。exit code 非 0、`summary.json` 欠落、`best.pt` 欠落、`final.pt` 欠落のいずれかなら未成功扱いにする
- `Stage1A` が未成功のあいだは `Stage1B` に進まない
- `Stage1A` が落ちたら partial save dir を退避し、同じ config で fresh rerun する
- 成功した `Stage1A` run が 1 本できるまで retry を継続する
- retry ごとに failure signature と退避した save dir 名を結果 markdown に残す

### 手順2. Stage1A smoke eval を 1 本だけ回す

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
  --config-json configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json
```

位置づけ:

- full 3-run external eval ではない
- checkpoint load と metric emission の smoke
- runner では non-blocking

### 手順3. Stage1B を同じ 3 run で学習する

前提:

- `Stage1A best.pt` が存在すること
- `checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe`
  が fresh であること。既存 dir があれば退避してから始めること。

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.expert_cfm.train \
  --config-json configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json
```

完了条件:

- `checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe/summary.json`
  が書かれる
- `best.pt` と `last.pt` が残る

retry ルール:

- `Stage1B` は hard gate。exit code 非 0、`summary.json` 欠落、`best.pt` 欠落のいずれかなら未成功扱いにする
- `Stage1B` が未成功のあいだは `Stage2` に進まない
- `Stage1B` が落ちたら partial save dir を退避し、同じ safe config で fresh rerun する
- 成功した `Stage1B` run が 1 本できるまで retry を継続する
- retry ごとに failure signature と退避した save dir 名を結果 markdown に残す

### 手順4. Stage1B smoke eval を回す

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
  --config-json configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_eval_safe.json
```

位置づけ:

- `Stage1B` checkpoint load と perimeter smoke の sanity check
- runner では non-blocking

### 手順5. Stage2 を学習する

前提:

- 手順0の `samples_reasoning_sft.jsonl` が 3 run 分そろっていること
- `Stage1A best.pt` が存在すること
- `Stage1B safe best.pt` が存在すること

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.train \
  --config-json configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json
```

完了条件:

- `checkpoints/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb/summary.json`
  が書かれる
- `best.pt` が残る

補足:

- この config は `max_epochs = 2`

retry ルール:

- `Stage2 train` は hard gate。exit code 非 0、`summary.json` 欠落、`best.pt` 欠落のいずれかなら未成功扱いにする
- `Stage2` が未成功のあいだも run は完了扱いにしない
- `Stage2 train` が落ちたら partial save dir を退避し、同じ config で fresh rerun する
- 成功した `Stage2 train` run が 1 本できるまで retry を継続する
- retry ごとに failure signature と退避した save dir 名を結果 markdown に残す

### 手順6. Stage2 smoke eval を回す

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval \
  --config-json configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_eval.json
```

位置づけ:

- `perimeter_cw` の `samples_reasoning_sft.jsonl` に対する smoke eval
- `Stage2` full 3-run external eval の主張には使わない
- runner では non-blocking

### 手順7. Stage2 sample inference を 1 本だけ回す

前提:

- `Stage2 best.pt` があること
- `Stage1B safe best.pt` があること

コマンド:

```bash
cd /home/masa/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference \
  --config-json configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_stage1b_safe.json
```

位置づけ:

- `reasoning_text` が出るか
- handoff 後の sample trajectory が最低限生成できるか
- runner では non-blocking

## 成功判定

full success:

- `Stage1A train` 完了
- `Stage1B train` 完了
- `Stage2 preprocess` 完了
- `Stage2 train` 完了
- `Stage2 smoke eval` と `Stage2 sample inference` も出力完了

partial progress:

- 外部都合で途中停止する可能性はある
- `Stage1A train` と `Stage1B train` が成功済みであることを前提に、その先が全部終わらなくてもよい
- 特に次の状態は `partial progress` とみなす
  - `Stage1B` 完了、`Stage2` 未着手
  - `Stage2 preprocess` 完了、`Stage2 train` 進行中
  - `Stage2 train` が途中まで進み、最後に確認できた epoch / step が分かる

failed:

- `Stage2 preprocess` が契約違反で止まり、そのまま解消できない
- `Stage1A` が学習開始できない
- 実行を止めた時点で `Stage1A` 成功 run が 1 本もない
- 実行を止めた時点で `Stage1B` 成功 run が 1 本もない
- 生成された artifact がなく、どこまで進んだかも追えない

## 失敗時の扱い

`Stage2 preprocess` が落ちた場合:

- `Stage1A` 以降は始めない
- dataset contract の失敗箇所を結果 markdown に残す

`Stage1A` が落ちた場合:

- partial save dir を退避する
- failure signature を結果 markdown に残す
- 同じ config で fresh rerun する
- `Stage1B` / `Stage2` には進まない
- 実行を止めた時点で成功 run が 1 本もなければ failed とする

`Stage1B` が落ちた場合:

- partial save dir を退避する
- failure signature を結果 markdown に残す
- 同じ safe config で fresh rerun する
- 成功するまで `Stage2` には進まない
- 実行を止めた時点で成功 run が 1 本もなければ failed とする

`Stage2` が落ちた場合:

- partial save dir を退避する
- failure signature を結果 markdown に残す
- 同じ config で fresh rerun する
- 実行を止めた時点で最後に確認できた epoch / step / `best.pt` / `last.pt` の状態を残す

rerun した場合:

- どの path を退避したか
- どの rerun が最終的な正本になったか
- fresh rerun にした理由

を必ず結果 markdown に書く。

## 記録ルール

今回の結果記録先:

- [stage1-stage2-ignore-rule-completion-run-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-results.md)

最低限残す項目:

- 開始日時 / 終了日時
- 実際に使った config
- `Stage2 preprocess` の出力有無と record 数
- `Stage1A` / `Stage1B` / `Stage2` の `summary.json` 主要数値
- `Stage1B` の retry 回数、各 retry の失敗理由、退避した save dir
- smoke eval の主要数値
- sample inference の有無と代表 sample の結果
- 途中停止したなら、その時点の最後の状態
