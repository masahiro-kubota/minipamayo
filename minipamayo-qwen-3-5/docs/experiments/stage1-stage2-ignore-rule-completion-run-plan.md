# Stage1 / Stage2 Ignore Rule Completion Run Plan

この計画書は、`ignore_rule_data` の 3 run を使って、`Stage1A -> Stage1B -> Stage2` を完了するまで進める completion run の実行方針を固定する。今回の run では `Stage1A`, `Stage1B`, `Stage2 train` を hard gate とし、各 train が成功するまで fresh rerun を継続する。

2026-04-03 更新:

- 学習が不十分な checkpoint は基本的に直進してしまい、`perimeter_cw` 全体の eval だと見かけ上ごまかせる
- そのため、decision-bearing な eval は `perimeter_cw` の `curve_holdout` に統一する
- 既存の perimeter-wide smoke eval artifact があっても参考情報にとどめ、正式な記録は curve-only eval で置き換える

結果の記録先:

- [stage1-stage2-ignore-rule-completion-run-results.md](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-results.md)

## 目的

今回の目的は次の順で優先する。

1. `Stage1A` canonical VLM CE を full 3-run データで完走させる
2. `Stage1B` expert CFM safe profile を、上の `Stage1A best.pt` から確実に通す
3. `Stage2` reasoning SFT の train 用 preprocess と train を完了させる
4. `Stage1A` / `Stage1B` / `Stage2` の curve-only eval と `Stage2` curve sample inference を最後まで残す

目標は full completion であり、timebox は置かない。

## 今回使うデータ

raw data の正本:

- [ignore_rule_data](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)

今回固定する 3 run:

- `20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8`

Stage1 canonical extraction config:

- [ignore_rule_data_k64_dt01.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json)

Stage1 train JSONL:

- [weave_ccw samples.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8/samples.jsonl)
  - `6423 samples`
- [weave_cw samples.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8/samples.jsonl)
  - `4676 samples`
- [perimeter_cw samples.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8/samples.jsonl)
  - `6167 samples`

Stage1 full train sample 数:

- `17266`

curve-focused eval holdout:

- [split_manifest.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/split_manifest.json)
- [curve_holdout.jsonl](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/curve_holdout.jsonl)
  - `569 samples`

Stage2 は 2 系統の JSONL を使う。

- train 用:
  上の full 3-run `samples.jsonl` から attempt-scoped の `samples_reasoning_sft.jsonl` を 3 run 分生成して学習する
- curve eval 用:
  `curve_holdout.jsonl` から attempt-scoped の `samples_reasoning_sft.jsonl` を 1 本生成し、`Stage2 eval` と `Stage2 sample inference` に使う

## 固定する config

### Stage1A

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)

curve eval:

- [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)

補足:

- `Stage1A curve eval` は `best.pt` を使う
- eval 対象は `perimeter_cw` 全体ではなく `curve_holdout.jsonl`

### Stage1B

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe.json)

curve eval:

- [ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json)

sample inference:

- [ignore_rule_data_k64_dt01_completion_001_sample_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/inference/canonical/ignore_rule_data_k64_dt01_completion_001_sample_safe.json)

safe profile の意図:

- 学習データ、epoch 数、model 幅は canonical full config と同じ
- `num_workers=0` に固定して dataloader 起因の不安定さを避ける
- `save_dir` と artifact 出力先を `completion_001` 専用 path に分けて、古い run と混ざらないようにする
- 今回の completion run では fast profile より安全性を優先し、この safe profile を正本として使う

curve eval の補足:

- eval 対象は `curve_holdout.jsonl`
- `include_pid_override=true` を有効にして、速度制御の崩れに引きずられず lateral の改善を見やすくする

### Stage2

train reasoning JSONL preprocess:

- [ignore_rule_data_k64_dt01_completion_001.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001.json)

curve eval reasoning JSONL preprocess:

- [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)

train:

- [ignore_rule_data_k64_dt01_completion_001_12gb.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)

curve eval:

- [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)

curve sample inference:

- [ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json)

補足:

- `Stage2 train` は `Stage1A best.pt` を使う
- `Stage2 curve eval` は `curve_holdout.jsonl` 由来の reasoning JSONL を使う
- `Stage2 curve sample inference` は `Stage2 best.pt` に加えて `Stage1B safe best.pt` も使う
- `Stage2 train` 自体は `Stage1A` 依存だが、今回の bookkeeping では `Stage1A` と `Stage1B` の成功を hard gate として `Stage1A -> Stage1B -> Stage2` に固定する

## Artifact Hygiene

- train / eval / inference の checkpoint と出力 artifact は、すべて `completion_001` 専用 config を使い、共有 canonical save dir には書かない
- `Stage2 train preprocess` は毎回 fresh に再生成する。出力先は `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001/<run>/samples_reasoning_sft.jsonl`
- `Stage2 curve eval preprocess` も毎回 fresh に再生成する。出力先は `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001_curve_eval/perimeter_cw_holdout_v1/samples_reasoning_sft.jsonl`
- fresh run を始める前に、対象 `save_dir` と eval / inference の `output_json`, `progress_json` がすでに存在する場合は削除ではなく退避する。`Stage1A curve eval` はこれに加えて `per_sample_jsonl` も退避対象に含める。推奨は `<path>_bak_<YYYYMMDD_HHMMSS>` へ rename
- curve eval / curve inference は `output_json` と `progress_json` の config 明示を必須にし、W&B online logging も必須にする。`Stage1A curve eval` は `per_sample_jsonl` の config 明示も必須にする。W&B init 失敗や path 設定漏れは command failure と同じ扱いにする
- 追加で、eval / inference の `wandb_project`, `wandb_run_name`, `progress_every_samples`, `progress_every_seconds` は config 明示を必須にする。`Stage1A` と `Stage2` の image budget、`Stage1B` の `flow_steps`、`Stage2 inference` の sampling params、`Stage1B` の PID override gains は silent default を禁止する
- `Stage1A`, `Stage1B`, `Stage2 train` は retry 時にも同じ config を使い、毎回 partial save dir を退避してから rerun する
- この plan は fresh-only。in-place resume は定義しない。再実行は退避後の fresh rerun として扱う

## Runbook

runner entrypoints:

- [ignore_rule_run.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/ops/ignore_rule_run.py)
- [ignore_rule_paths.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/ops/ignore_rule_paths.py)
- [ignore-rule-run-entrypoints.md](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/docs/experiments/ignore-rule-run-entrypoints.md)

runner の責務:

- `uv run python -m minipamayo_qwen35.ops.ignore_rule_run full` は `Stage2 train preprocess -> Stage1A train retry loop -> Stage1A curve eval -> Stage1B train retry loop -> Stage1B curve eval -> Stage2 train retry loop -> Stage2 curve eval -> Stage2 curve sample inference` を自動で連鎖する
- `full` subcommand は eval command を直書きせず、curve eval が必要なタイミングで同じ module の `eval` subcommand を subprocess で呼ぶ
- `uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval` は `Stage1A curve eval`, `Stage1B curve eval`, `Stage2 curve eval preprocess`, `Stage2 curve eval`, `Stage2 curve sample inference` の正本 entrypoint とする
- `Stage2` の train 用 preprocess 出力と attempt-scoped `save_dir` は `full` subcommand が開始前に退避する
- eval / inference 出力と `Stage2` curve eval preprocess 出力は `eval` subcommand が開始前に退避する
- `Stage2 train preprocess` の出力 line count を `6423 / 4676 / 6167` で検証し、一致しなければ failed で止める
- 各 stage の stdout / stderr を個別 log に保存しつつ、master log にも集約する
- 各 stage の exit code を `state/<stage>.exitcode` に保存する
- run 全体の状態を `run.status.json` と `run.exitcode` に保存し、`run.status.json` の `current_stage` を stage 開始ごとに更新する
- `Stage1A` は hard gate。`MAX_STAGE1A_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する
- `Stage1B` は hard gate。`MAX_STAGE1B_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する
- `Stage2 train` も hard gate。`MAX_STAGE2_ATTEMPTS=0` のとき無制限 retry、正数のときその回数まで retry する
- `--start-stage stage1b` を指定した場合、`Stage1A` の既存 `best.pt` / `final.pt` / `summary.json` を前提に、`Stage1B -> Stage2` だけ fresh rerun する
- `--start-stage stage1b` では、`Stage1B train` の前に `eval --target-stage stage1a` を 1 回流してから進める
- `--start-stage stage2` を指定した場合、`Stage1A` / `Stage1B` の既存 artifact を前提に、eval を含めて `Stage2` 以降だけ fresh rerun する

固定 path:

- log root:
  `/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001`
- master log:
  `/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/master.log`
- run status:
  `/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json`
- run exit code:
  `/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.exitcode`
- per-stage exit codes:
  `/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/state/*.exitcode`

推奨起動方法:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
tmux new-session -d -s ignore-rule-completion-001 \
  'uv run python -m minipamayo_qwen35.ops.ignore_rule_run full \
    --max-stage1a-attempts 0 --stage1a-retry-sleep-s 30 \
    --max-stage1b-attempts 0 --stage1b-retry-sleep-s 30 \
    --max-stage2-attempts 0 --stage2-retry-sleep-s 30'
```

`Stage1A` がすでに成功済みで、`Stage1B` から切り直す場合:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
tmux new-session -d -s ignore-rule-completion-stage1b-001 \
  'uv run python -m minipamayo_qwen35.ops.ignore_rule_run full \
    --attempt-name completion_ignore_rule_stage1b_restart_001 \
    --session-name ignore-rule-completion-stage1b-001 \
    --start-stage stage1b \
    --max-stage1b-attempts 0 --stage1b-retry-sleep-s 30 \
    --max-stage2-attempts 0 --stage2-retry-sleep-s 30'
```

fallback:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
nohup uv run python -m minipamayo_qwen35.ops.ignore_rule_run full \
  --max-stage1a-attempts 0 --stage1a-retry-sleep-s 30 \
  --max-stage1b-attempts 0 --stage1b-retry-sleep-s 30 \
  --max-stage2-attempts 0 --stage2-retry-sleep-s 30 >/dev/null 2>&1 &
echo $!
```

監視コマンド:

```bash
tail -f /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/master.log
```

```bash
cat /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.status.json
cat /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/artifacts/run_logs/completion_ignore_rule_full_001/run.exitcode
```

補足:

- train stage は blocking。`Stage2` の 2 種類の preprocess は失敗時に chain を止め、`Stage1A train`, `Stage1B train`, `Stage2 train` は成功するか retry 上限に達するまで rerun する
- curve eval と curve sample inference は non-blocking。失敗しても後続 train は止めず、log と exit code に残す
- `watch` には `uv run python -m minipamayo_qwen35.ops.ignore_rule_run watch --attempt-name ...` を使う
- `Stage1A` 成功前は `partial progress` にしない
- `Stage1B` retry は transient failure を想定した安全策であり、failure signature は結果 markdown に残す

## 実行順

### 手順0. Stage2 train 用 reasoning JSONL を生成する

理由:

- `Stage2 train` だけ dataset 契約が違う
- 前処理は短く終わるので、長い train を始める前に fail-fast できる

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
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

### 手順0b. Stage2 curve eval 用 reasoning JSONL を生成する

理由:

- `Stage2 eval` と `Stage2 sample inference` は full run ではなく curve-only holdout を見る
- `Stage1A/B` と `Stage2` の eval slice をそろえる

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.preprocess \
  --config-json configs/stage2/reasoning_sft/data/ignore_rule_data_k64_dt01_completion_001_curve_eval.json
```

期待する出力:

- `datasets/processed/stage2/reasoning_sft/ignore_rule_data_k64_dt01_completion_001_curve_eval/perimeter_cw_holdout_v1/samples_reasoning_sft.jsonl`
  が 1 本生成されること
- line count が `569` で一致すること

skip 条件:

- なし
- 既存 `samples_reasoning_sft.jsonl` があっても fresh に再生成する

### 手順1. Stage1A を full 3-run で学習する

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
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

### 手順2. Stage1A curve eval を回す

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.vlm_ce.eval \
  --config-json configs/stage1/vlm_ce/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json
```

位置づけ:

- `curve_holdout` に対する decision-bearing eval
- 学習不十分でも直進で見かけ上ごまかせない slice を見る
- runner では non-blocking だが、正式な記録はこの curve eval を使う

### 手順3. Stage1B を同じ 3 run で学習する

前提:

- `Stage1A best.pt` が存在すること
- `checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_k64_dt01_completion_001_12gb_safe`
  が fresh であること。既存 dir があれば退避してから始めること

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
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

### 手順4. Stage1B curve eval を回す

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage1.expert_cfm.eval \
  --config-json configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json
```

位置づけ:

- `curve_holdout` に対する decision-bearing eval
- `include_pid_override=true` の指標も含めて lateral の改善を見る
- runner では non-blocking

### 手順5. Stage2 を学習する

前提:

- 手順0の train 用 `samples_reasoning_sft.jsonl` が 3 run 分そろっていること
- 手順0bの curve eval 用 `samples_reasoning_sft.jsonl` が生成済みであること
- `Stage1A best.pt` が存在すること
- `Stage1B safe best.pt` が存在すること

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
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

### 手順6. Stage2 curve eval を回す

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.eval \
  --config-json configs/stage2/reasoning_sft/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json
```

位置づけ:

- `curve_holdout` 由来の `samples_reasoning_sft.jsonl` に対する eval
- `Stage2` full 3-run external eval の代替ではなく、curve slice の意思決定確認に使う
- runner では non-blocking

### 手順7. Stage2 curve sample inference を 1 本だけ回す

前提:

- `Stage2 best.pt` があること
- `Stage1B safe best.pt` があること

コマンド:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.stage2.reasoning_sft.inference \
  --config-json configs/stage2/reasoning_sft/inference/canonical/ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json
```

位置づけ:

- `reasoning_text` が出るか
- curve holdout sample で handoff 後の trajectory が最低限生成できるか
- runner では non-blocking

## 成功判定

full success:

- `Stage1A train` 完了
- `Stage1B train` 完了
- `Stage2 train preprocess` 完了
- `Stage2 curve eval preprocess` 完了
- `Stage2 train` 完了
- `Stage1A` / `Stage1B` / `Stage2` の curve eval 出力完了
- `Stage2 curve sample inference` 出力完了

partial progress:

- 外部都合で途中停止する可能性はある
- `Stage1A train` と `Stage1B train` が成功済みであることを前提に、その先が全部終わらなくてもよい
- 特に次の状態は `partial progress` とみなす
  - `Stage1B` 完了、`Stage2` 未着手
  - `Stage2` の 2 種類の preprocess 完了、`Stage2 train` 進行中
  - `Stage2 train` が途中まで進み、最後に確認できた epoch / step が分かる

failed:

- `Stage2 train preprocess` または `Stage2 curve eval preprocess` が契約違反で止まり、そのまま解消できない
- `Stage1A` が学習開始できない
- 実行を止めた時点で `Stage1A` 成功 run が 1 本もない
- 実行を止めた時点で `Stage1B` 成功 run が 1 本もない
- 生成された artifact がなく、どこまで進んだかも追えない

## 失敗時の扱い

`Stage2 train preprocess` または `Stage2 curve eval preprocess` が落ちた場合:

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

- [stage1-stage2-ignore-rule-completion-run-results.md](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1-stage2-ignore-rule-completion-run-results.md)

最低限残す項目:

- 開始日時 / 終了日時
- 実際に使った config
- `Stage2 train preprocess` と `Stage2 curve eval preprocess` の出力有無と record 数
- `Stage1A` / `Stage1B` / `Stage2` の `summary.json` 主要数値
- `Stage1B` の retry 回数、各 retry の失敗理由、退避した save dir
- curve eval の主要数値
- curve sample inference の有無と代表 sample の結果
- 途中停止したなら、その時点の最後の状態
