# Stage1 Curve Eval Interpretation

2026-04-03 時点の `Stage1A` curve eval と、その解釈メモ。

## 前提

`Stage1A` の入力契約では、`command` は record に入っているが、モデル prompt には明示的に入っていない。

- dataset には `command` がある:
  [dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/dataset.py)
- ただし prompt は固定文:
  `"Output the future trajectory as action tokens in order. Do not provide explanations."`
  [prompt.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/contract/prompt.py)

このため、T 字路や交差点進入前に「左へ行くか右へ行くか」の決定手がかりが画像だけでは足りない場面では、進入前の左右選択ミスをそのまま強く罰するのは不自然である。

一方で、曲がり始めた後は、その方向へ追従できるべきである。

## 現在ある結果

旧 perimeter-wide eval:

- artifact:
  [ignore_rule_data_k64_dt01_completion_001_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_12gb.json)
- dataset:
  `perimeter_cw` 全体 `6167 samples`
- metrics:
  - `teacher_forced_token_accuracy = 0.7694`
  - `autoregressive_token_accuracy = 0.5003`
  - `action_mae_kappa = 0.008804`
  - `ade_m = 2.5438`
  - `fde_m = 7.2628`

curve-only eval:

- artifact:
  [ignore_rule_data_k64_dt01_completion_001_curve_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.json)
- progress:
  [ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json)
- per-sample:
  [ignore_rule_data_k64_dt01_completion_001_curve_eval.per_sample.jsonl](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/eval/stage1/vlm_ce/canonical/ignore_rule_data_k64_dt01_completion_001_curve_eval.per_sample.jsonl)
- dataset:
  `perimeter_cw curve_holdout` `569 samples`
- metrics:
  - `teacher_forced_token_accuracy = 0.5685`
  - `autoregressive_token_accuracy = 0.3402`
  - `action_mae_kappa = 0.02609`
  - `ade_m = 7.7326`
  - `fde_m = 21.6674`

## 観察

curve-only に切ると大きく悪化している。少なくとも「直線区間が多いことで overall が過大評価される」という仮説は正しかった。

per-sample 集計では、軽いカーブと重いカーブで差が大きい。

- 軽いカーブ:
  - `small`: `43 samples`, `mean FDE = 1.03`
  - `mid`: `38 samples`, `mean FDE = 2.04`
- 重いカーブ:
  - `large`: `488 samples`, `mean FDE = 25.01`
  - `mean kappa_mae = 0.0293`
  - `mean autoregressive_token_accuracy = 0.310`

最終 lateral 符号一致は `354 / 569 = 62.2%` しかなく、左右方向そのものを外すケースが相当ある。

- `gt pos -> pred pos = 97`
- `gt pos -> pred neg = 104`
- `gt neg -> pred pos = 111`
- `gt neg -> pred neg = 257`

worst 50 sample は `nominal_cruise` のみで、`right` と `lanefollow` が大半だった。

- holdout 全体:
  - `lanefollow = 474`
  - `right = 64`
  - `left = 31`
- worst 50:
  - `right = 30`
  - `lanefollow = 16`
  - `left = 4`

## 解釈

今の curve-only eval は有用だが、次の 2 種類の失敗を一緒に罰している。

1. 交差点進入前の曖昧さ
   - 入力に `command` が入っていない以上、左か右かの決定根拠が弱い
   - Alpamayo 系の前提と整合的に、ここはある程度 ambiguous でも不思議ではない

2. 曲がり始めた後の追従失敗
   - これは本当に性能不足
   - とくに strong curve で大きく崩れている

したがって、現在の `ADE/FDE` は悲観的すぎる可能性がある一方で、strong curve に対して弱いこと自体は事実である。

## 今すぐ確かめるべきこと

### 1. turn onset 以降だけで再評価する

GT から `turn onset` を定義し、その時点以降だけを評価する。

候補:

- `|kappa_gt| >= threshold`
- `|yaw change| >= threshold`

見る指標:

- `post_onset_ade_m`
- `post_onset_fde_m`
- `post_onset_kappa_mae`
- `post_onset_sign_agreement`

### 2. commit latency を測る

`turn onset` から何 step 後に、予測が GT の曲がり方向へ安定して一致するかを見る。

見る指標:

- `commit_latency_steps`
- `commit_failure_rate`

### 3. Stage1B でどこまで救えるかを見る

`Stage1B curve eval` の `pid_override` 指標を確認する。

- ここで大きく改善する:
  longitudinal 側の崩れが大きい可能性
- ここでも改善しない:
  `Stage1A` の lateral token 自体が弱い可能性

## 改善策の候補

### A. curve block を重く学習させる

最優先候補。

- curve block の oversampling
- high-curvature sample の loss weight 増加
- 少なくとも `perimeter_cw` の比重を上げる

### B. 単一 run で成立確認する

混合分布で壊れているのか、そもそも曲がれないのかを分ける。

候補:

- `weave_ccw` 単独
- `perimeter_cw` 単独

まず `Stage1A -> Stage1B` までで見る。

### C. Stage1A の loss を strong curve 寄りにする

候補:

- `kappa` 側の重みを上げる
- endpoint / late-horizon を重くする

### D. 入力契約を見直す

根本策ではあるが、すぐには重い。

- `command` を Stage1 に入れる
- 少なくとも分岐意図があるタスクでは prompt 契約を見直す

## 現時点の結論

- `Stage1A` が進入前に迷うこと自体は、現在の入力契約なら不自然ではない
- ただし、strong curve で大きく壊れていること自体は実結果として確認された
- したがって次の正しい切り分けは、
  1. `post-turn-onset` 指標を出す
  2. `Stage1B curve eval` の `pid_override` と比較する
  3. そのうえで curve-weighted 学習か single-run 実験へ進む

この順で進めるのがよい。
