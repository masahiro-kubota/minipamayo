## 目的

`Stage1B` の横方向能力を切り出して評価する。

今回は `kappa_only` の新しい学習系を増やさず、

- `Stage1A`: canonical (`accel + kappa`)
- `Stage1B`: canonical (`accel + kappa`)
- 評価時だけ: `predicted accel` を捨てて `longitudinal PID` で置換

の形で進める。

狙いは、

- `Stage1B` がカーブを曲がれるか
- `ignore_rule_data` のような単純化シナリオで、横方向制御だけ見ても成立するか

を確認すること。

## データ

入力データの正本は以下を使う。

- [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)

対象 run:

- `20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8`
- `20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8`

実験では、上記 raw data から通常どおり `Stage1` 用 JSONL を再抽出して使う。
今回の実施では、full extraction はすでに済ませてある。

データ使用方針は、最初から full を使うのではなく、少ないデータから段階的に増やす。

- 初期確認
  - `probe16` のような小さな subset を使う
  - 目的は `PID override` 実装確認と、評価指標が正しく出るかの確認
- 小規模学習
  - まずは単純な 1 run だけで学習する
  - ただし初手からその full run は使わない
  - 第一候補は `perimeter_cw` の小さな subset
  - subset の sample ladder は `128 -> 512 -> 2048` とする
- 中規模学習
  - 1 run でうまくいかなければ、別 run を追加する
  - `perimeter_cw -> weave_cw -> weave_ccw` の順で増やす
- 本評価
  - 上の段階で不十分な場合だけ `ignore_rule_data` の 3 run 全部を使う
  - つまり [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data) 配下の全データを比較評価に使う

すでに full extraction は済ませてあるので、以後は再抽出ではなく

- 使用する JSONL を絞る
- 必要なら `probe16` などの subset を使う

方向で進める。

## なぜ kappa_only ではなく PID 置換か

今の repo では `kappa_only` は `Stage1A` の discrete target 実験としては存在するが、
`Stage1B` は canonical `(accel, kappa)` を前提にしている。

特に [stage1b_action_expert.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/stage1b_action_expert.py) は
`2 action channels` を前提にしているため、`Stage1B` をそのまま `kappa_only` 化すると
action space, normalization, rollout, checkpoint contract をまとめて崩す。

そのため今回は、

- 学習系は canonical のまま維持する
- 下流の evaluator だけで accel を PID に差し替える

方が自然で、切り分けも明確になる。

## 仮説

仮説は次の 3 つ。

1. `ignore_rule_data` では、縦方向を PID に固定しても横方向の評価は十分に意味を持つ。
2. `Stage1B` が出す `kappa` が妥当なら、accel を捨ててもカーブ追従は成立する。
3. 逆に曲がれない場合、その失敗は `Stage1B` の横方向条件付け不足か、`Stage1A -> Stage1B` handoff の情報不足である可能性が高い。

## 実験条件

### 学習

学習は canonical のまま行う。

- `Stage1A`: canonical VLM CE
- `Stage1B`: canonical expert CFM

まず `probe16` のような小さな subset で短い確認を行い、その後も
いきなり full には行かず、単純な 1 run から順に広げる。

### 推論

比較する推論条件は 3 本。

1. canonical baseline
   - `Stage1B` の `(accel, kappa)` をそのまま rollout
2. lateral-only evaluation
   - `Stage1B` の `kappa` を使う
   - `accel` は捨てて PID で置換
3. optional oracle sanity check
   - GT もしくは expert-derived longitudinal target を使い
   - 横方向だけ `Stage1B` に任せる

最低限 1 と 2 を比較する。

## 実装方針

新しい学習 entrypoint は増やさない。

追加するのは evaluator / inference 側だけにする。

必要な実装は次のとおり。

1. `Stage1B` inference/eval に longitudinal PID 置換モードを追加する
2. `predicted action` から `kappa` だけを残し、`accel` を PID で埋める
3. PID 置換後の `(accel, kappa)` を既存 action-space rollout に流す
4. canonical baseline と同じ指標で比較できるようにする

## データカリキュラム

今回の方針では、必要以上に多いデータで最初から学習しない。

推奨順は次のとおり。

1. `probe16`
   - 実装確認だけ
   - 既存の canonical `probe16` checkpoint を使って、PID override と eval/inference が壊れていないことを確認する
2. `perimeter_cw` の小さな subset
   - sample ladder は `128 -> 512 -> 2048`
   - まず `128` で始め、必要なら `512`, さらに必要なら `2048` に増やす
   - 一番単純な curve-following 確認
3. `perimeter_cw` full
   - subset で不十分な場合だけ広げる
4. `perimeter_cw + weave_cw`
   - 右回りの variation を足す
5. `perimeter_cw + weave_cw + weave_ccw`
   - 左右両方向の一般化を見る
6. full `ignore_rule_data`
   - 上の段階で不足したときだけ使う

この順にする理由は、

- まず curve-following の成立可否だけ見たい
- 初手から全 3 run を混ぜると、失敗時の原因切り分けがしにくい
- `perimeter_cw` 単独でも 6000 超 sample あるので、初手は `128 -> 512 -> 2048` の subset ladder から入った方が速い
- うまくいった段階で止めれば、必要以上に多いデータを使わずに済む

からである。

## curve-only holdout の定義

今回の成功判定は `overall` ではなく `curve-only holdout` を主に使う。

理由は、

- 直線区間が多いと `overall ADE/FDE` が過大評価になりやすい
- 今回見たいのは `Stage1B` が本当にカーブを曲がれるかだから

である。

`curve-only` は sample 単位で切るのではなく、threshold に掛かった sample を
anchor にして前後秒へ広げた block として切る。

まず raw MCAP から再抽出した canonical GT を見て anchor threshold を決め、その後
anchor block を前後秒へ展開して holdout を作る。

判定に使う候補量は次の 2 つ。

- horizon 内の `max |kappa_gt|`
- horizon 終端の `|yaw change|`

閾値決定手順:

1. `ignore_rule_data` の再抽出済み canonical samples から上の 2 量の分布を集計する
   - 集計には [curve_thresholds.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/curve_thresholds.py) を使う
   - 例:
     `uv run python -m minipamayo_qwen35.stage1.preprocess.curve_thresholds --config-json /home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json --block-kappa-threshold 0.08 --block-yaw-threshold 0.5 --block-pre-seconds 1.0 --block-post-seconds 2.0`
2. `perimeter_cw`, `weave_cw`, `weave_ccw` ごとに histogram / percentile と block coverage を見る
3. 直線 sample を十分落としつつ、curve sample を十分残す anchor threshold を固定する
4. anchor threshold に掛かった sample を contiguous block にまとめ、前後秒へ展開する
5. その block 定義を本実験の全 run で共通に使う

現時点の目安:

- `perimeter_cw`
  - `max |kappa_gt|`: p75 `0.0829`, p90 `0.1162`
  - `|yaw change|`: p75 `0.4859 rad`, p90 `1.5046 rad`
- `weave_cw`
  - `max |kappa_gt|`: p75 `0.1052`, p90 `0.1539`
  - `|yaw change|`: p75 `1.3097 rad`, p90 `1.5471 rad`
- `weave_ccw`
  - `max |kappa_gt|`: p75 `0.1080`, p90 `0.1539`
  - `|yaw change|`: p75 `1.2217 rad`, p90 `1.5457 rad`

この分布から、現在の運用値は次のとおり。

- anchor mode:
  - `or`
- anchor 条件:
  - `max |kappa_gt| >= 0.08`
  - または `|yaw change| >= 0.5 rad`
- expansion:
  - `pre = 1.0s`
  - `post = 2.0s`

この設定は、現在の threshold JSON / plot artifact と一致する。

holdout の切り方は random split ではなく、時系列 block 単位にする。
少なくとも `curve-only holdout` は、threshold に掛かった sample そのものだけでなく
カーブ侵入前と脱出後を含む block 単位で切る。

### 評価データの抽出方法

今回の `curve-only holdout` は、次の手順で作る。

1. canonical `Stage1` samples から curve block 候補を集計する
   - コマンド:
     `uv run python -m minipamayo_qwen35.stage1.preprocess.curve_thresholds --config-json /home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json --block-kappa-threshold 0.08 --block-yaw-threshold 0.5 --block-pre-seconds 1.0 --block-post-seconds 2.0`
   - 出力先:
     [ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_thresholds/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json)

2. block 抽出結果を地図上に重ねて確認する
   - コマンド:
     `uv run python -m minipamayo_qwen35.stage1.preprocess.plot_curve_blocks --curve-json /home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_thresholds/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json`
   - overview:
     [overview.png](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_block_plots/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/overview.png)
   - run 別 plot:
     [weave_ccw](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_block_plots/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8.png)
     [weave_cw](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_block_plots/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8.png)
     [perimeter_cw](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_block_plots/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8.png)

3. 生成された `curve_blocks.blocks` を holdout 候補の正本にする
   - 各 block は `start_sample_index`, `end_sample_index`, `start_sample_id`, `end_sample_id` を持つ
   - train / holdout の分離は sample 単位ではなく、この block 単位で行う

4. 実験ではこの block 群だけを `curve-only holdout` として評価する
   - `overall` は補助指標
   - 主指標はこの block 群に対する ADE / FDE / lateral error とする

## PID の入力

PID は少なくとも以下を使う。

- current ego speed `v0`
- desired speed profile もしくは fixed target speed
- `dt`

最初の実装は単純でよい。

- target speed を固定値にする
- stop / slow-down rule は入れない

初期値は `24 km/h` にする。

理由:

- `ignore_rule_data` の raw MCAP summary では 3 run とも平均速度がほぼ `23.9 km/h`
- 抽出済み `Stage1` sample の `v0 * 3.6` を見ても、全体と curve-only block の中心はほぼ `24 km/h`

この simple PID で curve tracking が成立するかを見る。

補足:

- 最初から広い sweep はしない
- まず `24 km/h` だけで確認する
- 結果が微妙なときだけ `23 / 24 / 25 km/h` の狭い sweep を追加する
- sweep の目的は最適化というより、結論を 1 点の target speed に依存させないこと

必要なら次段で、

- command-aware target speed
- curvature-aware speed cap

を足す。

## 評価指標

主指標:

- `curve-only holdout ADE`
- `curve-only holdout FDE`
- `curve-only holdout max lateral error`
- `curve-only holdout` での定性確認
  - 曲がり切れずに膨らむか
  - 不自然な蛇行が出るか
  - 右回り / 左回りの両方で成立するか

補助指標:

- overall ADE
- overall FDE

追加であるとよいもの:

- predicted `kappa` と GT `kappa` の相関
- curve 区間だけ切った lateral error
- 最大横偏差
- `cw / ccw` bucket ごとの成績
- `gentle / medium / sharp` bucket ごとの成績

## 実験結果の記録

実験結果は、計画書とは別に次の markdown へ蓄積する。

- [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)

運用:

- `probe16` 技術確認も含め、各 rung の終了後に 1 セクション追記する
- 追記単位は
  - `Stage1A train`
  - `Stage1A gate`
  - `Stage1B train`
  - `Stage1B canonical vs PID`
  をまとめた 1 セットにする
- 少なくとも次を残す
  - 使った config
  - checkpoint path
  - `Stage1A gate` 指標
  - `Stage1B canonical` 指標
  - `Stage1B + PID` 指標
  - 次の rung へ進むかどうか

この計画書は「何をやるか」を管理し、結果の時系列記録と比較表は結果ファイル側で管理する。

## Stage1A gate

`Stage1B` に進む前に、`Stage1A` が離散 trajectory token を最低限学習できているかを
`curve-only holdout` で確認する。

見るもの:

- `teacher_forced_token_accuracy`
- `autoregressive_token_accuracy`
- `action_mae_kappa`
- `ade_m`
- `fde_m`

使う runner:

- [runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/eval/runner.py)

考え方:

- `Stage1A` の離散 token と decode 後 trajectory が `curve-only holdout` で明らかに崩れているなら、
  その条件付けから `Stage1B` だけで横方向性能を回復したとしても解釈しにくい
- そのため、`Stage1B` 実験の前に `Stage1A` を gate として確認する

運用:

- `Stage1A` eval が明らかに悪い場合は、その data size では `Stage1B` へ進まない
- その場合は `128 -> 512 -> 2048` の ladder を次段へ進めるか、split / holdout 定義を見直す

## 成功条件

この実験の主成功条件は、`curve-only holdout` で次を満たすこと。

1. canonical baseline と比べて、PID 置換版でもカーブ追従が大きく崩れない
2. 少なくとも `curve-only holdout` では、横方向は十分使えると確認できる
3. 逆に崩れる場合も、`Stage1B lateral weakness` と `longitudinal coupling loss` のどちらが原因か切り分けられる

`overall ADE/FDE` は参考値として見るが、成功判定の主基準にはしない。

## 失敗時の解釈

PID 置換で大きく悪化した場合、候補は主に 3 つ。

1. `Stage1B` の `kappa` 自体が弱い
2. `accel` を捨てたことで速度プロファイルが変わり、同じ `kappa` でも軌道が崩れた
3. `ignore_rule_data` でも longitudinal-lateral coupling が無視できない

このときは、

- fixed target speed PID
- curvature-aware target speed PID
- canonical baseline

を並べて比較する。

## 実施手順

### 手順1. canonical Stage1 dataset を用意する

目的:
- `ignore_rule_data` raw から canonical `Stage1` samples を揃える

使うもの:
- raw data: [ignore_rule_data](/home/masa/minipamayo/minipamayo-qwen-3-5/datasets/raw/ignore_rule_data)
- extraction config: [ignore_rule_data_k64_dt01.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/data/ignore_rule_data_k64_dt01.json)
- extractor: [extract.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/extract.py)

出力:
- processed samples は `datasets/processed/stage1/ignore_rule_data_k64_dt01/.../samples.jsonl`
- 各 run の `extract_summary.json`

補足:
- 今回の実施では full extraction はすでに完了している
- 以後の実験では、この再抽出済み `samples.jsonl` を subset / holdout の正本として使う

### 手順2. Stage1B の PID override を実装する

目的:
- `Stage1B` の予測 `(accel, kappa)` から `kappa` だけを使い、`accel` を longitudinal PID で置換できるようにする
- その際の初期 `target speed` は `24 km/h` 固定にする

触る場所:
- inference runner: [inference.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/expert_cfm/inference.py)
- eval runner: [eval.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/expert_cfm/eval.py)
- action rollout 側:
  [record_adapter.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/contract/record_adapter.py)
  と
  [stage1b_action_expert.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/stage1b_action_expert.py)
  の契約に合わせる

到達条件:
- canonical baseline
- PID 置換版
を同じ runner から比較できる
- 初期条件として `24 km/h` 固定で rollout / 指標 / 可視化が通る

### 手順3. curve-only holdout の threshold と block 定義を固定する

目的:
- `curve-only holdout` の定義を sample 単位ではなく `anchor + 前後秒` の block として固定する

使うもの:
- 集計 script: [curve_thresholds.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/curve_thresholds.py)
- plot script: [plot_curve_blocks.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/plot_curve_blocks.py)

固定する初期値:
- anchor: `max |kappa_gt| >= 0.08`
- `pre = 1.0s`
- `post = 2.0s`

正本出力:
- threshold JSON:
  [ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_thresholds/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json)
- overview plot:
  [overview.png](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_block_plots/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2/overview.png)

到達条件:
- 地図上で見て、侵入前から脱出後までを含む curve block が概ね妥当に抽出されている

### 手順4. perimeter_cw の train / curve-only holdout split を作る

目的:
- 最初の本命 run を `perimeter_cw` に固定して、train と `curve-only holdout` を block 単位で分ける

入力:
- threshold/block 正本:
  [ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_thresholds/ignore_rule_data_k64_dt01__mode-or__kappa-0p08__yaw-0p5__pre-1__post-2.json)
- `perimeter_cw` samples:
  `/media/masa/ssd_data/minipamayo_data/minipamayo-qwen-3-5/datasets/processed/stage1/ignore_rule_data_k64_dt01/20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8/samples.jsonl`

方針:
- holdout は `curve_blocks.blocks` の block 単位で取る
- train 側は、それ以外の sample から作る
- random split は使わない

使う script:
- [build_curve_split.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/preprocess/build_curve_split.py)

現在の split 設定:
- `holdout_stride = 4`
- `holdout_offset = 0`
- `subset_sizes = 128,512,2048`
- 出力先:
  [perimeter_cw_holdout_v1](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1)
- manifest:
  [split_manifest.json](/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1/preprocess/curve_splits/perimeter_cw_holdout_v1/split_manifest.json)

この step の出力:
- `curve-only holdout` block manifest
- `perimeter_cw` train subset manifest
  - `128 sample`
  - `512 sample`
  - `2048 sample`

以後の rung は、この manifest 群を正本として使う

### 手順5. 既存 probe16 checkpoint で PID override の技術確認をする

目的:
- 学習を回し直す前に、PID override と評価指標が壊れていないことを確認する
- ここでは新しい学習はしない

使うもの:
- 既存 `Stage1A` checkpoint:
  [best.pt](/media/masa/ssd_data/minipamayo_data/minipamayo-qwen-3-5/checkpoints/stage1/vlm_ce/canonical/ignore_rule_data_probe16_12gb/best.pt)
- 既存 `Stage1B` checkpoint:
  [best.pt](/media/masa/ssd_data/minipamayo_data/minipamayo-qwen-3-5/checkpoints/stage1/expert_cfm/canonical/ignore_rule_data_probe16_12gb/best.pt)
- Stage1B inference config:
  [ignore_rule_data_probe16_pid_sample.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/inference/canonical/ignore_rule_data_probe16_pid_sample.json)
- Stage1B eval config:
  [ignore_rule_data_probe16_pid_eval.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/eval/canonical/ignore_rule_data_probe16_pid_eval.json)

到達条件:
- canonical baseline と PID 置換版の両方が落ちずに走る
- 指標と可視化が出る

注意:
- この step は interface / metric の技術確認だけに使う
- `probe16` の結果は、最終的な性能判断には使わない
- 実行後は結果を
  [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)
  の `probe16 smoke` 欄に追記する

### 手順6. perimeter_cw 128 sample rung を回す

目的:
- 最小の本命 dataset で、`Stage1A -> Stage1A gate -> Stage1B -> PID 比較評価` の 1 サイクルを回す

対象データ:
- `perimeter_cw` train split のうち `128 sample`
- 手順4で作った `128 sample` train subset manifest を使う

ベースにするもの:
- `Stage1A` train config の雛形:
  [ignore_rule_data_probe16_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_probe16_12gb.json)
  または
  [ignore_rule_data_k64_dt01_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_12gb.json)
- `Stage1B` train config の雛形:
  [ignore_rule_data_probe16_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_probe16_12gb.json)
  または
  [ignore_rule_data_k64_dt01_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_12gb.json)

使う config:
- `configs/stage1/vlm_ce/train/canonical/perimeter_cw_curve128_12gb.json`
- `configs/stage1/vlm_ce/eval/canonical/perimeter_cw_curve128_eval.json`
- `configs/stage1/expert_cfm/train/canonical/perimeter_cw_curve128_12gb.json`
- `configs/stage1/expert_cfm/eval/canonical/perimeter_cw_curve128_eval.json`
- `configs/stage1/expert_cfm/inference/canonical/perimeter_cw_curve128_sample.json`

この rung でやること:

1. `Stage1A` を学習する
2. [runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage1/vlm_ce/eval/runner.py) で
   `Stage1A` を `curve-only holdout` で評価して gate 判定する
3. gate を通ったら、その `Stage1A` checkpoint から `Stage1B` を学習する
4. canonical baseline と PID 置換版を比較評価する

重要:
- 各 rung は独立に回す
- 前の rung の `Stage1A` / `Stage1B` checkpoint は流用しない
- data size を増やしたら、その data size で `Stage1A` から作り直す

この rung で確認する主指標:
- `Stage1A`
  - `teacher_forced_token_accuracy`
  - `autoregressive_token_accuracy`
  - `action_mae_kappa`
  - `ade_m`
  - `fde_m`
- `Stage1B + PID`
  - `curve-only holdout ADE`
  - `curve-only holdout FDE`
  - `curve-only holdout max lateral error`
  - 定性 plot

次へ進む条件:
- `Stage1A` gate を通る
- それでも `Stage1B + PID` の結論がまだ曖昧なら、次の rung へ進む
- 実行後は結果を
  [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)
  の `perimeter_cw_curve128` 欄に追記する

### 手順7. perimeter_cw 512 sample rung を回す

目的:
- `128 sample` では判断できない場合に、同じ `perimeter_cw` で data size を増やして
  同じ実験サイクルを繰り返す

対象データ:
- `perimeter_cw` train split のうち `512 sample`
- 手順4で作った `512 sample` train subset manifest を使う

使う config:
- `configs/stage1/vlm_ce/train/canonical/perimeter_cw_curve512_12gb.json`
- `configs/stage1/vlm_ce/eval/canonical/perimeter_cw_curve512_eval.json`
- `configs/stage1/expert_cfm/train/canonical/perimeter_cw_curve512_12gb.json`
- `configs/stage1/expert_cfm/eval/canonical/perimeter_cw_curve512_eval.json`
- `configs/stage1/expert_cfm/inference/canonical/perimeter_cw_curve512_sample.json`

この rung でやること:
- 手順6と同じく
  - `Stage1A`
  - `Stage1A gate`
  - `Stage1B`
  - `PID 比較評価`
  を `512 sample` 条件で回す
- `128 sample` rung の checkpoint は使い回さない
- 実行後は結果を
  [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)
  の `perimeter_cw_curve512` 欄に追記する

### 手順8. perimeter_cw 2048 sample rung を回す

目的:
- `512 sample` でも不十分なら、同じ run のまま data size を増やして同じ実験サイクルを回す

対象データ:
- `perimeter_cw` train split のうち `2048 sample`
- 手順4で作った `2048 sample` train subset manifest を使う

使う config:
- `configs/stage1/vlm_ce/train/canonical/perimeter_cw_curve2048_12gb.json`
- `configs/stage1/vlm_ce/eval/canonical/perimeter_cw_curve2048_eval.json`
- `configs/stage1/expert_cfm/train/canonical/perimeter_cw_curve2048_12gb.json`
- `configs/stage1/expert_cfm/eval/canonical/perimeter_cw_curve2048_eval.json`
- `configs/stage1/expert_cfm/inference/canonical/perimeter_cw_curve2048_sample.json`

この rung でやること:
- 手順6と同じく
  - `Stage1A`
  - `Stage1A gate`
  - `Stage1B`
  - `PID 比較評価`
  を `2048 sample` 条件で回す
- `512 sample` rung の checkpoint は使い回さない
- 実行後は結果を
  [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)
  の `perimeter_cw_curve2048` 欄に追記する

### 手順9. perimeter_cw full rung を回す

目的:
- subset ladder でも判断できない場合に、同じ run の full data で見る

ベースにするもの:
- `Stage1A` train config の雛形:
  [ignore_rule_data_k64_dt01_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/vlm_ce/train/canonical/ignore_rule_data_k64_dt01_12gb.json)
- `Stage1B` train config の雛形:
  [ignore_rule_data_k64_dt01_12gb.json](/home/masa/minipamayo/minipamayo-qwen-3-5/configs/stage1/expert_cfm/train/canonical/ignore_rule_data_k64_dt01_12gb.json)

使う config:
- `configs/stage1/vlm_ce/train/canonical/perimeter_cw_full_12gb.json`
- `configs/stage1/vlm_ce/eval/canonical/perimeter_cw_full_eval.json`
- `configs/stage1/expert_cfm/train/canonical/perimeter_cw_full_12gb.json`
- `configs/stage1/expert_cfm/eval/canonical/perimeter_cw_full_eval.json`
- `configs/stage1/expert_cfm/inference/canonical/perimeter_cw_full_sample.json`

注意:
- ここでも holdout は step4 の curve block 定義に従う

この rung でやること:
- 手順6と同じ実験サイクルを、`perimeter_cw` full で回す
- subset rung の checkpoint は使い回さない
- 実行後は結果を
  [stage1b-lateral-pid-experiment-results.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage1b-lateral-pid-experiment-results.md)
  の `perimeter_cw_full` 欄に追記する

### 手順10. weave_cw, weave_ccw を順に追加する

目的:
- `perimeter_cw` だけでは見えない一般化を確認する

順番:
- `perimeter_cw + weave_cw`
- `perimeter_cw + weave_cw + weave_ccw`

評価:
- `cw / ccw` bucket を分けて見る
- `curve-only holdout` を主指標のまま維持する

この step でも:
- `Stage1A`
- `Stage1A gate`
- `Stage1B`
- `PID 比較評価`
の順番は変えない
- run を増やした各条件ごとに、結果ファイルへ独立したセクションを追加する

### 手順11. 必要なら full ignore_rule_data に広げる

目的:
- 上の段階でも判断できない場合だけ full data へ行く

入力:
- 3 run 全部
  - `perimeter_cw`
  - `weave_cw`
  - `weave_ccw`

注意:
- ここは最後の段階とし、初手では使わない

この step でも:
- `Stage1A`
- `Stage1A gate`
- `Stage1B`
- `PID 比較評価`
を同じ順で回す
- full data での結果も、結果ファイルの `full_ignore_rule_data` 欄へ追記する

### 手順12. target speed を必要時だけ再 tuning する

目的:
- `24 km/h` 固定では解釈しにくい場合にだけ、PID 置換版の longitudinal 設計を改善する

比較対象:
- fixed target speed PID
- curvature-aware speed cap
- 必要なら command-aware target speed

初手:

- まず `24 km/h` 固定で確認する
- 必要な場合だけ `23 / 24 / 25 km/h` を比較する

判断:
- `Stage1B lateral weakness`
- `longitudinal coupling loss`
のどちらが主要因かを見ながら詰める
