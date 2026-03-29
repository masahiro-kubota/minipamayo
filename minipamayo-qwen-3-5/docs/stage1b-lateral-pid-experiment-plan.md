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

特に [action_expert.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/models/action_expert.py) は
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

この分布から、初期候補としては

- `max |kappa_gt| >= 0.08`
  または
- `|yaw change| >= 0.5 rad`

あたりが自然だが、最終値は histogram と block coverage を見て固定する。

block 展開の初期値:

- anchor 条件:
  - `max |kappa_gt| >= 0.08`
- expansion:
  - `pre = 1.0s`
  - `post = 2.0s`

最初は `kappa` anchor を主に使い、`yaw change` は sanity check と bucket 分けに使う。

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

この simple PID で curve tracking が成立するかを見る。

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

## 実施順

1. `ignore_rule_data` raw から canonical Stage1 dataset を再抽出する
2. `Stage1B` inference/eval に PID longitudinal override を追加する
3. raw MCAP 由来 GT から `max |kappa_gt|` と `|yaw change|` の分布を見て、`curve-only` threshold を固定する
4. `perimeter_cw` の train / curve-only holdout split を作る
5. 既存の canonical `probe16` checkpoint を使って、`PID override` 実装と評価指標が正しく動くか確認する
6. `perimeter_cw` の `128 sample` で `Stage1A` を学習する
7. その `Stage1A` checkpoint から `Stage1B` を学習する
8. `128 sample` で canonical baseline と PID 置換版を比較評価する
9. まだ判断できなければ、同じ `perimeter_cw` で `512 sample` に増やして `Stage1A -> Stage1B -> 評価` を回す
10. さらに必要なら、同じ `perimeter_cw` で `2048 sample` に増やして `Stage1A -> Stage1B -> 評価` を回す
11. それでも不十分な場合だけ `perimeter_cw` full に広げる
12. さらに必要なら `weave_cw`, `weave_ccw` を順に追加する
13. それでも不十分な場合だけ full `ignore_rule_data` に広げる
14. 結果を見て `target speed` 設計を詰める
