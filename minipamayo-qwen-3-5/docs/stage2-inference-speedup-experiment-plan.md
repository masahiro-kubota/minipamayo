# Stage2 Inference Speedup Experiment Plan

## Goal

`Stage2` の online single-sample latency を下げるために、
まずは **精度を無視して speed-only で効く knob を切り分ける**。

現時点の基準値は scratch `1 epoch` の `Stage1A -> Stage1B -> Stage2` で、
RTX6000 上の single-sample warm 推論が次のとおり。

- 16 reasoning tokens: `0.282s`
- 32 reasoning tokens: `0.485s`
- 64 reasoning tokens: `0.887s`
- 128 reasoning tokens: `1.745s`
- 256 reasoning tokens: `3.305s`

この時点で、16 token でも約 `3.5 Hz` なので、
10 Hz (`0.100s`) を目指すには **約 2.8x** の追加短縮が必要。

## Working Hypothesis

今の latency は主に次の 2 つで決まっている。

1. `VLM` の autoregressive reasoning rollout
2. fixed cost としての prompt prefill + expert diffusion

scratch `Stage2` は `<|traj_future_start|>` を早く出せていないため、
token budget にほぼ比例して遅くなる。

粗い近似:

- `推論時間 ≈ 0.08s + 0.013s × reasoning token 数`

したがって、10 Hz を狙うには

- reasoning token を極小化する
- image/history 入力 token をかなり減らす
- diffusion steps も削る

の 3 方向を同時に見る必要がある。

## Current Stage2 Input Breakdown

same sample の current `Stage2` input prefix は約 `280 tokens`。

- image tokens: `180`
- history placeholder: `48`
- history start/end: `2`
- reasoning instruction text: `18`
- system prompt text: `12`
- assistant prefill `<|cot_start|>`: `1`
- chat template overhead: 約 `19`

優先度は、

1. output reasoning tokens
2. image tokens
3. history tokens
4. text prompt tokens

とみなす。

## Experiment Axes

### A. Reasoning Tokens

目的:

- `<|traj_future_start|>` を早く出せたと仮定したときの latency 上限を測る

方法:

- `Stage2 inference` の `max_reasoning_tokens` を絞る
- official inference path を使う

候補:

- `0`
- `1`
- `2`
- `4`
- `8`
- `16`

期待:

- もっとも効く
- ただし current scratch model は handoff 不成立なので、
  実際には「短く切る」だけで quality は壊れる

再学習要否:

- 不要

### B. Flow Steps

目的:

- expert diffusion の fixed cost をどこまで削れるか見る

方法:

- `flow_steps` を減らして official inference path を回す

候補:

- `2`
- `4`
- `6`
- `10` baseline

期待:

- fixed cost 側に効く
- reasoning token より効き幅は小さい可能性が高い

再学習要否:

- 不要

### C. Image Tokens

目的:

- prefix の大部分を占める image tokens (`180`) を削ったときの latency を測る

方法:

- benchmark-only path を作る
- canonical guard を外した専用 benchmark config で `image_min_pixels` / `image_max_pixels` を変更する
- 実際に何 tokens になったかを毎回記録する

目標水準:

- current `180`
- `128` 前後
- `96` 前後
- `64` 前後

注意:

- pixel budget と actual image token count は 1:1 ではないので、
  各 setting で実測 token count を保存する
- canonical inference には入れず、speed benchmark path に限定する

再学習要否:

- 不要

### D. History Tokens

目的:

- prefix の次に大きい history tokens (`48`) を削ったときの latency を測る

方法:

- benchmark-only path を作る
- prompt 側の history placeholder 数を `48 -> 16 / 8` に変える
- `fuse_traj_tokens` に入れる replacement token 数も同じ長さへ揃える
- 具体的には、history quantizer が作る `48` tokens のうち先頭 `N` 個だけ使う speed-only ablation にする

候補:

- `48` baseline
- `16`
- `8`

注意:

- これは quality を維持する方法ではない
- ただし speed-only で「history token が何 ms 分効いているか」を切り出すには十分
- placeholder 数と replacement token 数を一致させないと code path が壊れる

再学習要否:

- speed-only benchmark には不要
- quality を見たい本番案には必要

## Experiment Order

### Phase 0. Baseline Reproduction

まず current baseline を固定する。

- same sample
- same checkpoints
- official `Stage2` path
- 4070 Ti / RTX6000

保存項目:

- mean
- median
- p95
- actual stop position
- `<|traj_future_start|>` emitted or not

### Phase 1. No-Retrain Knobs

まず code patch なしで触れるものから回す。

1. reasoning tokens
2. flow steps

ここで、

- `reasoning 0-4`
- `flow_steps 2-6`

でも `0.1s` に遠いなら、
現行 contract のまま 10 Hz は難しいと判断しやすい。

### Phase 2. Benchmark-Only Prefix Compression

次に、benchmark path 限定で prefix を削る。

1. image tokens
2. history tokens

この phase では quality は見ない。
見るのは、

- prefix token count
- latency reduction
- reduction per removed token

だけでよい。

### Phase 3. Combined Best-Case

単独で効いた設定を組み合わせて、
best-case latency を測る。

候補の例:

- reasoning `0/1/2/4`
- flow steps `2/4`
- image tokens `64/96`
- history tokens `8/16`

ここで `0.1s` に届かなければ、
現行 Stage2 contract のまま 10 Hz は非現実と判断する。

## Metrics To Record

各 run で必ず残す。

- GPU
- checkpoint ids
- sample id
- total latency mean
- total latency median
- total latency p95
- actual input token count
- image token count
- history token count
- reasoning token budget
- actual generated reasoning length
- `<|traj_future_start|>` emitted or not
- flow steps

可能なら追加で残す。

- VLM rollout time
- diffusion sample time

## Decision Rules

### If 10 Hz is still impossible

次のどちらかへ進む。

1. online では text reasoning をやめて即 handoff に寄せる
2. VLM は低周波 planner にして、10 Hz は別 controller に任せる

### If image/history compression is dominant

次の本命は、

- compact history representation
- reduced vision token budget

を前提にした再学習になる。

### If reasoning tokens dominate overwhelmingly

次の本命は、

- `<|traj_future_start|>` を早く出す訓練
- reasoning を短くする supervision

になる。

## Deliverables

この計画の output は次の 2 つ。

1. speed-only benchmark table
2. 10 Hz 到達可否の判定

判定は次の 3 値で十分。

- `possible with current contract`
- `possible only with aggressive benchmark-only compression`
- `not realistic without changing the online architecture`
