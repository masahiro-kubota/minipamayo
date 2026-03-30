# Stage1 / Stage2 推論時間メモ

## 目的

`Stage1A` と `Stage2` の single-sample 推論時間を、
alignment 議論とは切り離して整理するためのメモ。

ここでの主眼は

- online control を想定した `batch=1` latency
- GPU を替えたときの効き方
- `VLM rollout` と `action expert` のどちらが支配的か

である。

## 計測条件

- single sample
- warm run ベース
- `uv run` で official code path を使う
- `Stage2` は official wrapper path を使う
- `Stage2` の scratch checkpoint は `Stage1A -> Stage1B -> Stage2` を各 `1 epoch` だけ学習したもの

## Stage1A

### RTX6000 実測

- total mean: `1.755s`
- prompt mean: `0.043s`
- generate mean: `1.707s`
- decode mean: `0.005s`

### 読み取り

- `Stage1A` はほぼ `generate(...)` 支配
- prompt 前処理と decode は小さい
- したがって、`Stage1A` の速度を決めている主因は
  `Qwen3.5-VL` の autoregressive action-token rollout

## Stage2 scratch

### 前提

scratch `Stage2` は free-running で `<|traj_future_start|>` を安定して出せていない。

そのため、想定していた

- 短い reasoning
- 早い expert handoff

にはなっておらず、`max_reasoning_tokens` 近くまで
VLM rollout を引っ張ってから最後で handoff する挙動になっている。

### 4070 Ti 実測

- 16 token budget: `0.290s`
- 32 token budget: `0.501s`
- 64 token budget: `0.919s`
- 128 token budget: `1.764s`
- 256 token budget: `3.463s`

### RTX6000 実測

- 16 token budget: `0.282s`
- 32 token budget: `0.485s`
- 64 token budget: `0.887s`
- 128 token budget: `1.745s`
- 256 token budget: `3.305s`

### 粗い近似

- 4070 Ti: `推論時間 ≈ 0.08s + 0.013s × reasoning token 数`
- RTX6000: `推論時間 ≈ 0.08s + 0.013s × reasoning token 数`

## 考察

### 1. scratch `Stage2` は handoff 不成立が律速

- `<|traj_future_start|>` を早く出せれば、
  VLM rollout を早く切って expert 側へ handoff できる
- いまの scratch `Stage2` はそれができない
- そのため、推論時間はほぼ reasoning token budget に比例する

### 2. RTX6000 にしてもあまり縮まらない

- 4070 Ti と RTX6000 の差は、ほぼ `1 token` あたりの傾き差にしか出ていない
  - 4070 Ti: 約 `13.2ms / token`
  - RTX6000: 約 `12.6ms / token`
- つまり current scratch `Stage2` は、
  GPU の大きな行列スループット差よりも、
  `batch=1` の逐次 reasoning rollout latency に支配されている

### 3. 支配項は action expert ではなく VLM

- `Stage1A` の分解計測は `generate` が支配的
- `Stage2` の budget sweep も token 数に対してほぼ線形
- したがって、current scratch `Stage2` の latency を決めている主因は
  action expert / diffusion sample より VLM reasoning rollout 側と見るのが自然

## いま言えること

- `0.8B` だから自動的に高速、ではない
- `batch=1` の VLM autoregressive decode は、
  GPU を強くしても劇的には縮まらない
- current scratch `Stage2` の改善で一番効くのは、
  GPU 交換よりも
  `<|traj_future_start|>` を早く安定して出せるようにすること
