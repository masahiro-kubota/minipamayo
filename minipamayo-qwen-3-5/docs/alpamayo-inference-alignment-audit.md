# Alpamayo 推論コードとの未整合メモ

## 前提

このメモは、`/home/masa/minipamayo/related_repos/alpamayo` と比較したときに、
`minipamayo-qwen-3-5` の `stage1` / `stage2` で **まだ整合していない部分だけ** を残したものである。

次の 2 点は意図的な差分として許容する。

- 過去画像を入れていない
- バックボーンとして `Qwen3.5-0.8B` を使っている

整合済みの項目はこのメモから除外している。

## 未整合の一覧

### 1. `stage2` の free-running handoff 成立性

`stage2 -> expert_cfm` の code path 自体はある。  
また `stage2` の target も、いまは

- `reasoning_text`
- `<|cot_end|>`
- `<|traj_future_start|>`
- `eos`

までに寄せている。

それでも未解決なのは、**学習済み `stage2` checkpoint が free-running で
`<|traj_future_start|>` を安定して出せるか** である。

現状の確認:

- smoke `1 epoch` の `stage2` checkpoint では
  - `Stage 2 reasoning rollout did not emit <|traj_future_start|> within the token budget`
  となる
- greedy / sampling の両方で、
  structured reasoning 断片を繰り返して boundary に到達しない例を確認済み
- same scratch `1 epoch` 系列 (`Stage1A -> Stage1B -> Stage2`) の
  budget sweep でも、`<|traj_future_start|>` を出せず token budget 依存で推論時間が伸びる

影響:

- 学習コードと handoff runner の配線確認は済んでいる
- ただし Alpamayo 推論コードと同じ「reasoning rollout のあと expert に handoff」が、
  いまの smoke 学習済み重みで安定成立するところまではまだ確認できていない

追加計測:

- 計測条件
  - scratch `1 epoch` の `Stage1A -> Stage1B -> Stage2`
  - same single sample
  - official `stage2` inference core と同じ wrapper path
  - `Stage2` は free-running で `<|traj_future_start|>` を出せず、warning を出したまま最後で handoff
- 4070 Ti 実測
  - 16 token budget: `0.290s`
  - 32 token budget: `0.501s`
  - 64 token budget: `0.919s`
  - 128 token budget: `1.764s`
  - 256 token budget: `3.463s`
- RTX6000 実測
  - 16 token budget: `0.282s`
  - 32 token budget: `0.485s`
  - 64 token budget: `0.887s`
  - 128 token budget: `1.745s`
  - 256 token budget: `3.305s`

読み取り:

- 4070 Ti / RTX6000 のどちらでも、scratch `Stage2` は `<|traj_future_start|>` を安定して出せていない
- そのため「短い reasoning のあと expert へ早く handoff」ではなく、
  `max_reasoning_tokens` 近くまで VLM rollout を引っ張ってから最後で handoff している
- 推論時間は概ね reasoning token budget に比例し、粗い近似としては
  - 4070 Ti: `推論時間 ≈ 0.08s + 0.013s × reasoning token 数`
  - RTX6000: `推論時間 ≈ 0.08s + 0.013s × reasoning token 数`
- RTX6000 にしても大幅には縮まず、
  現状の主要ボトルネックは GPU だけでなく「早い `<|traj_future_start|>` を出せないこと」にある

補足考察:

- 今回の sweep では、4070 Ti と RTX6000 の差は主に `1 token` あたりの傾き差にしか出ていない
  - 4070 Ti: 約 `13.2ms / token`
  - RTX6000: 約 `12.6ms / token`
- つまり current scratch `Stage2` は、GPU の大きな行列スループット差よりも
  `batch=1` の逐次 reasoning rollout latency に強く支配されている
- これは「0.8B と小さいので速いはず」という期待とずれるが、
  実際には小さい `Qwen3.5-VL` を single-sample / autoregressive decode で回しているため、
  1 token ごとの固定寄り cost が積み上がる
- action expert / diffusion sample も含まれてはいるが、
  budget sweep の形から見て支配項は VLM reasoning rollout 側である
  - 少なくとも current scratch `Stage2` の latency を決めている主因は expert より VLM である
- Stage1A 側の分解計測でも、RTX6000 で
  - prompt: `0.043s`
  - generate: `1.707s`
  - decode: `0.005s`
  となっており、VLM generate 支配という見立てと整合する

### 2. 依存ライブラリ stack

まだ一致していない主要差分:

- `transformers`
  - Alpamayo: `4.57.1`
  - こちら: `5.4.0`
- build backend
  - Alpamayo: `uv_build`
  - こちら: `setuptools`

補足:

- `torch==2.8.0`, `torchvision>=0.23.0`, `flash-attn>=2.8.3`, `physical_ai_av>=0.2.0`,
  `hydra-core`, `hydra-colorlog`, `einops`, `av` は揃った
- `transformers` 差分は、現在の `Qwen3.5-0.8B` 読み込み要件に引っ張られている

影響:

- `transformers` 差分により generation / KV-cache / attention / memory usage の挙動差が残る
- build backend 差分により environment 再現性の条件が完全一致ではない
- その結果、Alpamayo 公開実装と完全同条件の runtime 比較ができない

## 優先度順

### 優先度 A

- `stage2` が free-running で `<|traj_future_start|>` を安定して出すところまで確認する

### 優先度 B

- `transformers` と build backend を含む依存 stack をさらに揃える

## ひとことで言うと

過去画像なしとバックボーン差分を除けば、
`stage1` / `stage2` の **配線** はかなり Alpamayo に寄っている。

いま残っているのは主にこの 2 つである。

- `stage2` free-running handoff の成立性
- 依存ライブラリ stack
