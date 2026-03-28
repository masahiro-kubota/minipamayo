# Alpamayo 推論コードとの未整合メモ

## 前提

このメモは、`/home/masa/minipamayo/related_repos/alpamayo` と比較したときに、
`minipamayo-qwen-3-5` の `stage1` / `stage2` で **まだ整合していない部分だけ** を残したものである。

次の 2 点は意図的な差分として許容する。

- 過去画像を入れていない
- バックボーンとして `Qwen3.5-0.8B` を使っている

整合済みの項目はこのメモから除外している。

## 未整合の一覧

### 1. `action` の教師信号と rollout 評価が旧 dynamics 契約のまま

`stage1` / `stage2` の trajectory tokenization と `action_space` rollout は、
いまは Alpamayo 寄せの `UnicycleAccelCurvatureActionSpace` を使う。

一方で、学習用 JSONL に保存される `action` は extractor 側で依然として
旧 `inverse_dynamics_np` から作られている。

その結果、現在の canonical path は次の 2 系統を同時に持っている。

- Stage 1A discrete token supervision:
  - `ego_history_xyz/rot + ego_future_xyz/rot` から
    Alpamayo 寄せ `action_space.traj_to_action(...)` を通して token 化
- Stage 1B expert CFM / Stage 1A eval action MAE:
  - JSONL に保存された旧 `action` を教師として使用
- Stage 1A eval trajectory ADE/FDE:
  - 旧 `forward_dynamics_batch` を使用

影響:

- Stage 1A token CE と Stage 1B expert CFM が、完全に同じ `(a, kappa)` 契約を見ていない
- Stage 1A eval の `action_mae_*`, `ADE/FDE` も、tokenization / Stage 1B rollout と別契約になる
- そのため、現在の `stage1 -> stage2` は Alpamayo 比較以前に、
  repo 内部で action contract が二重化している

### 2. `stage2` の free-running handoff 成立性

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

影響:

- 学習コードと handoff runner の配線確認は済んでいる
- ただし Alpamayo 推論コードと同じ「reasoning rollout のあと expert に handoff」が、
  いまの smoke 学習済み重みで安定成立するところまではまだ確認できていない

### 3. 依存ライブラリ stack

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

いま残っているのは主にこの 3 つである。

- `stage1` 内部の action contract 二重化
- `stage2` free-running handoff の成立性
- 依存ライブラリ stack
