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

影響:

- 学習コードと handoff runner の配線確認は済んでいる
- ただし Alpamayo 推論コードと同じ「reasoning rollout のあと expert に handoff」が、
  いまの smoke 学習済み重みで安定成立するところまではまだ確認できていない

### 2. `expert_cfm` 本体の exact 同型性

`expert_cfm` は API と大枠構造を Alpamayo 側に寄せたが、
**公開実装と exact 同型** とまではまだ言えない。

比較対象:

- Alpamayo:
  - `models.action_in_proj`
  - `models.alpamayo_r1`
  - expert transformer config
- こちら:
  - `stage1/expert_cfm/model.py`

残っている差分:

- expert text config / hidden config の作り方
- action in projection の細部ハイパーパラメータ
- prompt cache の slicing / attention mask の扱い
- `hydra` config からそのまま instantiate する構成ではない

影響:

- 役割は近いが、expert そのものの capacity と数値特性が一致している保証はない
- trajectory 品質差が出たときに、data/recipe 以外に expert 本体差も候補に残る

### 3. `action_space` の数値契約

`action_space` の API surface 自体はかなり揃ったが、
**数値の中身** はまだ Alpamayo の `UnicycleAccelCurvatureActionSpace` と一致していない。

比較対象:

- Alpamayo:
  - `action_space/unicycle_accel_curvature.py`
  - `action_space/utils.py`
  - `geometry/rotation.py`
- こちら:
  - `stage1/expert_cfm/action_space.py`

残っている差分:

- smoothing / ridge / lambda を使った trajectory-to-action 推定ではない
- `theta_smooth`, `dxy_theta_to_v`, `solve_xs_eq_y` などの数値ルーチンがない
- rotation / geometry helper 群を共有していない

影響:

- 同じ future trajectory から得る `(a, kappa)` が Alpamayo と完全一致しない
- 同じ predicted action を rollout しても、ADE / FDE に action-space 由来の差が混じる

### 4. 依存ライブラリ stack

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

- `expert_cfm` の exact 同型性を高める
- `action_space` の数値契約を Alpamayo 側に寄せる

### 優先度 C

- `transformers` と build backend を含む依存 stack をさらに揃える

## ひとことで言うと

過去画像なしとバックボーン差分を除けば、
`stage1` / `stage2` の **配線** はかなり Alpamayo に寄っている。

いま残っているのは主にこの 4 つである。

- `stage2` free-running handoff の成立性
- expert / action-space の exact 数値契約
- 依存ライブラリ stack
