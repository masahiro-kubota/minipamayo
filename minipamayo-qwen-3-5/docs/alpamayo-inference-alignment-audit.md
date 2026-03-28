# Alpamayo 推論コードとの未整合メモ

## 前提

このメモは、`/home/masa/minipamayo/related_repos/alpamayo` と比較したときに、
`minipamayo-qwen-3-5` の `stage1` / `stage2` で **まだ整合していない部分だけ** を残したものである。

次の 2 点は意図的な差分として許容する。

- 過去画像を入れていない
- バックボーンとして `Qwen3.5-0.8B` を使っている

整合済みの項目はこのメモから除外している。

## 未整合の一覧

### 1. history trajectory tokenization の中身

history の shape と `input_ids` への fuse 方式は Alpamayo 側に寄せたが、
**history をどの `<i*>` 列に変換するか** の中身はまだ一致していない。

- Alpamayo:
  - `models.base_model.tokenize_history_trajectory(...)`
  - `hist_traj_tokenizer.encode(...)`
- こちら:
  - `stage1/tokenization/history.py`
  - `HistoryTrajectoryQuantizer`

残っている差分:

- history token の bin の切り方
- `hist_xyz / hist_rot -> token` の変換規約
- `tokenize_history_trajectory(...)` と同じ trajectory tokenizer を使っていない

影響:

- 同じ `ego_history_xyz / ego_history_rot` を入れても、
  VLM が見る history token 列の意味空間が Alpamayo と一致しない。

### 2. future discrete trajectory tokenization の中身

future token は `<i*>` 系に揃えたが、
**future trajectory をどの bin に量子化するか** はまだ Alpamayo の
`DiscreteTrajectoryTokenizer` と同一ではない。

- Alpamayo:
  - `action_space.traj_to_action(...)`
  - `DiscreteTrajectoryTokenizer.encode(...)`
  - `dims_min / dims_max / num_bins`
- こちら:
  - `ActionQuantizer`
  - 固定 `a_range / kappa_range`
  - Stage1A では `ActionQuantizer.encode_bin_ids(...)`

残っている差分:

- `traj_to_action` を通した tokenization ではなく、独自 quantizer を使っている
- `dims_min / dims_max` 契約が Alpamayo の tokenizer 設定と一致していない
- future trajectory token の「値の意味」が完全一致ではない

影響:

- `stage1A` の CE 教師信号を Alpamayo と 1 対 1 で比較できない。
- token loss / token accuracy の比較にも離散化仕様差が混ざる。

### 3. `stage2` の free-running handoff 成立性

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

### 4. `expert_cfm` 本体の exact 同型性

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

### 5. `action_space` の数値契約

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

### 6. 依存ライブラリ stack

まだ一致していない主要差分:

- `torch`
  - Alpamayo: `2.8.0`
  - こちら: `2.10.0+cu128`
- `transformers`
  - Alpamayo: `4.57.1`
  - こちら: `5.5.0.dev0`
- attention 実装
  - Alpamayo: `flash-attn`
  - こちら: なし
- dataset / AV 周辺
  - Alpamayo: `physical_ai_av` あり
  - こちら: なし

補足:

- `hydra`, `einops`, `av` は導入済み

影響:

- generation / KV-cache / attention / memory usage の挙動差
- Alpamayo 公開実装と完全同条件の runtime 比較ができない

## 優先度順

### 優先度 A

- history / future trajectory tokenization を Alpamayo tokenizer 契約へさらに寄せる
- `stage2` が free-running で `<|traj_future_start|>` を安定して出すところまで確認する

### 優先度 B

- `expert_cfm` の exact 同型性を高める
- `action_space` の数値契約を Alpamayo 側に寄せる

### 優先度 C

- `torch`, `transformers`, `flash-attn`, `physical_ai_av` を含む依存 stack を揃える

## ひとことで言うと

過去画像なしとバックボーン差分を除けば、
`stage1` / `stage2` の **配線** はかなり Alpamayo に寄っている。

いま残っているのは主にこの 4 つである。

- history / future tokenization の中身
- `stage2` free-running handoff の成立性
- expert / action-space の exact 数値契約
- 依存ライブラリ stack
