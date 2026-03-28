# Alpamayo 推論コードとの整合性監査

## 目的

`minipamayo-qwen-3-5` の `stage1` と `stage2` が、`/home/masa/minipamayo/related_repos/alpamayo` の公開推論コードとどこまで整合しているかを確認する。

今回の監査では、次の 2 点は意図的な差分として許容する。

- 過去画像を入れていない
- バックボーンとして `Qwen3.5-0.8B` を使っている

それ以外について、学習コード、推論コード、型、special token、ライブラリ構成の整合性を確認した。

## 結論

`stage2` まで来れば Alpamayo の推論コードと直接比較できる、という状態にかなり近づいている。ただし、まだ完全一致ではない。

すでに揃った部分は大きい。

- prompt special token に `<|cot_start|>`, `<|traj_future_start|>` などを追加した
- history tensor shape を `single_traj_group` 前提に正規化した
- history は `input_ids` の placeholder 置換で fuse する形に寄せた
- `stage2 -> expert_cfm` の end-to-end handoff runner を追加した
- `expert_cfm` に `action_space` / `diffusion` ラッパーを追加した
- 依存ライブラリも `hydra-core`, `hydra-colorlog`, `einops`, `av` を追加した

まだ残る本質差分は次のとおり。

- Alpamayo と同一の `helper.create_message(...)` 契約にはまだ達していない
- history token の量子化仕様と trajectory token `<i*>` 契約はまだ別
- `expert_cfm` は周辺 API を寄せたが、Alpamayo の expert 本体と完全同型ではない
- `torch` / `transformers` / `flash-attn` / `physical_ai_av` などの依存 stack はまだ一致していない

したがって、現状は「Stage2 までの handoff 契約はかなり寄ったが、公開 Alpamayo 実装と同一 stack ではまだない」と判断するのが正確。

## 監査対象

### Alpamayo 側

- `related_repos/alpamayo/src/alpamayo_r1/helper.py`
- `related_repos/alpamayo/src/alpamayo_r1/test_inference.py`
- `related_repos/alpamayo/src/alpamayo_r1/models/base_model.py`
- `related_repos/alpamayo/src/alpamayo_r1/models/alpamayo_r1.py`
- `related_repos/alpamayo/src/alpamayo_r1/action_space/`
- `related_repos/alpamayo/pyproject.toml`

### こちら側

- `src/minipamayo_qwen35/stage1/prompt.py`
- `src/minipamayo_qwen35/stage1/tokenization/history.py`
- `src/minipamayo_qwen35/stage1/data/dataset.py`
- `src/minipamayo_qwen35/stage1/vlm_ce/train/runner.py`
- `src/minipamayo_qwen35/stage1/vlm_ce/eval/runner.py`
- `src/minipamayo_qwen35/stage1/vlm_ce/inference/helper.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/action_space.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/diffusion.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/model.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/train/runner.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/eval/runner.py`
- `src/minipamayo_qwen35/stage1/expert_cfm/inference/runner.py`
- `src/minipamayo_qwen35/stage2/reasoning_sft/dataset.py`
- `src/minipamayo_qwen35/stage2/reasoning_sft/train/runner.py`
- `src/minipamayo_qwen35/stage2/reasoning_sft/inference/runner.py`
- `pyproject.toml`

## すでに揃っている部分

### 1. 画像 token budget

画像 budget は Alpamayo と同じ `min_pixels=163840`, `max_pixels=196608` に固定している。

### 2. history tensor shape

canonical `stage1` / `stage2` dataset は Alpamayo と同じ single-traj-group 形に正規化して扱う。

- `ego_history_xyz`: `[B, 1, T, 3]`
- `ego_history_rot`: `[B, 1, T, 3, 3]`

### 3. history の注入方式

history は `input_ids` の placeholder token を discrete history token ID に置換する形へ修正済み。

以前の `inputs_embeds` 上書きからは脱しており、この点は Alpamayo 側へ寄った。

### 4. prompt special token

canonical prompt 側には次を持たせている。

- `<|cot_start|>`
- `<|cot_end|>`
- `<|traj_future_start|>`
- `<|traj_future_end|>`

また Stage2 では reasoning 用 user text と `assistant` prefill `<|cot_start|>` を入れる。

### 5. Stage2 から expert への handoff

canonical runner を追加済み。

- `src/minipamayo_qwen35/stage2/reasoning_sft/inference/runner.py`

この runner は

- Stage2 VLM で reasoning を rollout
- `<|traj_future_start|>` まで進める
- その cache を `stage1.expert_cfm` に handoff
- continuous action / trajectory を生成

という形で、Alpamayo に近い end-to-end 推論経路を持つ。

### 6. expert 側の周辺 API

`expert_cfm` には Alpamayo 風の周辺 API を追加した。

- `stage1/expert_cfm/action_space.py`
- `stage1/expert_cfm/diffusion.py`

`FlowMatchingDiffusion.sample(...)` と `UnicycleAccelCurvatureActionSpace.action_to_traj(...)` を通す構造になっている。

## まだ整合していない部分

### 1. helper / message builder はまだ別物

Alpamayo の `helper.create_message(...)` は

- system
- user
  - 複数画像
  - history placeholder
  - CoT 指示
- assistant
  - `<|cot_start|>`

の契約をそのまま持つ。

一方、こちらの `stage1.vlm_ce.inference.helper.create_message(...)` はまだ簡略版で、

- 画像
- 任意の `user_text`

を組み立てる helper に留まっている。

canonical `stage1.prompt` はかなり寄っているが、`helper.py` と 1 対 1 で一致しているわけではない。

### 2. history token の量子化仕様

history は `input_ids` 置換型にしたが、量子化仕様そのものは独自実装である。

- Alpamayo: `tokenize_history_trajectory(...)`
- こちら: `HistoryTrajectoryQuantizer`

したがって、「history を discrete token にする」という方針は同じでも、token の意味空間はまだ一致していない。

### 3. trajectory token `<i*>` 契約

Alpamayo は future discrete trajectory token として `<i*>` 系を持つ。

こちらは `Stage1TokenRegistry` による独自 action token 語彙であり、Alpamayo の trajectory token 契約とは一致していない。

これは `stage1A` / `stage2` の教師信号比較で残る差分である。

### 4. Stage2 の supervised target

Stage2 の handoff runner は追加したが、学習 target はまだ Alpamayo の推論経路と完全一致ではない。

現在は `reasoning_text + <|traj_future_start|> + action token supervision` 系で学習しているため、

- 推論経路は寄った
- 学習教師はまだ完全一致ではない

という状態である。

### 5. expert 本体は完全同型ではない

`expert_cfm` の外側の契約は寄せたが、expert 本体まで Alpamayo と同型ではない。

まだ違う点は次のとおり。

- Alpamayo の `AutoModel.from_config` / expert transformer stack と完全同型ではない
- `hydra` instantiate 契約を使っていない
- local normalization / metadata 契約が残っている

つまり、「同じ役割を持つ近縁実装」ではあるが、「そのまま同型」とは言えない。

### 6. action_space は full API ではない

現在の `stage1/expert_cfm/action_space.py` は

- history から初期速度を推定
- unicycle rollout で trajectory を返す

という最小 API を持つ。

ただし、まだ Alpamayo の full API との差がある。

- `traj_to_action` を含んでいない
- action 正規化統計の持ち方が違う
- `geometry` 周辺の補助実装がない

### 7. ライブラリ stack

Alpamayo 側の主要依存:

- `torch==2.8.0`
- `transformers==4.57.1`
- `flash-attn`
- `hydra-core`
- `hydra-colorlog`
- `einops`
- `physical_ai_av`
- `av`

こちらの現在値:

- `torch==2.10.0`
- `transformers` は GitHub `main`
- `flash-linear-attention==0.4.2`
- `hydra-core` あり
- `hydra-colorlog` あり
- `einops` あり
- `physical_ai_av` なし
- `av` あり

依存 stack の差はまだ大きい。特に `transformers` pin と `flash-attn` / `physical_ai_av` は未整合である。

## 優先度順の残タスク

### 優先度 A

- `stage1.vlm_ce.inference.helper` を Alpamayo helper 契約に合わせる
- `<i*>` trajectory token と future boundary token の tokenizer 契約を整理する
- Stage2 target を Alpamayo の推論経路にもっと揃える

### 優先度 B

- `expert_cfm` の expert 本体を Alpamayo の expert stack にさらに寄せる
- `action_space` を full API に近づける
- `diffusion` 周辺の API を Alpamayo 命名に寄せる

### 優先度 C

- `torch`, `transformers`, `flash-attn`, `physical_ai_av` を含む依存 stack を見直す
- processor / tokenizer 初期化方法を Alpamayo helper とさらに近づける

## 監査上の判断

今のコードは、

- 「Alpamayo の推論契約にかなり寄せた近縁実装」

とは言える。

ただし、

- 「過去画像なし・バックボーン差分を除けば、Alpamayo 推論コードと完全に同じ」

とはまだ言えない。

比較可能性をさらに高めるために必要なのは、主に次の 4 つである。

- helper / tokenizer / special token 契約の整理
- history token 量子化仕様の整理
- expert / diffusion / action_space stack の同型化
- 依存ライブラリ stack の整理

## 補足

本監査は「何がまだ一致していないか」を整理するためのものであり、差分のすべてが直ちに悪いという意味ではない。

ただし、

- 型を整合させたい
- Alpamayo と意図しない差分を減らしたい
- 学習コードと推論コードの整合性を強く取りたい

という目的に対しては、上の残差分は無視できない。

## 実行確認メモ

2026-03-29 時点で、smoke 実行で確認できたことは次のとおり。

- `Stage1B` smoke train は完走した
  - `best_val_cfm_loss = 1.7816`
  - `peak_reserved_gib = 2.691`
- `Stage1B` inference は通った
  - `flow_steps = 10`
  - `ADE = 3.4728m`
  - `FDE = 7.7718m`
- `Stage2` smoke train は現行 contract で回し直した
  - `target_layout = reasoning_then_traj_future_start_then_action_tokens`
  - `val_loss = 3.7437`

ただし、`Stage2 -> expert_cfm` の free-running handoff 推論は、smoke 1epoch checkpoint ではまだ成功していない。

- failure:
  - `Stage 2 reasoning rollout did not emit <|traj_future_start|> within the token budget`
  - `max_reasoning_tokens = 256`

したがって、

- code path 自体は存在する
- `Stage1B` 単体は動く
- しかし `Stage2` の free-running rollout が handoff token を安定して出すところまでは、smoke checkpoint では未確認

という状態である。
