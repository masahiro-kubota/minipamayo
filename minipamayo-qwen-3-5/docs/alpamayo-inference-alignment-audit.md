# Alpamayo 推論コードとの未整合メモ

## 前提

このメモは、`/home/masa/minipamayo/related_repos/alpamayo` と比較したときに、`minipamayo-qwen-3-5` の `stage1` / `stage2` で **まだ整合していない部分だけ** を残したものである。

次の 2 点は意図的な差分として許容する。

- 過去画像を入れていない
- バックボーンとして `Qwen3.5-0.8B` を使っている

整合済みの項目はこのメモから除外している。

## 未整合の一覧

### 1. `stage1.vlm_ce.inference.helper` の message contract

Alpamayo の `helper.create_message(...)` 契約と、こちらの `stage1.vlm_ce.inference.helper.create_message(...)` はまだ一致していない。

残っている差分:

- Alpamayo は `frames: torch.Tensor` の 4D 入力を前提にしているが、こちらは `list[Any]` を受ける
- Alpamayo helper そのものと同じ message builder ではない
- `helper.py` の `system / user / assistant-prefill` 契約をそのまま再現していない
- canonical prompt 本体は寄っているが、inference helper の API はまだ簡略版
- Alpamayo の `get_processor(tokenizer)` 契約とも一致していない
  - Alpamayo: `BASE_PROCESSOR_NAME` 固定 + 外から tokenizer を差し込む
  - こちら: checkpoint 由来の `processor_path` をそのまま読む

影響:

- `apply_chat_template(...)` に入る message object が完全一致しない
- helper レベルでの比較がしにくい

### 2. history token の量子化仕様

history は `input_ids` 置換型に寄せたが、量子化仕様そのものは独自実装である。

- Alpamayo: `tokenize_history_trajectory(...)`
- こちら: `HistoryTrajectoryQuantizer`

残っている差分:

- history token の意味空間
- bin の切り方
- history trajectory tokenization の実装契約

影響:

- 同じ history を入れても token 列の意味が一致しない

### 3. future trajectory token `<i*>` 契約

Alpamayo は future discrete trajectory token として `<i*>` 系を使う。

こちらは `Stage1TokenRegistry` による独自 action token 語彙を使っている。

残っている差分:

- Alpamayo の `<i*>` trajectory token 契約を持っていない
- future boundary token と future payload token の扱いが異なる

影響:

- `stage1A` / `stage2` の教師信号を Alpamayo と直接比較できない

### 4. `stage2` の supervised target

`stage2 -> expert_cfm` の handoff runner 自体はあるが、`stage2` の学習 target はまだ Alpamayo の推論経路と完全一致ではない。

現在の target:

- `reasoning_text + <|traj_future_start|> + action token supervision`

残っている差分:

- Stage2 の教師信号設計が Alpamayo の最終推論契約と完全には一致しない
- `stage2` が free-running で `<|traj_future_start|>` を安定して出すところまではまだ確認できていない

実行確認上の未解決:

- smoke 1 epoch の `stage2` checkpoint では
  - `Stage 2 reasoning rollout did not emit <|traj_future_start|> within the token budget`
  で handoff が失敗した

影響:

- code path はある
- ただし end-to-end handoff が安定に成立するところまでは未確認

### 5. `expert_cfm` の expert 本体

`expert_cfm` は周辺 API を寄せたが、Alpamayo の expert 本体と完全同型ではない。

残っている差分:

- Alpamayo の `AutoModel.from_config` / expert transformer stack と完全同型ではない
- `hydra` instantiate 契約を expert 本体では使っていない
- local normalization / metadata 契約が残っている

影響:

- 「同じ役割の近縁実装」ではあるが、「同じモデル構成」とはまだ言えない

### 6. `action_space` の API

現在の `stage1/expert_cfm/action_space.py` は最小限の rollout API しか持っていない。

残っている差分:

- `traj_to_action` がない
- action 正規化統計の持ち方が違う
- `geometry` 周辺の補助 API がない

影響:

- Alpamayo の `action_space` をそのまま置き換え対象として比較できない

### 7. 依存ライブラリ stack

まだ一致していない主要差分:

- `torch`
  - Alpamayo: `2.8.0`
  - こちら: `2.10.0`
- `transformers`
  - Alpamayo: `4.57.1`
  - こちら: GitHub `main`
- attention 実装
  - Alpamayo: `flash-attn`
  - こちら: `flash-linear-attention`
- dataset / AV 周辺
  - Alpamayo: `physical_ai_av` あり
  - こちら: なし

影響:

- generation / cache / attention 周辺の挙動差
- Alpamayo 公開実装と完全同条件の比較ができない

## 優先度順

### 優先度 A

- `stage1.vlm_ce.inference.helper` を Alpamayo helper 契約に合わせる
- `<i*>` trajectory token と future token 契約を整理する
- `stage2` target を Alpamayo 推論経路にさらに寄せる
- `stage2` free-running で `<|traj_future_start|>` が出るところまで確認する

### 優先度 B

- `expert_cfm` の expert 本体を Alpamayo の expert stack にさらに寄せる
- `action_space` を full API に近づける

### 優先度 C

- `torch`, `transformers`, `flash-attn`, `physical_ai_av` を含む依存 stack を見直す

## ひとことで言うと

過去画像なしとバックボーン差分を除いても、`stage2` までで「ある程度比べられる」とは言える。

ただし、まだ残っているのは主にこの 4 つ。

- helper / tokenizer / future token 契約
- `stage2` の target と free-running handoff 成立性
- expert / action_space の同型性
- 依存ライブラリ stack
