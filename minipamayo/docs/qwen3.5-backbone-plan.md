# Qwen3.5-0.8B バックボーン移行方針

## 1. 目的

`cosmos-reason-mini` を初期値として使う現行 MiniPamayo 系列とは別に、**`Qwen3.5-0.8B` を VLM バックボーンとして使う系統**をこのブランチで検証する。

狙いは 2 つ:

1. 単発の視覚理解性能を引き上げる
2. 将来的に Alpamayo 風の VLA に接続できる、より強い VLM 初期値を持つ

このブランチでは、`Qwen3.5-0.8B` を **drop-in replacement** と見なさない。  
`Qwen2.5-0.5B + DINOv2 + Adapter` 前提の既存実装をそのまま流用するのではなく、**Qwen3.5 専用の系統として段階的に組み直す**。

---

## 2. 現状認識

### 2.1 現行 MiniPamayo の前提

現行 docs / code は以下を強く前提としている。

- 外部 vision encoder: `DINOv2 ViT-B/14`
- Vision-LLM bridge: `Adapter`
- LLM: `Qwen2.5-0.5B`
- Stage 2 Expert 条件付け: `Qwen2.5` の `past_key_values`
- Stage 1 action token: `Qwen2.5` の vocab / tokenizer に追加

特に Stage 2 は **Qwen2.5 の KV-cache 形状**に強く依存している。

- `num_hidden_layers=24`
- `num_kv_heads=2`
- `head_dim=64`

### 2.2 Qwen3.5-0.8B 側の差分

`Qwen3.5-0.8B` は、単に「Qwen2.5 の少し強い版」ではない。

- **ネイティブ VLM** であり、内蔵 vision encoder を持つ
- text backbone は **Qwen3.5 hybrid architecture**（Gated DeltaNet + Gated Attention）
- `text_config.hidden_size=1024`
- `num_hidden_layers=24`
- `num_attention_heads=8`
- `num_key_value_heads=2`
- `head_dim=256`

重要なのは、**Stage 2 でそのまま Qwen2.5 用 Expert を流用できない**こと。

---

## 3. 基本方針

### 3.1 採用方針

このブランチでは、`Qwen3.5-0.8B` の **内蔵 vision encoder をそのまま使う**。

つまり、まずは以下を採用する。

- `DINOv2 + Adapter` を前提にしない
- `Qwen3.5-0.8B` を 1 つの VLM バックボーンとして扱う
- action / reasoning 学習側を Qwen3.5 に合わせて作り直す

### 3.2 非採用方針

当面は以下をやらない。

- `Qwen3.5` の text backbone だけを抜き出して、既存の `DINOv2 + Adapter` に繋ぐ
- 現行 `cosmos-reason-mini` の checkpoint を部分流用する
- Stage 2 の KV-cache Expert を最初から Alpamayo 忠実に再実装する

理由は、最初からそこまで踏み込むと、**どこで性能が良くなったか / 壊れたか切り分けにくくなる**ため。

---

## 4. 開発方針

### 4.1 まずは Stage 1 相当まで持っていく

最初の目標は、`Qwen3.5-0.8B` を使って **離散 action token を自己回帰生成できるところまで**持っていくこと。

ここでは以下を優先する。

- 画像 → Qwen3.5 → action token の学習
- 既存 Stage 1 の「離散 action token 化」は踏襲
- action token から `(a, κ)` を decode して軌道を復元

### 4.2 Stage 0 回帰は原則スキップ

`Qwen3.5-0.8B` は既に強い VLM なので、この系統では **Stage 0 Phase 3/4 の回帰 warm-up は原則スキップ**する。

まずは:

`Qwen3.5-0.8B -> Stage 1 相当（離散 token）`

から入る。

必要になった場合のみ、後から簡易回帰ヘッドを足して比較する。

### 4.3 Stage 2 は別問題として切り出す

Stage 2 はこのブランチの最重要難所なので、**Stage 1 が成立するまで保留**する。

優先順位は以下。

1. `Qwen3.5 + action token` が学習できること
2. action token decode だけでも軌道品質が出ること
3. その後に Flow / Expert を再設計すること

---

## 5. 実装ステップ

### Step A: Qwen3.5 バックボーンの最小統合

目的:

- `Qwen3.5-0.8B` を学習コードから読み込めるようにする
- 画像 + プロンプト + generate / forward の最小経路を固める

実装項目:

- `Qwen3.5-0.8B` 用 model loader
- processor / tokenizer を使った画像前処理
- `inputs_embeds` ではなく、まずは **公式 processor ベース**で動作確認
- hidden states / past_key_values を取得できるか確認

Exit 条件:

- 学習コード外で 1 バッチの forward が安定して通る
- `hidden_states` / `past_key_values` の形状を記録できる

### Step B: Stage 1 相当の action token 学習

目的:

- `Qwen3.5` の語彙に action token を追加し、離散 action を自己回帰生成できるようにする

実装項目:

- hardcoded な `Qwen2.5` tokenizer / vocab 前提を除去
- `vocab_offset` を **動的に計算**
- action token の追加と `resize_token_embeddings()`
- teacher forcing による CE 学習

Exit 条件:

- loss が下がる
- action token が collapse しない
- dequantize 後の軌道が最低限妥当

### Step C: Stage 3 相当の reasoning + action joint 学習

目的:

- CoC / reasoning token と action token を同じ系列で扱えるかを見る

実装項目:

- reasoning token + action token の joint SFT
- tokenizer / generation 周りの `Qwen2.5` 固定箇所を除去
- eval / visualization の token parse を Qwen3.5 対応

Exit 条件:

- reasoning と action が同時に生成される
- action 側が reasoning 学習で壊れない

### Step D: Stage 2 再設計

目的:

- `Qwen3.5` で連続軌道 decoder をどう作るか決める

候補は 3 つ。

1. **hidden states 条件付け Expert**
   - まずは最も実装しやすい
   - `past_key_values` 直結ではなく hidden states を条件にする

2. **Qwen3.5 専用 KV-cache Expert**
   - Alpamayo に最も近い
   - ただし hybrid attention / delta 部分まで理解が必要

3. **Stage 2 を当面スキップ**
   - action token decode のみで軌道出力
   - VLA 最小版として先に全体をつなぐ

当面の推奨は **1 または 3**。

### Step E: RL / Post-training

これは最後に回す。

理由:

- backbone 交換の影響が大きい
- まず supervised 系が成立しないと RL の比較が難しい

---

## 6. コード変更の優先順位

優先度高:

- model loader
- tokenizer / vocab offset の動的化
- Stage 1 train / eval
- Stage 3 / Stage 4 の `Qwen2.5` 固定部分

優先度中:

- 可視化
- reasoning prompt まわり
- checkpoint save/load 形式の整理

優先度低:

- 既存 `DINOv2 + Adapter` 系との共存を綺麗に抽象化すること

最初は分岐実装でもよい。  
抽象化は **Qwen3.5 系が最低限勝てる見込みが見えてから**で十分。

---

## 7. 主なリスク

### 7.1 Stage 2 の再利用不可

最も大きいリスク。  
既存の `Qwen2.5` 前提 Expert は、そのままでは使えない可能性が高い。

### 7.2 Catastrophic interference

強い汎用 VLM に action / CoC を入れるため、元の視覚理解が壊れる可能性がある。  
特に full fine-tune では要注意。

### 7.3 推論速度

`Qwen3.5-0.8B` は現時点で fast path は有効化できているが、**CoC を毎回生成する VLA** としてはまだ重い可能性がある。  
Stage 1 の段階ではまず品質を見て、速度はその後に詰める。

### 7.4 画像前処理の不整合

`DINOv2 224x224 固定` 前提の既存データパイプラインを、そのまま `Qwen3.5` に流用しない。  
processor 主導の resize / patchify を尊重する。

---

## 8. 当面の意思決定

このブランチでは、まず以下を正式方針とする。

1. `Qwen3.5-0.8B` は **ネイティブ VLM として使う**
2. `cosmos-reason-mini` の直接置換ではなく、**別バックボーン系統**として進める
3. まずは **Stage 1 相当**を成立させる
4. **Stage 2 は後回し**
5. 既存コードとの共通化は後回しでよい

---

## 9. 最初の実装タスク

次に着手する具体タスクは以下。

1. `Qwen3.5-0.8B` 用の最小学習ラッパーを追加
2. hidden states / past_key_values の shape dump を取る
3. Stage 1 の action token 学習を Qwen3.5 用に複製または分岐
4. `vocab_offset` を固定値から動的計算に置き換える
5. `image.png` レベルの簡易 qualitative ではなく、nuScenes / CARLA の action 学習で本当に勝てるかを見る

以上。
