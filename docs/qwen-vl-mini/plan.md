# Qwen2.5-VL Mini 実装計画

## 全体方針

DINOv2 + SmolLM2 から汎用 VLM を構築する。Qwen2.5-VL の学習パイプライン（5 段階）を簡略化し、LLaVA-1.5 方式の **2 段階**で進める。詳細は [設計書](design.md) を参照。

```
Phase 1 (基盤) → Phase 2 (Stage 1: Feature Alignment) → Phase 3 (Stage 2: Visual Instruction Tuning)
```

完了後、学習済み重みを Cosmos Reason Mini に引き継ぐ。

---

## Phase 1: 基盤構築

### 1.1 モジュール実装

- [ ] **Vision Encoder**: DINOv2 ViT-S/14 ロード + forward
  - `facebook/dinov2-small`
  - 入力: (B, 3, 224, 224) → 出力: (B, 256, 384)
- [ ] **Adapter**: 2層 MLP（初期実装）
  - Linear(384, 960) → GELU → Linear(960, 960)
  - 入力: (B, 256, 384) → 出力: (B, 256, 960)
- [ ] **LLM**: SmolLM2-360M ロード
  - visual tokens を embedding シーケンスの先頭に注入
  - テキスト入力（指示 + 質問）と結合
- [ ] 統合モデル `QwenVLMini` クラス
  - forward: 画像 + テキスト指示 → テキスト出力
  - generate: 自己回帰テキスト生成
  - Loss 計算: 回答部分のみに cross-entropy を適用

### 1.2 動作確認

- [ ] ランダム重みで forward pass が通ることを確認
- [ ] generate（自己回帰生成）が動作することを確認
- [ ] VRAM 使用量の実測

### 1.3 Exit 条件

- [ ] forward + generate が動作する
- [ ] OOM しない

---

## Phase 2: Stage 1 — Feature Alignment

### 目的

Adapter が DINOv2 の視覚特徴を SmolLM2 の入力空間にマッピングすることを学ぶ。

### 2.1 データ準備

- [ ] LLaVA-CC3M-Pretrain-595K のダウンロード
  - HuggingFace: `liuhaotian/LLaVA-CC3M-Pretrain-595K`
  - 画像 + キャプションのペア、約 595K サンプル
- [ ] DataLoader 実装
  - 画像: リサイズ (224×224) + ImageNet 正規化
  - テキスト: "Describe the image briefly.\n" + キャプション
  - Loss マスク: キャプション部分のみ（指示テキスト部分は無視）
- [ ] 単体テスト: バッチが正しく取り出せることを確認

### 2.2 学習

- [ ] **Adapter のみ trainable**、DINOv2 + SmolLM2 は frozen
- [ ] ハイパーパラメータ:
  - 学習率: 1e-3（Adapter のみなので大きめ）
  - スケジューラ: cosine with warmup
  - micro-batch=4, grad_accum=4（≈ global batch 16）
  - エポック: 1
  - bf16 mixed precision
  - AdamW (β1=0.9, β2=0.95)
- [ ] wandb ロギング
- [ ] 学習時間の見積もり: Adapter のみなので高速（数時間〜半日）

### 2.3 評価

- [ ] テスト画像を入力してキャプションを生成
- [ ] 生成テキストの定性的評価（画像に関連した内容が出るか）
- [ ] Stage 1 の前後で生成テキストを比較
- [ ] POPE で定量評価（ランダム回答 50% を上回るか）

### 2.4 Exit 条件

- [ ] Loss が安定して下がる
- [ ] 画像を入力すると（不完全でも）画像に関連したテキストが生成される
- [ ] ランダムなテキストではなく、画像の内容に応じた出力が変化する

---

## Phase 3: Stage 2 — Visual Instruction Tuning

### 目的

画像について質問に答えたり詳細に説明したりする VLM 能力を獲得する。

### 3.1 データ準備

- [ ] LLaVA-Instruct-150K のダウンロード
  - HuggingFace: `liuhaotian/LLaVA-Instruct-150K`
  - 画像 + 会話（質問-回答）のペア、約 150K サンプル
- [ ] DataLoader 実装
  - 会話形式: `[visual_tokens] User: {質問}\nAssistant: {回答}`
  - Loss マスク: Assistant の回答部分のみ
- [ ] （任意）ShareGPT4V-100K も混合してデータ品質を向上

### 3.2 学習

- [ ] **DINOv2 + Adapter + LLM を全解凍**（Qwen2.5-VL Phase 2 と同方式）
- [ ] Stage 1 の学習済み Adapter 重みを初期値として使用
- [ ] ハイパーパラメータ:
  - 学習率: 2e-5（全パラメータ共通。DINOv2 の特徴崩壊が見られたら layer-wise lr decay を検討）
  - スケジューラ: cosine with warmup
  - micro-batch=1, grad_accum=16
  - エポック: 1
  - bf16 mixed precision
  - gradient checkpointing ON（DINOv2 + LLM）
  - AdamW (β1=0.9, β2=0.95)
- [ ] wandb ロギング

### 3.3 Adapter トークン圧縮（推奨）

256 ビジョントークンは SmolLM2-360M のコンテキスト長を大量に消費するため、**トークン圧縮の導入は早期に検討すべき**。Qwen2.5-VL でも隣接 4 パッチをグループ化して 4 倍圧縮している。

- [ ] **方式 A: 隣接パッチグループ化**（Qwen2.5-VL 方式）
  - 隣接 4 パッチを結合 → MLP で射影
  - 256 → 64 トークン。実装が容易
- [ ] **方式 B: Cross-Attention Pooling**
  - learnable query 16 個、DINOv2 パッチ特徴を key/value
  - 256 → 16 トークン。最も圧縮率が高い
- [ ] 圧縮ありと圧縮なしの性能比較
- [ ] トークン圧縮を変更した場合、Stage 1 からやり直す必要あり

### 3.4 評価

- [ ] 定性的評価:
  - 「この画像を説明してください」→ 詳細な記述が生成されるか
  - 「画像に何が写っていますか」→ 物体を列挙できるか
  - 「この画像の天気は？」→ 適切に回答できるか
- [ ] 定量的評価（lmms-eval 使用）:
  - **Tier 1（必須）**: POPE（物体存在判定）、ScienceQA-IMG（多肢選択）
  - **Tier 2（推奨）**: VQAv2（一般 QA）、GQA（シーン理解）
  - 参考目標: SmolVLM-256M（ScienceQA 73.8%, POPE ~75%）に近づくか

### 3.5 Exit 条件

- [ ] Loss が安定して下がる
- [ ] 画像に対する質問に（大雑把にでも）妥当な回答が生成される
- [ ] Stage 1 より明確に改善している

---

## Cosmos Reason Mini への引き継ぎ

- [ ] 学習済み重み（DINOv2 + Adapter + SmolLM2）を保存
- [ ] Cosmos Reason Mini でロードし、forward pass が通ることを確認
- [ ] 引き継ぎ時の Adapter 方式（MLP or Cross-Attention）を記録

---

## 実装優先順位まとめ

```
Phase 1 (基盤: モジュール実装)
    ↓
Phase 2 (Stage 1: Feature Alignment — Adapter のみ)
    ↓
Phase 3 (Stage 2: Visual Instruction Tuning — Adapter + LLM)
    ↓
→ Cosmos Reason Mini に引き継ぎ
```

**最短経路**: Phase 1 → 2 → Cosmos Reason Mini（Stage 2 スキップ、Alignment だけで先に進む）
**推奨経路**: Phase 1 → 2 → 3 → Cosmos Reason Mini

---

## リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| DINOv2 特徴と SmolLM2 の空間が遠すぎる | Alignment が進まない | 学習率を上げる、MLP の層数を増やす |
| CC3M のダウンロードが困難 | データ不足 | ShareGPT4V-PT 等の代替データを使用 |
| SmolLM2 の生成能力が低い | VQA 回答の品質が悪い | 短い回答に限定、会話形式を簡素化 |
| Stage 2 で LLM のテキスト能力が劣化 | 一般的な言語能力の喪失 | 学習率を小さく保つ、テキストのみのデータも混合 |
| 256 ビジョントークンが LLM に多すぎる | 計算コスト増、性能低下 | トークン圧縮（隣接4パッチグループ化 or Cross-Attention Pooling） |
| DINOv2 にテキスト対応がない（CLIP と異なる） | Adapter の学習負担が大きい | Stage 1 で十分な学習を行う。学習率やデータ量の調整で対応 |
