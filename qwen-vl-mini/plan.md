# Qwen2.5-VL Mini 実装計画

## 全体方針

DINOv2 ViT-B/14 + Qwen2.5-0.5B から汎用 VLM を構築する。Qwen2.5-VL の学習パイプライン（5 段階）を簡略化し、LLaVA-1.5 方式の **2 段階**で進める。詳細は [設計書](design.md) を参照。

```
Phase 1 (基盤) → Phase 2 (Stage 1: Feature Alignment) → Phase 3 (Stage 2: Visual Instruction Tuning)
```

完了後、学習済み重みを Cosmos Reason Mini に引き継ぐ。

---

## Phase 1: 基盤構築

### 1.1 モジュール実装

- [ ] **Vision Encoder**: DINOv2 ViT-B/14 ロード + forward
  - `facebook/dinov2-base`
  - 入力: (B, 3, 224, 224) → 出力: (B, 256, 768)
  - **深層レイヤーのみ使用を検討**: COMM Table 4 より DINOv2 は深層に有用情報が集中。ViT-B/14（12 層）なら後半 6 層（7-12 層）の Mean を使用（§10.12 参照）
  - **CLS トークンではなくパッチトークン全体を出力**: LLaVA §4.1 に準拠
- [ ] **Adapter**: 2層 MLP（初期実装）
  - Linear(768, 3072) → GELU → Linear(3072, 896)（COMM [arXiv:2310.08825] Table 6: Ratio 4 が最適）
  - 入力: (B, 256, 768) → 出力: (B, 256, 896)
  - ※ Pixel Shuffle 導入時は `nn.Linear(768 * r^2, 896, bias=False)` の 1 層 Linear に置換（SmolVLM 実装）
- [ ] **LLM**: Qwen2.5-0.5B ロード
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

Adapter が DINOv2 の視覚特徴を Qwen2.5-0.5B の入力空間にマッピングすることを学ぶ。

### 2.1 データ準備

- [ ] LLaVA-CC3M-Pretrain-595K のダウンロード
  - HuggingFace: `liuhaotian/LLaVA-CC3M-Pretrain-595K`
  - 画像 + キャプションのペア、約 595K サンプル
  - **DINOv2 は text-alignment がないため、558K で Alignment が不十分な場合は ShareGPT4V-PT 1,246K に増量を検討**（TinyLLaVA Figure 7, Cambrian-1 Figure 7）
  - ただし Qwen2.5-0.5B は 494M と小型のため、データ過多によるハルシネーション増加にも注意（TinyLLaVA §4.2.2: TinyLlama 1.1B で POPE 劣化の事例）
- [ ] DataLoader 実装
  - 画像: リサイズ (224×224) + ImageNet 正規化
  - テキスト: "Describe the image briefly.\n" + キャプション
  - Loss マスク: キャプション部分のみ（指示テキスト部分は無視）
  - **合成キャプション（synthetic captions）は alt-text より品質が高い**: Idefics2 Table 6 で +3.1pt
- [ ] 単体テスト: バッチが正しく取り出せることを確認

### 2.2 学習

- [ ] **Adapter のみ trainable**、DINOv2 + Qwen2.5-0.5B は frozen
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
  - **システムプロンプト + メディアマーカーを含む入力テンプレート**: SmolVLM Finding 6 で小規模 VLM の性能大幅向上を確認
    ```
    System: You are a visual agent and should provide concise answers.
    Here is an image: [visual_tokens]
    User: {質問}
    Assistant: {回答}
    ```
  - **Loss マスク: completion 部分（Assistant の回答）のみ**: SmolVLM §3.2 でユーザープロンプトマスクの有効性を確認。マスクしないとモデルが表面的な繰り返しを学習してしまう
  - **VQA データには formatting prompt を付加**: "Answer the question using a single word or phrase"（LLaVA-1.5 §3.2 Table 1b）で short-answer overfit を回避
  - **データの多様性を確保**: Conversation + Detail description + Complex reasoning の 3 種混合が最高性能（LLaVA Table 4）
- [ ] （任意）ShareGPT4V-100K も混合してデータ品質を向上
- [ ] （任意）GPT4V-annotated データ追加: ShareGPT-4V 20K + LAION-GPT-V 10K + ALLaVA 300K で +1.4pt（Imp Table 1 §3.2）
- [ ] （任意）OCR/Chart データ追加: DVQA + ChartQA + DocVQA + AI2D + InfographicVQA = 32K（Imp §3.1）。DINOv2 の OCR 弱点を補うために特に重要

### 3.2 学習

- [ ] **DINOv2 + Adapter + LLM を全解凍**（Qwen2.5-VL Phase 2 と同方式）
- [ ] Stage 1 の学習済み Adapter 重みを初期値として使用
- [ ] ハイパーパラメータ:
  - LLM + Adapter 学習率: **2e-5**（全論文で一致）
  - **DINOv2 学習率: 1e-5**（メイン LR の半分。Cambrian-1 Table 23 準拠）
  - スケジューラ: cosine with warmup（warmup ratio 0.03）
  - micro-batch=1, grad_accum=16
  - エポック: **2**（Imp [arXiv:2405.12107] Table 1 §2.2: 1ep は学習不足 (71.6)、2ep が最適 (72.1)。**3 エポック以上は過学習リスク** -0.4pt）
  - bf16 mixed precision
  - gradient checkpointing ON（DINOv2 + LLM）
  - AdamW (β1=0.9, β2=0.95)
  - **weight decay: 0.1**（LLaVA-Phi §3.1: 小型モデルでは weight decay による正則化が重要。LLaVA/LLaVA-1.5 の 0 とは異なる）
- [ ] **過学習対策**:
  - **NEFTune（Noisy Embedding Fine-Tuning）**: 入力埋め込みにノイズ注入で汎化向上（Idefics2 §4.2）。過学習の兆候が見られたら導入
  - 画像解像度のランダムスケールアップ（Idefics2 §4.2）
  - multi-turn 会話のシャッフル（Idefics2 §4.2）
- [ ] **チェックポイントを 25 ステップごとに保存**（SmolVLM 知見: 最適点は訓練終了時とは限らない）
- [ ] **訓練不安定時の段階的対策**（設計書 §10.4 参照）:
  1. まず全解凍で試行（DINOv2 lr=**1e-5**、LLM lr=**2e-5**）
  2. 発散したら DINOv2 の後半 6 層のみ解凍（TinyLLaVA [arXiv:2402.14289] Table A1: Share recipe）
  3. それでも不安定なら LoRA rank=256 を LLM に適用（Imp [arXiv:2405.12107] Table 1 §2.1: rank=256 が最適、Idefics2 [arXiv:2405.02246] Table 3: LoRA で +9.2pt）
  4. DINOv2 を frozen にしたまま adapter + LLM のみ学習（COMM の知見）
  - **推奨手順**: まず frozen で設計を固め（データ、Adapter、ハイパーパラメータの検証）→ 設計が固まったら unfreeze で最終訓練（Cambrian-1 方式: +4.88pt avg, Vision-Centric +11.47pt）
  - **注意**: DINOv2 解凍で訓練速度 50-55% 低下（Cambrian-1 Appendix F）
- [ ] wandb ロギング

### 3.3 Adapter トークン圧縮（推奨）

256 ビジョントークンはコンテキスト長を大量に消費するため、**トークン圧縮の導入は早期に検討すべき**。Qwen2.5-VL でも隣接 4 パッチをグループ化して 4 倍圧縮している。

- [ ] **方式 A: 隣接パッチグループ化**（Qwen2.5-VL 方式）
  - 隣接 4 パッチを結合 → MLP で射影
  - 256 → 64 トークン。実装が容易
- [ ] **方式 B: Cross-Attention Pooling**
  - learnable query 16 個、DINOv2 パッチ特徴を key/value
  - 256 → 16 トークン。最も圧縮率が高い
- [ ] **方式 C: Pixel Shuffle r=2**（SmolVLM [arXiv:2504.05299] Figure 3 右の知見を適用）
  - Space-to-Depth 変換で空間特徴をチャネル方向に再配置 → Linear で射影
  - 256 → 64 トークン（16×16 → 8×8）。SmolVLM-500M は SigLIP 1,024 パッチに r=4 で 64 トークン。DINOv2 は 256 パッチなので r=2 で同等
  - SmolVLM 実装: `nn.Linear(768 * r^2, 896, bias=False)` — 単一 Linear 層
  - パラメータフリー（Pixel Shuffle 部分）で実装が容易
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

- [ ] 学習済み重み（DINOv2 + Adapter + Qwen2.5-0.5B）を保存
- [ ] Cosmos Reason Mini でロードし、forward pass が通ることを確認
- [ ] 引き継ぎ時の Adapter 方式（MLP or Cross-Attention）を記録

---

## 実装優先順位まとめ

```
Phase 1 (基盤: モジュール実装)
    ↓
Phase 2 (Stage 1: Feature Alignment — Adapter のみ)
    ↓
Phase 3 (Stage 2: Visual Instruction Tuning — DINOv2 + Adapter + LLM)
    ↓
→ Cosmos Reason Mini に引き継ぎ
```

**最短経路**: Phase 1 → 2 → Cosmos Reason Mini（Stage 2 スキップ、Alignment だけで先に進む）
**推奨経路**: Phase 1 → 2 → 3 → Cosmos Reason Mini

---

## リスクと対策

| リスク | 影響 | 対策 | 根拠 |
|---|---|---|---|
| DINOv2 特徴と Qwen2.5-0.5B の空間が遠すぎる | Alignment が進まない | MLP ratio 4 を使用（Linear は -10.8pt）。データを 558K → 1,246K に増量 | COMM Table 6, TinyLLaVA Fig 7 |
| Stage 2 で全解凍すると発散 | 訓練が進まない | DINOv2 lr=1e-5 で開始 → 後半 6 層のみ解凍 → LoRA rank=256 → frozen | §10.4 段階的対策 |
| Stage 2 で LLM のテキスト能力が劣化 | 一般的な言語能力の喪失 | LoRA rank=256 で LLM を保護、weight decay=0.1 | Imp Table 1, LLaVA-Phi §3.1 |
| Stage 2 で過学習 | 汎化性能の低下 | 2 エポック以内、NEFTune、25 ステップごとの checkpoint | Imp Table 1 §2.2, Idefics2 §4.2 |
| 256 ビジョントークンが LLM に多すぎる | 計算コスト増、8k 上限超のリスク | Pixel Shuffle r=2 で 64 トークンに圧縮 | SmolVLM Finding 2-3, Idefics2 Table 4 |
| DINOv2 にテキスト対応がない（CLIP と異なる） | OCR/テキスト系タスクが弱い | OCR/Chart データ 32K を追加。Pre-alignment を十分に行う | Cambrian-1 Table 12, Eagle Table 5 |
| DINOv2 解凍でハルシネーション増加 | POPE スコア低下 | POPE を常時監視、Share recipe（後半のみ解凍）で緩和 | TinyLLaVA Table A1 |
| CC3M のダウンロードが困難 | データ不足 | ShareGPT4V-PT 等の代替データを使用 | — |

---

## 小規模 VLM 論文からの注意事項

実装時に特に注意すべき点。詳細な数値とテーブルは設計書 §10 を参照。

### アーキテクチャ（Phase 1）

1. **Adapter は 2 層 MLP を維持**: TinyLLaVA [arXiv:2402.14289] Figure 6 で MLP > Resampler を確認。COMM [arXiv:2310.08825] Table 6 で DINOv2 には 2 層 MLP が必須（Linear 比 +15.7pt）。4 層以上は逆効果
2. **Adapter の隠れ層は Ratio 4（入力の 4 倍）が最適**: COMM [arXiv:2310.08825] Table 6 で確認。`Linear(768, 3072) → GELU → Linear(3072, 896)` を推奨。Pixel Shuffle 導入時は 1 層 Linear で十分（SmolVLM 実装）
3. **DINOv2 に Linear Projector は使うな**: COMM Table 6 で ratio4 比 **-10.8pt**。text-alignment なしのエンコーダには非線形変換が必須
4. **トークン圧縮は Pixel Shuffle r=2 を第一候補に**: SmolVLM [arXiv:2504.05299] Figure 3 右で小規模 VLM には積極的な圧縮が有効と報告。DINOv2 の 256 パッチには r=2 で 64 トークン（SmolVLM-500M の SigLIP 1,024 パッチ + r=4 = 64 トークンと同等の圧縮比）
5. **64 トークンへの圧縮は性能を損なわない**: Idefics2 [arXiv:2405.02246] Table 9 で 64 トークンが 7B-13B 競合モデルを凌駕。積極的なトークン圧縮を正当化
6. **DINOv2 は深層レイヤーのみ使用**: COMM [arXiv:2310.08825] Table 4 で Mean(19-24)=71.7 vs Mean(all)=69.1（-2.6pt）。浅層マージは性能劣化（CLIPとは逆）。ViT-B/14（12 層）なら後半 6 層（7-12 層）
7. **CLS トークンではなくパッチトークン全体を使用**: LLaVA §4.1 に準拠。penultimate layer（最終層の一つ前）も検討（LLaVA Table 8: +0.96%）
8. **ViT-LLM サイズバランスは適切**: SmolVLM [arXiv:2504.05299] Figure 3 左で 428M ViT + 135M LM は性能低下。DINOv2 86M + Qwen2.5-0.5B 494M（~15%）は SmolVLM-500M（~21%）と同等で妥当
9. **DINOv2 の特徴特性を理解**: COMM [arXiv:2310.08825] Table 1 で DINOv2 はグラウンディングに強い（CLIP 比 +7.5pt）が VQA では劣る（-5.7pt）。OCR/テキスト認識は根本的弱点（Cambrian-1 Table 12: OCRBench=3.10, ChartQA=16.48）。ただし自己教師あり学習モデル中で全カテゴリ 1 位（Cambrian-1 Table 2）

### Stage 1: Feature Alignment（Phase 2）

10. **DINOv2 はフローズンで正しい**: LLaVA-1.5 [arXiv:2310.03744] と TinyLLaVA [arXiv:2402.14289] の Stage 1 と同方式。DINOv2 の汎用特徴は高品質なので Adapter のみで十分
11. **DINOv2 は text-alignment がないため Pre-training がさらに重要**: LLaVA Table 8 で Pre-training スキップは -5.11%。DINOv2 ではさらに深刻。Eagle Table 5 でも Pre-alignment 必須と確認
12. **558K で不足なら ShareGPT4V-PT 1,246K に増量**: TinyLLaVA Figure 7 で ShareGPT4V が一貫して良い結果。Cambrian-1 Figure 7 でデータ量増加が DINOv2-CLIP ギャップ縮小に有効
13. **合成キャプションは alt-text より高品質**: Idefics2 Table 6 で +3.1pt
14. **LLM-SFT データの再利用は避ける**: SmolVLM [arXiv:2504.05299] Figure 7 左で画像タスク -6.5%、動画タスク -3.7% の劣化を確認

### Stage 2: Visual Instruction Tuning（Phase 3）

15. **全解凍で発散したら段階的に対処**: TinyLLaVA [arXiv:2402.14289] Table A1 の Share recipe（前半層凍結）→ Imp [arXiv:2405.12107] Table 1 §2.1 の LoRA rank=256（平均 71.6、Full FT 71.2 を上回る）→ Idefics2 [arXiv:2405.02246] Table 3 でも LoRA が +9.2pt で安定化を確認 → DINOv2 frozen のまま adapter + LLM のみ（COMM の知見）
16. **DINOv2 解凍時の lr は 1e-5（メイン LR の半分）**: Cambrian-1 Table 23 準拠。解凍で Vision-Centric +11.47pt だが訓練速度 50-55% 低下
17. **まず frozen で設計を固め、最終訓練で unfreeze**: Cambrian-1 方式。DINOv2 は解凍時の改善幅が特に大きい（+4.88pt avg）
18. **2 エポック以内**: Imp [arXiv:2405.12107] Table 1 §2.2 で 2ep が最適（72.1）。3ep で -0.4pt（TextVQA -1.7, SQA-I -1.2 が主因）
19. **weight decay = 0.1**: LLaVA-Phi §3.1。小型モデルでは weight decay による正則化が重要（大型モデルの 0 とは異なる）
20. **NEFTune（Noisy Embedding Fine-Tuning）で過学習防止**: Idefics2 §4.2。入力埋め込みにノイズ注入で汎化向上。画像解像度のランダムスケールアップ、multi-turn 会話のシャッフルも併用
21. **SFT 時にユーザープロンプトをマスク（completion のみで loss 計算）**: SmolVLM §3.2。マスクしないとモデルが表面的な繰り返しを学習
22. **システムプロンプト + メディアマーカーが性能を大幅向上**: SmolVLM [arXiv:2504.05299] Finding 6。Stage 2 で入力テンプレートにシステムプロンプトとメディアイントロ/アウトロを含める
23. **VQA データには formatting prompt を付加**: "Answer the question using a single word or phrase"（LLaVA-1.5 §3.2）で short-answer overfit を回避
24. **チェックポイント 25 ステップごとに保存**: SmolVLM [arXiv:2504.05299] の方式。最適点は訓練終了時とは限らない
25. **位置トークンを追加する場合は学習可能な埋め込みを使用**: SmolVLM [arXiv:2504.05299] §2.2 Finding 5 で文字列トークンは「OCR loss plague」を引き起こすと報告
26. **CoT データは 0.02-0.05% 以下**: SmolVLM [arXiv:2504.05299] Figure 7 中央で高比率は画像タスクを顕著に劣化させると報告
27. **ViT 解凍は POPE 低下リスクあり**: TinyLLaVA [arXiv:2402.14289] Table A1 で Share recipe は TextVQA を +2.6~+3.5pt 改善する一方、POPE を -0.4~-1.0pt 低下（ハルシネーション増加）
28. **小規模 LM は 8k トークン超で学習不安定**: SmolVLM [arXiv:2504.05299] Finding 2 で 135M/360M LM が 8k 超で不安定と報告。トークン圧縮の必要性を補強
29. **DINOv2 は ViT-g/14 からの蒸留モデル**: DINOv2 Table 4, 17。教師モデルの知識を効率的に保持（ImageNet 82.1 vs 86.5）
30. **小型 LLM では ViT fine-tuning が有効（大型 LLM とは逆）**: TinyLLaVA §4.2.2。ただし訓練パラメータ増加でハルシネーション増加リスクあり
