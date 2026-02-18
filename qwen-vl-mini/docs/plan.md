# Qwen2.5-VL Mini 実装計画

## 全体方針

DINOv2 ViT-B/14 + Qwen2.5-0.5B から汎用 VLM を構築する。Qwen2.5-VL の学習パイプライン（5 段階）を簡略化し、LLaVA-1.5 方式の **2 段階**で進める。詳細は [設計書](design.md) を参照。

```
Phase 1 (基盤) → Phase 2 (Stage 1: Feature Alignment) → Phase 3 (Stage 2: Visual Instruction Tuning)
```

完了後、学習済み重みを Cosmos Reason Mini に引き継ぐ。

### 実装アプローチ: Simplest-First Fail-Fast

design.md / plan.md には論文サーベイに基づく多数の「検討」事項（336x336 解像度、[CLS] トークン連結、トークン圧縮、GLU 活性化、深層レイヤー選択、vision block 追加 等）が記載されているが、**Phase 1 ではすべて最もシンプルな選択で実装し、まず動くものを作る**。改善は実測に基づいて段階的に行う。

**Phase 1 の初期構成（最小構成）**:

| 項目 | 選択 | 理由 |
|---|---|---|
| 入力解像度 | **224×224** | DINOv2 のデフォルト。336 は後で試す |
| Vision Encoder 出力 | **全 12 層の最終層パッチトークン** | 深層選択・Layerscale 統合は後で試す |
| [CLS] トークン | **使わない**（パッチトークンのみ） | LLaVA §4.1 準拠。連結は後で試す |
| Adapter | **2 層 MLP, GELU** | COMM Table 6 準拠。GLU は後で試す |
| トークン圧縮 | **なし**（256 トークンそのまま） | まず動作確認。Pixel Shuffle は後で導入 |
| Vision block 追加 | **なし** | dino-meets-text の知見だが初期は不要 |

**改善サイクル**: Phase 1 完了後、Phase 2/3 の学習結果を見ながら一つずつ変更を試し、効果を実測で確認する。**同時に複数変更しない**（どの変更が効いたか分からなくなる）。

具体的な Phase 1 の実装手順は [phase1-implementation.md](phase1-implementation.md) を参照。

---

## Phase 1: 基盤構築

### 1.1 モジュール実装

- [ ] **Vision Encoder**: DINOv2 ViT-B/14 ロード + forward
  - `facebook/dinov2-base`（register token なし版を使用。`dinov2-base-reg` を使う場合は register token 4 個を出力から除外する処理が必須）
  - **入力前処理**: ImageNet 正規化（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]）を適用。公式リポジトリの transform 設定と一致させること（§10.15 参照）
  - 入力: (B, 3, 224, 224) → 出力: (B, 256, 768)。**336x336 への解像度アップを検討**（576 トークン、COMM §5 で採用。position embedding を双線形補間で拡大）
  - **深層レイヤーのみ使用を検討**: COMM Table 4 より DINOv2 は深層に有用情報が集中。ViT-B/14（12 層）なら後半 6 層（7-12 層）の Layerscale 統合を使用（§10.12, §10.21 COMM MFM 参照）
  - **CLS トークンではなくパッチトークン全体を出力**: LLaVA §4.1 に準拠。ただし **[CLS] トークンを global context として prepend する設計も検討**（dino-meets-text Table 2: [CLS]+avg-pooled が全タスク最適）
  - **bf16 必須**: fp16 では小規模モデルで loss NaN が発生する（MoE-LLaVA Appendix A.2）。TF32 も明示的に有効化（§10.15 参照）
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

具体的な実装手順は [phase2-implementation.md](phase2-implementation.md) を参照。

### 目的

Adapter が DINOv2 の視覚特徴を Qwen2.5-0.5B の入力空間にマッピングすることを学ぶ。

### 2.1 データ準備

- [ ] LLaVA-CC3M-Pretrain-595K のダウンロード
  - HuggingFace: `liuhaotian/LLaVA-CC3M-Pretrain-595K`
  - 画像 + キャプションのペア、約 595K サンプル
  - **DINOv2 は text-alignment がないため、558K で Alignment が不十分な場合は ShareGPT4V-PT 1,246K に増量を検討**（TinyLLaVA Figure 7, Cambrian-1 Figure 7）
  - ただし Qwen2.5-0.5B は 494M と小型のため、データ過多によるハルシネーション増加にも注意（TinyLLaVA §4.2.2: TinyLlama 1.1B で POPE 劣化の事例）
- [ ] DataLoader 実装
  - 画像: リサイズ (224×224 or 336×336) + ImageNet 正規化（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]）
  - テキスト: **11 種類の質問プロンプトからランダム選択**（LLaVA Table 11）。"Describe the image concisely." / "Provide a brief description of the given image." / "Summarize the visual content of the image." 等
  - Loss マスク: キャプション部分のみ（指示テキスト部分は `ignore_index=-100` でマスク）
  - **合成キャプション（synthetic captions）は alt-text より品質が高い**: Idefics2 Table 6 で +3.1pt
  - **Pre-training データの飽和点は ~1000K**（ShareGPT4V Figure 6）。558K→1246K の増量は妥当だが、それ以上はコスト対効果低い
- [ ] 単体テスト: バッチが正しく取り出せることを確認

### 2.2 学習

- [ ] **Adapter のみ trainable**、DINOv2 + Qwen2.5-0.5B は frozen
- [ ] ハイパーパラメータ:
  - 学習率: 1e-3（Adapter のみなので大きめ。MLP 使用時は Linear の 2e-3 から半減。LLaVA-1.5 Table 9）
  - スケジューラ: cosine with warmup（warmup ratio 0.03）。**warmup なしも試す価値あり**（OpenVLA §3.4: warmup の効果が見られなかったケース）
  - micro-batch=4, grad_accum=4（≈ global batch 16。**目標 batch size=256** に近づけるよう grad_accum を調整）
  - エポック: 1
  - **bf16 mixed precision**（fp16 禁止: loss NaN リスク。MoE-LLaVA App.A.2）
  - **TF32 有効化**: `torch.backends.cuda.matmul.allow_tf32 = True`
  - AdamW (β1=0.9, β2=0.95)
  - **gradient clipping: max_grad_norm=1.0**
- [ ] wandb ロギング（**データソースごとの loss を個別トラッキング**: OpenVLA §3.3）
- [ ] 学習時間の見積もり: Adapter のみなので高速（数時間〜半日。LLaVA-Phi: 8xA100 で 1.5 時間）

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

具体的な実装手順は [phase3-implementation.md](phase3-implementation.md) を参照。

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
  - **Loss マスク: completion 部分（Assistant の回答）のみ**: `ignore_index=-100` をシステムプロンプト + ユーザー質問 + visual tokens に適用。Qwen2.5 の場合 `<|im_start|>assistant` 〜 `<|im_end|>` のみ loss 対象
  - **VQA データには formatting prompt を付加**（LLaVA-1.5 Appendix A.2 のデータセット別使い分け）:
    - VQAv2, GQA, TextVQA, POPE → "Answer the question using a single word or phrase."
    - ScienceQA, MMBench → "Answer with the option's letter from the given choices directly."
    - 自由回答系 → フォーマット指示なし
  - **Answer Machine Phenomenon 対策**: 短答 QA 過多で会話能力喪失を防ぐため、フォーマットプロンプトで短答/会話を明示的に分離（Cambrian-1 §5.3）
  - **データの多様性を確保**: 推奨比率 General ~35%, Language/text-only ~15%, Science ~10%, OCR ~9%, Counting ~7%, Math+Code ~5%（Cambrian-1 Table 20 参考）
  - **同一画像の QA ペアを単一マルチターン会話に統合**: 画像エンコード重複を削減（LLaVA-1.5 Appendix A.2）
  - **バッチ内モダリティ分離**: 言語のみ/視覚付きを同一バッチに混在させない。25% 高速化（LLaVA-1.5 Appendix A.2）
  - **テキスト会話は 2048 トークンで truncate**（split ではなく。LLaVA-1.5 Appendix A.2）
  - **50 語未満のテキストはフィルタ除外**（Cambrian-1 Appendix E.3）
- [ ] （任意）SFT データの **3-5% を ShareGPT4V キャプションに置換**するだけで全体性能向上（ShareGPT4V Figure 2）
- [ ] （任意）GPT4V-annotated データ追加: ShareGPT-4V 20K + LAION-GPT-V 10K + ALLaVA 300K で +1.4pt（Imp Table 1 §3.2）
- [ ] （任意）OCR/Chart データ追加: DVQA 10K + ChartQA 4K + DocVQA 10K + AI2D 4K + InfographicVQA 4K = 32K（Imp §3.1, Table 2）。DINOv2 の OCR 弱点を補うために特に重要
- [ ] （任意）**text-only instruction data を 10-15% 混合**: OpenHermes-2.5, MetaMathQA 等で LLM の catastrophic forgetting を緩和（Idefics2 Appendix A.2.1）
- [ ] （任意）**TextCaps (22K) は TextVQA と同じ画像セット**。zero-shot 評価したい場合は除去（Imp §4.3）

### 3.2 学習

- [ ] **DINOv2 + Adapter + LLM を全解凍**（Qwen2.5-VL Phase 2 と同方式）
- [ ] Stage 1 の学習済み Adapter 重みを初期値として使用
- [ ] ハイパーパラメータ:
  - LLM + Adapter 学習率: **2e-5**（全論文で一致）
  - **DINOv2 学習率: 1e-5**（メイン LR の半分。Cambrian-1 Table 23 準拠。Eagle では SFT と同じ 2e-5 も使用）
  - スケジューラ: cosine with warmup（warmup ratio 0.03）。**warmup なしも試す価値あり**（OpenVLA §3.4）
  - micro-batch=1, grad_accum=16（**目標 batch size=128-256**）
  - エポック: **2**（Imp [arXiv:2405.12107] Table 1 §2.2: 1ep は学習不足 (71.6)、2ep が最適 (72.1)。**3 エポック以上は過学習リスク** -0.4pt）
  - **bf16 mixed precision**（fp16 禁止。TF32 有効化）
  - gradient checkpointing ON（DINOv2 + LLM）
  - AdamW (β1=0.9, β2=0.95)
  - **weight decay: 0.1**（LLaVA-Phi §3.1: 小型モデルでは weight decay による正則化が重要。LLaVA/LLaVA-1.5 の 0 とは異なる）
  - **gradient clipping: max_grad_norm=1.0**（全論文で明示記載なし → Qwen2.5 デフォルト値を採用。DINOv2 unfreeze 安定化に寄与）
- [ ] **過学習対策**:
  - **NEFTune（Noisy Embedding Fine-Tuning）**: 入力埋め込みにノイズ注入で汎化向上（Idefics2 §4.2）。過学習の兆候が見られたら導入
  - 画像解像度のランダムスケールアップ（Idefics2 §4.2）
  - multi-turn 会話のシャッフル（Idefics2 §4.2）
- [ ] **チェックポイントを 25 ステップごとに保存**（SmolVLM 知見: 最適点は訓練終了時とは限らない）
- [ ] **訓練不安定時の段階的対策**（設計書 §10.4, §10.15, §10.18 参照）:
  1. まず全解凍で試行（DINOv2 lr=**1e-5**、LLM lr=**2e-5**）
  2. 発散したら DINOv2 の後半 6 層のみ解凍（TinyLLaVA [arXiv:2402.14289] Table A1: Share recipe。**コネクタは Stage 1 の学習済み重みで初期化**）
  3. それでも不安定なら LoRA rank=256 を LLM に適用（Imp [arXiv:2405.12107] Table 1 §2.1: rank=256 が最適、Idefics2 [arXiv:2405.02246] Table 3: LoRA で +9.2pt）
  4. **FFN-only fine-tuning**: Attention 層凍結、FFN 層のみ学習で 75% 時間で同等性能（MoE-LLaVA Table 5a）
  5. **Sandwich fine-tuning**: VE + token embedding + LLM 最終層のみ解凍（OpenVLA Table 1: LoRA rank=32 で Full FT にほぼ匹敵）
  6. DINOv2 を frozen にしたまま adapter + LLM のみ学習（COMM の知見）
  - **推奨手順**: まず frozen で設計を固め（データ、Adapter、ハイパーパラメータの検証）→ 設計が固まったら unfreeze で最終訓練（Cambrian-1 方式: +4.88pt avg, Vision-Centric +11.47pt）
  - **注意**: DINOv2 解凍で訓練速度 50-55% 低下（Cambrian-1 Appendix F）
  - **0.5B LLM の特性**: 小型 LLM では ViT unfreeze が POPE 改善する可能性あり（大型 LLM とは逆。TinyLLaVA §4.2.2）
- [ ] wandb ロギング（**データソースごとの loss/accuracy を個別モニタリング**。学習が進まないデータソースは途中除外。OpenVLA §3.3）

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
- [ ] **評価プロトコル**（§10.19 参照）:
  - **greedy decoding（temperature=0, do_sample=False）**: LLaVA-1.5 Appendix A.3。全ベンチマークで統一し再現性を確保
  - **データセット別評価プロンプト**:
    - VQAv2, GQA, TextVQA, POPE → "Answer the question using a single word or phrase."
    - ScienceQA, MMBench → "Answer with the option's letter from the given choices directly."
    - 自由回答系（LLaVA-Bench, MM-Vet）→ フォーマット指示なし
  - **ストップワード設定**: VQA 系は `\n` で生成停止。多肢選択はオプション文字後に停止
  - **ベンチマーク結果のノイズ ±5pt**: プロンプト・シード・データ構成で変動する。改善が 5pt 未満の場合は有意差なしと判断

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
| fp16 で loss NaN | 学習が進まない | bf16 必須。TF32 明示的有効化 | MoE-LLaVA App.A.2 |
| Answer Machine Phenomenon | 短答のみ学習し会話能力喪失 | データ比率管理、formatting prompt で短答/会話分離 | Cambrian-1 §5.3 |
| DINOv2 の DocVQA 弱点（11%） | テキスト読解系タスク低性能 | OCR/Chart 32K データ追加、text-only mixing 10-15% | §10.21, Imp §3.1 |
| text-only データ過多 | negative transfer で視覚タスク劣化 | 14% 上限を厳守 | SmolVLM §3.3 |

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

### 実装・学習の落とし穴（全 Phase 共通）

31. **bf16 必須、fp16 禁止**: MoE-LLaVA Appendix A.2 で Qwen-1.8B クラスの小型モデルが fp16 で loss NaN を報告。全 Phase で bf16 mixed precision を使用。TF32 も明示的に有効化（`torch.backends.cuda.matmul.allow_tf32 = True`）
32. **DINOv2 register token に注意**: `dinov2-base-reg` を使う場合、4 個の register token を出力から除外必須。推奨は `dinov2-base`（register token なし）を使用（§10.15）
33. **ImageNet 正規化を忘れずに適用**: DINOv2 は mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]。公式リポジトリの transform 設定と一致させること（§10.15）
34. **gradient clipping = 1.0**: DINOv2 unfreeze 時の安定化に寄与。全 Phase で適用（§10.15）

### DINOv2 活用の追加テクニック

35. **[CLS]+avg-pooled patch 連結を検討**: dino-meets-text Table 2 で分類・セグメンテーション・検索の全タスクで最適。パッチトークンのみより高品質な表現（§10.16）
36. **DINOv2 上に 1-2 層の learnable vision block 追加を検討**: dino-meets-text Table 3 で検索 +7pt 改善。frozen DINOv2 の出力に learnable Transformer block を追加する設計（§10.16）
37. **GLU 活性化関数 > 標準 MLP**: SAIL Table 1 で +12.4%（ImageNet zero-shot）。Adapter の GELU を GLU に置換する候補（§10.18）

### データ・学習の追加知見

38. **Answer Machine Phenomenon に注意**: 短答 QA データ過多で会話能力が喪失する。フォーマットプロンプトで短答/会話を明示的に分離し、データ比率を管理（Cambrian-1 §5.3、§10.17）
39. **text-only instruction data は 14% 以下**: SmolVLM §3.3 で超過すると negative transfer が発生。10-15% を推奨（§10.17）
40. **Stage 1 での VE 解凍は高品質データが条件**: Cambrian-1 Figure 7 で低品質データ + 解凍は逆効果。ShareGPT4V-PT 以上の品質が必要（§10.18）

### 評価プロトコル

41. **評価は greedy decoding（temperature=0, do_sample=False）**: LLaVA-1.5 Appendix A.3。再現性確保のため必須（§10.19）
42. **データセット別評価プロンプトを使い分ける**: VQAv2 は "Answer the question using a single word or phrase."、ScienceQA は "Answer with the option's letter..."、自由回答は指示なし（§10.19）
43. **ストップワードを設定**: VQA 系は `\n` で停止。多肢選択はオプション文字で停止（§10.19）
44. **ベンチマーク結果には ±5pt のノイズ**: 同一手法でもプロンプト・シード・データで 5pt 変動する。絶対値より傾向に注目（§10.19）
