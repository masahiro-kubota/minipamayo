# Qwen2.5-VL Mini 設計書 v0.3

## 1. 目的

Qwen2.5-VL の学習パイプラインに倣い、DINOv2 + Qwen2.5-0.5B から**汎用 VLM（Vision-Language Model）**を構築する。

「画像を見てテキストを生成する」基礎能力を獲得させることが目的。この VLM が後段の全パイプラインの土台となる:

```
Qwen2.5-VL Mini（本設計書）
  → Cosmos Reason Mini（Physical AI SFT + RL）
    → MiniPamayo（行動予測 Stage 0〜4）
```

### なぜ VLM 構築が必要か

Cosmos-Reason1 は **Qwen2.5-VL（既に完成した VLM）** の上に Physical AI SFT をかけている。つまり SFT 開始時点で既に「画像→テキスト」の基礎能力がある。

一方、DINOv2 + Qwen2.5-0.5B の素の状態は:
- DINOv2: 自己教師あり学習のみ。テキストとの対応付けなし
- Adapter: ランダム初期化
- Qwen2.5-0.5B: テキスト LLM のみ。画像を見たことがない

この状態でいきなり Physical AI SFT（運転 QA）をかけると、モデルは同時に 3 つを学ぶ必要があり困難:
1. Adapter が視覚特徴を LLM に伝える方法（視覚-言語アライメント）
2. 画像→テキスト生成の基礎能力（VLM 能力）
3. 運転ドメインの知識（Physical AI SFT の本来の目的）

Qwen2.5-VL Mini で 1 と 2 を先に解決し、Cosmos Reason Mini では 3 に集中できるようにする。

### Qwen2.5-VL との対応関係

実際の Qwen2.5-VL は **3 段階の Pre-training + 2 段階の Post-training（SFT + DPO）= 計 5 段階**で学習される:

| Phase | Qwen2.5-VL | 学習対象 | トークン数 | seq長 |
|---|---|---|---|---|
| Phase 1 | Visual Pre-Training | **ViT のみ**（LLM frozen） | 1.5T | 8,192 |
| Phase 2 | Multimodal Pre-Training | **ViT + LLM 全解凍** | 2T | 8,192 |
| Phase 3 | Long-Context Pre-Training | ViT + LLM | 0.6T | 32,768 |
| Phase 4 | SFT | LLM + Merger（**ViT frozen**） | ~200万エントリ | — |
| Phase 5 | DPO | LLM + Merger（ViT frozen） | — | — |

Qwen2.5-VL Mini では、Phase 1〜2 を小規模に再現し、Phase 3（Long-Context）はスキップ、Phase 4〜5 は Cosmos Reason Mini 側で実施する:

| 観点 | Qwen2.5-VL-7B | Qwen2.5-VL Mini | 備考 |
|---|---|---|---|
| Vision Encoder | 独自 ViT（CLIP 事前学習、675M） | DINOv2 ViT-B/14（自己教師あり、86M） | 下記 §1.1 参照 |
| Merger/Adapter | 2層 MLP + 隣接4パッチグループ化 | 2層 MLP（初期）→ トークン圧縮（改善） | 下記 §3.2 参照 |
| LLM | Qwen2.5-7B | Qwen2.5-0.5B | 同じ decoder-only、同じ Qwen ファミリー |
| Stage 1 | Visual Pre-Training（**ViT のみ**、LLM frozen） | Feature Alignment（**Adapter のみ**、他 frozen） | 同思想、対象が異なる |
| Stage 2 | Multimodal Pre-Training（**全パラメータ**） | Visual Instruction Tuning（**全パラメータ**） | **同方式** |
| Stage 3 | Long-Context Pre-Training | —（スキップ） | Mini では不要 |
| Post-training | SFT + DPO | — | Cosmos Reason Mini で実施 |
| データ規模 | Pre-training 合計: **4.1T トークン** | ~100K〜600K サンプル | 桁違いの差 |

### 1.1 Vision Encoder の事前学習方法の違い

この差異はアーキテクチャ上最も重要な点:

| | Qwen2.5-VL | Qwen2.5-VL Mini |
|---|---|---|
| ViT | 独自設計 ViT（Window Attention、2D-RoPE） | DINOv2 ViT-B/14（標準 ViT） |
| 事前学習 | **CLIP**（テキストとのコントラスト学習） | **DINO v2**（自己教師あり、テキストなし） |
| テキスト対応付け | あり（CLIP で学習済み） | **なし** |

Qwen2.5-VL の ViT は CLIP 事前学習済みなので、視覚特徴が既にテキスト空間と部分的に対応している。一方 DINOv2 はテキストとの対応が一切ないため、**Adapter が橋渡しすべきギャップが大きい**。

この差異を補うため:
- Stage 1（Feature Alignment）で Adapter にギャップを橋渡しさせる
- Stage 2 で DINOv2 を含む全パラメータを解凍し、end-to-end で適応させる（Qwen2.5-VL と同方式）

### 1.2 DINOv2 を frozen にするか解凍するか

Qwen2.5-VL は Phase 2 以降で ViT を全解凍する。Mini でも同方式を採用し **Stage 2 で DINOv2 を解凍**する。

**DINOv2 を frozen にする場合（LLaVA-1.5 方式）**:
- 利点: 汎用視覚特徴を壊すリスクがない。VRAM 節約
- 欠点: DINOv2 はテキスト対応がないため、Adapter だけでギャップを埋める必要がある。Qwen2.5-VL / Qwen3-VL の方式と異なる

**DINOv2 を Stage 2 で解凍する場合（Qwen2.5-VL 方式、本設計で採用）**:
- 利点: fine-tune でテキストと相性の良い特徴に適応できる。本家と同方式。DINOv2 ViT-B/14 は 86M なので VRAM への影響は限定的（+勾配 ~172MB、+オプティマイザ ~688MB）
- 欠点: fine-tune で特徴が壊れるリスクがある。データが ~600K と少ないので過学習の可能性
- 対策: Stage 2 の学習率を小さく保つ（2e-5）。DINOv2 にさらに小さい学習率を設定する layer-wise lr decay も検討可能

### 1.3 MRoPE（位置エンコーディング）の違い

MRoPE（Multimodal Rotary Position Embedding）は **LLM 側**の位置エンコーディング。ViT ではなく LLM が visual tokens を受け取ったときに「このトークンは画像のどの位置から来たか」を伝える仕組み。

| | Qwen2.5-VL | Qwen2.5-VL Mini |
|---|---|---|
| LLM の位置エンコーディング | **MRoPE**（3 成分: temporal, height, width） | **標準 1D RoPE**（Qwen2.5-0.5B のデフォルト） |
| 画像パッチの空間位置 | LLM が位置エンコーディングで認識 | LLM は位置エンコーディングでは認識しない |

Qwen2.5-VL の MRoPE は通常の 1D RoPE を 3 成分に分解する:
- **temporal（時間）**: 動画のフレーム時刻
- **height（高さ）**: 画像パッチの y 座標
- **width（幅）**: 画像パッチの x 座標

テキストトークンの場合は 3 成分すべて同じ値にして通常の 1D RoPE と等価になる。

Qwen2.5-0.5B は標準の 1D RoPE のみなので、画像パッチの空間位置情報は位置エンコーディングでは LLM に伝わらない。ただし **DINOv2 の出力特徴自体にパッチの空間情報が暗黙的に含まれている**（パッチ順序が空間的に並んでいる + DINOv2 が空間関係を学習済み）ため、致命的ではない。

> **補足**: Qwen2.5-0.5B は Qwen2.5-VL-7B と同じ Qwen ファミリーだが、VL 版（MRoPE 対応）は 3B 以上のみ。0.5B のテキスト版は標準 RoPE のため、MRoPE の導入にはアーキテクチャ変更が必要。本設計では見送る。

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| Vision Encoder | DINOv2 ViT-B/14（86M params） |
| Adapter | MLP（初期）→ トークン圧縮方式（改善） |
| LLM | Qwen2.5-0.5B（494M params） |
| 入力 | 画像（224×224） |
| 出力 | テキスト（キャプション、VQA 回答） |

---

## 3. アーキテクチャ

MiniPamayo / Cosmos Reason Mini と同じアーキテクチャ:

```
┌──────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Image    │────▶│  DINOv2 ViT-B/14 │────▶│   Adapter     │────▶│ Qwen2.5-0.5B │
│ (224×224) │     │  (Vision Enc.)   │     │  (768→896)    │     │  494M (LLM)  │
└──────────┘     └──────────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                               ┌──────▼───────┐
                                                               │ テキスト出力  │
                                                               │ (caption/QA) │
                                                               └──────────────┘
```

### 3.1 Vision Encoder — DINOv2 ViT-B/14

- モデル: `facebook/dinov2-base`（86M params）
- 入力: RGB 224×224
- 出力: パッチ特徴 (256 patches × 768 dim)
- hidden dim 768 は Qwen2.5-0.5B の 896 に近く（1.2倍）、Adapter の射影ギャップが小さい
- **Stage 1 では frozen**、**Stage 2 で解凍**（Qwen2.5-VL と同方式。詳細は §1.2 参照）

### 3.2 Adapter（Vision → LLM Projector）

| 段階 | 方式 | 入力 → 出力 | 備考 |
|---|---|---|---|
| 初期（fail-fast） | 2層 MLP | 256×768 → 256×896 | シンプルな次元変換のみ |
| 改善 | トークン圧縮付き MLP or Cross-Attention | 256×768 → N×896（N<256） | トークン数を削減 |

**Qwen2.5-VL の Merger との対応**:

Qwen2.5-VL は **隣接 4 パッチをグループ化 → 結合 → 2 層 MLP** で次元変換とトークン圧縮を同時に行う。これにより画像トークン数が 4 分の 1 に削減される。

初期実装ではまず 2 層 MLP で次元変換のみ行い動作確認する。ただし **256 トークンは多い**ため、トークン圧縮の導入は早期に検討すべき:

| 圧縮方式 | トークン数 | 方式 | 備考 |
|---|---|---|---|
| なし（初期） | 256 | MLP のみ | 最もシンプル、動作確認用 |
| 隣接パッチグループ化 | 64 | Qwen2.5-VL 方式（4パッチ→1） | 実装が容易 |
| Cross-Attention Pooling | 16 | learnable query で圧縮 | 最も圧縮率が高い |

### 3.3 Language Model — Qwen2.5-0.5B

- モデル: `Qwen/Qwen2.5-0.5B`（494M params）
- hidden_dim: 896, num_layers: 24, num_attention_heads: 14, num_kv_heads: 2（GQA）, vocab_size: 151,646
- Alpamayo 0.5B と**同一の LLM**
- GQA（2 KV heads）により KV cache が効率的
- **Stage 1 では frozen**。テキスト LLM としての能力を保持したまま Adapter だけ学習する

---

## 4. 学習パイプライン

Qwen2.5-VL の 5 段階学習を簡略化した **2 段階構成**:

```
Stage 1: Feature Alignment（特徴アライメント）
    Vision Encoder: frozen
    Adapter: trainable  ← ここだけ学習
    LLM: frozen
    データ: 画像キャプションペア
    ↓
Stage 2: Visual Instruction Tuning（視覚指示調整）
    Vision Encoder: trainable  ← 解凍
    Adapter: trainable
    LLM: trainable  ← 解凍
    データ: 視覚 QA・会話データ
```

**Qwen2.5-VL との対応**:
- Stage 1 → Qwen2.5-VL Phase 1（Visual Pre-Training）に対応。Qwen2.5-VL では ViT を学習するが、Mini では DINOv2 は frozen のまま Adapter のみ学習する
- Stage 2 → Qwen2.5-VL Phase 2（Multimodal Pre-Training）に対応。**全パラメータ解凍で同方式**
- Qwen2.5-VL Phase 3（Long-Context）→ スキップ（Mini では不要）
- Qwen2.5-VL Phase 4〜5（SFT + DPO）→ Cosmos Reason Mini 側で実施

### 4.1 Stage 1: Feature Alignment（特徴アライメント）

#### 目的

Adapter が DINOv2 の視覚特徴を Qwen2.5-0.5B の入力空間に正しくマッピングすることを学ぶ。

#### Qwen2.5-VL での対応

Qwen2.5-VL の Phase 1（Visual Pre-Training）:
- **ViT のみ trainable**、LLM は frozen
- 1.5T トークン（Image Caption, Knowledge, OCR データ）
- 目的: 視覚空間と言語空間のアライメント基盤を構築

Mini との違い: Qwen2.5-VL は独自 ViT を CLIP 初期化から fine-tune するため ViT が trainable。Mini では DINOv2 ViT-B/14 が既に高品質な視覚特徴を持つので frozen とし、Adapter のみ学習する。

#### Qwen2.5-VL Mini での実装

- **Adapter のみ trainable**（Vision Encoder + LLM は frozen）
- LLM を frozen にすることで、テキスト生成能力を壊さずに Adapter の学習に集中できる
- データ: 画像キャプションペア

#### データ

| データセット | サンプル数 | 内容 | 備考 |
|---|---|---|---|
| LLaVA-CC3M-Pretrain-595K | 595K | CC3M のフィルタリング済みキャプション | LLaVA-1.5 の Stage 1 データ |
| ShareGPT4V-PT | 1.2M | 高品質キャプション | より高品質だがサイズ大 |

**推奨**: LLaVA-CC3M-Pretrain-595K から開始。十分小さい（~数 GB）ので RTX 4090 で高速に処理可能。

#### 学習設定

| 項目 | LLaVA-1.5 | Qwen2.5-VL Mini |
|---|---|---|
| trainable | Projector のみ | Adapter のみ |
| frozen | ViT + LLM | DINOv2 + Qwen2.5-0.5B |
| データ | CC3M-595K | CC3M-595K（同じ） |
| 学習率 | 1e-3 | 1e-3 |
| バッチサイズ | 256 | micro-batch=4, grad_accum=4（≈16） |
| エポック | 1 | 1 |
| 精度 | bf16 | bf16 |

Adapter のみの学習なのでパラメータが少なく（~1M）、VRAM 消費は最小。micro-batch を大きめにできる。

#### 入出力

```
入力:  [visual_tokens] + "Describe the image briefly."
出力:  [キャプションテキスト]
Loss:  cross-entropy（next-token prediction、キャプション部分のみ）
```

### 4.2 Stage 2: Visual Instruction Tuning（視覚指示調整）

#### 目的

画像について質問に答えたり、詳細に説明したりする VLM としての能力を獲得する。

#### Qwen2.5-VL での対応

Qwen2.5-VL の Phase 2（Multimodal Pre-Training）:
- **全パラメータ trainable**（ViT + Merger + LLM）
- 2T トークン（Pure text, Interleaved, VQA, Video, Grounding, Agent 等の多タスクデータ）
- 目的: 視覚と言語の深い接続を構築

#### Qwen2.5-VL Mini での実装

- **DINOv2 + Adapter + LLM を全解凍**（Qwen2.5-VL Phase 2 と同方式）
- DINOv2 を解凍することで、テキスト対応のない視覚特徴をタスクに適応させる
- DINOv2 の特徴崩壊を防ぐため、学習率を小さく保つ（詳細は §1.2 参照）
- データ: 視覚 QA、画像会話、詳細キャプション

#### データ

| データセット | サンプル数 | 内容 | 備考 |
|---|---|---|---|
| LLaVA-v1.5-mix665k | 665K | VQA, OCR, 会話等の混合 | LLaVA-1.5 の Stage 2 データ |
| LLaVA-Instruct-150K | 150K | GPT-4 生成の視覚会話 | よりコンパクト |
| ShareGPT4V | 100K | 高品質な詳細記述 | 品質重視 |

**推奨**: LLaVA-Instruct-150K から開始。小規模で品質が高い。

#### 学習設定

| 項目 | LLaVA-1.5 | Qwen2.5-VL Mini |
|---|---|---|
| trainable | Projector + LLM | **DINOv2 + Adapter + Qwen2.5-0.5B（全解凍）** |
| frozen | ViT | — |
| データ | LLaVA-mix665k | LLaVA-Instruct-150K |
| 学習率 | 2e-5 | 2e-5 |
| バッチサイズ | 128 | micro-batch=1, grad_accum=16 |
| エポック | 1 | 1 |
| 精度 | bf16 | bf16 |
| gradient checkpointing | — | ON（LLM） |

#### 入出力

```
入力:  [visual_tokens] + "What is happening in this image?"
出力:  [詳細な回答テキスト]
Loss:  cross-entropy（next-token prediction、回答部分のみ）
```

---

## 5. 入出力仕様

### 5.1 入力

| 入力 | 形状 | 備考 |
|---|---|---|
| 画像 | RGB 224×224 | |
| テキスト指示 | 自然言語 | キャプション要求、質問等 |

### 5.2 出力

| Stage | 出力 | 備考 |
|---|---|---|
| Stage 1 | 画像キャプション | 短い記述文 |
| Stage 2 | VQA 回答 / 詳細記述 | 質問に応じた回答 |

---

## 6. 評価

### 6.1 定性的評価

- 画像を入力して生成テキストの妥当性を目視確認
- 「画像の内容を説明してください」→ 内容に即した記述が出るか
- 「画像に何人いますか」→ 正しい回答が出るか
- Stage 1 前後、Stage 2 前後で同じ画像に対する出力を比較

### 6.2 定量的評価

~582M の小規模 VLM に適したベンチマークを選定。SOTA を目指すのではなく、「VLM として機能するか」の確認が目的。

#### 評価フレームワーク

**lmms-eval**（推奨）: HuggingFace 公式の VLM 評価フレームワーク。60 以上のベンチマークに対応し、CLI でカスタムモデルの評価が容易。

#### ベンチマーク優先度

| 優先度 | ベンチマーク | タスク | 指標 | 備考 |
|---|---|---|---|---|
| **Tier 1（必須）** | POPE | 物体存在判定（Yes/No） | Accuracy | 小規模 LLM でも評価しやすい形式 |
| Tier 1 | ScienceQA-IMG | 科学問題（多肢選択） | Accuracy | 選択式で評価が安定 |
| **Tier 2（推奨）** | VQAv2 | 一般的な画像 QA | VQA Accuracy | VLM の標準ベンチマーク |
| Tier 2 | GQA | シーン理解 QA | Accuracy | VQAv2 と相補的 |
| **Tier 3（余裕あれば）** | TextVQA | 画像内テキスト読解 | VQA Accuracy | OCR 能力（Mini では対象外） |
| Tier 3 | MME | 知覚 + 認知 14 サブタスク | スコア合計 | 総合評価だが実行コスト高 |
| Tier 3 | MMBench | 総合能力（多肢選択） | Accuracy | 総合評価 |

#### 参考スコア（SmolVLM-256M）

同規模の SmolVLM-256M のスコアを参考値として記載:

| ベンチマーク | SmolVLM-256M |
|---|---|
| ScienceQA-IMG | 73.8% |
| TextVQA | 50.2% |
| POPE | ~75% |

Mini の目的は技術理解であり SOTA ではないが、ランダム回答を大きく上回ることを最低限の目標とする。

---

## 7. VRAM 見積もり

### Stage 1（Adapter のみ学習）

| コンポーネント | サイズ | 備考 |
|---|---|---|
| DINOv2 ViT-B/14（frozen、推論のみ） | ~172 MB | 勾配・オプティマイザ不要 |
| Qwen2.5-0.5B（frozen、推論のみ） | ~988 MB | 勾配・オプティマイザ不要 |
| Adapter（trainable） | ~4 MB | |
| Adapter オプティマイザ | ~16 MB | 小さい |
| Activation | ~1-2 GB | |
| **合計** | **~2-3 GB** | **非常に軽量** |

### Stage 2（DINOv2 + Adapter + Qwen2.5-0.5B 全解凍）

bf16 学習時の固定コスト: **N × 12 bytes**（パラメータ 2B + AdamW 1st moment 4B + 2nd moment 4B + 勾配 2B）

| コンポーネント | サイズ | 備考 |
|---|---|---|
| 全パラメータ (582M × 12 bytes) | ~6.98 GB | DINOv2 86M + Adapter ~2M + Qwen2.5-0.5B 494M |
| Activation（checkpointing ON） | ~2-3 GB | |
| **合計** | **~10 GB** | |

**結論**: 両 Stage とも RTX 4090 (24 GB) で余裕あり。Stage 1 は特に軽量。

---

## 8. 全体パイプラインにおける位置付け

```
Qwen2.5-VL Mini（本設計書）     Cosmos Reason Mini         MiniPamayo
┌───────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│ Stage 1: Alignment    │     │ Physical AI SFT     │     │ Stage 0: 回帰        │
│   Adapter のみ学習     │────▶│  運転シーン理解 QA   │────▶│ Stage 1: 離散化       │
│ Stage 2: Instruction  │     │ Physical AI RL      │     │ Stage 2: Flow        │
│   全パラメータ解凍     │     │  MCQ + GRPO         │     │ Stage 3: CoC SFT     │
└───────────────────────┘     └─────────────────────┘     │ Stage 4: RL          │
                                                          └──────────────────────┘
汎用 VLM 構築               運転ドメイン特化             行動予測
```

- Qwen2.5-VL Mini で「画像→テキスト」の基礎 VLM 能力を獲得
- その重み（DINOv2 + Adapter + Qwen2.5-0.5B）を Cosmos Reason Mini に引き継ぎ
- Cosmos Reason Mini で運転ドメインに特化した Physical AI SFT + RL を実施
- その重みを MiniPamayo Stage 0 に引き継ぎ

---

## 9. 補足: Qwen3-VL からの知見

Qwen3-VL（Qwen2.5-VL の後継）の論文から、本設計に関連する知見をまとめる。設計の根拠補強として参照。

### 9.1 S0（Merger のみ学習）フェーズの導入

Qwen3-VL では Pre-training が **4 段階**に拡張され、最初の S0 で **Merger のみ学習（ViT + LLM は frozen）** するフェーズが明示的に設けられた:

| Stage | 学習対象 | トークン数 | 内容 |
|---|---|---|---|
| S0 | **Merger のみ** | ~67B | Vision-Language Alignment |
| S1 | 全パラメータ | ~1T | Multimodal Pre-Training |
| S2 | 全パラメータ | ~1T | Long-Context Pre-Training |
| S3 | 全パラメータ | ~100B | Ultra-Long-Context Adaptation |

Qwen2.5-VL Mini の Stage 1（Adapter のみ学習）は、Qwen3-VL の S0 と**完全に同じ思想**。最新の VLM 設計でもこのアプローチが採用されていることの裏付け。

### 9.2 Vision Encoder の変遷

| | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| ViT | 独自設計（CLIP 事前学習） | **SigLIP-2**（SigLIP2-SO-400M）を継続学習 |
| 変更理由 | — | 既存の高品質 ViT を活用し、継続学習で適応 |

Qwen3-VL が既製の SigLIP-2 を fine-tune する方式に移行したことは、「高品質な事前学習済み ViT を frozen or 軽微な fine-tune で使う」という Mini のアプローチとも方向性が近い。

### 9.3 その他の参考知見

- **Square-root-normalized per-token loss**: Qwen3-VL で導入。テキストデータとマルチモーダルデータの loss 寄与をバランス調整。Mini で Pure text データを混ぜる場合に参考になる
- **DeepStack**: ViT の複数層から特徴を抽出し LLM の複数層に残差接続で注入。Mini では不要だが、Adapter の改善方向として将来的に参考になり得る
- **Interleaved MRoPE**: MRoPE の周波数成分を均一分散。位置エンコーディングの改善として参考
