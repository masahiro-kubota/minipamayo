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

---

## 10. 小規模 VLM 論文からの知見（設計への反映）

以下は Qwen2.5-VL Mini の設計に直接影響する知見を、出典論文とともに記載する。論文 PDF は [paper/](paper/) ディレクトリに格納。

### 10.1 Adapter 設計: MLP > Resampler

**TinyLLaVA [arXiv:2402.14289] Figure 6**: CLIP ViT + TinyLlama + LLaVA-1.5 データで MLP connector と Resampler connector を比較。**MLP が Resampler を全体的な性能で上回る**。全後続実験で MLP を採用。具体的な数値差は図中のバーチャートのみで、テーブルでの記載はない。

**COMM [arXiv:2310.08825] Table 6**: DINOv2 ViT-L を VLM のビジョンエンコーダとして使う場合の射影層アブレーション。**2 層 MLP が最適**で、1 層 Linear に対して RefCOCO+ test-A で **+15.7pt**（77.5 vs 61.8）、RefCOCO val で **+11.8pt**（74.7 vs 62.9）の大差。4 層以上にすると急激に性能が崩壊する（4 層で -23.8pt、8 層でほぼ壊滅）。「Ratio 4」は隠れ層次元が入力の 4 倍を意味する。

**COMM [arXiv:2310.08825] Table 6（Ratio アブレーション）**: 2 層 MLP の隠れ層次元は **Ratio 4（入力の 4 倍）が最適**。Ratio 8 (77.4) や Ratio 16 (76.2) でも Ratio 4 の 1 層 MLP (75.3) と同等以上だが、2 層 MLP + Ratio 4 (77.5) が最高。

**SmolVLM [arXiv:2504.05299] Idefics3 ソースコード**: Pixel Shuffle 導入時の MLP は **`nn.Linear(input, output, bias=False)` の単一 Linear 層**。Pixel Shuffle で次元を拡張した後は非線形 MLP が不要になる。

**Idefics2 [arXiv:2405.02246] Table 11, Appendix A.1.3**: Perceiver Resampler 以外の pooling 戦略（simple linear layer, Mapping Network）は Perceiver Resampler に大幅に劣る。ただし Perceiver Resampler は追加パラメータが大きいため（3 層）、~600M の小型モデルでは軽量な MLP の方が適切。

**Idefics2 [arXiv:2405.02246] Appendix A.1.3**: Perceiver Resampler のレイヤー数は **3 層**が最適。レイヤー数増加は統計的に有意な改善なし。

**設計への反映**: §3.2 の初期実装は `Linear(768, 3072) → GELU → Linear(3072, 896)` とする（COMM Table 6: Ratio 4 が最適）。**Pixel Shuffle を導入する場合は、Pixel Shuffle + 1 層 Linear（バイアスなし）で十分**（SmolVLM 実装）。Perceiver Resampler は高い圧縮性能を持つが、~600M モデルには追加パラメータが重いため MLP を優先する。

### 10.2 トークン圧縮: Pixel Shuffle r=4 が有力

**SmolVLM [arXiv:2504.05299] Figure 3 右**: ~500M モデルでは **Pixel Shuffle r=4（16 倍圧縮）** が最適。大型モデル（~2B）の r=3 よりも積極的な圧縮が有効。SmolVLM-500M の実装（HuggingFace config.json + Idefics3 ソースコードから確認）:
- SigLIP-B/16 出力: 32×32 = 1,024 トークン × 768 次元
- Pixel Shuffle r=4: `(B, 1024, 768)` → reshape `(B, 32, 32, 768)` → `(B, 8, 8, 768×16)` = `(B, 64, 12288)`
- MLP 射影: `nn.Linear(12288, 960, bias=False)` — **単一 Linear 層（バイアスなし）**
- 最終出力: 64 トークン × 960 次元

**Idefics2 [arXiv:2405.02246] Table 11**: Fully autoregressive + frozen バックボーンにおける Vision-Language Connector 比較。Perceiver（729→64 トークン）が Linear Projection (44.5) や Mapping Network (51.8) に対して **60.3**（+8.5pt）。Table 4 では 64 トークン (71.7) が 128 トークン (71.2) と同等以上。

**設計への反映**: §3.2 のトークン圧縮候補。**DINOv2 ViT-B/14 は 16×16 = 256 トークン**なので r=4 だと 1×1 = 1 トークンになってしまう点に注意。r=2 で 4×4 = 16 トークンが現実的:

| 方式 | トークン数 | 実装 | 備考 |
|---|---|---|---|
| なし（初期） | 256 | MLP のみ | 動作確認用 |
| 隣接パッチグループ化 | 64 | Qwen2.5-VL 方式（4 パッチ→1） | r=2 相当 |
| **Pixel Shuffle r=2** | **64** | Space-to-Depth + Linear | SmolVLM 方式を 256 パッチに適用 |
| Pixel Shuffle（4 パッチグループ + r=2） | **16** | グループ化 → Space-to-Depth + Linear | 最も積極的な圧縮 |
| Cross-Attention Pooling | 任意 | learnable query | 柔軟だが学習が必要 |

> **注意**: SmolVLM-500M は SigLIP-B/16（512×512、32×32 = 1,024 パッチ）に r=4 を適用して 64 トークンに圧縮している。DINOv2 ViT-B/14 は 224×224、16×16 = 256 パッチなので、同じ 64 トークンを得るには r=2 で十分。

**SmolVLM [arXiv:2504.05299] Finding 2**: 小規模 LM（135M, 360M）は**コンテキスト長 8k トークンを超えると学習が不安定**になる。1.7B LM は 16k まで安定。Qwen2.5-0.5B（494M）は中間に位置するため、ビジョントークン + テキストの合計が 8k を超えないようトークン圧縮が重要。256 ビジョントークン + テキストでは上限に余裕があるが、将来的にマルチ画像や長いテキスト指示を扱う場合にリスクとなる。

**Idefics2 [arXiv:2405.02246] Table 9**: instruction-tuned Idefics2（8B）はわずか **64 視覚トークンで LLaVA-NeXT 13B、DeepSeek-VL 7B、MM1-Chat 7B を凌駕**（MMMU 43.5, MathVista 51.6, TextVQA 70.4, MMBench 76.8）。320 トークンにしても大差なし（TextVQA +2.6pt のみ）。**64 トークンへの圧縮は性能を損なわないどころか有利**という強いエビデンス。

**Idefics2 [arXiv:2405.02246] Table 3**: LoRA 適用時、fully autoregressive (69.5) が cross-attention (67.3) を **+2.2pt 逆転**。Cross-Attention Pooling を Adapter として使う場合、LoRA との相性に注意が必要。

**Idefics2 [arXiv:2405.02246] Table 5, Finding 5**: **アスペクト比保持**が有効。正方形リサイズ (768px) とアスペクト比保持 (378-768px) で性能はほぼ同等（73.1 vs 72.1）だが、アスペクト比保持は GPU メモリ節約に寄与。

**Idefics2 [arXiv:2405.02246] Table 9, Finding 6**: **Image splitting（サブ画像分割）** が特に OCR タスクで効果的。画像を 4 つのクロップ + オリジナルに分割し、各画像 64 トークン（計 320 トークン）として入力。TextVQA と DocVQA で特に大きな改善。50% のサンプルにのみ適用しても 100% と同等の効果。

**LLaVA-1.5 [arXiv:2310.03744] §3.4, Figure 2**: 高解像度対応は**グリッド分割 + 独立エンコード + グローバルコンテキスト連結**方式で実現。グローバルコンテキストの連結が重要（+0.9 GQA, +71 MME, +3.2 MM-Vet）。

**LLaVA-1.5 [arXiv:2310.03744] §5.2**: **高解像度化でハルシネーションが減少**。入力解像度が低いと訓練データの詳細を識別できず、モデルがハルシネーションを「学習」してしまう。

### 10.3 Vision Encoder と LLM のバランス

**SmolVLM [arXiv:2504.05299] Figure 3 左**: 428M エンコーダ (SigLIP-SO400M) + 135M LM (SmolLM2-135M) で性能が著しく低下。360M LM に 428M エンコーダを使うとパラメータ 66% 増加に対し +11.6% の改善のみ → 小さいエンコーダ (93M SigLIP-B/16) の方が効率的。

**Idefics2 [arXiv:2405.02246] Table 1, 2**: LLM を LLaMA-1-7B → Mistral-7B に変更で **+5.1pt**。Vision Encoder を CLIP-ViT-H → SigLIP-SO400M に変更で **+3.3pt**。さらに EVA-CLIP-5B (4.4B) と SigLIP-SO400M (400M) の差はわずか 0.5pt で、パラメータ 11 倍でも Vision Encoder 側の改善は限定的。

**Imp [arXiv:2405.12107] Table 1, §1.1**: **LLM の品質がビジュアルエンコーダより重要**。同じ 2.7B スケールで Phi-2 は MobileLLAMA より平均スコア 68.6 vs 65.1 と大幅に優れる。LLM の事前学習データの質が VLM 性能に直結する。

**LLaVA-Phi [arXiv:2401.02330] §4, Figure 3**: **基盤 LLM の事前訓練分野が特定タスクの成功/失敗を決定的に左右する**。Phi-2（コード生成訓練）は数学 OCR で正確な計算が可能だが、LLaVA-1.5-13B は数字と数学記号の認識に失敗。**Qwen2.5-0.5B は多言語・コード・数学を含むバランスの取れた事前訓練**が行われており、幅広いタスクへの適応が期待できる。

**Imp [arXiv:2405.12107] Table 3**: Imp-2B（**Qwen-1.5 1.8B** + SigLIP-SO400M）のベンチマーク。特に **MMBCN（中国語ベンチマーク）63.8** で他の LLM（Phi-2: 49.4、MobileLLaMA: 27.1）を大幅に上回る。多言語 LLM の能力がそのまま VLM に引き継がれる証拠。

**設計への反映**: DINOv2 ViT-B/14（86M）+ Qwen2.5-0.5B（494M）のエンコーダ比率 ~15% は、SmolVLM-500M（SigLIP-B 93M + SmolLM2 360M、~21%）と同等で適切なバランス。Qwen2.5-0.5B は同ファミリーの Qwen-1.5 1.8B の多言語能力を縮小版で受け継いでおり、特にアジア言語タスクでの優位性が期待される。

### 10.4 DINOv2 のフローズン戦略と訓練安定性

**TinyLLaVA [arXiv:2402.14289] Table A1**: 「Share recipe」（前半 12 層凍結、後半のみ解凍、lr=2e-5）の効果。TextVQA で +2.6~+3.5pt 改善する一方、**POPE は全モデルで低下**（Sig-TL -1.0pt、Sig-Phi -0.7pt）。ViT 解凍はハルシネーション増加のリスクがある。

**Idefics2 [arXiv:2405.02246] Table 3**: Fully autoregressive + frozen (60.3) → LoRA (69.5) で **+9.2pt**（論文本文では "+12.9 points increase" と記述、実験条件の差異あり）。完全 unfreeze は訓練発散を引き起こし安定化不可能。LoRA の具体的な rank 等のハイパーパラメータは論文中に未開示。最終モデルでは DoRA（LoRA の変種）を使用。

**Imp [arXiv:2405.12107] Table 1 §2.1**: LoRA rank のアブレーション。Full-parameter FT (71.2) < LoRA rank=128 (71.4) < **rank=256 (71.6)** > rank=512 (71.5)。小規模 LLM では完全微調整よりも LoRA の方が過学習を防ぎ、メモリ効率も高い。

**COMM [arXiv:2310.08825] 訓練設定**: COMM は **DINOv2 を frozen にしたまま** LLM + alignment 層 + MFM モジュールのみを学習し、高い性能を達成（Table 2: RefCOCO val 91.73, test-A 94.06 等）。DINOv2 を解凍せずとも十分な VLM 性能が得られる可能性を示唆。ただし COMM は 7B LLM を使用しており、0.5B の Mini では LLM 容量が限られるため frozen DINOv2 だけでは不足する可能性もある。

**Imp [arXiv:2405.12107] Table 1 §2.1（LoRA vs Full FT ベンチマーク別）**: LoRA rank=256 は平均で Full FT を +0.4pt 上回るが、**MMBench では -0.5pt 劣る**（唯一の劣位指標）。LoRA 適用時は MMBench 系タスクの性能に注意。

**Cambrian-1 [arXiv:2406.16860] Finding 4, §3.3**: 「ビジョンエンコーダのアンフリーズは広く有益であり、特に **SSL モデル（DINOv2）は vision-centric ベンチマークで特に恩恵**を受ける」。DINOv2 ViT-L/14@336 の Frozen → Unfrozen: Average +4.88, Vision-Centric **+11.47**。解凍時のビジョンエンコーダ lr=**1e-5**。ただし訓練速度は 50-55% 低下。

**TinyLLaVA [arXiv:2402.14289] §4.2.2**: **小型 LLM では ViT の fine-tuning が有効**（大型 LLM とは逆パターン）。大型 LLM では ViT fine-tuning が性能を劣化させるとの報告があるが、小型 LLM では逆に改善する。ただし**訓練パラメータ増加がハルシネーション増加のリスク**と直結（POPE 低下）。

**Eagle [arXiv:2408.15998] Table 3**: DINOv2 ViT-L/14-Reg@448 の Frozen avg=520.7 → Unfrozen avg=**537.3 (+16.6)**。解凍の効果を独立して確認。

**設計への反映**: §1.2 の Stage 2 で DINOv2 を全解凍する方針は維持するが、訓練が不安定な場合の段階的対策:
1. まず全解凍で試行（DINOv2 lr=**1e-5**、LLM lr=**2e-5**。Cambrian-1 Table 23 準拠）
2. 発散したら DINOv2 の**後半 6 層のみ解凍**（TinyLLaVA Share recipe 準拠。ただし POPE 低下に注意）
3. それでも不安定なら **LoRA rank=256** を LLM に適用（Imp Table 1 §2.1 が根拠。Idefics2 でも LoRA による安定化を確認。MMBench -0.5pt に注意）
4. DINOv2 を frozen にしたまま adapter + LLM のみ学習も選択肢（COMM の知見）

**推奨手順**: まず frozen で設計を固め（データ、Adapter、ハイパーパラメータの検証）、**設計が固まったら unfreeze で最終訓練**（Cambrian-1 方式）。解凍時は 50-55% の訓練速度低下を見込む。

### 10.5 エポック数と過学習

**Imp [arXiv:2405.12107] Table 1 §2.2**: エポック数のアブレーション（SigLIP-SO400M + Phi-2、LoRA rank=256）。

| エポック | 平均スコア | VQAv2 | TextVQA | SQA-I | POPE |
|---|---|---|---|---|---|
| 1 | 71.6 | 79.9 | 57.9 | 71.0 | 87.8 |
| **2** | **72.1** | 81.2 | 59.4 | 71.2 | 87.8 |
| 3 | 71.7 | 81.5 | 57.7 | 70.0 | 87.5 |

1→2 で +0.5pt、2→3 で **-0.4pt**（TextVQA -1.7、SQA-I -1.2 が主因）。**2 エポックが最適**。Imp は 1 エポックでは「学習不足（undertrained）」と明示的に診断しており、**Stage 2 のデフォルトは 2 エポックとすべき**。

**SmolVLM [arXiv:2504.05299]**: チェックポイントを **25 最適化ステップごとに保存**し、最適点は訓練終了時とは限らないことを前提に設計。

**設計への反映**: §4.1, §4.2 のエポック数は 1 で設計済みだが、2 エポック目を追加する余地がある。ただし 3 エポック以上は避ける。チェックポイントは 25 ステップごとに保存し、POPE + ScienceQA の重み付き平均で最適を選択。

### 10.6 位置トークンの設計

**SmolVLM [arXiv:2504.05299] §2.2 Finding 5**: 文字列ベースの位置トークン（`<row_1_col_2>` 等）は「OCR loss plague」を引き起こし、学習が不安定になる。**学習可能な位置埋め込み**で安定化。

**設計への反映**: DINOv2 のパッチ順序は空間的に並んでおり、§1.3 で述べたように暗黙的な位置情報を持つ。明示的な位置トークンを追加する場合は、文字列ではなく学習可能な埋め込みを使用する。

### 10.7 DINOv2 の特徴特性

**COMM [arXiv:2310.08825] Table 1**: DINOv2 単独（MFM なし）の VLM 性能。グラウンディング (Avg REC 54.8) では CLIP (47.3) を **+7.5pt** 上回るが、VQA (63.1 vs 68.8) やキャプション生成 (Flickr30k CIDEr 68.9 vs 80.7) では CLIP に劣る。MFM 適用後も DINOv2 w/ MFM (72.8) が CLIP w/ MFM (70.0) をグラウンディングで **+2.8pt** 上回る。

**COMM [arXiv:2310.08825] MFM モジュール**: DINOv2 ViT-L（24 層）からは**深い層（19-24 層）のみ**を使用。浅い層はセマンティック情報が不足し、全層の平均 Mean(all) は Mean(19-24) より顕著に劣る（Figure 3、数値はグラフのみ）。DINOv2 ViT-B/14（12 層）に適用する場合は後半 6 層（7-12 層）が対応。

**COMM [arXiv:2310.08825] Table 7**: 画像のみで事前学習された他のモデル（MAE, DeiT）との比較。MAE は DINOv2 に対して RefCOCO+ test-A で -9.4pt、DeiT は -50.0pt と壊滅的。DINOv2 の self-supervised contrastive learning が VLM に有効な特徴を獲得していることの根拠。

**Cambrian-1 [arXiv:2406.16860] Table 2**: 23 種のビジョンエンコーダを体系的評価。DINOv2 は**自己教師あり学習モデルの中で全カテゴリ 1 位**、全体でも第 5 位。言語教師あり（CLIP/SigLIP 系）を除けば最強のエンコーダ。

**Cambrian-1 [arXiv:2406.16860] Figure 17**: DINOv2 ViT-L/14@336 の Frozen → Unfrozen の効果。Average **+4.88pt**、Vision-Centric **+11.47pt**（MMMU +11.11、RealWorldQA +21.99）。**DINOv2 は解凍時の改善幅が特に大きい**。解凍時のビジョンエンコーダ学習率は **1e-5**（メイン LR 2e-5 の半分）。ただし訓練速度は **50-55% 低下** する（Appendix F）。

**Cambrian-1 [arXiv:2406.16860] Figure 7**: DINOv2 はデータ量を増やすことで CLIP とのギャップを縮められる。0M → 0.5M → 1.2M Adapter データで一貫して性能向上。5M instruction tuning データで Unfrozen DINOv2 avg=47.40 を達成。

**Cambrian-1 [arXiv:2406.16860] Table 12**: DINOv2 ViT-L/14@336（Frozen）の弱点の具体値。OCRBench=**3.10**、ChartQA=**16.48**、DocVQA=**11.90**。OCR/テキスト認識は DINOv2 の根本的弱点であり、テキスト系タスクでは CLIP の半分以下。

**DINOv2 [arXiv:2304.07193] Table 4, 17**: DINOv2 ViT-B/14 は **ViT-g/14 からの蒸留モデル**。ImageNet-1k Linear 精度 ViT-B=82.1、ViT-L=84.5、ViT-g=86.5。ViT-B は ViT-g の知識を効率的に保持しつつパラメータを 1/13 に圧縮。

**Eagle [arXiv:2408.15998] Table 5**: Pre-alignment 段階を経ない non-text-aligned encoder は有意に低い性能。DINOv2 のような自己教師あり学習エンコーダには **Pre-alignment（Stage 1）が必須**。

**設計への反映**: DINOv2 はテキスト対応がないため VQA・キャプションでは CLIP に劣るが、空間的な詳細情報に強い。Stage 1 の Feature Alignment で十分な学習を行い、このギャップを Adapter で補う必要がある。DINOv2 はグラウンディング・3D 理解に強みを持つため、自動運転シーンの空間理解には特に適している。解凍時の改善幅が大きいことから、Stage 2 での解凍は特に重要。

### 10.8 データ品質に関する知見

**SmolVLM [arXiv:2504.05299] Figure 7 左**: LLM-SFT テキストデータ（SmolTalk）の再利用は、画像タスクで最大 **-6.5%**、動画タスクで **-3.7%** の性能低下を引き起こす。LLM-SFT データの再利用は避け、新しいテキスト SFT データを使用すべき。

**SmolVLM [arXiv:2504.05299] Figure 7 中央**: CoT（Chain-of-Thought）データは全体の **0.02-0.05%** が最適。高い比率では性能が顕著に劣化、特に画像タスクで悪化。小規模 VLM の限られた容量を CoT データが圧迫するため。

**Idefics2 [arXiv:2405.02246] Table 6**: 合成キャプション (52.9) が人手の alt テキスト (49.8) を **+3.1pt** 上回る。Web 上の alt テキストはノイジーで品質が低い。

**Imp [arXiv:2405.12107] Table 1, Section 3.2**: GPT4V-annotated データ（ShareGPT-4V 20K + LAION-GPT-V 10K + ALLaVA 300K = 計 330K）の追加で平均スコア **71.8 → 73.2 (+1.4pt)**。特にキャプションと会話データの相乗効果が大きい。OCR & chart データ（DVQA, ChartQA, DocVQA, AI2D, InfographicVQA = 計 32K）も TextVQA, ScienceQA を大幅改善。

**TinyLLaVA [arXiv:2402.14289] Figure 7**: Stage 1 Pre-training データとして ShareGPT4V (1,246K) が LLaVA-1.5 (558K) より一貫して良い結果を示す。ただし **TinyLlama (1.1B) は大量データで POPE が顕著に劣化**。パラメータ不足により大量データへの適合が不十分で、ハルシネーション増加が生じる（§4.2.2）。

**Cambrian-1 [arXiv:2406.16860] Figure 7**: DINOv2 はデータ量を増やすことで CLIP とのギャップを縮められる。**DINOv2 は text-alignment がないため、Stage 1 の Alignment データが CLIP 以上に重要**。558K で不足なら 1,246K への増量を検討すべき。

**LLaVA [arXiv:2304.08485] Table 8, Ablation (iii)**: Pre-training をスキップすると精度が **90.92% → 85.81% (-5.11%)**。**DINOv2 はテキスト対応がないため、Pre-training スキップの影響は CLIP 以上に深刻**と予想される。

**LLaVA [arXiv:2304.08485] Table 4**: 3 種類の Instruction-following データ混合が最高性能。Conversation + Detail description + Complex reasoning の全 3 種 = **85.1%**、Detail + Complex のみ = 81.9%、Conversation のみ = 73.8%。**データの多様性が重要**。

**LLaVA-1.5 [arXiv:2310.03744] §5.1, Figure 4**: データの **50% にランダムダウンサンプリングしても 98% 以上の性能を維持**。30% まで減らしても MMBench, ScienceQA, POPE では低下なし。**小規模実験での検証が有効**であることの根拠。

**LLaVA-1.5 [arXiv:2310.03744] §3.2, Table 1b**: VQA データに **Response formatting prompt**（"Answer the question using a single word or phrase"）を付加するだけで、short-answer VQA と long-form 会話の両立が可能。InstructBLIP のような short-answer overfit を回避。

**設計への反映**: Stage 1 の CC3M-595K データは LLaVA [arXiv:2304.08485] でフィルタリング済みのため品質は確保されている。ただし **DINOv2 は text-alignment がないため、558K では不足する可能性がある**。まず 558K で開始し、Alignment が不十分な場合は **ShareGPT4V-PT 1,246K に増量**を検討する（TinyLLaVA Figure 7）。ただし Qwen2.5-0.5B は 494M と小型のため、データ過多によるハルシネーション増加にも注意（TinyLlama の事例）。Stage 2 で追加データを混合する場合は、CoT データの割合を 0.05% 以下に抑え、テキストのみのデータも新規データを使用する。データの多様性（会話 + 詳細記述 + 推論）を確保し、VQA データには formatting prompt を付加する。

### 10.9 プロンプト設計と Loss マスク

**SmolVLM [arXiv:2504.05299] Finding 6**: **システムプロンプト**（例: "You are a visual agent and should provide concise answers."）と**メディアイントロ/アウトロトークン**（例: "Here is an image..." / "Given this image..."）が小規模 VLM の性能を大幅に向上させる。特に動画タスクで顕著。さらに、**SFT 時にはユーザープロンプト部分をマスクし completion 部分のみで学習する**ことで過学習を抑制。

**SmolVLM [arXiv:2504.05299] §3.2**: **ユーザープロンプトマスク（completion のみで loss 計算）** の効果。Multimodal QA では質問が繰り返し的であり、マスクしないとモデルが表面的な繰り返しを学習してしまう。画像タスク・動画タスク両方で一貫した性能向上を確認。

**LLaVA-1.5 [arXiv:2310.03744] §3.2, Table 1b**: **Response formatting prompt** の追加で short-answer VQA と long-form 会話を両立。VQA データに "Answer the question using a single word or phrase" を付加するだけで overfit を回避。

**SmolVLM [arXiv:2504.05299] §3.1, Finding 5**: サブ画像位置を示す**文字列ベースの位置トークン**（`<row_1_col_2>` 等）は「**OCR loss plague**」を引き起こす — 学習初期の loss plateau 後に OCR 性能が一切改善しなくなる現象。**学習可能な位置埋め込み（learned positional tokens）** に切り替えることで学習収束が改善し、OCR 精度・汎化性能が向上。特に 256M/500M の小型モデルで差が顕著。

**設計への反映**:
- §5 の入出力仕様の改善として、Stage 2 でシステムプロンプトとメディアマーカーを含む入力テンプレートを設計する
- Loss マスクはシステムプロンプト + ユーザー質問部分を除外し、アシスタント回答部分のみに適用する（§4.2 と一貫）
- VQA データには formatting prompt を付加して short-answer overfit を回避
- 位置トークンを追加する場合は文字列ではなく学習可能な埋め込みを使用

### 10.11 訓練テクニック: 正則化と最適化

**LLaVA-Phi [arXiv:2401.02330] §3.1**: 小型 VLM (Phi-2 2.7B) では **weight decay = 0.1** を使用。LLaVA / LLaVA-1.5 の weight decay = 0 とは異なる。小型モデルではパラメータが少なく過学習しやすいため、weight decay による正則化がより重要になる。Optimizer は Adam (momentum 0.9, 0.98, epsilon 1e-7)。

**Idefics2 [arXiv:2405.02246] §4.2**: **NEFTune（Noisy Embedding Fine-Tuning）** を Instruction Fine-tuning 時に適用して過学習を防止。入力埋め込みにノイズを注入することで汎化性能を向上。さらに**画像解像度のランダムスケールアップ**と **multi-turn 会話のシャッフル**も過学習対策として併用。

**Idefics2 [arXiv:2405.02246] §4.2**: 最終モデルでは LoRA ではなく **DoRA（Weight-Decomposed Low-Rank Adaptation）** を使用。DoRA は LoRA の変種で、重みを方向と大きさに分解して適応する。

**LLaVA [arXiv:2304.08485] Appendix C**: 全 Stage で **Adam optimizer, cosine learning rate schedule, warmup ratio 3%** を使用。精度は **BF16 + TF32**。

**LLaVA-1.5 [arXiv:2310.03744] Table 9**: MLP projector 使用時の Stage 1 学習率は **1e-3**（Linear 時の 2e-3 から半減）。MLP は Linear より表現力が高いが学習が不安定になりやすいため、学習率を下げる必要がある。

**Cambrian-1 [arXiv:2406.16860] Table 23**: ビジョンエンコーダ解凍時の学習率設定。Adapter 事前訓練 lr=**1e-3**, Instruction Tuning lr=**2e-5**, **ビジョンエンコーダ lr=1e-5**（メイン LR の半分）。

**Imp [arXiv:2405.12107] §5**: 8×A100 GPU (40GB) で 32 時間以内に完了。4-bit 量子化でモデルサイズ 2.3GB、性能低下は軽微（SQA: 72.88 vs 73.03@8bit）。

**設計への反映**:
- Stage 2 の optimizer 設定: **weight decay = 0.1**（LLaVA-Phi の小型モデル向け設定）
- Stage 2 で過学習の兆候が見られたら **NEFTune** を導入（Idefics2 知見）
- DINOv2 解凍時の学習率: **1e-5**（Cambrian-1 Table 23 に準拠、メイン LR 2e-5 の半分）
- MLP projector のため Stage 1 学習率は **1e-3**（LLaVA-1.5 Table 9 に準拠）

### 10.12 Adapter の入力: パッチトークン vs CLS トークン

**LLaVA [arXiv:2304.08485] §4.1**: CLIP ViT の **grid features（全パッチトークン）** を使用。CLS トークンではなくパッチトークン全体を Projector への入力とする。

**LLaVA [arXiv:2304.08485] Table 8, Ablation (i)**: **penultimate layer（最終層の一つ前）** の特徴が最終層より +0.96% 高い（90.92 vs 89.96）。最終層はグローバルで抽象的な性質に集中し、一つ前の層の方が局所的な画像詳細に有用。

**COMM [arXiv:2310.08825] Table 4**: DINOv2 では**深層レイヤーのみ**使用すべき。ViT-L（24 層）で Mean(19-24)=71.7 vs Mean(all)=69.1（**-2.6pt**）。浅層特徴はセマンティック情報が不足。CLIPとは逆のパターン。

**設計への反映**:
- DINOv2 の出力として **CLS トークンではなくパッチトークン全体**（256 個）を Adapter に入力
- DINOv2 ViT-B/14（12 層）では**後半 6 層（7-12 層）の出力**を使用することを検討（ViT-L の 19-24 層に対応）
- penultimate layer の使用も試す価値がある（LLaVA Table 8）

### 10.13 2 段階訓練の具体的設定値（全論文の横断サマリ）

全論文で一貫する 2 段階訓練の設定値をまとめる:

**Stage 1（Feature Alignment / Pre-training）**:

| パラメータ | 推奨値 | 根拠 |
|-----------|--------|------|
| 学習率 | **1e-3**（MLP 使用時） | LLaVA-1.5 Table 9, TinyLLaVA, LLaVA-Phi 共通 |
| バッチサイズ | **256** | LLaVA-1.5 Table 9, LLaVA-Phi §3.1, TinyLLaVA 共通 |
| エポック | **1** | 全 4 論文（LLaVA, LLaVA-1.5, TinyLLaVA, LLaVA-Phi）で一致 |
| 訓練対象 | **Projector のみ** | LLM + Vision Encoder = frozen |
| optimizer | **AdamW** | LLaVA-1.5 Table 9 |
| lr schedule | **cosine decay** | LLaVA Appendix C, LLaVA-1.5 Table 9 |
| warmup ratio | **0.03** | LLaVA-1.5 Table 9 |
| weight decay | **0**（Stage 1 は Adapter のみなので正則化不要） | LLaVA Appendix C |
| DeepSpeed | Stage 2 | LLaVA-1.5 Table 9 |
| 精度 | **bf16** | 全論文共通 |
| データ | **558K** image-caption pairs（最低ライン）| LLaVA-1.5 Table 3 |

**Stage 2（Instruction Tuning / SFT）**:

| パラメータ | 推奨値 | 根拠 |
|-----------|--------|------|
| 学習率 | **2e-5** | 全論文で一致 |
| DINOv2 学習率 | **1e-5**（メイン LR の半分） | Cambrian-1 Table 23 |
| バッチサイズ | **128-256** | LLaVA-1.5: 128, LLaVA-Phi: 256, TinyLLaVA-base: 128 |
| エポック | **2** | Imp Table 1 §2.2: 1ep=71.6, 2ep=72.1, 3ep=71.7（過学習） |
| LLM 更新方法 | **LoRA rank=256** or 全解凍 | Imp Table 1: LoRA > Full FT, Idefics2: unfreeze は発散リスク |
| Vision Encoder | **まず frozen → 最終最適化で unfreeze** | Cambrian-1 Figure 17: +4.88pt avg |
| optimizer | **AdamW** | 全論文共通 |
| lr schedule | **cosine decay** | 全論文共通 |
| warmup ratio | **0.03** | LLaVA-1.5 Table 9 |
| weight decay | **0.1**（小型モデル） | LLaVA-Phi §3.1 |
| 精度 | **bf16** | 全論文共通 |
| Loss マスク | **completion 部分のみ** | SmolVLM §3.2 |
| データ | **665K** 混合（VQA + 会話 + OCR + 詳細記述） | LLaVA-1.5 Table 7 |

### 10.14 小規模 VLM の実績（設計思想の妥当性根拠）

**Imp [arXiv:2405.12107] Table 3**: Imp-3B（Phi-2 + SigLIP-SO400M、2 エポック + LoRA rank=256）が LLaVA-1.5-7B を多数のベンチマークで凌駕。特に MM-Vet +13.1pt (43.3 vs 30.2)、MMB +6.8pt (72.9 vs 66.1)。LLaVA-1.5-13B も VQAv2 で上回る (81.2 vs 80.0)。

**TinyLLaVA [arXiv:2402.14289] Table 3**: TinyLLaVA-3.1B（Phi-2 + SigLIP、Share recipe）が LLaVA-1.5-7B を VQAv2 (79.9 vs 78.5)、TextVQA (59.1 vs 58.2)、MMBench (66.9 vs 64.3) で上回る。

**LLaVA-Phi [arXiv:2401.02330] §3.1**: LLaVA-Phi（CLIP ViT-L/14 + Phi-2 = 3B）は Pre-training **1.5 時間** + Fine-tuning **8 時間** = 合計 **9.5 時間**（8×A100）で完了。小型モデルの訓練は非常に高速。600M モデルならさらに短時間で完了する見込み。

**LLaVA-Phi [arXiv:2401.02330] Table 1**: LLaVA-Phi (3B) は ScienceQA-IMG で **68.4%** を達成し、LLaVA-1.5-7B (66.8) と InstructBLIP (60.5) を上回る。これは Phi-2 がコード生成と数学コーパスで事前訓練されていることに起因。

**設計思想への裏付け**: 3B クラスのモデルが適切な学習戦略（LoRA、2 エポック、高品質データ）で 7B/13B を凌駕できるという事実は、~600M の Qwen2.5-VL Mini で「実用的な VLM を構築する」という設計思想が妥当であることを示す。ただし 3B と 600M の間には大きなギャップがあるため、性能目標は控えめに設定する。訓練時間は非常に短いため、多数の実験イテレーションが可能。

### 10.15 実装上の落とし穴と注意事項

実装時にバグや学習失敗の原因になりうる具体的な注意点をまとめる。

**MoE-LLaVA [arXiv:2401.15947] Appendix A.2**: 小規模モデル（Qwen-1.8B クラス）を **fp16 で学習すると loss が NaN になる**ことがある。fp16 のダイナミックレンジの狭さに起因し、パラメータ数が少ないモデルで発生しやすい。Qwen2.5-0.5B（494M）はさらに小さいため、**bf16 が必須**。bf16 が使えない GPU（V100 等）では fp16 の loss scaling 設定に細心の注意が必要。

**LLaVA [arXiv:2304.08485] Appendix C**: 学習精度は **BF16 + TF32** の両方を有効化。TF32 は matmul 精度設定として `torch.backends.cuda.matmul.allow_tf32 = True` および `torch.backends.cudnn.allow_tf32 = True` で明示的に有効化すべき。

**DINOv2 [arXiv:2304.07193] / dino-meets-text**: `facebook/dinov2-base`（register token なし）の出力は **[CLS] + 256 パッチトークン = 257 トークン**。一方 `facebook/dinov2-base-reg`（register token あり版）は **[CLS] + 4 register + 256 パッチトークン = 261 トークン**。register token は「potential register tokens are discarded as they are not used」（dino-meets-text §3.1）と明記されており、**reg 版使用時は register token を出力から明示的に除外**してから Adapter に入力する必要がある。

**DINOv2 入力前処理**: DINOv2 論文本文には正規化パラメータが明示されていない。公式リポジトリおよび HuggingFace 実装では **ImageNet 正規化**（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）がデフォルト。**誤った正規化は特徴量の品質を大幅に劣化させる**ため、必ず確認すること。

**Idefics2 [arXiv:2405.02246] §3.2, Table 3**: Fully autoregressive アーキテクチャで事前学習済み backbone を**全パラメータ unfreeze すると loss divergence が発生**する。学習率を下げても安定化不可能。LoRA を使うことで安定した学習が可能になる。Stage 2 で全解凍する際は、**まず LoRA で安定することを確認してから full FT を試す**のが安全。

**gradient clipping**: LLaVA 系論文ではいずれも gradient clipping 値を明示記載していない。LLaVA コードベースのデフォルト値が暗黙的に使用されている。Qwen2.5 系のデフォルト **max_grad_norm=1.0** を採用し、特に DINOv2 unfreeze 時の安定化に寄与させることを推奨。

**DINOv2 ViT-B/14 のアーキテクチャ確認**: Appendix B Table 17 より、蒸留版 ViT-B/14 は **standard MLP FFN**（SwiGLU ではない）。Embedding dim=768, Heads=12, Blocks=12, Stochastic depth drop rate=**0**（蒸留版）。推論時に dropout は不要。

**設計への反映**:
- bf16 必須。fp16 は使用禁止
- TF32 を明示的に有効化
- `facebook/dinov2-base`（register token なし）を使用。reg 版を使う場合は register token 除外コードが必要
- ImageNet 正規化を適用。公式リポジトリの transform 設定を確認
- gradient clipping = 1.0 を設定
- Stage 2 の全解凍は loss diverge リスクあり。LoRA → full FT の順で試行

### 10.16 DINOv2 の特徴量の詳細な性質

§10.7 を補完する、DINOv2 の特徴量のより詳細な性質。

**dino-meets-text Table 2**: DINOv2 の画像表現として **[CLS] + avg-pooled パッチの結合が全タスク最適**。[CLS] 単体は classification に強い（IN1K=79.2）が segmentation が壊滅的（ADE=8.3）。avg-pooling 単体は逆。結合（concat）すると IN1K=79.2, COCO=34.7, ADE=18.2 と全タスクで最良。

**dino-meets-text Table 3**: 凍結した DINOv2 の上に **2 つの学習可能な vision transformer block を追加**すると retrieval が大幅向上（COCO: 35.1→42.1, +7pt）。1 block で +6.1、2 blocks で +7.4。DINOv2 の特徴空間をテキスト空間に近づける「vision adapter」として有効。

**DINOv2 [arXiv:2304.07193] §4, Appendix B.1**: KoLeo 正則化（`L_koleo = -(1/n) Σ log(d_{n,i})`、重み 0.1）により、DINOv2 の**特徴空間は一様に広がる**性質を持つ。これは MLP Adapter での射影時に有利（特徴量が潰れにくい）。DINO head と iBOT head は**パラメータ非共有**で、[CLS] トークンとパッチトークンが異なる空間にある。

**DINOv2 [arXiv:2304.07193] §7.5, Figure 9**: パッチ特徴量の **PCA 第 1 成分を閾値処理すると前景/背景が自動分離**される。PCA 第 2-3 成分は物体の「パーツ」（翼、体、頭など）に対応。DINOv2 の特徴は空間的に非常に構造化されており、**パッチトークンの空間的順序を保持し、pooling で潰さない設計が重要**。

**DINOv2 [arXiv:2304.07193] §4, §6.6**: 高解像度適応は **position embedding を双線形補間**して実現。518x518 で追加 10K イテレーションの fine-tuning。224→416 への短期高解像度訓練でも、最初から 416 で訓練した場合とほぼ同等の性能。ViT-B/14 で 518x518 の場合 518/14=37 → 37x37=**1369 トークン**となり小規模 LLM には重すぎる。

**COMM [arXiv:2310.08825] §5**: 入力解像度を 224→**336x336** にアップすると fine-grained perception 能力が向上。336/14=24 → 24x24=**576 パッチトークン**。1369 トークンの 518x518 に比べ現実的。

**DINOv2 [arXiv:2304.07193] Table 16**: 蒸留版 ViT-B/14 は **LayerScale 初期値 1e-5** で訓練。各ブロック出力のスケールが小さいため、特徴量の数値範囲が影響を受ける。

**SAIL Figure 2**: DINOv2 の **k-NN クラスタリング品質**がクロスモーダルアライメント性能と強く相関（Pearson r=0.991）。DINOv2 の k-NN 性能が高いことが、テキストとのアライメント学習を容易にしている理論的根拠。

**SAIL Figure 2 右**: 言語モデル（LLM）の **MTEB スコアとアライメント品質に Pearson r=0.994 の相関**。LLM の言語能力が弱いとアライメントも弱くなるため、Qwen2.5-0.5B の容量制約下では Stage 1 にデータ・ステップを追加投入する必要がある可能性。

**設計への反映**:
- パッチトークンに加え、[CLS] トークンを global context として含める設計を検討。具体的には [CLS] を先頭に prepend して 257 トークンを Adapter に入力
- DINOv2 と Adapter の間に 1-2 層の軽量 Transformer block を追加する「vision adapter」を検討。Stage 1 で学習させる
- 入力解像度は **336x336（576 トークン）** を第一候補に。トークン圧縮（Pixel Shuffle r=2 で 144 トークン、r=3 で 64 トークン）と組み合わせ
- パッチトークンの空間的順序を保持する設計を維持

### 10.17 データ処理の実践的テクニック

学習データの構築・前処理に関する実践的な知見。

**LLaVA [arXiv:2304.08485] Table 11, Appendix E**: Stage 1 の質問プロンプトは**11 種類のバリエーション**からランダムサンプリング。「Describe the image concisely.」「Provide a brief description of the given image.」「Summarize the visual content of the image.」など。同じ意味だが自然言語表現を変えることで、**特定フレーズへの過学習を防止**。

**LLaVA [arXiv:2304.08485] Appendix E**: CC3M フィルタリングは**名詞句ベースの概念カバレッジ**で実施。SpaCy で名詞句抽出 → 頻度 3 未満はスキップ → 残りの名詞句ごとにランダムサンプリング → 595K に絞り込み。概念カバレッジを維持しつつデータ量を削減。

**LLaVA-1.5 [arXiv:2310.03744] Appendix A.2**: 同一画像の QA ペアを**単一マルチターン会話に統合**。画像エンコードの重複を減らし学習効率を向上。

**LLaVA-1.5 [arXiv:2310.03744] Appendix A.2**: **バッチ内モダリティ分離で 25% 高速化**。言語のみ会話は視覚付き会話より長い傾向があり、混在させると padding が増える。各バッチは単一モダリティからのみサンプリング。最終性能に影響なし。

**LLaVA-1.5 [arXiv:2310.03744] Appendix A.2**: ShareGPT データの処理では **2048 トークンを超える会話は分割ではなく切り詰め（truncate）**。分割すると文脈が断裂し学習の質が低下する可能性。

**SmolVLM [arXiv:2504.05299] §3.3**: テキストのみのデータ比率は **14% 上限**。それ以上は negative transfer で画像タスク -6.5%、動画タスク -3.7% の劣化。

**ShareGPT4V [arXiv:2311.12793] Figure 6**: Pre-training 用高品質合成キャプションの**飽和点は ~1000K**。100K→600K で急激な向上、600K→1000K でゆるやか、1000K→1200K でほぼ飽和。558K→1246K の増量計画は妥当な範囲。

**ShareGPT4V [arXiv:2311.12793] Figure 2**: SFT データの**わずか 3.5%（23K）を高品質 ShareGPT4V キャプションに置換するだけで全体性能が一貫して向上**。全体を高品質化する必要はなく、description 系タスクの一部（3-5%）の置換で十分。

**Imp [arXiv:2405.12107] §4.3, Table 1**: **TextCaps (22K) は TextVQA と同じ画像セット**。除去すると TextVQA -4.8pt だが、これが真の zero-shot 性能。zero-shot 評価したい場合は訓練データから TextCaps を必ず除去。

**Cambrian-1 [arXiv:2406.16860] Appendix E.3**: データエンジンの品質基準として **50 語未満のテキストサンプルをフィルタ除外**。シンプルだが効果的。

**Cambrian-1 [arXiv:2406.16860] Table 20**: SFT データの最適バランシング比率（1.35M 実験）: **General 34.5%, Language 27.2%, Science 10.0%, OCR 8.7%, Counting 7.2%, Math 4.5%, Code 4.5%**。OCR 比率を上げすぎると General/Vision-Centric が劣化。

**Cambrian-1 [arXiv:2406.16860] §5.3, Table 18**: **Answer Machine Phenomenon** — VQA ベンチマーク最適化データ（短答 QA 過多）で訓練するとモデルが会話能力を喪失。10 種のフォーマットプロンプトを各データセットに割り当てて短答/会話を明示的に分離して緩和。

**OpenVLA [arXiv:2406.09246] §3.3**: 複数データソース混合時、**データソースごとの loss/accuracy を個別にモニタリング**。学習が進まないデータソースを特定・途中除外する仕組みが重要。OpenVLA では DROID データセットが低 accuracy のまま推移したため、学習の最後 1/3 で除外。

**Idefics2 [arXiv:2405.02246] Appendix A.2.1**: Vision-language データに加え、**text-only instruction data**（OpenHermes-2.5, MetaMathQA, OrcaMath 等）を**全トークンの約 25%** 混合。LLM のテキスト能力保持（catastrophic forgetting 対策）と数学能力向上に寄与。

**設計への反映**:
- Stage 1: 質問プロンプトを 11 種類用意しランダム割り当て
- Stage 2: 同一画像 QA を multi-turn 統合。バッチ内モダリティ分離。テキスト会話は 2048 トークンで truncate
- Stage 2 データ比率: General ~35%, Language/text-only ~15-25%, Science ~10%, OCR ~9%, Counting ~7%, Math+Code ~5%
- テキストデータ比率は 14% 以下。text-only instruction data を 10-15% 混合して catastrophic forgetting を緩和
- SFT データの 3-5% を ShareGPT4V キャプションに置換
- Pre-training データは ~1000K で飽和
- データソースごとの loss を wandb で個別トラッキング。学習が進まないデータは途中除外
- TextCaps は TextVQA 評価用に除去
- 50 語未満のテキストはフィルタ除外

### 10.18 学習テクニック補足

§10.11 を補完する、追加の学習テクニック。

**OpenVLA [arXiv:2406.09246] Table 1**: **Sandwich fine-tuning** — Vision Encoder + token embedding + LLM 最終層のみ解凍、LLM バックボーンは凍結。Full FT (69.7%) に対し Sandwich (62.1%) だが、**LoRA rank=32 (68.2%) で Full FT にほぼ匹敵**。メモリ効率と性能のバランスが良い選択肢。

**MoE-LLaVA [arXiv:2401.15947] Table 5a**: **FFN-only fine-tuning** — Attention 層を凍結し FFN 層のみ学習すると、全パラメータ fine-tune とほぼ同等の性能で**学習時間が 75% に削減**（20h vs 27h）。Stage 2 でメモリ不足の場合の代替策。

**TinyLLaVA [arXiv:2402.14289] §4.1.2**: **Share Recipe の具体設定** — Pre-training 段階で ViT の前半 12 層を凍結し残りを解凍。コネクタは **Base Recipe の事前学習済み重みで初期化**（ゼロからではない）。lr=2e-5, batch=256。ViT-B/14（12 層）に適用するなら**前半 6 層凍結、後半 6 層解凍**が対応。

**TinyLLaVA [arXiv:2402.14289] §4.2.2**: **小型 LLM（~1B）では ViT unfreeze が POPE を改善する可能性**がある。大型 LLM（~3B）では unfreeze で POPE が低下する逆パターン。Qwen2.5-0.5B は最も小さいクラスのため、unfreeze によるハルシネーション低下が期待できる。

**ShareGPT4V [arXiv:2311.12793] Table 6**: 高品質キャプション使用時は **Stage 1 で ViT 後半レイヤーも解凍可能**。24 層中 12 層目から解凍が最良（MME^P=1567.4）。全層解凍より部分解凍のほうが良い。DINOv2 ViT-B/14 では後半 6 層解凍が対応。ただし Stage 1 の設計変更を伴うため慎重に検証。

**LLaVA [arXiv:2304.08485] Table 8**: ScienceQA で **reasoning-first CoT（理由→答え）は answer-first の半分のエポックで同等精度**（6ep vs 12ep で 89.77%）。推論タスクでは「まず理由を述べさせ、次に答えを出させる」フォーマットが収束を加速。

**SAIL Table 1**: アライメント層として **GLU（Gated Linear Unit with ReLU）が Linear/MLP より大幅に優秀**。ImageNet-1K 零ショットで +12.4%（33.2→45.4）、T2I retrieval で +6.4%。Qwen2.5-VL の ViT も SwiGLU FFN を採用しており、Adapter の活性化関数として **SwiGLU/GEGLU への変更**を検討する価値がある。

**OpenVLA [arXiv:2406.09246] §3.4**: 固定学習率 2e-5 が最良で、**学習率 warmup の効果が見られなかった**ケースあり。小規模データセットでは warmup 期間が学習ステップを浪費する可能性。warmup なしの設定も試す価値がある。

**Qwen2.5-VL [qwen2.5-vl.pdf] §2.3.4**: post-training（SFT + DPO）では **ViT パラメータは凍結**。十分に align された ViT は SFT 段階では凍結して LLM 側のみ調整するのが本家の方式。

**Qwen2.5-VL [qwen2.5-vl.pdf] §2.3.3**: 推論タスクに対して **rejection sampling** を使用。中間モデルで回答を生成し、正解と一致するもののみ保持。コードスイッチング・過度な長さ・繰り返しパターンをフィルタリング。CoT データの品質確保に有効。

**Qwen2.5-VL [qwen2.5-vl.pdf] §2.3.2**: SFT データの **2 段階フィルタリング**。Stage 1: ドメイン特化分類モデルで 8 ドメイン 30 サブカテゴリに自動分類。Stage 2: ドメインごとのルールベース + モデルベースフィルタリング。繰り返しパターン除去、不完全レスポンス除去、コードスイッチング除去が重要。

**設計への反映**:
- Stage 2 でメモリ不足時: FFN-only FT（75% 時間削減）または Sandwich FT を検討
- Share Recipe 適用時: Adapter を Stage 1 の学習済み重みで初期化してから ViT 後半 6 層を解凍
- 0.5B LLM では ViT unfreeze が POPE 改善に寄与する可能性。ただし安全策として部分 unfreeze から開始
- 推論タスクのデータでは reasoning-first フォーマットを採用
- Adapter 活性化関数として SwiGLU/GEGLU を検討（GLU 系 > MLP）
- warmup なし設定も探索対象に含める

### 10.19 評価プロトコル

VLM の評価に関する具体的な設定・プロトコル。

**LLaVA-1.5 [arXiv:2310.03744] Appendix A.3**: 評価時は**再現性のため greedy decoding を使用**。beam search や sampling は使わない。

**Idefics2 [arXiv:2405.02246] Appendix A.3**: 全評価は **batch size=1, greedy decoding**。タスク別プロンプト:
- Multi-choice（MMMU, MathVista, MMBench）: `"Question: {question}\nChoices:\nA. {choice_a}\nB. {choice_b}\n...\nAnswer with the letter."`
- Open-ended（TextVQA, DocVQA, VQAv2）: `"Question: {question}\nGive a very brief answer."`

**Idefics2 [arXiv:2405.02246] Appendix A.3**: **ストップワード**設定: `Question`, `User`, `<end_of_utterance>`, `<eos>` で生成停止。Qwen2.5 の場合は `<|im_end|>` や `<|endoftext|>` が対応。

**LLaVA-1.5 [arXiv:2310.03744] Appendix A.2**: データセット別の **response formatting prompt** の具体的使い分け:
- VQAv2, GQA, TextVQA, MME, POPE → "Answer the question using a single word or phrase."
- ScienceQA, MMBench, SEED-Bench → "Answer with the option's letter from the given choices directly."
- VizWiz → "When the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."
- LLaVA-Bench, MM-Vet → フォーマット指示なし（自由回答）

**Idefics2 [arXiv:2405.02246] Appendix A.3**: VQAv2 等の open-ended evaluation は ground truth と完全一致判定のため、「large」vs「big」等の言い換えで不正解。**5pt 程度の差は評価ノイズの範囲**。

**設計への反映**:
- 評価時は greedy decoding、batch size=1
- Qwen2.5 のストップワード `<|im_end|>` を設定
- タスクに応じた formatting prompt を付与（学習時・評価時で一致させる）
- open-ended 評価の 5pt 程度の差はノイズとして解釈

### 10.20 SAIL / dino-meets-text: DINOv2 のテキストアライメント研究

DINOv2 とテキストのアライメントに特化した研究から得られた知見。

**SAIL Table 5**: SAIL でアライメント学習した DINOv2-L を LLaVA-1.5 に統合した結果、**7 タスク中 5 タスクで CLIP-L/14 を上回った**。GQA +1.55、VizWiz +5.88、POPE +0.76、VQAv2 +2.37。ただし TextVQA と MMBench では CLIP が優位。DINOv2 は事前アライメントを施せば CLIP なしでも競争力のある VLM が構築できる。

**SAIL Table 1**: contrastive loss として InfoNCE の代わりに **Sigmoid（SigLIP 形式）を使うと大幅改善**。ImageNet +5.3%、T2I +9.3%、I2T +13.5%。DINOv2 とテキストの事前アライメントを行う場合は SigLIP 形式の Sigmoid Loss を推奨。

**SAIL Table 1**: 各画像に元の短いキャプション + ShareGPT4 生成の高品質合成キャプションを positive pair として追加する **Multi-Pos 手法**で retrieval が向上（COCO T2I +4.0%、I2T +8.7%）。Stage 1 データで同一画像に短いキャプションと長い詳細キャプションの両方を含めることでアライメント品質が向上。

**dino-meets-text Table 4**: データキュレーションは**テキスト側（頻出テキストの確率的除去）とイメージ側（DINOv2 特徴の k-means バランシング）の両方向**から行うと全タスクで性能向上。片方のみより両方で改善（81.4/45.4/20.6 vs 80.9/43.7/20.4 or 80.8/43.9/20.5）。

**設計への反映**: 直接的には Stage 1 の Adapter 学習で補うが、将来的に SAIL 的な事前アライメント（DINOv2 → テキスト空間への射影学習）を Stage 0 として追加する選択肢がある。その場合は SigLIP 形式の Sigmoid Loss + Multi-Pos + 両方向キュレーションが推奨設定。

### 10.21 追加論文からの補足知見

**MoE-LLaVA [arXiv:2401.15947] Table 2, 6**: Stage II のデータを LLaVA-FT 665k から **Hybrid-FT 964k**（SViT-157k + LVIS-220k + LRV-331k + MIMIC-IT-256k を追加）に拡充すると大幅性能向上。LLM から LVLM への変換には十分なデータ量が必要。特に SViT（空間理解）、LRV（ハルシネーション軽減）が小規模モデルで効果的。

**OpenVLA [arXiv:2406.09246] §3.1**: Prismatic VLM のバックボーンは **DINOv2 + SigLIP の二重エンコーダ**で、それぞれの出力を**チャネル方向に結合**。DINOv2 の低レベル空間情報と SigLIP の高レベルセマンティック情報を融合。将来的に SigLIP-Base 等を追加してチャネル結合する拡張が検討可能。

**OpenVLA [arXiv:2406.09246] Table 1**: **Sandwich fine-tuning**（VE + token embedding + 最終層解凍、LLM バックボーン凍結）は Frozen VE (47.0%) → Sandwich (62.1%) → Full FT (69.7%)。LoRA rank=32 (68.2%) でも Full FT にほぼ匹敵。

**ShareGPT4V [arXiv:2311.12793] Figure 8**: 合成キャプション生成時、**画像のデータソースごとに異なる専用プロンプト**を使用。ランドマーク画像には位置情報を促すプロンプト、テキスト画像には OCR 内容への言及を促すプロンプト。全画像に同一プロンプトを使うより品質が大幅向上。

**Qwen2.5-VL [qwen2.5-vl.pdf] §2.1**: Merger の具体構造は**空間的に隣接する 4 パッチをグルーピング → 結合 → 2 層 MLP で射影**。入力チャネル = 4 × vision_dim、出力 = LLM hidden_size。Pixel Shuffle r=2 と概念的に同等。

**Qwen2.5-VL [qwen2.5-vl.pdf] Table 1**: 3B モデルでは **Embedding Tying**（入力 embedding と出力 projection の重み共有）を使用。7B/72B では不使用。Qwen2.5-0.5B は小規模のため Embedding Tying が有効になっている可能性が高い。LoRA 適用時に入力 embedding と lm_head が連動する点に注意。

**Cambrian-1 [arXiv:2406.16860] Table 22**: SVA のアテンション分布分析。DINOv2 は **DocVQA（文書画像）に対する貢献が最低（11.0%）** で、ConvNeXt が 44.5% を占める。一方 GQA（自然画像）では 24.1% と健闘。DINOv2 単体 VLM では文書理解が弱点となることを定量的に確認。

**Eagle [arXiv:2408.15998] Table 4**: 5 つの fusion 戦略の比較で **Channel Concatenation が最高性能**（avg 681.5）。Sequence Append (675.0)、LLaVA-HR (678.7)、Mini-Gemini (672.5)、Deformable Attention (674.3) を上回る。複雑な fusion architecture は不要で、単純な連結で十分。

**COMM [arXiv:2310.08825] §4, Equation (1)**: MFM モジュールの具体構造 — 各レイヤー出力に **LayerNorm → Linear → 学習可能なスカラー重み（α, β）の加重和**。DINOv2 側は深層のみ（ViT-L で layer 19-24）を使用。ViT-B/14 に適用する場合は **layer 7-12 の Layerscale 統合**が対応。各レイヤーの寄与度を自動調整する仕組み。

**Cambrian-1 [arXiv:2406.16860] Table 23**: 最終 Cambrian モデルの **Adapter lr は 1e-4**（探索実験の 1e-3 より低い）。大規模データ（2.5M）で事前学習する場合は lr=1e-4 がより安定する可能性。

---

### 参考スコア比較

| ベンチマーク | SmolVLM-256M | SmolVLM-500M | LLaVA-Phi (3B) | Imp-3B | Qwen2.5-VL Mini 目標 |
|---|---|---|---|---|---|
| ScienceQA | 73.8% | 80.0% | 68.4% | 72.9% | ~75%（ランダム 25% を大幅に上回る） |
| POPE | — | — | 85.0% | 88.0% | ~85% |
| VQAv2 | — | — | 71.4% | 81.2% | 参考値 |
| TextVQA | 50.2% | 60.2% | 48.6% | 59.4% | ~50%（DINOv2 の OCR 弱点を考慮） |
| MME | — | — | 1335 | 1434 | ~1300 |
| MMBench | — | — | 59.8% | 72.9% | 参考値 |
| MMMU | 29.0% | 33.7% | — | — | 参考値 |
| DocVQA | 58.3% | 70.5% | — | — | — |
| MathVista | 35.9% | 40.1% | — | — | — |
| OCRBench | 52.6 | 61.0 | — | — | —（DINOv2 の OCR 弱点により低い可能性） |

> SmolVLM-500M (SigLIP-B 93M + SmolLM2 360M = 453M) と Qwen2.5-VL Mini (DINOv2 86M + Qwen2.5-0.5B 494M = 582M) はパラメータ数が近い。ただし DINOv2 はテキスト対応がないため、OCR/テキスト系タスクで同等スコアの達成は容易ではない。一方、グラウンディング・空間理解タスクでは DINOv2 の強みが発揮される可能性がある。

---

### 論文一覧（paper/ ディレクトリ）

| ファイル名 | 論文 | arXiv |
|---|---|---|
| llava.pdf | Visual Instruction Tuning (LLaVA) | 2304.08485 |
| llava-1.5.pdf | Improved Baselines with Visual Instruction Tuning (LLaVA-1.5) | 2310.03744 |
| dinov2.pdf | DINOv2: Learning Robust Visual Features without Supervision | 2304.07193 |
| tinyllava.pdf | TinyLLaVA: A Framework of Small-scale Large Multimodal Models | 2402.14289 |
| imp.pdf | Imp: Highly Capable Large Multimodal Models for Mobile Devices | 2405.12107 |
| smolvlm.pdf | SmolVLM: Redefining Small and Efficient Multimodal Models | 2504.05299 |
| idefics2-what-matters.pdf | What Matters When Building Vision-Language Models? (Idefics2) | 2405.02246 |
| comm-clip-to-dino.pdf | From CLIP to DINO: Visual Encoders Shout in Multi-Modal LLMs (COMM) | 2310.08825 |
| llava-phi.pdf | LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model | 2401.02330 |
| moe-llava.pdf | MoE-LLaVA: Mixture of Experts for Large Vision-Language Models | 2401.15947 |
| cambrian-1.pdf | Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs | 2406.16860 |
| eagle.pdf | Eagle: Exploring The Design Space for Multimodal LLMs | 2408.15998 |
| sail.pdf | SAIL: Steering Approaches for Improved Language Models | — |
| dino-meets-text.pdf | When DINO Meets Text | — |
| openvla.pdf | OpenVLA: An Open-Source Vision-Language-Action Model | 2406.09246 |
| sharegpt4v.pdf | ShareGPT4V: Improving Large Multi-Modal Models with Better Captions | 2311.12793 |
