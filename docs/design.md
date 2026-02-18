# MiniPamayo 設計書 v0.1

## 1. 目的

- Alpamayo（VLA：Vision-Language-Action）の**技術的理解**を目的に、同種の構成要素と学習段取りを RTX 4090 単体で再現する
- 目標は「SOTA 性能」ではなく、**学習が回り、段階的に "回帰→Flow" まで到達**すること

---

## 2. 制約と前提

| 項目 | 値 |
|---|---|
| GPU | RTX 4090（24 GB VRAM） |
| 凍結 | **なし**（全モジュール trainable） |
| カメラ | **1 台** |
| 解像度 | 224×224（必要に応じて縮小可） |
| 視覚トークン | **16〜32 tokens / image** |

---

## 3. アーキテクチャ

```
┌──────────┐     ┌──────────────────┐     ┌──────────┐     ┌─────────────┐
│  Camera   │────▶│  DINOv2 ViT-S/14 │────▶│  Adapter  │────▶│  SmolLM2    │
│ (224×224) │     │  (Vision Enc.)   │     │ (Pooling) │     │  360M (LLM) │
└──────────┘     └──────────────────┘     └──────────┘     └──────┬──────┘
                                                                  │
                                                     ┌────────────┴────────────┐
                                                     │                         │
                                              ┌──────▼──────┐          ┌──────▼──────┐
                                              │ Action Head  │          │  Flow Head   │
                                              │ (MLP回帰)    │          │ (Flow Match) │
                                              │ [Stage 0]    │          │ [Stage 2]    │
                                              └─────────────┘          └─────────────┘
```

### 3.1 Vision Encoder — DINOv2 ViT-S/14

- モデル: `facebook/dinov2-small`（ViT-S/14、21M params）
- 入力: RGB 224×224
- 出力: パッチ特徴 (16×16)=256 patches × 384 dim
- trainable（凍結なし）

### 3.2 Vision → LLM Adapter（視覚トークン圧縮）

- 入力: DINOv2 パッチ特徴 (256 × 384)
- 出力: **N_vis tokens (16〜32) × d_llm (960)**
- 方式（実装容易性で選択、後で置換可）:

| 優先度 | 方式 | 概要 |
|---|---|---|
| 1 | Cross-Attention Pooling | learnable query 16/32 個で DINOパッチに attend |
| 2 | MLP + Attention Pool | 簡易版 |
| 3 | 平均 Pool + Linear | 最小実装（fail-fast 用） |

**初期実装**: まず方式 3（平均 Pool + Linear）で全パイプラインを通し、後で方式 1 に置き換える。

### 3.3 Language Model — SmolLM2-360M

- モデル: `HuggingFaceTB/SmolLM2-360M`
- アーキテクチャ: decoder-only Transformer
  - hidden_dim: 960
  - num_layers: 32
  - num_heads: 15
  - vocab_size: 49,152
- trainable（凍結なし）
- **必須要件**: KV-cache を取り出せること（Stage 2 Flow の条件付けに使用）

### 3.4 Action Head（2段階）

#### Stage A — MLP 回帰ヘッド（Stage 0 で使用）

- 入力: LLM 最終層 hidden state
- 出力: `[steer, throttle]`（2D 連続値）or waypoint 列 `[(x1,y1),...,(xK,yK)]`
- Loss: Huber / L2

#### Stage B — Flow Matching ヘッド（Stage 2 で使用）

- 条件付け:
  - **Option A（軽量）**: LLM 最終層 hidden states
  - **Option B（Alpamayo 寄り）**: LLM KV-cache
- 出力: trajectory（waypoints）を生成
- Flow steps: 初期は 10〜20

---

## 4. 入出力仕様

### 4.1 入力

| 入力 | 形状 | 備考 |
|---|---|---|
| 画像 | RGB 224×224 | カメラ 1 台 |
| テキスト | なし or 固定短文 | 初期はなしでも可 |
| (任意) 低次元状態 | speed, yaw rate 等 | 後から追加可 |

### 4.2 出力

| Stage | 出力 | 形状 |
|---|---|---|
| Stage 0（回帰） | steer, throttle | (2,) |
| Stage 0（回帰） | waypoints | (K, 2) |
| Stage 2（Flow） | trajectory | (K, 2) — Flow で生成 |

---

## 5. VRAM 見積もり（概算）

| コンポーネント | パラメータ数 | bf16 サイズ | 備考 |
|---|---|---|---|
| DINOv2 ViT-S/14 | 21M | ~42 MB | |
| SmolLM2-360M | 362M | ~724 MB | |
| Adapter | ~1M | ~2 MB | 方式による |
| Action Head (MLP) | <1M | ~2 MB | |
| **パラメータ合計** | **~385M** | **~770 MB** | |
| オプティマイザ状態 (AdamW) | — | ~3.1 GB | params×8 bytes (fp32 momentum + variance) |
| 勾配 | — | ~770 MB | params と同サイズ (bf16) |
| Activation（checkpointing ON） | — | ~2-4 GB | バッチサイズ依存 |
| **合計推定** | — | **~7-9 GB** | micro-batch=1 時 |

**結論**: 24 GB に対して十分余裕あり。micro-batch=2〜4 も試行可能。

---

## 6. Alpamayo との対応表

| 要素 | Alpamayo | MiniPamayo |
|---|---|---|
| LLM | Qwen2.5-0.5B | SmolLM2-360M |
| Vision Encoder | DINOv2 (ViT-B/L) | DINOv2 ViT-S/14 |
| カメラ | マルチカメラ＋時系列 | **1 台** |
| Adapter | 視覚トークン圧縮 | 同思想（16〜32 tokens） |
| Action Head | Flow Matching | **回帰 → Flow**（段階的） |
| 条件付け | KV-cache ベース | 同思想（hidden or KV-cache） |
| 学習順序 | Flow 含むパイプライン | **回帰 first → Flow**（fail-fast） |

---

## 7. 実装上の推奨初期設定

```yaml
# 4090 で通すためのデフォルト
image_size: 224
n_visual_tokens: 16
text_input: null  # or fixed short prompt
flow_steps: 10    # Stage 2 開始時
micro_batch_size: 1
grad_accumulation_steps: 16
precision: bf16
gradient_checkpointing: true  # DINO / LLM / Flow すべて
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine_with_warmup
```
