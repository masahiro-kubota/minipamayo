# Stage 1: 離散トークン化 — 具体的実装プラン

## 目的

Alpamayo の **Dual Representation** 戦略の前半（論文 §3.2.2, §5.1）を実装する。
Stage 0 で回帰ヘッドにより学習した制御入力 (a, κ) を**均一量子化**して離散トークンに変換し、LLM の自己回帰フレームワーク（cross-entropy, teacher forcing）で行動を生成する。

Stage 3（CoC SFT）で推論トークンと行動トークンを**同じシーケンス**で扱うための前提条件であり、RL ポストトレーニング（Stage 4）で離散トークンを通じて直接勾配を流すための基盤でもある。

### Alpamayo 論文との対応

| Alpamayo 論文 | MiniPamayo Stage 1 | 差分 |
|---|---|---|
| §3.2.2 制御ベース表現の離散化 | 同一方式（均一量子化、128 トークン） | なし |
| §5.1 Dual Representation（学習時離散） | 同一思想（cross-entropy + teacher forcing） | なし |
| §5.1 離散トークンで推論と軌道が共通空間を共有 | 同一（Qwen2.5-0.5B の語彙に特殊トークン追加） | なし |
| §5.1 RL 時に離散トークンを通じて勾配を流す | Stage 4 で実装予定 | Stage 1 では基盤のみ |
| §5.1 Flow Matching による推論時デコード | Stage 2 で実装予定 | Stage 1 では離散のみ |

### Dual Representation のメリット（Alpamayo 論文 §5.1）

1. **推論と軌道の共通空間**: Reasoning トークンと行動トークンが同一のトークン空間を共有し、密な結合が可能
2. **RL での直接勾配**: ポストトレーニング時に離散トークンを通じて直接勾配を流せる
3. **強い教師信号**: 離散表現が車両ダイナミクスの強い教師信号となる
4. **高速推論**: Flow Matching による推論時デコードは 128 トークンの自己回帰生成より高速

---

## 前提条件

- Stage 0（制御ベース回帰）が完了していること
  - 学習済みチェックポイント（Vision Encoder + Adapter + LLM + Action Head）が利用可能
  - 制御ベース表現の GT 制御列 (a, κ) が利用可能
- GT 制御列のデータ分布（加速度 a、曲率 κ の範囲）が既知であること
  - Stage 0 のデータ前処理で統計量を算出済み

---

## 離散トークン化の設計

### 量子化方式: 均一量子化

制御入力 (aᵢ, κᵢ) を所定の範囲で均一量子化する。Alpamayo §3.2.2 と同一の方式。

```
bin_index = clamp(floor((value - v_min) / (v_max - v_min) * N_bins), 0, N_bins - 1)
value_reconstructed = v_min + (bin_index + 0.5) * (v_max - v_min) / N_bins
```

### 加速度 a の量子化

| パラメータ | 値 | 備考 |
|---|---|---|
| 範囲 | [-4.0, 4.0] m/s² | データ分布から決定（急ブレーキ〜急加速をカバー） |
| ビン数 (N_bins_a) | 256 | 分解能: ~0.031 m/s² |
| 分解能 | (4.0 - (-4.0)) / 256 = 0.03125 m/s² | 十分な精度 |

### 曲率 κ の量子化

| パラメータ | 値 | 備考 |
|---|---|---|
| 範囲 | [-0.1, 0.1] 1/m | 最小回転半径 10m に対応。一般道レベル |
| ビン数 (N_bins_κ) | 256 | 分解能: ~0.00078 1/m |
| 分解能 | (0.1 - (-0.1)) / 256 ≈ 0.00078 1/m | 十分な精度 |

> **注**: 量子化範囲はデータ分布の 99.5 パーセンタイル等に基づいて調整する。範囲外の値はクランプ。

### 離散トークンの構成

- 予測ホライズン: **6.4 秒**（64 waypoints @ 10Hz）— Alpamayo と同一
- 各タイムステップで 2 値 (a, κ) を予測
- 合計: **64 × 2 = 128 離散トークン**
- トークン順序: `[a₁, κ₁, a₂, κ₂, ..., a₆₄, κ₆₄]`（交互配置）

### LLM 語彙への特殊トークン追加

Qwen2.5-0.5B の語彙（vocab_size: 151,646）に離散アクショントークンを追加する。

| 方式 | トークン数 | 語彙サイズ | 備考 |
|---|---|---|---|
| **方式 A: 共有ビン** | N_bins (= 256) | 151,646 + 256 = 151,902 | a と κ で同じビン ID を共有。位置で区別 |
| 方式 B: 独立ビン | N_bins_a + N_bins_κ (= 512) | 151,646 + 512 = 152,158 | a と κ で別のトークン ID |

**採用: 方式 A（共有ビン）**。a と κ はシーケンス内の位置（偶数/奇数）で区別でき、トークン数が少ないほど学習が安定する。

### トークン ID マッピング

```
# 加速度 a のビン index → トークン ID
action_token_id = VOCAB_SIZE_ORIGINAL + bin_index

# 曲率 κ のビン index → トークン ID（共有ビンの場合は同じ）
action_token_id = VOCAB_SIZE_ORIGINAL + bin_index

# 逆変換: トークン ID → ビン index
bin_index = token_id - VOCAB_SIZE_ORIGINAL
```

### Embedding Layer と LM Head の拡張

```python
# 1. Embedding Layer の拡張
model.resize_token_embeddings(VOCAB_SIZE_ORIGINAL + N_BINS)
# 新規トークンの embedding は小さいランダム値で初期化

# 2. LM Head の拡張
# resize_token_embeddings が lm_head も自動で拡張する（transformers ライブラリ）
# 新規トークンの出力バイアスはゼロ初期化
```

---

## プロジェクト構成（Stage 1 で追加・変更するファイル）

```
minipamayo/
├── src/
│   └── minipamayo/
│       └── models/
│           ├── discrete_head.py        # 新規: 量子化・逆量子化、トークン ID マッピング
│           ├── dynamics.py             # 変更: 量子化パラメータの追加
│           └── minipamayo.py           # 変更: 離散トークンモードの追加
├── configs/
│   └── stage1.yaml                     # 新規: Stage 1 ハイパーパラメータ
└── scripts/
    └── train_stage1.py                 # 新規: Stage 1 学習スクリプト
```

### models/discrete_head.py

量子化・逆量子化とトークン ID マッピングを担当するモジュール。

```python
"""離散トークン化: 制御入力 (a, κ) ↔ 離散トークン ID の変換。"""
import torch
import torch.nn as nn

class DiscreteActionTokenizer:
    """制御入力 (a, κ) の均一量子化・逆量子化。"""

    def __init__(
        self,
        n_bins: int = 256,
        a_range: tuple[float, float] = (-4.0, 4.0),
        kappa_range: tuple[float, float] = (-0.1, 0.1),
        vocab_offset: int = 151_646,  # Qwen2.5-0.5B の元の語彙サイズ
    ):
        self.n_bins = n_bins
        self.a_range = a_range
        self.kappa_range = kappa_range
        self.vocab_offset = vocab_offset

    def quantize(self, values: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        """連続値 → ビン index。"""
        normalized = (values - v_min) / (v_max - v_min)
        bins = torch.clamp((normalized * self.n_bins).long(), 0, self.n_bins - 1)
        return bins

    def dequantize(self, bins: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        """ビン index → 連続値（ビン中心）。"""
        return v_min + (bins.float() + 0.5) * (v_max - v_min) / self.n_bins

    def encode(self, a: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """制御入力 (a, κ) → 離散トークン ID 列。
        Args:
            a: (B, T) 加速度
            kappa: (B, T) 曲率
        Returns:
            token_ids: (B, T*2) トークン ID 列 [a₁, κ₁, a₂, κ₂, ...]
        """
        a_bins = self.quantize(a, *self.a_range)
        k_bins = self.quantize(kappa, *self.kappa_range)
        # 交互配置
        B, T = a.shape
        token_ids = torch.stack([a_bins, k_bins], dim=-1).reshape(B, T * 2)
        return token_ids + self.vocab_offset

    def decode(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """離散トークン ID 列 → 制御入力 (a, κ)。
        Args:
            token_ids: (B, T*2) トークン ID 列
        Returns:
            a: (B, T) 加速度
            kappa: (B, T) 曲率
        """
        bins = token_ids - self.vocab_offset
        B = bins.shape[0]
        bins = bins.reshape(B, -1, 2)
        a = self.dequantize(bins[..., 0], *self.a_range)
        kappa = self.dequantize(bins[..., 1], *self.kappa_range)
        return a, kappa
```

### models/dynamics.py への追加

```python
# 既存の dynamics.py に量子化パラメータを追加

# 量子化パラメータ（データ分布から決定）
QUANTIZATION_CONFIG = {
    "n_bins": 256,
    "a_range": (-4.0, 4.0),       # m/s²
    "kappa_range": (-0.1, 0.1),   # 1/m
    "n_waypoints": 64,
    "dt": 0.1,                    # 10Hz
}
```

---

## 実装ステップ

### Step 1: 量子化パラメータの決定

- [ ] GT 制御列のデータ分布を可視化（ヒストグラム）
- [ ] 加速度 a の 0.5 / 99.5 パーセンタイルから量子化範囲を算出
- [ ] 曲率 κ の 0.5 / 99.5 パーセンタイルから量子化範囲を算出
- [ ] 量子化誤差の統計量を計算（平均絶対誤差、最大誤差）
- [ ] N_bins = {128, 256, 512} での量子化誤差を比較

```bash
cd minipamayo && uv run python -c "
import torch, json
# GT 制御列のロードと分布分析
# → a_range, kappa_range を決定
"
```

### Step 2: DiscreteActionTokenizer の実装

- [ ] `models/discrete_head.py` を新規作成
- [ ] 量子化・逆量子化の単体テスト
- [ ] ラウンドトリップテスト: encode → decode で元の値に近いことを確認
- [ ] 量子化誤差が ADE/FDE に与える影響を理論値として算出

### Step 3: LLM 語彙拡張

- [ ] `model.resize_token_embeddings(VOCAB_SIZE_ORIGINAL + N_BINS)` で語彙拡張
- [ ] 新規トークンの embedding 初期化方法を確認
  - 既存 embedding の平均・分散に合わせたランダム初期化
- [ ] LM head の対応する重みも拡張されていることを確認
- [ ] 拡張前後で既存トークンの embedding が変わらないことを確認

### Step 4: 学習ループの実装

- [ ] 入力シーケンス構成:
  - `[visual_tokens(16), BOS, action_token_1, ..., action_token_128]`
  - BOS トークン（自己回帰開始のマーカー）を追加
- [ ] Loss: cross-entropy（LLM の標準 next-token prediction）
  - action token 部分のみで loss を計算（visual token 部分は無視）
- [ ] Teacher forcing: GT 離散トークン列をシフトして入力
- [ ] Stage 0 の学習済み重みを初期値としてロード
  - Action Head（MLP 回帰）は使用しない（離散トークンに置き換え）
  - Vision Encoder + Adapter + LLM の重みは引き継ぐ
- [ ] 勾配制御: Vision Encoder + Adapter + LLM すべて trainable（設計書 §3.7）

### Step 5: デコードパイプラインの実装

- [ ] 自己回帰生成: `[visual_tokens, BOS]` → 128 トークンを逐次生成
- [ ] 離散トークン → 連続制御入力への逆量子化
- [ ] 連続制御入力 → 軌道への変換（ユニサイクルダイナミクス Euler 積分）
- [ ] 可視化: 予測軌道 vs GT を画像上にプロット

### Step 6: 学習の実行と評価

- [ ] 小規模データで overfitting 確認（数十サンプルで loss が 0 に近づくか）
- [ ] フルデータで学習
- [ ] ADE/FDE を Stage 0（回帰版）と比較
- [ ] 離散トークンの分布分析（生成されたトークンの頻度ヒストグラム）

---

## 評価

### 定量評価

| 指標 | 内容 | Stage 0 との比較 |
|---|---|---|
| Cross-entropy loss | 学習曲線の収束 | — |
| ADE (Average Displacement Error) | 全 waypoint の平均位置誤差 | 劣化が小さいことを確認 |
| FDE (Final Displacement Error) | 最終 waypoint の位置誤差 | 同上 |
| 量子化誤差 | encode → decode のラウンドトリップ誤差 | — |

### 量子化ビン数の影響調査

| N_bins | 加速度 分解能 | 曲率 分解能 | 追加トークン数 | ADE | FDE |
|---|---|---|---|---|---|
| 128 | 0.0625 m/s² | 0.00156 1/m | 128 | TBD | TBD |
| **256** | **0.03125 m/s²** | **0.00078 1/m** | **256** | TBD | TBD |
| 512 | 0.01563 m/s² | 0.00039 1/m | 512 | TBD | TBD |

### 離散トークンの分布分析

- 生成されたトークンの頻度ヒストグラム
  - 加速度: 0 付近（定速走行）にピークがあることを確認
  - 曲率: 0 付近（直進）にピークがあることを確認
- GT 分布と生成分布の KL ダイバージェンスを計算
- 稀なビン（急ブレーキ、急旋回）の生成精度を個別に確認

---

## ハイパーパラメータ

```yaml
# Stage 1: 離散トークン化
stage: 1

# 量子化
n_bins: 256
a_range: [-4.0, 4.0]        # m/s²
kappa_range: [-0.1, 0.1]    # 1/m
n_waypoints: 64
n_action_tokens: 128         # 64 * 2

# モデル
vision_encoder: facebook/dinov2-base
llm: Qwen/Qwen2.5-0.5B
image_size: 224
n_visual_tokens: 16
vocab_size_extended: 151902   # 151646 + 256

# 勾配制御（設計書 §3.7）
trainable_vision: true
trainable_adapter: true
trainable_llm: true

# 学習
init_from: checkpoints/stage0/best.pt   # Stage 0 の学習済み重み
micro_batch_size: 1
grad_accumulation_steps: 16
effective_batch_size: 16
precision: bf16
gradient_checkpointing: true

# Optimizer
optimizer: AdamW
learning_rate: 5.0e-5         # Stage 0 より小さめ（既学習重みの破壊を防ぐ）
weight_decay: 0.01
warmup_ratio: 0.05
scheduler: cosine_with_warmup
max_epochs: 10

# Loss
loss: cross_entropy
label_smoothing: 0.0           # 初期値。必要に応じて 0.1 に調整
ignore_index: -100             # visual token 部分を無視
```

---

## VRAM 見積もり

設計書 §5.3 より、Stage 0-1 は同一の VRAM プロファイル:

| 項目 | 値 |
|---|---|
| Trainable params | 582M（Vision 86M + Adapter ~2M + LLM 494M） |
| 固定コスト (N×12) | 6.98 GB |
| Activation + overhead | ~3 GB |
| **合計** | **~10 GB** |

Stage 0 との差分: Action Head MLP (<1M) が不要になり、語彙拡張 (+256 tokens) が追加されるが、VRAM への影響は無視できる。

---

## Exit 条件

| 条件 | 基準 |
|---|---|
| Cross-entropy loss | 学習曲線が安定して下がり、収束すること |
| ADE/FDE 劣化 | Stage 0（回帰版）比で **20% 以内**の劣化 |
| デコード軌道の妥当性 | 可視化で軌道が入力画像に対しておおよそ妥当 |
| 量子化誤差 | ラウンドトリップ誤差が ADE/FDE の劣化に支配的でないこと |
| OOM なし | RTX 4090 (24 GB) 以内で学習が完走すること |

---

## 完了状況

| Step | 状態 | 備考 |
|------|------|------|
| Step 1: 量子化パラメータの決定 | 未着手 | |
| Step 2: DiscreteActionTokenizer の実装 | 未着手 | |
| Step 3: LLM 語彙拡張 | 未着手 | |
| Step 4: 学習ループの実装 | 未着手 | |
| Step 5: デコードパイプラインの実装 | 未着手 | |
| Step 6: 学習の実行と評価 | 未着手 | |
