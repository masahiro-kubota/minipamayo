# Stage 2: Flow Matching — Trajectory Decoder 設計書

## 1. 目的

Dual Representation の推論側を完成させる。Stage 0/1 で学習した VLM（Vision Encoder + Adapter + LLM）の内部表現を条件として、**Conditional Flow Matching (CFM)** で連続的かつ多様な軌道を生成する。Alpamayo 論文における「Trajectory Decoder」に相当するモジュールの設計・実装・学習方針を定める。

Alpamayo 論文 §5.1 より:
> "At inference time, we bypass the discrete-token head and instead use a dedicated trajectory decoder based on flow matching to produce continuous trajectories."

離散トークンによる自己回帰生成に対して、Flow Matching は以下の優位性を持つ（論文 §3.2.2, §6.6）:

- **精度**: 連続空間で直接軌道を生成するため、量子化誤差がない
- **快適性**: ジャーク（加加速度）が小さく、滑らかな軌道を生成
- **推論速度**: 12 トークンの自己回帰生成より、10 step の Euler 積分の方が高速
- **多様性**: ノイズからの生成により、同一条件から複数の妥当な軌道をサンプリング可能

---

## 2. Alpamayo 論文との対応表

| 観点 | Alpamayo 論文 | MiniPamayo Stage 2 | 論文参照 |
|---|---|---|---|
| デコード方式 | Flow Matching（推論時） | 同一 | §3.2.2 |
| Dual Representation | 学習: 離散トークン / 推論: Flow | 同一 | §5.1 |
| 条件付け | VLM の KV-cache に stop-gradient | VLM の KV-cache（past_key_values）に stop-gradient | §5.1 |
| Trajectory Decoder 規模 | 2B（10B 版） / 不明（0.5B 版） | ~146M（LLM 494M の ~30%） | §5.1 |
| Flow の優位性 | 精度・快適性・速度すべてで自己回帰に勝る | 同じ検証を実施 | §6.6 |
| 離散トークン vs Flow | Flow が minADE6 で 15-20% 改善 | 比較評価を実施 | §6.6 Table 7 |
| 勾配制御 | VLM KV-cache に stop-gradient | VLM 全体を frozen | §5.1 |

---

## 3. 前提条件

Stage 2 の開始には以下が完了していること:

- [x] **Stage 0（回帰）**: VLM パイプラインが動作し、制御ベース表現 (a, κ) で loss が収束
- [x] **Stage 1（離散トークン）**: 離散アクショントークンの学習が完了し、cross-entropy loss が収束
- [ ] **制御ベース表現の安定**: GT 制御列の逆算精度が十分であり、ADE/FDE が意味のある値を示す
- [ ] **KV-cache の取り出し**: Qwen2.5-0.5B から中間表現（KV-cache or hidden states）を取得できることを確認済み

Stage 0/1 の学習済み重み（Vision Encoder + Adapter + LLM）が Stage 2 の VLM 初期値となる。

---

## 4. Trajectory Decoder の設計

### 4.1 サイジング: ~146M params

Alpamayo の Expert/VLM 比率（~25%）に合わせたサイジング:

- LLM (Qwen2.5-0.5B): 494M params
- Trajectory Decoder: **~146M params**（146M / 494M ≈ 30%）

この規模であれば Stage 2 の VRAM 見積もり（後述 §8）に十分収まる。

### 4.2 アーキテクチャ: 小さな Transformer

Trajectory Decoder は LLM と同系列の Transformer アーキテクチャを採用するが、hidden / MLP サイズを縮小する。

#### 構成パラメータ

| パラメータ | LLM (Qwen2.5-0.5B) | Trajectory Decoder (Expert) | 備考 |
|---|---|---|---|
| hidden_dim | 896 | **640** | Expert の Q heads 数で決定（heads × head_dim） |
| num_layers | 24 | **24** | VLM と一致（レイヤーごとの KV-cache 対応） |
| num_attention_heads | 14 | **10** | Expert の Q heads（自由に設定可能） |
| num_kv_heads | 2 | **2** | VLM と一致（KV-cache 互換性制約） |
| head_dim | 64 | **64** | VLM と一致（KV-cache 互換性制約） |
| intermediate_size (MLP) | 4,864 | **2,560** | |
| **推定パラメータ数** | **494M** | **~146M** | Expert/VLM ≈ 30% |

> **KV-cache 互換性制約**: Expert は VLM の `past_key_values` を直接消費するため、`num_kv_heads=2` と `head_dim=64` が VLM と一致する必要がある。`num_hidden_layers=24` も VLM と一致が必要（レイヤーごとの KV-cache 対応）。Expert の Q heads（`num_attention_heads`）は自由に設定可能で、`hidden_size = num_attention_heads × head_dim` の制約から有効な構成は 512/8, 640/10, 768/12, 896/14 heads。

#### 入力構成

Decoder の各 Transformer ブロックへの入力は以下の要素から構成される:

1. **ノイズ付き制御入力 a_t**: (batch, seq_len, action_dim) — 拡散過程のタイムステップ t における中間状態
2. **タイムステップ embedding**: スカラー t ∈ [0, 1] を高次元に射影
3. **条件情報**: LLM からの hidden states or KV-cache（§4.4 参照）

#### 出力

- **velocity field v_Θ(a_t, o)**: (batch, seq_len, action_dim) — Flow の速度場予測
- 推論時: Euler 積分で a_t を更新し、連続的な制御入力列 (a, κ) を生成

### 4.3 入力エンコーディング（Fourier Feature V2 + MLP）

アクション (a, κ) とタイムステップ t を Expert の hidden dim に射影する。Alpamayo §5.2 の ActionInProj に準拠。

```
per-waypoint action (a_i, κ_i) + timestep t
  → Fourier Feature V2（各次元を独立にエンコード）
    - a_i → FourierEncoderV2(dim=20, max_freq=100) → (20,)
    - κ_i → FourierEncoderV2(dim=20, max_freq=100) → (20,)
    - t   → FourierEncoderV2(dim=20, max_freq=100) → (20,)
  → Concat → (60,)
  → MLPEncoder(RMSNorm + SiLU, 4 layers, hidden=1024)
  → LayerNorm
  → (hidden_size,) per waypoint = Expert の入力トークン列 (B, K, hidden_size)
```

タイムステップ t は AdaLN による各ブロックへの注入ではなく、各 waypoint の入力に直接 concat して Expert トークンに統合する方式。Alpamayo と同一。

### 4.4 条件付け方式

VLM の内部表現を Trajectory Decoder に渡す方式として 2 つの選択肢がある。

#### 採用方式: VLM KV-cache → past_key_values（Alpamayo §5.1 準拠）

```
VLM forward (use_cache=True) → past_key_values (KV-cache)
  → Expert に past_key_values として直接渡す
  → Expert の attention が VLM の KV-cache を key/value として消費
```

- **Alpamayo と同一方式**: VLM の `past_key_values` を Expert に直接渡す
- Expert は Qwen2 Transformer（`AutoModel.from_config(Qwen2Config(...))`）で構築
- Expert の `num_kv_heads=2`, `head_dim=64` が VLM と一致するため、KV-cache をネイティブに消費可能
- Non-causal attention（`attention_mask` をゼロテンソルで bidirectional）
- Position IDs は VLM シーケンス長の continuation（Expert トークンは VLM の続き）
- 各 step_fn 後に Expert が追加したトークンの KV-cache を crop（Alpamayo パターン）

Alpamayo 論文 §5.1:
> "we apply a stop-gradient to the KV-cache produced by the VLM"

MiniPamayo では VLM を `@torch.no_grad()` で実行し、KV-cache に stop-gradient を適用して同等の勾配制御を実現。

---

## 5. Flow Matching の詳細

### 5.1 Conditional Flow Matching (CFM) の概要

Flow Matching は、ノイズ分布 p_0 からデータ分布 p_1 への確率的輸送を学習する生成モデルフレームワーク。Diffusion とは異なり、直線的な輸送パスを使うため学習・推論ともに効率的。

### 5.2 Gaussian OT (Optimal Transport) パス

タイムステップ t ∈ [0, 1] に対して、ノイズ ε ~ N(0, I) とデータサンプル a（GT 制御入力列）の間を直線的に補間する:

```
a_t = t * a + (1 - t) * ε
```

ここで:
- t = 0: 純粋なノイズ ε
- t = 1: データサンプル a（GT 制御入力列）
- 中間の t: ノイズとデータの線形補間

### 5.3 Target velocity field

Gaussian OT パスに対応する target velocity field は:

```
u(a_t | a) = a - ε
```

これは「ノイズからデータへ向かう方向」を表す。

### 5.4 CFM Loss

Trajectory Decoder v_Θ が予測する velocity field と target field の MSE:

```
L_CFM = E_{t~Beta(α,β), ε~N(0,I), a~p_data} [ || v_Θ(a_t, t, c) - (a - ε) ||² ]
```

ここで c は条件情報（LLM hidden state sequence）。t は shifted beta distribution `Beta(α=2, β=5)` からサンプリング（Alpamayo §5.2 準拠）。一様分布 U(0,1) と比較して、t が小さい領域（ノイズに近い状態）に重点を置くことで学習効率が向上する。

#### 学習手順（1 ステップ）

1. データバッチから GT 制御入力列 a を取得
2. ノイズ ε ~ N(0, I) をサンプリング
3. タイムステップ t ~ Beta(α, β) をサンプリング（shifted beta distribution）
4. 中間状態を計算: a_t = t * a + (1 - t) * ε
5. VLM を forward して条件情報 c を取得（**stop-gradient**）
6. Decoder で velocity field を予測: v_Θ(a_t, t, c)
7. Loss を計算: || v_Θ - (a - ε) ||²
8. Decoder のパラメータのみ更新

### 5.5 推論パイプライン（Euler 積分）

推論時はノイズから出発し、学習した velocity field に沿って Euler 積分で軌道を生成する。

```
入力:  ε ~ N(0, I)  (純粋なノイズ)
ステップ数: N = 10  (δt = 0.1)

for i in range(N):
    t = i / N
    a_t = a_t + δt * v_Θ(a_t, t, c)

出力: a_1 ≈ a_N  (生成された制御入力列)
```

- **デフォルト: 10 steps（δt = 0.1）**
- 必要に応じて steps 数を増やして精度向上（20, 50 steps）
- 同一条件 c に対して異なるノイズ ε から開始すれば、**多様な軌道をサンプリング可能**

---

## 6. 勾配制御

### 6.1 方針

Stage 2 では **Vision Encoder + Adapter + LLM は完全に frozen（stop-gradient）**とし、Trajectory Decoder のみを trainable とする。

| モジュール | パラメータ数 | 状態 | 備考 |
|---|---|---|---|
| DINOv2 ViT-B/14 | 86M | **frozen** | 推論のみ |
| Adapter | ~2M | **frozen** | 推論のみ |
| Qwen2.5-0.5B (LLM) | 494M | **frozen** | KV-cache / hidden states を出力 |
| Trajectory Decoder (Expert) | ~146M | **trainable** | CFM loss で学習 |

### 6.2 Alpamayo 論文との整合

Alpamayo 論文 §5.1:
> "we apply a stop-gradient to the KV-cache produced by the VLM to prevent gradients from the expert back-propagating into the VLM weights"

MiniPamayo では VLM 全体を `torch.no_grad()` + `requires_grad_(False)` で frozen にし、KV-cache を Expert に `past_key_values` として渡すことで同等の勾配制御を実現する。

### 6.3 実装上の注意

- VLM の forward は `@torch.no_grad()` で実行（メモリ節約 + 計算高速化）
- VLM の出力 KV-cache を Expert に `past_key_values` として渡す（stop-gradient が暗黙的に適用）
- Expert 内部の gradient checkpointing はオプションで適用（VRAM に余裕があるため不要な可能性が高い）

---

## 7. プロジェクト構成

```
minipamayo/
├── src/
│   └── minipamayo/
│       ├── models/
│       │   ├── trajectory_decoder.py   # Flow Matching Expert (Qwen2 Transformer)
│       │   │   ├── TrajectoryDecoder       — メインモジュール (KV-cache 条件付け)
│       │   │   ├── FourierEncoderV2        — Fourier Feature エンコーダ
│       │   │   ├── ActionInProj            — per-waypoint Fourier + MLP 入力射影
│       │   │   └── cfm_loss()              — Conditional Flow Matching loss
│       │   └── ...
│       ├── train_stage2.py             # Stage 2 学習ループ
│       ├── eval_stage2.py              # Stage 2 評価 (ADE/FDE/多様性)
│       └── ...
```

---

## 8. VRAM 見積もり

### 8.1 コンポーネント別

| コンポーネント | パラメータ数 | 状態 | メモリ | 計算式 |
|---|---|---|---|---|
| VLM (Vision + Adapter + LLM) | 582M | frozen（推論のみ） | ~1.16 GB | 582M × 2 bytes (bf16) |
| Trajectory Decoder (Expert) | 146M | trainable | ~1.75 GB | 146M × 12 bytes (param + optim + grad) |
| Activation（Decoder 学習分） | — | — | ~3.00 GB | gradient checkpointing 込み |
| **合計** | | | **~6 GB** | |

### 8.2 RTX 4090 との比較

- RTX 4090 VRAM: **24 GB**
- Stage 2 使用量: **~6 GB**
- 余裕: **~18 GB**

全 Stage の中で最も VRAM 消費が小さい。VLM が frozen のため、学習対象が Decoder のみであることが要因。

### 8.3 バッチサイズの拡大余地

余裕が大きいため、micro-batch サイズを Stage 0/1 より大きくできる:

| micro_batch_size | 追加 activation | 推定合計 VRAM |
|---|---|---|
| 1 | ~3 GB | ~6 GB |
| 4 | ~8 GB | ~11 GB |
| 8 | ~14 GB | ~17 GB |

**推奨: micro_batch_size=4 から開始**し、OOM が出なければ増やす。

---

## 9. 実装ステップ（チェックリスト）

### Phase 1: CFM 基本実装

- [ ] Gaussian OT パスの実装（a_t = t*a + (1-t)*ε）
- [ ] Target velocity field の計算（u = a - ε）
- [ ] CFM loss 関数の実装（MSE）
- [ ] 単体テスト: 既知の分布（2D ガウシアン等）で Flow が学習できることを確認

### Phase 2: Trajectory Decoder Expert（Qwen2 Transformer）

- [ ] FourierEncoderV2（per-dim Fourier feature encoding）
- [ ] ActionInProj（per-waypoint Fourier + MLP 入力射影、Alpamayo §5.2 準拠）
- [ ] TrajectoryDecoder 統合クラス（Qwen2Config ベースの Expert Transformer）
- [ ] パラメータ数の確認: ~146M に収まること
- [ ] 単体テスト: ランダム入力で forward / backward が通ること

### Phase 3: KV-cache 条件付けの実装

- [ ] VLM forward（frozen, use_cache=True）から KV-cache を取得するパイプライン
- [ ] KV-cache を Expert に `past_key_values` として渡す条件付け（Alpamayo §5.1 準拠）
- [ ] `@torch.no_grad()` による stop-gradient の確認
- [ ] Non-causal attention mask + position IDs の continuation

### Phase 4: 学習ループ

- [ ] VLM（frozen）+ Decoder（trainable）の統合学習ループ
- [ ] VRAM 使用量の実測・記録
- [ ] gradient checkpointing の Decoder への適用（必要に応じて）
- [ ] wandb ロギング: CFM loss, learning rate, VRAM 使用量

### Phase 5: 推論パイプライン

- [ ] ノイズ生成 → 10 step Euler 積分 → 制御入力列の生成
- [ ] 制御入力列 → ユニサイクルダイナミクス → waypoint 軌道への変換
- [ ] 可視化: 生成軌道をカメラ画像上にプロット

### Phase 6: 多様性評価

- [ ] 同一入力に対して異なるノイズから複数軌道をサンプリング（K=5, 10）
- [ ] 軌道のばらつきを可視化・定量化
- [ ] 多様性 vs 精度のトレードオフを確認

---

## 10. 評価

### 10.1 精度指標

| 指標 | 説明 | 比較対象 |
|---|---|---|
| **ADE** (Average Displacement Error) | 全タイムステップの平均変位誤差 | Stage 0（回帰）, Stage 1（離散） |
| **FDE** (Final Displacement Error) | 最終タイムステップの変位誤差 | 同上 |
| **minADE_K** | K 本サンプル中の最小 ADE | Flow 固有（K=5, 10） |
| **minFDE_K** | K 本サンプル中の最小 FDE | Flow 固有（K=5, 10） |

### 10.2 マルチモーダル性（多様性）

- 同一入力から K 本の軌道をサンプリングし、軌道間の分散を計算
- 交差点等の分岐シナリオで、直進・左折・右折の軌道が出現するかを定性的に確認
- **期待**: 回帰（Stage 0）は常に 1 本の平均的軌道しか出力できないが、Flow は複数の妥当なモードを捉える

### 10.3 Flow steps 数の影響

| steps 数 | δt | 推論時間（相対） | 精度（想定） |
|---|---|---|---|
| 10 | 0.1 | 1× | ベースライン |
| 20 | 0.05 | 2× | やや改善 |
| 50 | 0.02 | 5× | 収束に近い |

### 10.4 推論速度比較

| 方式 | 処理内容 | 推論時間（想定） |
|---|---|---|
| Stage 0（回帰） | MLP 1 回 forward | 最速 |
| Stage 1（離散） | 12 トークン自己回帰 | 最遅 |
| **Stage 2（Flow）** | **10 step Euler 積分** | **中間** |

Alpamayo 論文 §6.6 では Flow が 12 トークン自己回帰より高速であることが示されている。MiniPamayo でも同様の傾向を確認する。

### 10.5 快適性指標

| 指標 | 説明 |
|---|---|
| **Jerk** (加加速度) | 制御入力の 3 階差分の RMS |
| **Curvature smoothness** | 曲率 κ の変化率の RMS |

Alpamayo 論文 §6.6 Table 8 では、Flow が離散トークンよりジャークが小さいことが示されている。

---

## 11. ハイパーパラメータ

```yaml
# Stage 2: Flow Matching
# ─────────────────────────

# Trajectory Decoder Expert（~146M params）
decoder_hidden_dim: 640
decoder_num_layers: 24
decoder_num_heads: 10
decoder_num_kv_heads: 2         # VLM と一致（KV-cache 互換性制約）
decoder_head_dim: 64            # VLM と一致
decoder_intermediate_size: 2560
decoder_dropout: 0.0

# Flow Matching
flow_steps_train: 1          # 学習時は 1 step（t をランダムサンプリング）
flow_steps_inference: 10     # 推論時の Euler 積分ステップ数
noise_schedule: "linear"     # Gaussian OT path（直線補間）
time_sampling: "shifted_beta" # Beta(alpha=2, beta=5) — Alpamayo §5.2
beta_a: 2.0                  # Beta distribution alpha parameter
beta_b: 5.0                  # Beta distribution beta parameter

# 条件付け（Alpamayo §5.1 準拠）
conditioning: "kv_cache"  # VLM の past_key_values を Expert に直接渡す

# 入力エンコーディング（Fourier Feature V2 + MLP, Alpamayo §5.2）
num_fourier_feats: 20        # Fourier encoding 次元数
fourier_max_freq: 100.0      # Fourier encoding 最大周波数
mlp_hidden_size: 1024        # ActionInProj MLP hidden dim
mlp_num_layers: 4            # ActionInProj MLP 層数

# 学習
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine_with_warmup
warmup_steps: 500
max_epochs: 30               # データ量に応じて調整
micro_batch_size: 4          # VRAM に余裕があるため Stage 0/1 より大きく
grad_accumulation_steps: 4
max_grad_norm: 1.0
precision: bf16

# VLM（frozen）
vlm_checkpoint: "checkpoints/stage1/best.pt"  # Stage 0/1 の学習済み重み
vlm_precision: bf16

# アクション表現
prediction_horizon: 6        # 6 waypoints @ dt=0.5s = 3s
action_dim: 2                # (acceleration, curvature)

# 評価
eval_num_samples: 10         # 多様性評価時のサンプル数
eval_flow_steps: [10, 20, 50]  # steps 数の比較
```

---

## 12. Exit 条件

Stage 2 の完了判定基準:

| 条件 | 基準 | 必須/推奨 |
|---|---|---|
| **CFM loss 収束** | loss が安定して下がり、plateauに達する | 必須 |
| **ADE/FDE が Stage 0 と同等以上** | 回帰版と比較して劣化しない | 必須 |
| **多様性の確認** | 同一入力から複数の異なる妥当な軌道が生成される | 必須 |
| **推論速度** | 12 トークン自己回帰より高速（or 同等） | 推奨 |
| **快適性** | ジャークが離散トークン版より改善 | 推奨 |
| **OOM しない** | RTX 4090 (24 GB) 以内で学習・推論が完了 | 必須 |
| **Flow steps の影響** | 10 steps で十分な精度が出ることを確認 | 推奨 |

---

## 13. Full nuScenes (v1.0-trainval) 学習結果

### 設定

- データ: nuScenes v1.0-trainval, **シーン単位分割** (公式 split)
  - Train: 23,230 samples (700 scenes), Val: 4,969 samples (150 scenes)
- K=6 (dt=0.5s), Decoder のみ trainable（VLM frozen）
- lr=1e-4, cosine w/ warmup, 10 epochs
- Phase 4 checkpoint（scene split 版）から初期化

### 学習推移

| Epoch | Train CFM | Val CFM | Time |
|---|---|---|---|
| 初期 | — | 2.070 | — |
| 1 | 1.074 | 1.120 | 3.2min |
| 2 | 0.971 | 1.067 | 3.2min |
| 3 | 0.932 | 1.046 | 3.2min |
| 4 | 0.916 | 1.026 | 3.2min |
| 5 | 0.906 | 1.014 | 3.2min |
| 6 | 0.899 | 1.029 | 3.2min |
| 7 | 0.897 | 0.995 | 3.2min |
| **8** | **0.879** | **0.986** (Best) | 3.2min |
| 9 | 0.873 | 1.001 | 3.2min |
| 10 | 0.876 | 0.988 | 3.2min |
| **合計** | | | **33min** |

### 分析

- Val CFM は Epoch 8 の 0.986 が Best。初期値 2.070 から 52.4% 改善
- Train-Val gap: 0.879 vs 0.986（+12.2%）。多少の overfitting だが許容範囲
- Epoch 6 で Val が一時上昇（1.029）後、Epoch 7-8 で回復。学習率が下がり安定
- Decoder のみ trainable のため学習が高速（1 epoch = 3.2min, 全10 epochs = 33min）
