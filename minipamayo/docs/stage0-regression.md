# Stage 0: パイプライン検証（MLP 回帰 → 制御ベース表現）

## 目的

MiniPamayo の最初のステージとして、**行動予測パイプライン全体が動作すること**を検証する。

1. **Phase 3（fail-fast 最小検証）**: MLP 回帰ヘッドで `[steer, throttle]` (2,) を予測し、勾配が DINO → Adapter → LLM → Action Head の全経路に流れることを確認する
2. **Phase 4（制御ベース表現への移行）**: Action Head の出力を (a, κ) × 64 waypoints に拡張し、Adapter を Cross-Attention Pooling に置換する

### Alpamayo 論文との対応表

| Alpamayo 0.5B | MiniPamayo Stage 0 | 差分 |
|---|---|---|
| Action Injection（VLM に制御入力予測を注入） | Phase 3: MLP 回帰 → Phase 4: 制御ベース表現 | 同思想（段階的に導入） |
| 制御ベース表現 (a, κ) × 64 @ 10Hz | Phase 4 で同一表現を採用 | **同一** |
| ユニサイクルダイナミクス + Euler 積分 | 同一の積分方式 | **同一** |
| GT 逆算: 最小二乗法 + Tikhonov 正則化 | 同一の逆算方式 | **同一** |
| 予測ホライズン 6.4秒 | 6.4秒（64 waypoints @ 10Hz） | **同一** |
| DINOv2 → Projector → Qwen2.5-0.5B | DINOv2 ViT-B/14 → Adapter → Qwen2.5-0.5B | **同系列** |
| マルチカメラ 7台 + 時系列 | 単一カメラ（CAM_FRONT） | 差分（MiniPamayo の制約） |
| 内部データ 80K 時間 | nuScenes 公開データ | 差分（技術理解目的なので許容） |

---

## 前提条件

- Cosmos Reason Mini のドメイン SFT（または Qwen2.5-VL Mini Stage 2.1）が完了していること
  - 学習済み重み（Vision Encoder + Adapter + LLM）が MiniPamayo Stage 0 の初期値となる
- nuScenes データセットがダウンロード・展開済みであること
  - パイプライン検証: v1.0-mini（~4 GB）
  - 本番学習: v1.0-trainval（~300 GB）
- RTX 4090（24 GB VRAM）が利用可能であること
- wandb アカウントがセットアップ済みであること

---

## データ設計

全体のデータ戦略は設計書 §6 を参照。Stage 0 では主に nuScenes を使用し、Phase 4（本番学習）で comma2k19 を追加データとして活用可能。

### 主要データソース: nuScenes

| 項目 | nuScenes Mini | nuScenes Trainval |
|---|---|---|
| 用途 | パイプライン検証 | 本番学習 |
| シーン数 | 10 | 850 (train) + 150 (val) |
| キーフレーム数 | ~400 | ~34,000 (train) + ~6,000 (val) |
| サイズ | ~4 GB | ~300 GB |
| 画像解像度 | 1600×900 | 1600×900 |
| カメラ | CAM_FRONT のみ使用 | 同左 |

### 追加データソース: comma2k19（Phase 4 以降）

| 項目 | comma2k19 |
|---|---|
| 用途 | 本番学習の追加データ（設計書 §6 Phase A） |
| データ量 | 33h |
| カメラ | フロント 1 台 |
| 制御信号 | steering_angle, wheel_speeds |
| ライセンス | MIT |

> **注**: comma2k19 は ego pose ではなく CAN bus（steering_angle, wheel_speeds）から GT 制御列を算出する必要がある。nuScenes と異なるデータパイプラインの実装が必要。

### 入力データ

| 入力 | 形状 | 説明 |
|---|---|---|
| CAM_FRONT 画像 | RGB 224×224 | 1600×900 を 224×224 にリサイズ、ImageNet 正規化 |
| ego pose | (x, y, z, qw, qx, qy, qz) | nuScenes の `ego_pose` テーブルから取得 |

### GT アクションの設計

#### Phase 3（fail-fast）: [steer, throttle] (2,)

最小限のパイプライン検証用。ego pose の差分から近似的に算出する。

- **steer**: 連続する 2 フレーム間の yaw 変化量（ラジアン）
- **throttle**: 連続する 2 フレーム間の速度変化量（m/s）

```python
# ego pose から steer/throttle を近似算出
yaw_curr = quaternion_to_yaw(ego_pose_curr["rotation"])
yaw_next = quaternion_to_yaw(ego_pose_next["rotation"])
steer = normalize_angle(yaw_next - yaw_curr)

pos_curr = np.array(ego_pose_curr["translation"][:2])
pos_next = np.array(ego_pose_next["translation"][:2])
dt = (timestamp_next - timestamp_curr) / 1e6  # μs → s
speed = np.linalg.norm(pos_next - pos_curr) / dt
throttle = speed - speed_prev  # 加速度近似
```

#### Phase 4（制御ベース表現）: (a, κ) × 64

Alpamayo 論文 §3.2.2 に準拠した制御ベース表現。ego pose 軌道から最小二乗法で逆算する。

**ユニサイクルダイナミクス（Euler 積分）**:

```
x_{i+1} = x_i + Δt/2 * (v_i cos θ_i + v_{i+1} cos θ_{i+1})
y_{i+1} = y_i + Δt/2 * (v_i sin θ_i + v_{i+1} sin θ_{i+1})
θ_{i+1} = θ_i + Δt * κ_i * v_i + Δt²/2 * κ_i * a_i
v_{i+1} = v_i + Δt * a_i
```

- Δt = 0.1s（10Hz）— Alpamayo 論文値
  - **実装上の注記**: nuScenes keyframe は 2Hz のため、実装では dt=0.5s を使用。将来 10Hz 補間を実装した場合に dt=0.1s に戻す
- 予測ホライズン: 64 waypoints = 6.4秒

**GT 制御列の逆算手順**:

1. nuScenes の ego pose 列（キーフレーム 2Hz）を取得
2. 10Hz に補間（線形補間 + slerp for quaternion）
3. 各フレームの (x, y, θ, v) を計算
4. 既知の (x, y, θ, v) 軌道からユニサイクルダイナミクスの逆問題を解く
5. 最小二乗法 + Tikhonov 正則化で (a, κ) 制御列を推定
   - 正則化パラメータ λ で制御入力の滑らかさを調整

```python
def inverse_dynamics_np(positions, headings, dt=0.5, v_threshold=0.1, lambda_reg=1e-2):
    """(x, y, θ) 軌道から (a, κ) 制御列をTikhonov正則化で逆算する。"""
    # 速度: 位置の差分
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt

    # Tikhonov 正則化による加速度推定
    # 有限差分行列 D で微分を表現し、正則化項で平滑化
    # a = (D^T D + λI)^{-1} D^T b
    D = finite_difference_matrix(K, dt)
    raw_a = D @ speeds
    a = np.linalg.solve(D.T @ D + lambda_reg * np.eye(K+1), D.T @ raw_a)
    a = D @ a  # 正則化された加速度

    # 曲率: heading 差分と速度から算出、Tikhonov 正則化で平滑化
    # L は平滑化行列（隣接差分）
    raw_kappa = np.diff(headings) / (speeds_clipped * dt)
    kappa = np.linalg.solve(np.eye(K) + lambda_reg * L.T @ L, raw_kappa)

    return a, kappa  # (K,), (K,)
```

### 前処理パイプライン

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])
```

---

## プロジェクト構成（Stage 0 で追加するファイル）

```
minipamayo/
├── pyproject.toml                              # 新規: プロジェクト設定
├── src/minipamayo/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py                   # 新規: DINOv2 ViT-B/14 ラッパー
│   │   ├── adapter.py                          # 新規: Vision→LLM Adapter
│   │   ├── llm.py                              # 新規: Qwen2.5-0.5B ラッパー
│   │   ├── action_head.py                      # 新規: MLP 回帰ヘッド
│   │   ├── dynamics.py                         # 新規: ユニサイクルダイナミクス
│   │   └── minipamayo.py                       # 新規: 統合モデル
│   ├── data/
│   │   ├── __init__.py
│   │   ├── nuscenes_dataset.py                 # 新規: nuScenes データセット
│   │   ├── action_label.py                     # 新規: GT 制御列の逆算
│   │   └── transforms.py                       # 新規: 画像前処理
│   └── training/
│       ├── __init__.py
│       ├── trainer.py                          # 新規: 学習ループ
│       └── losses.py                           # 新規: Huber loss 等
├── configs/
│   └── stage0.yaml                             # 新規: Stage 0 設定
├── scripts/
│   ├── train_stage0.py                         # 新規: 学習スクリプト
│   └── eval_stage0.py                          # 新規: 評価スクリプト
├── data/                                       # データディレクトリ（.gitignore）
│   └── nuscenes/                               # nuScenes データ
└── checkpoints/                                # チェックポイント（.gitignore）
    └── stage0/
```

---

## 実装ステップ

### Phase 3: fail-fast（最小回帰）

パイプライン全体の動作検証。`[steer, throttle]` (2,) の回帰で勾配が流れることを確認する。

#### 3.1 モジュール実装

- [x] **Vision Encoder** (`models/vision_encoder.py`)
  - `facebook/dinov2-base` をロード
  - 入力: (B, 3, 224, 224) → 出力: (B, 256, 768) パッチ特徴
  - gradient checkpointing 対応
  - 86M params, trainable

- [x] **Adapter** (`models/adapter.py`)
  - Phase 3: 平均 Pool + Linear（最小実装）
    - (B, 256, 768) → 平均Pool → (B, 768) → Linear → (B, 896)
    - 視覚トークン 1 個として LLM に注入
  - ~0.7M params, trainable

- [x] **LLM** (`models/minipamayo.py` に統合)
  - `Qwen/Qwen2.5-0.5B` をロード
  - 視覚トークンを embedding 空間に直接注入（input_embeds 経由）
  - 最終層 hidden state を取り出す
  - gradient checkpointing 対応
  - 494M params, trainable

- [x] **Action Head** (`models/action_head.py`)
  - Phase 3: MLP (896 → 256 → 2)
    - 入力: LLM 最終層 hidden state
    - 出力: [steer, throttle] (2,)
  - <0.3M params, trainable

#### 3.2 統合モデル MiniPamayo クラス

- [x] `models/minipamayo.py` に統合モデルを実装
  ```python
  class MiniPamayo(nn.Module):
      def __init__(self, vision_encoder, adapter, llm, action_head):
          ...

      def forward(self, images, attention_mask=None):
          # 1. Vision Encoder
          patch_features = self.vision_encoder(images)  # (B, 256, 768)
          # 2. Adapter
          visual_tokens = self.adapter(patch_features)   # (B, N_vis, 896)
          # 3. LLM
          llm_hidden = self.llm(visual_tokens)           # (B, seq_len, 896)
          # 4. Action Head
          action = self.action_head(llm_hidden[:, -1])   # (B, 2)
          return action
  ```
- [x] ドメイン SFT の学習済み重み（Vision Encoder + Adapter + LLM）をロード
- [x] 全モジュール trainable（勾配が全経路に流れることを確認）

#### 3.3 データセット

- [x] `data/nuscenes_dataset.py` を実装
  - nuScenes Mini から CAM_FRONT 画像 + ego pose を読み込み
  - GT アクション: ego pose 差分から [steer, throttle] を近似算出
  - 画像前処理: 224×224 リサイズ + ImageNet 正規化
  - `__getitem__` → (image, action) タプル

#### 3.4 学習ループ

- [x] `train_stage0.py` に学習ループを実装（`training/trainer.py` ではなく単一スクリプトに統合）
  - micro-batch = 1
  - gradient accumulation = 16（実効バッチサイズ 16）
  - bf16 mixed precision（`torch.amp.autocast`）
  - gradient checkpointing（DINOv2 + LLM）
  - AdamW optimizer（lr=1e-4, weight_decay=0.01）
  - Huber loss（δ=1.0）
  - cosine annealing with warmup（warmup_ratio=0.1）

#### 3.5 wandb ロギング

- [x] 以下をロギング（stdout + wandb オプション）:
  - train/loss（step ごと）
  - train/lr（step ごと）
  - 各モジュールの grad norm（VE, Adapter, LLM, Action Head 個別）
  - VRAM 使用量（学習終了時）
  - [ ] 推論サンプル（数ステップごとに予測 vs GT を記録）— 未実装

#### 3.6 VRAM 実測

- [x] 学習開始直後の VRAM 使用量を `torch.cuda.max_memory_allocated()` で記録
- [x] 見積もり: ~10 GB → 実測: **5.69 GB**（見積もりより大幅に少ない）
- [x] 24 GB 以内であることを確認

### Phase 4: 制御ベース表現

パイプラインが安定したら、Alpamayo と同一の制御ベース表現に移行する。

#### 4.1 ユニサイクルダイナミクス実装

- [ ] `models/dynamics.py` を実装
  - `forward_dynamics(a, kappa, x0, y0, theta0, v0, dt)`: 制御入力 → 軌道変換
  - `inverse_dynamics(positions, headings, velocities, dt, lambda_reg)`: 軌道 → 制御入力の逆算
  - Euler 積分:
    ```
    x_{i+1} = x_i + Δt/2 * (v_i cos θ_i + v_{i+1} cos θ_{i+1})
    y_{i+1} = y_i + Δt/2 * (v_i sin θ_i + v_{i+1} sin θ_{i+1})
    θ_{i+1} = θ_i + Δt * κ_i * v_i + Δt²/2 * κ_i * a_i
    v_{i+1} = v_i + Δt * a_i
    ```
  - 逆算: 最小二乗法 + Tikhonov 正則化
  - 単体テスト: forward → inverse → forward の往復で誤差が十分小さいことを確認

#### 4.2 Action Head 拡張

- [ ] Action Head の出力を `[steer, throttle]` (2,) → `(a, κ)` (64, 2) に拡張
  - MLP: (896 → 512 → 256 → 128) — 64 waypoints × 2 values
  - 出力を (64, 2) にリシェイプ
- [ ] Loss: Huber loss on (a, κ) 制御入力列
  ```python
  loss = F.huber_loss(pred_controls, gt_controls, delta=1.0)  # (B, 64, 2)
  ```

#### 4.3 GT 制御列の逆算パイプライン

- [ ] `data/action_label.py` を実装
  - nuScenes の ego pose 列（2Hz キーフレーム）を取得
  - 10Hz に補間（線形 + quaternion slerp）
  - 各フレームの (x, y, θ, v) を算出
  - `inverse_dynamics()` で (a, κ) 制御列を逆算
  - 逆算結果を事前計算してキャッシュ（JSON / HDF5）

#### 4.4 Adapter 改善（平均 Pool + Linear → Cross-Attention Pooling）

- [ ] `models/adapter.py` に `CrossAttentionAdapter` を追加
  - learnable query: 16 個 (16, 896)
  - DINOv2 パッチ特徴 (256, 768) を key/value に使用
  - Multi-Head Cross-Attention + FFN
  - 出力: (B, 16, 896) — 16 視覚トークンとして LLM に注入
  - ~5M params（attention + FFN）

- [ ] 性能比較:
  - 平均 Pool + Linear（Phase 3）vs Cross-Attention Pooling（Phase 4）
  - 比較指標: loss 低下速度、最終 loss、ADE/FDE

#### 4.5 評価指標（ADE/FDE）

- [ ] 制御入力 (a, κ) → `forward_dynamics()` → waypoint (x, y) に変換
- [ ] **ADE (Average Displacement Error)**: 全 waypoints の L2 距離の平均
  ```python
  ade = torch.mean(torch.norm(pred_waypoints - gt_waypoints, dim=-1))
  ```
- [ ] **FDE (Final Displacement Error)**: 最終 waypoint の L2 距離
  ```python
  fde = torch.norm(pred_waypoints[:, -1] - gt_waypoints[:, -1], dim=-1).mean()
  ```
- [ ] wandb にログ: eval/ade, eval/fde

#### 4.6 可視化（予測軌道 vs GT をカメラ画像上にプロット）

- [ ] 予測 (a, κ) → forward_dynamics → (x, y) waypoints
- [ ] GT (a, κ) → forward_dynamics → (x, y) waypoints
- [ ] BEV（鳥瞰図）上に両軌道をプロット
- [ ] カメラ画像上に軌道を射影してオーバーレイ（nuScenes のカメラ内部・外部パラメータを使用）
- [ ] wandb に画像をログ

---

## ハイパーパラメータ

### Phase 3（fail-fast）

```yaml
# モデル
vision_encoder: facebook/dinov2-base       # ViT-B/14, 86M params
llm: Qwen/Qwen2.5-0.5B                     # 494M params
adapter: mean_pool_linear                   # 平均 Pool + Linear (768→896)
n_visual_tokens: 1                          # 最小実装
action_output: 2                            # [steer, throttle]

# 学習
micro_batch_size: 1
grad_accumulation_steps: 16
precision: bf16
gradient_checkpointing: true
optimizer: AdamW
learning_rate: 1.0e-4
weight_decay: 0.01
scheduler: cosine_with_warmup
warmup_ratio: 0.1
max_epochs: 10                              # fail-fast なので小さく
loss: huber                                 # delta=1.0

# 勾配制御
vision_encoder_trainable: true
adapter_trainable: true
llm_trainable: true
action_head_trainable: true
max_grad_norm: 1.0

# データ
image_size: 224
dataset: nuscenes
nuscenes_version: v1.0-mini                 # パイプライン検証
camera: CAM_FRONT
```

### Phase 4（制御ベース表現）

```yaml
# モデル変更点
adapter: cross_attention_pooling            # Cross-Attention Pooling
n_visual_tokens: 16                         # 16 queries
action_output: 128                          # (a, κ) × 64 waypoints

# アクション表現
action_representation: unicycle_control     # 制御ベース表現
prediction_horizon: 64                      # 64 waypoints
prediction_hz: 10                           # 10Hz
prediction_seconds: 6.4                     # 6.4秒

# GT 逆算
inverse_dynamics_lambda: 1.0e-3             # Tikhonov 正則化パラメータ

# 学習
nuscenes_version: v1.0-trainval             # 本番データ
max_epochs: 20
learning_rate: 5.0e-5                       # Phase 3 より小さく
```

---

## Exit 条件

### Phase 3 Exit 条件（fail-fast）

| 条件 | 基準 | 備考 |
|---|---|---|
| **勾配伝播** | 全モジュール（DINO, Adapter, LLM, Action Head）に non-zero gradient | wandb の per-module grad norm で確認 |
| **Loss 低下** | 学習 loss が初期値から安定して低下 | overfitting でも可（パイプライン検証が目的） |
| **OOM なし** | VRAM 使用量 24 GB 以内 | `torch.cuda.max_memory_allocated()` で実測 |
| **入力依存性** | 異なる画像に対して異なる出力が得られる | 固定出力でないことを確認 |

### Phase 4 Exit 条件（制御ベース表現）

| 条件 | 基準 | 備考 |
|---|---|---|
| **Loss 低下** | 制御ベース表現で loss が安定して低下 | (a, κ) の Huber loss |
| **ADE/FDE** | 意味のある値を示す（ランダムより大幅に良い） | 絶対値より低下傾向を重視 |
| **可視化** | 予測軌道がおおよそ妥当な方向を向いている | BEV プロット + カメラ画像オーバーレイ |
| **Adapter 比較** | Cross-Attention Pooling が平均 Pool + Linear と同等以上 | loss / ADE / FDE で比較 |
| **VRAM** | 24 GB 以内 | Cross-Attention Pooling 追加後も収まること |

### Phase 4 → Stage 1 移行条件

Phase 4 の Exit 条件をすべて満たしたうえで:

- 制御ベース表現の GT 逆算が安定していること（forward → inverse → forward の round-trip 誤差が十分小さい）
- ADE/FDE が改善傾向にあること（完全な収束は不要）
- 可視化で軌道の向きが概ね正しいこと
