# Alpamayo 公開コード vs MiniPamayo 実装比較

Alpamayo 公開リポジトリ（推論コードのみ）の全ソースコードと MiniPamayo の対応実装を
比較し、将来の実装改善の参考としてまとめる。

> **対象**: `related_repos/alpamayo/src/alpamayo_r1/` の全ファイル vs `src/minipamayo/` の対応ファイル

---

## 1. ディレクトリ構造の対応関係

```
alpamayo_r1/                          minipamayo/
├── action_space/                     ├── models/
│   ├── action_space.py (ABC)         │   ├── dynamics.py ← 対応
│   ├── unicycle_accel_curvature.py   │   │   (forward/inverse dynamics)
│   ├── utils.py (Tikhonov solver)    │   │
│   ├── discrete_action_space.py      │   ├── discrete_head.py ← 対応
│   │   (DiscreteTrajectoryTokenizer) │   │   (DiscreteActionTokenizer)
│   └── __init__.py                   │   │
├── diffusion/                        │   ├── trajectory_decoder.py ← 対応
│   ├── base.py (BaseDiffusion ABC)   │   │   (TrajectoryDecoder + cfm_loss/sample)
│   └── flow_matching.py              │   │
├── models/                           │   ├── minipamayo.py ← 対応
│   ├── base_model.py (ReasoningVLA)  │   │   (MiniPamayo: DINOv2+Adapter+Qwen)
│   ├── alpamayo_r1.py (Expert分離)   │   │
│   ├── action_in_proj.py (Fourier)   │   │   ← FourierFeatureV2 として実装済み
│   ├── delta_tokenizer.py (履歴)     │   │   ← MiniPamayo には対応なし
│   └── token_utils.py                │   │
├── geometry/rotation.py              │   │   ← MiniPamayo は pyquaternion で代替
├── config.py                         │   │
├── helper.py                         │   │
├── test_inference.py                 │   │
└── load_physical_aiavdataset.py      ├── data/
                                      │   ├── nuscenes_trajectory_dataset.py
                                      │   ├── coc_dataset.py
                                      │   └── coc_labeling.py
                                      ├── rewards.py ← Alpamayoに対応なし(非公開)
                                      ├── train_stage*.py
                                      └── eval_stage*.py
```

---

## 2. Action Space（動力学変換）

### 2.1 逆動力学（軌跡 → action GT）

| | Alpamayo (`unicycle_accel_curvature.py` + `utils.py`) | MiniPamayo (`dynamics.py`) |
|---|---|---|
| 実装言語 | **PyTorch** (バッチ対応, GPU) | **NumPy** (CPU, 単一サンプル) |
| 速度の復元 | **台形積分の逆問題**: `Δp = dt/2 × (v_t u_t + v_{t+1} u_{t+1})` を最小二乗で解く | **有限差分**: `v = \|\|Δp\|\| / dt` |
| v0 拘束 | 履歴から `estimate_t0_states()` で推定し、Lagrange 乗数で固定 | なし（差分 v0 をそのまま使用） |
| heading 平滑化 | `theta_smooth()`: rotation matrix → yaw → unwrap → **3次 Tikhonov** で平滑化 | なし（`Quaternion.yaw_pitch_roll` を直接使用） |
| 加速度 | **2次 Tikhonov** (`w_smooth2=1.0`): jerk の smoothness を保証 | **2次 Tikhonov** (`a_lambda/dt⁴`): **Alpamayo 準拠** |
| 曲率 | `dθ / (dt·v + dt²·a/2)` を **2次 Tikhonov** で解く | `dθ / (dt·v + dt²·a/2)` を **2次 Tikhonov** で解く: **Alpamayo 準拠** |
| 角度正規化 | `atan2(sin, cos)` (`round_2pi`) | `atan2(sin, cos)`: **Alpamayo 準拠** |
| 正規化 | `accel_mean/std`, `curvature_mean/std` で正規化 | `accel_mean/std`, `kappa_mean/std` で正規化: **Alpamayo 準拠** |
| ソルバー | Cholesky 分解 (`torch.linalg.cholesky`) + 動的 ridge | `np.linalg.solve` + 固定 ridge |

#### Alpamayo の速度復元の詳細

```python
# Alpamayo: dxy_theta_to_v() — 台形積分の逆問題
# Δp_t = dt/2 * (v_t u_t + v_{t+1} u_{t+1}) を解く
# u_t = [cos θ_t, sin θ_t]
# → 2N 本の連立方程式 (各 Δp が xy 2成分) を N+1 個の v について最小二乗
# + 3次 Tikhonov 正則化 (w_smooth3=1.0) で速度変化を滑らかに
```

```python
# MiniPamayo: inverse_dynamics_np() — 有限差分 + 2次 Tikhonov
# v_t = ||p_{t+1} - p_t|| / dt（速度復元は有限差分のまま）
# 加速度: raw_a = Δv/dt → (I + λ/dt⁴·D₂ᵀD₂ + ridge·I)⁻¹ raw_a
# 曲率: s = dt·v + dt²/2·a → (SᵀS + λ/dt⁴·D₂ᵀD₂ + ridge·I)⁻¹ Sᵀ·Δθ
```

#### Tikhonov 正則化の次数の意味

| 次数 | 正則化対象 | 物理的意味 | Alpamayo の使い方 | MiniPamayo |
|---|---|---|---|---|
| 1次 | `Δx` (1次差分) | x の変化量を小さく | — | — |
| 2次 | `Δ²x` (2次差分) | x の変化の滑らかさ | **加速度・曲率** (`w_smooth2=1.0`) | **加速度・曲率** (`a_lambda/kappa_lambda`) |
| 3次 | `Δ³x` (3次差分) | 変化の変化の滑らかさ | **速度・heading** (`w_smooth3=1.0`) | — |

**残存差分**: 速度復元（有限差分 vs 台形逆問題）、heading 平滑化（なし vs 3次 Tikhonov）、v0 拘束（なし vs Lagrange）、ソルバー（np.linalg.solve vs Cholesky+動的ridge）。K=6 では影響小。

### 2.2 順動力学（action → 軌跡）

| | Alpamayo (`action_to_traj()`) | MiniPamayo (`forward_dynamics_batch()`) |
|---|---|---|
| 速度積分 | `v = v0 + cumsum(a·dt)` (ベクトル化) | ループ: `v_new = v + dt·a` |
| heading 積分 | `θ = cumsum(κ·v·dt + κ·a·dt²/2)` (ベクトル化) | ループ: `θ_new = θ + dt·κ·v + dt²/2·κ·a` |
| 位置積分 | `x = cumsum(v_t cos θ_t + v_{t+1} cos θ_{t+1})·dt/2` | 同上の台形積分 |
| 正規化 | 逆正規化してから積分 | 逆正規化してから積分: **Alpamayo 準拠** |
| 出力 | 3D xyz + 3×3 回転行列 | 2D xy のみ |

**数学的には同じ台形ユニサイクルモデル**。MiniPamayo はループで逐次計算、Alpamayo は cumsum でベクトル化。

### 2.3 物理制約チェック

| | Alpamayo | MiniPamayo |
|---|---|---|
| `is_within_bounds()` | **あり**: `a ∈ [-9.8, 9.8]`, `κ ∈ [-0.2, 0.2]` (正規化前の値でチェック) | **なし** |
| 用途 | GRPO で物理的に不可能なサンプルを reject/penalize | — |

---

## 3. 離散化（Tokenization）

### 3.1 Alpamayo: `DiscreteTrajectoryTokenizer`

```
軌跡 (xyz, rot) → ActionSpace.traj_to_action() → 正規化済み action (N, 2)
  → dims_min/dims_max で [0, 1] にスケール
  → round(× (num_bins-1)) → [0, num_bins-1]
  → flatten → token IDs
```

- `num_bins` = 768（設定可能）
- `ActionSpace` をラップ → 軌跡の入出力を直接扱える
- 正規化パラメータを外部 config で指定

### 3.2 Alpamayo: `DeltaTrajectoryTokenizer`

```
履歴軌跡の Δxyz を直接量子化 (位置差分ベース)
  → xyz_min/max で [0, 1] にスケール
  → round → token IDs
  → predict_yaw=True なら Δyaw も追加
```

- `num_bins` = 1000
- 履歴軌跡の入力用（VLM の入力に `<traj_history>` として注入）

### 3.3 MiniPamayo: `DiscreteActionTokenizer`

```
interleaved action [a_0, κ_0, a_1, κ_1, ...]
  → 偶数 index: a_range=(-6.0, 6.0) で量子化
  → 奇数 index: kappa_range=(-0.1, 0.1) で量子化
  → 共有 256 bins, vocab_offset=151936
```

### 3.4 差分詳細

| | Alpamayo | MiniPamayo |
|---|---|---|
| bins 数 | 768 | 256 |
| 量子化対象 | **正規化後**の値 → 分布が均一に近い | **生値** → 多くの値が中心付近に集中 |
| a の量子化精度 | 768 bins × 正規化で均等 | 256 bins / 12.0 = **0.047 m/s²/bin** |
| κ の量子化精度 | 同上 | 256 bins / 0.2 = **0.00078 rad/m/bin** |
| トークン配置 | `(a_0, κ_0), (a_1, κ_1), ...` を flatten (N×2) | interleaved: `a_0, κ_0, a_1, κ_1, ...` (K×2) |
| 対称的な量子化 | `dims_min/dims_max` でチャネル別に範囲指定 | a と κ でハードコーデッドされた異なる範囲 |

---

## 4. Flow Matching (Diffusion)

### 4.1 Alpamayo

```
diffusion/
├── base.py       — BaseDiffusion ABC, StepFn Protocol
└── flow_matching.py — FlowMatching (Euler sampler, num_inference_steps=10)
```

- `x_dims` = ActionSpace の次元 (例: `(64, 2)`)
- step_fn = **Expert Transformer** の forward（VLM の KV-cache を past_key_values として使用）
- `sample()`: `randn(B, *x_dims)` → Euler 積分 → action 空間のサンプル

### 4.2 MiniPamayo

```
models/trajectory_decoder.py
├── TrajectoryDecoder  — Expert (24-layer Qwen2) + Fourier Feature V2 + KV-cache conditioning
├── cfm_loss()         — Gaussian OT loss with Beta(2, 5) schedule, action normalization
└── cfm_sample()       — Euler sampler (10-20 steps), action denormalization
```

- `action_dim` = K × 2（フラット 1D、例: 12）
- 条件: VLM の **KV-cache** を `past_key_values` として Expert に渡す（Alpamayo 準拠）
- Expert: Qwen2.5-0.5B の `text_config` から構築した 24 層 Transformer（`embed_tokens` 削除）

### 4.3 差分詳細

| | Alpamayo | MiniPamayo |
|---|---|---|
| Denoiser 実体 | **Expert Transformer** (VLM とは別モデル、VLM の text config ベース) | **Expert Transformer** (Qwen2.5-0.5B の text_config ベース、24層): **Alpamayo 準拠** |
| 入力エンコーディング | **Fourier Feature V2** (対数間隔周波数) + MLP per waypoint | **Fourier Feature V2** (対数間隔周波数) + MLP per waypoint: **Alpamayo 準拠** |
| Time embedding | **Fourier Feature V2** | **Fourier Feature V2**: **Alpamayo 準拠** |
| 条件付け方法 | VLM の **KV-cache** を `past_key_values` として Expert に渡す | VLM の **KV-cache** を `past_key_values` として Expert に渡す: **Alpamayo 準拠** |
| Non-causal attention | `is_causal=False` (config.expert_non_causal_attention) | `is_causal=False`: **Alpamayo 準拠** |
| KV-cache 管理 | `DynamicCache.crop()` ネイティブメソッド | `DynamicCache.crop()`: **Alpamayo 準拠** |
| Action 正規化 | `accel_mean/std`, `curvature_mean/std` で正規化 | `accel_mean/std`, `kappa_mean/std` で正規化: **Alpamayo 準拠** |
| Action 形状 | `(N, 2)` — waypoint ごとに 2D | `(K×2,)` — フラット 1D |
| 学習 loss | 非公開 | Gaussian OT CFM loss + Beta(2,5) schedule |
| 推論ステップ数 | 10 | 10-20 |

#### Fourier Feature V2 の詳細

```python
# Alpamayo / MiniPamayo 共通: FourierFeatureV2
# 各アクション次元を個別に Fourier encoding → concat → MLP
class FourierFeatureV2(nn.Module):
    freqs = logspace(0, log10(100), steps=half_dim)  # 対数間隔
    forward(x) → cat[sin(x·freqs·2π), cos(x·freqs·2π)] × √2

# 各 waypoint に対して:
#   Fourier(a_i) || Fourier(κ_i) || Fourier(timestep) → MLP (4層, 1024 hidden, RMSNorm+SiLU) → token embedding
```

MiniPamayo は Alpamayo と同等の Fourier Feature V2 を実装済み。

---

## 5. モデルアーキテクチャ

### 5.1 Alpamayo: VLM + Expert 分離

```
VLM (Qwen3-VL-8B)
  │ autoregressive generation
  │ CoC テキスト: <cot_start>...<cot_end><meta_action_start>...<meta_action_end>
  │ <traj_future_start> が出たら停止
  │
  └─→ KV-cache を保存 (past_key_values)
        │
Expert Transformer (text_config ベースの小型 LM)
  │ 入力: action_in_proj(noisy_action, timestep) → token embeddings
  │ 条件: VLM の KV-cache (past_key_values)
  │ attention_mask: VLM コンテキストの一部をマスク
  │ 出力: action_out_proj(hidden) → velocity field
  │
Flow Matching sampler (Euler ×10 steps)
  │ 複数サンプル生成 (num_traj_samples)
  │
ActionSpace.action_to_traj() → 軌跡 (xyz, rot)
```

- `ExpertLogitsProcessor`: VLM 生成時に trajectory token の logits を `-inf` にマスク（テキスト生成に干渉しない）
- Expert は VLM とは独立した小型 LM（embed_tokens は削除）
- multi-sample: KV-cache を再利用して trajectory だけ複数回サンプル

### 5.2 MiniPamayo: VLM + Expert 分離（Alpamayo 準拠）

```
VLM: DINOv2 → Adapter → Qwen2.5-0.5B
  │
  ├── Stage 1: LLM が直接 discrete action tokens を生成
  ├── Stage 2: KV-cache → Expert Transformer (Flow Matching)  ← Alpamayo 準拠
  ├── Stage 3: CoC テキスト + action tokens を同時生成 (SFT)
  └── Stage 4: GRPO で LLM を RL ポストトレーニング

Expert Transformer (text_config ベース, 24層 Qwen2)
  │ 入力: FourierFeatureV2(noisy_action, timestep) → token embeddings
  │ 条件: VLM の KV-cache (past_key_values), is_causal=False
  │ 出力: action_out_proj(hidden) → velocity field
  │
Flow Matching sampler (Euler ×10 steps)
  │ denormalize → 軌跡 (a, κ)
  │
forward_dynamics → waypoints (x, y)
```

### 5.3 主要な構造差

| | Alpamayo | MiniPamayo |
|---|---|---|
| テキスト/軌跡の分離 | VLM (テキスト) → Expert (軌跡) の 2 段階 | VLM → Expert の 2 段階: **Alpamayo 準拠** |
| Multi-sample 効率 | KV-cache 再利用で軌跡だけ再サンプル | KV-cache clone-crop-reuse で同様に実装可能 |
| 履歴 trajectory 入力 | `DeltaTrajectoryTokenizer` で離散化し VLM 入力に含める (48 tokens) | なし（画像 + v0 のみ） |
| 軌跡生成時の VLM テキスト | ExpertLogitsProcessor で trajectory tokens をマスク | Stage 2 は VLM とは別に Expert を学習 |

---

## 6. 特殊トークンとチャットテンプレート

### 6.1 Alpamayo

```
System: "You are a driving assistant..."
User: [images]<traj_history_start><traj_history>×48<traj_history_end>
      output the chain-of-thought reasoning..., then output the future trajectory.
Assistant: <cot_start>{reasoning}<cot_end>
           <meta_action_start>{meta_action}<meta_action_end>
           <traj_future_start>{trajectory tokens}<traj_future_end>
```

特殊トークン一覧 (SPECIAL_TOKENS_KEYS):
`prompt_start/end`, `image_start/pre_tkn/end`, `traj_history_start/pre_tkn/end`,
`cot_start/end`, `meta_action_start/end`, `traj_future_start/pre_tkn/end`,
`traj_history`, `traj_future`, `image_pad`, `vectorized_wm*`, `route_start/pad/end`,
`question_start/end`, `answer_start/end`

### 6.2 MiniPamayo

```
<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{visual_tokens} Speed: {v0:.1f} m/s. What driving action should be taken?<|im_end|>
<|im_start|>assistant
{CoC reasoning text}{action tokens}<|im_end|>
```

### 6.3 差分

| | Alpamayo | MiniPamayo |
|---|---|---|
| CoC セクション | `<cot_start>...<cot_end>` で明示的に区切り | assistant ターン内にテキストとして出力 |
| Meta-action | `<meta_action_start>...<meta_action_end>` で独立セクション | CoC reasoning text 内に含まれる |
| 軌跡開始マーカー | `<traj_future_start>` で VLM → Expert に切り替え | 区切りなし（action tokens が直接続く） |
| 履歴軌跡 | `<traj_history>` ×48 tokens | なし |
| 速度情報 | 履歴軌跡の離散トークンに暗黙的に含まれる | `Speed: {v0} m/s` としてテキストで明示 |

---

## 7. ユーティリティ

### 7.1 Geometry (`geometry/rotation.py`)

Alpamayo は rotation matrix, yaw, 座標変換のユーティリティを提供:
- `so3_to_yaw_torch/np`: 3×3 回転行列 → yaw 角
- `rotation_matrix_torch`: yaw → 2×2 回転行列
- `rot_2d_to_3d` / `rot_3d_to_2d`: 2D/3D 回転行列の変換
- `round_2pi_torch`: 角度の正規化
- `stable_gramschmidt`: Gram-Schmidt 直交化

MiniPamayo は `pyquaternion.Quaternion` で代替。

### 7.2 Helper (`helper.py`)

- `create_message()`: Qwen3-VL 用のメッセージ構築（マルチカメラ画像 + 履歴軌跡プレースホルダ）
- `get_processor()`: Qwen3-VL プロセッサの構築
- `to_device()`: 再帰的なデバイス/dtype 変換

### 7.3 データロード (`load_physical_aiavdataset.py`)

- NVIDIA `physical_ai_av` パッケージからデータをロード
- 4 カメラ × 4 フレーム + ego pose (history 16 + future 64 steps @ 10Hz)
- ローカル座標系（t0 の ego pose を原点）に変換
- MiniPamayo は `NuScenesTrajectoryDataset` で nuScenes から直接ロード

---

## 8. Alpamayo にあって MiniPamayo にないもの

| 機能 | Alpamayo のファイル | 用途 | MiniPamayo での状態 |
|---|---|---|---|
| **Expert 分離** | `models/alpamayo_r1.py` | VLM と trajectory decoder の分離 | ✅ **実装済み** (KV-cache 条件付け) |
| **Fourier Feature V2** | `models/action_in_proj.py` | noisy action の高精度エンコーディング | ✅ **実装済み** |
| **is_causal=False** | `models/alpamayo_r1.py` | Expert の non-causal attention | ✅ **実装済み** |
| **DynamicCache.crop()** | `models/alpamayo_r1.py` | KV-cache の crop | ✅ **実装済み** |
| **Action 正規化** | `unicycle_accel_curvature.py` | mean/std で正規化 | ✅ **実装済み** |
| **2次 Tikhonov** | `action_space/utils.py` | jerk smoothness 保証 | ✅ **実装済み** |
| **曲率の運動学的分母** | `unicycle_accel_curvature.py` | `s = dt·v + dt²/2·a` | ✅ **実装済み** |
| **DeltaTrajectoryTokenizer** | `models/delta_tokenizer.py` | 履歴軌跡の離散化 | 未実装（履歴入力未対応） |
| **ExpertLogitsProcessor** | `models/alpamayo_r1.py` | テキスト生成時の trajectory token マスク | 未実装（現段階では不要） |
| **token_utils** | `models/token_utils.py` | traj token 抽出、special token パース | 未実装 |
| **is_within_bounds()** | `action_space/unicycle_accel_curvature.py` | 物理制約チェック | 未実装（GRPO 時に追加予定） |
| **台形逆問題 (速度復元)** | `action_space/utils.py` | 高品質な GT action 抽出 | 未実装（K=6 では影響小） |
| **heading 3次 Tikhonov** | `action_space/utils.py` | heading の平滑化 | 未実装（K=6 では影響小） |

---

## 9. MiniPamayo にあって Alpamayo にないもの

| 機能 | MiniPamayo のファイル | 備考 |
|---|---|---|
| **Reward 関数** | `rewards.py` | Alpamayo の学習コードは非公開 |
| **CoC ラベリング** | `data/coc_labeling.py` | OpenAI API で自動ラベリング |
| **Beta schedule** | `models/trajectory_decoder.py` | `Beta(2, 5)` shifted time sampling（Alpamayo の schedule は非公開） |
| **Obstacle collision check** | `rewards.py` | `has_collision()` + inflated OBB |
| **Stage 別学習設定** | `models/minipamayo.py` | `set_stage0()` ~ `set_stage4()` |

---

## 10. 改善候補の優先順位

### ✅ 実装済み

1. ~~**Expert 分離アーキテクチャ**~~ → KV-cache 条件付け、24 層 Qwen2 Expert
2. ~~**Fourier Feature V2 encoding**~~ → 対数間隔周波数 + 4 層 MLP
3. ~~**正規化パラメータの導入**~~ → accel/kappa の mean/std 正規化
4. ~~**Tikhonov 正則化を 2 次に変更**~~ → `_second_order_diff_matrix` + `λ/dt⁴` scaling
5. ~~**曲率の運動学的分母**~~ → `s = dt·v + dt²/2·a`
6. ~~**is_causal=False**~~ → Expert forward に明示
7. ~~**DynamicCache.crop()**~~ → ネイティブメソッドに移行

### 未実装（今後の候補）

#### 高優先（GRPO に直結）

1. **`is_within_bounds()` の追加**
   - 実装コスト: 低（数行）
   - GRPO で物理的に不可能な action をペナルティ/reject
   - Alpamayo のデフォルト: `a ∈ [-9.8, 9.8]`, `κ ∈ [-0.2, 0.2]`

#### 中優先（K 拡大時に重要）

2. **逆動力学の速度復元を台形逆問題に変更**
   - 実装コスト: 高
   - 現状の K=6 ではインパクト小、K=64 に拡大した場合に大きな品質差
   - 参照: `utils.py` `dxy_theta_to_v()`

3. **heading の 3 次 Tikhonov 平滑化**
   - Alpamayo は `theta_smooth()` で heading を平滑化してから曲率計算
   - K=6 では heading noise の影響小

4. **bins 数の増加** (256 → 512 or 768)
   - 実装コスト: 低（パラメータ変更のみ）
   - LLM 語彙サイズ増加とのトレードオフ

#### 低優先

5. **履歴 trajectory の入力**
   - Alpamayo: 過去 16 ステップを `DeltaTrajectoryTokenizer` で離散化
   - 現状: v0 のみ → 加減速の文脈が不足

6. **Action を (N, 2) 形状で扱う**
   - 現在のフラット 1D → waypoint ごとの 2D に変更
   - Transformer が空間的構造を捉えやすくなる

### 不要（MiniPamayo のスケールでは過剰）

- `ActionSpace` 抽象基底クラスの導入（ユニサイクルモデルのみ）
- `BaseDiffusion` 抽象クラスの導入（Flow Matching のみ）
- 3D (xyz + 回転行列) 対応（2D で十分）
- Hydra 設定管理（argparse で十分）
- `geometry/rotation.py` の完全再実装（`pyquaternion` で十分）
