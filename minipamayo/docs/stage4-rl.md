# Stage 4: RL ポストトレーニング — 実装計画

## 目的

SFT（Stage 3）だけでは残る推論品質の問題を RL で改善する。

- **データバイアス**: auto-labeling 由来の不完全な因果推論の学習
- **推論-行動の不一致**: 「止まる」と推論しながら止まらない等
- **ハルシネーション**: 視覚的根拠のない推論の生成

GRPO（Group Relative Policy Optimization）により、3 要素の報酬信号を用いてポリシーを最適化する。

---

## Alpamayo 論文との対応

| 観点 | Alpamayo（§5.3） | MiniPamayo Stage 4 |
|---|---|---|
| アルゴリズム | GRPO（softmax-weighted REINFORCE + KL） | **同一**（Alpamayo §5.3.2 準拠） |
| 報酬: 推論品質 | LRM（DeepSeek-R1, Cosmos-Reason）画像+GT+PRED | マルチモーダル LLM API（デフォルト: gpt-4o）画像+GT+PRED |
| 報酬: CoC-Action 一貫性 | ルールベース（バイナリ）| **同一** |
| 報酬: 軌道品質 | -(λ_L2·L2 + λ_coll·I[coll] + λ_jerk·J) ペナルティ形式 | **同一** |
| 勾配制御 | LLM のみ trainable | **同一** |
| Reference policy | SFT モデルを frozen で保持 | **同一** |
| データキュレーション | 内部予測と外部報酬の不一致が大きいサンプルを優先 | 全データ使用（データ量が小さいため） |
| 改善効果（Alpamayo） | 推論品質 +45%、一貫性 +37% | — |
| Cosmos Reason Mini での検証 | MCQ 正解率 38.3% → 86.7%（+48.4%） | 参考値 |

---

## 前提条件

- Stage 3（CoC 推論 SFT）が完了していること
- LLM が CoC 推論トレース + 離散軌道トークンを自己回帰生成できる状態
- SFT チェックポイントが保存されていること（reference policy として使用）

---

## 勾配制御

設計書 §3.7 に定義された方針:

| モジュール | 状態 | 備考 |
|---|---|---|
| Vision Encoder (DINOv2 ViT-B/14) | **frozen** | 勾配を流さない |
| Adapter | **frozen** | 勾配を流さない |
| LLM (Qwen2.5-0.5B) | **trainable** | GRPO で更新対象 |
| Flow Head (Trajectory Decoder) | **frozen** | 勾配を流さない |
| Reference policy (SFT モデル全体) | **frozen** | KL 計算用に保持 |

### VRAM 見積もり

| 項目 | サイズ | 備考 |
|---|---|---|
| LLM trainable (494M × 12 bytes) | 5.93 GB | params + optimizer + grads |
| Reference VLM frozen (582M × 2 bytes) | 1.16 GB | bf16 推論のみ |
| Activation + overhead | ~3 GB | gradient checkpointing 適用 |
| **合計** | **~10 GB** | RTX 4090 (24 GB) に対して ~14 GB の余裕 |

ロールアウト生成時は KV-cache が追加で必要だが、逐次生成のため VRAM はピーク ~10 GB を超えない想定。

---

## GRPO アルゴリズム

### 概要

GRPO は PPO の critic（value function）を排し、グループ内の相対報酬で advantage を推定する手法。VLM の RL に適している（大規模な critic を別途学習する必要がない）。

### 手順

1. **ロールアウト**: 現在のポリシー `π_θ` から各プロンプトに対して K 個の応答をサンプリング
2. **報酬計算**: 各応答に対して 3 要素の報酬を計算し、合成報酬 R(o_i) を算出
3. **Advantage 計算**: グループ G = {o_1, ..., o_K} 内の相対的な advantage（std 正規化なし）
   ```
   A_i = r_i - r̄  (group mean)
   ```
4. **ポリシー更新**: softmax-weighted policy gradient + KL 正則化（Alpamayo §5.3.2）
   ```
   L_GRPO = -Σ softmax(β_grpo * A_i) × (log π_θ(τ_i) - λ_KL * KL(π_θ || π_ref))
   ```
   ここで `β_grpo` は softmax temperature、`λ_KL` は KL ペナルティ係数。
   PPO-style の clipping は使用しない。

### KL 正則化

- Reference policy: Stage 3 SFT モデルを frozen で保持
- KL 項により、RL でポリシーが SFT モデルから過度に逸脱することを防ぐ
- λ_KL（KL 係数）でその強度を調整

---

## 報酬信号（3 要素）

### 1. 推論品質 (r_reason)

LRM（大規模推論モデル）による 0-5 スケールの比較採点（Alpamayo §5.3.2 準拠）。

**入力（3要素）**:
- **画像**: 運転シーンのカメラ映像（視覚的根拠の検証に使用）
- **GT CoC**: データセットからの正解推論トレース（比較基準）
- **PRED CoC**: 現在のポリシーが生成した推論トレース（評価対象）

**評価観点**:
- **行動一貫性**: PRED が GT と整合する運転行動を記述しているか
- **因果推論品質**: CoC の原則に従い、シーンで観察可能な因果要因を正しく特定しているか

**MiniPamayo での実装**:
- マルチモーダル LLM API（デフォルト: gpt-4o）— Vision API で画像入力に対応
- **オフライン計算**: API レイテンシのためオンライン計算は非現実的
  1. ロールアウト生成後、全応答をバッチで API に送信
  2. 採点結果をディスクキャッシュ（SHA256 ハッシュキー）
  3. ポリシー更新時にキャッシュから読み出し

**プロンプト**（Alpamayo 論文原文準拠）:
```
You are an expert evaluator for autonomous driving reasoning traces.
The reasoning trace describes what the ego vehicle should be doing and
the reasons and factors that lead to the behavior. Your task is to score
how well a predicted reasoning trace (PRED) aligns with the ground truth
(GT) in terms of behavior consistency and causal reasoning.

The attached image shows the driving scene from the front camera.
Use this visual context to verify the reasoning.

[Ground Truth Reasoning (GT)]
{gt_reasoning}

[Predicted Reasoning (PRED)]
{pred_reasoning}

Scoring rubric (0-5):
5 Behavior & causal reasoning fully consistent.
4 Behavior correct; causal reasoning mostly consistent.
3 Behavior roughly correct, but incomplete or slightly incorrect reasoning.
2 Behavior partially incorrect or reasoning largely inconsistent.
1 Behavior is wrong or contradicts GT.
0 Completely unrelated or opposite.

Score (0-5):
```

### 2. CoC-Action 一貫性 (r_consistency)

バイナリ報酬。推論トレースの意図と実際の予測軌道が一致しているかを検証する。

**手順**（Alpamayo §5.3.2: モデルの推論と行動の内部一貫性を評価）:
1. 予測軌道（離散トークン → 連続制御入力 → waypoint 列）から **meta-action** を抽出
2. **モデルが生成した**推論トレースから Driving Decision（`go_straight`, `turn_left`, `stop` 等）をパース
3. 両者を照合（GT decision ではなく、予測テキスト内の decision を使用）:
   - **一致 → r_consistency = 1**
   - **不一致 → r_consistency = 0**

**meta-action 抽出のルール例**（2軸: 縦方向 + 横方向を独立に抽出）:

縦方向 meta-action:
| 軌道特徴 | meta-action |
|---|---|
| 最終速度 ≈ 0 | `stop` |
| 大きな減速パターン（速度低下 > 閾値） | `yield` |
| 上記以外 | `go_straight` |

> 注: `follow_lead` は軌道のみから判定困難（先行車情報が必要）。consistency check では `go_straight` と同等に扱う。

横方向 meta-action:
| 軌道特徴 | meta-action |
|---|---|
| 累積 heading 変化 > +30° | `turn_left` |
| 累積 heading 変化 < -30° | `turn_right` |
| 横方向変位 > 閾値 かつ heading 変化 < 30° | `lane_change_left/right` |
| 上記以外 | `lane_keeping` |

### 3. 低レベル軌道品質 (r_traj)

予測軌道の物理的品質をペナルティ形式で評価する（Alpamayo §5.3.2 準拠）。

```
r_traj = -(λ_L2 · ||x_pred - x_expert||²_2 + λ_coll · I[collision] + λ_jerk · J(x_pred))
```

**3a. L2 距離ペナルティ**:
- 予測軌道とエキスパート軌道の全 waypoint 間 L2 距離の二乗和
- `λ_L2 = 1.0`（デフォルト）

**3b. 衝突ペナルティ（バイナリ指示関数）**:
- 予測軌道上のいずれかの waypoint が障害物と衝突するか判定
- nuScenes annotation（bounding box）+ ego 車両マージンで inflated OBB 判定
- **バイナリ**: 1 つでも衝突があれば `I[collision] = 1`、なければ `0`
- `λ_coll = 5.0`（デフォルト）

**3c. ジャーク抑制ペナルティ**:
- 加速度列の 1 次差分の絶対値和: `J(x_pred) = Σ|a_{i+1} - a_i|`
- `λ_jerk = 0.1`（デフォルト）

> 注: r_traj は負のペナルティ値。値が 0 に近いほど高品質な軌道。

### 合成報酬

3 要素を重み付き和で合成:
```
R(o_i) = w_reason * (r_reason / 5) + w_consistency * r_consistency + w_traj * r_traj
```

r_reason は 0-5 スケールを [0,1] に正規化。r_traj は負のペナルティ項。

---

## プロジェクト構成

Stage 4 で追加するファイル:

```
minipamayo/src/minipamayo/
├── training/
│   ├── grpo.py              # GRPO 実装（ロールアウト、advantage、ポリシー更新）
│   ├── rewards.py           # 3 報酬関数（r_reason, r_consistency, r_traj）
│   └── meta_action.py       # 軌道 → meta-action 抽出
├── scripts/
│   └── compute_reason_reward.py  # 推論品質報酬のオフライン計算
└── configs/
    └── stage4.yaml          # ハイパーパラメータ
```

### 主要クラス / 関数

**`train_stage4.py`**（単一スクリプトに統合）:
```python
# GRPO の主要処理フロー:
# 1. ロールアウト: K 個の応答をサンプリング
# 2. 報酬計算: 3 要素の合成報酬
# 3. Advantage: A_i = r_i - r̄（グループ平均）
# 4. ポリシー更新: softmax-weighted policy gradient + KL penalty
```

**`rewards.py`**:
```python
class ReasoningReward:
    """推論品質報酬（外部 LLM API 呼び出し）。"""

    def __init__(self, api_client, cache_dir):
        ...

    def compute_batch(self, reasoning_traces, actions):
        """バッチで推論品質を採点（オフライン計算）。"""
        ...


class ConsistencyReward:
    """CoC-Action 一貫性報酬。"""

    def __init__(self, meta_action_extractor):
        ...

    def compute(self, reasoning_trace, predicted_trajectory):
        """推論意図と軌道の一致を判定（バイナリ）。"""
        ...


class TrajectoryReward:
    """低レベル軌道品質報酬。"""

    def compute(self, pred_traj, expert_traj, obstacles=None):
        """L2 + 衝突 + ジャーク の合成報酬。"""
        ...
```

**`meta_action.py`**:
```python
def extract_meta_action(trajectory, velocities, thresholds):
    """予測軌道から 2 軸の meta-action を抽出。

    Args:
        trajectory: (N, 2) の waypoint 列 (x, y)
        velocities: (N,) の速度列
        thresholds: 各 meta-action の閾値辞書

    Returns:
        meta_action: dict {"longitudinal": str, "lateral": str}
    """
    ...
```

---

## 実装ステップ

### Step 1: GRPO の基本実装

- [ ] ロールアウト生成（temperature sampling で K 個の応答をサンプリング）
- [ ] グループ内相対 advantage の計算（`A_i = r_i - r̄`、std 正規化なし）
- [ ] Softmax-weighted policy gradient の実装
- [ ] KL 正則化の実装（SFT モデルとの KL divergence）
- [ ] 単体テスト: ダミー報酬で advantage 計算が正しいことを確認

### Step 2: 報酬関数の実装

- [ ] `r_reason`: 外部 LLM API 呼び出しのラッパー + キャッシュ機構
- [ ] `r_consistency`: meta-action 抽出 + 推論意図との照合
- [ ] `r_traj`: L2 距離 + 衝突ペナルティ + ジャーク抑制
- [ ] 合成報酬の重み付き和
- [ ] 単体テスト: 各報酬関数が期待通りの値を返すことを確認

### Step 3: 軌道 → meta-action 抽出ロジック

- [ ] 離散トークン → 連続制御入力 → waypoint 列へのデコード
- [ ] waypoint 列から heading 変化、横方向変位、移動距離を算出
- [ ] ルールベースの meta-action 分類
- [ ] 閾値のチューニング（学習データの分布から決定）

### Step 4: 推論品質報酬のオフライン計算パイプライン

- [ ] ロールアウト結果の保存フォーマット定義
- [ ] バッチ API 呼び出しスクリプト（レートリミット対応）
- [ ] キャッシュ機構（同一応答の再計算を回避）
- [ ] 採点結果の統計確認（スコア分布の可視化）

### Step 5: 学習ループ

- [ ] ロールアウト → 報酬計算 → ポリシー更新の統合
- [ ] r_reason のオフライン / r_consistency + r_traj のオンライン混合
- [ ] wandb / tensorboard ロギング（報酬推移、KL、policy loss）
- [ ] チェックポイント保存（best reward + 定期保存）

### Step 6: Cosmos Reason Mini での知見活用

- [ ] Train/eval データ分割でリーク防止
- [ ] reward hacking の監視（報酬が上がるが実際の品質が低下していないか）
- [ ] KL ダイバージェンスの推移を監視

---

## 評価

### SFT（Stage 3）との比較

| 指標 | 説明 |
|---|---|
| 推論品質スコア | 外部 LLM API による 0-5 スケール採点の平均 |
| CoC-Action 一貫性率 | 推論意図と予測軌道が一致する割合 |
| 軌道品質（ADE / FDE） | 予測軌道とエキスパート軌道の誤差 |
| L2 距離 | 各 waypoint の L2 誤差の平均 |
| 衝突率 | 予測軌道上の衝突 waypoint の割合 |
| ジャーク | 制御入力の 2 次差分の平均 |

### 追加の評価観点

- **推論と行動の一貫性チェック**: 定性的に「止まる」と推論したケースで実際に stop 軌道が生成されるかを確認
- **KL 正則化の強度調査**: β を変えたときの reward vs KL のトレードオフ
- **報酬の各要素の推移**: r_reason, r_consistency, r_traj それぞれが学習を通じて改善しているか
- **Reward hacking の検出**: 合成報酬が高いが個別報酬のバランスが崩れていないか

---

## ハイパーパラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| K（ロールアウト数） | 4〜8 | VRAM 制約。逐次生成のため VRAM は K に比例しない |
| β_grpo（softmax temperature） | 0.1 | softmax(β·A) の温度パラメータ |
| λ_KL（KL ペナルティ係数） | 0.05 | SFT モデルからの逸脱を制御 |
| 学習率 | 1e-6〜5e-6 | RL では SFT より小さい学習率を使う |
| Temperature（サンプリング） | 0.7〜1.0 | ロールアウト生成時 |
| w_reason（推論品質の重み） | 0.4 | チューニング対象 |
| w_consistency（一貫性の重み） | 0.3 | チューニング対象 |
| w_traj（軌道品質の重み） | 0.3 | チューニング対象 |
| λ_L2（L2 距離ペナルティ） | 1.0 | r_traj 内の L2 ペナルティ係数 |
| λ_coll（衝突ペナルティ） | 5.0 | r_traj 内のバイナリ衝突ペナルティ係数 |
| λ_jerk（ジャークペナルティ） | 0.1 | r_traj 内のジャークペナルティ係数 |
| r_reason モデル | gpt-4o | マルチモーダル LLM API（--reason_model で変更可） |
| Batch size | 1〜2 | VRAM 制約 |
| Gradient accumulation | 8〜16 | 実効バッチサイズを確保 |
| Max sequence length | 2048 | CoC 推論 + 離散軌道トークンを含む |
| Precision | bf16 | |
| Gradient checkpointing | 有効 | LLM に適用 |

---

## Exit 条件

- [ ] 3 つの報酬（推論品質、CoC-Action 一貫性、軌道品質）のすべてで Stage 3（SFT）より改善
- [ ] 推論-行動の不一致率が減少（定性的にも確認）
- [ ] KL が発散しない（SFT モデルからの逸脱が制御されている）
- [ ] 報酬の各要素が学習を通じて単調に近い改善傾向を示す
- [ ] Reward hacking の兆候がない

---

## 学習ループの擬似コード

```python
# 初期化
policy = load_checkpoint("stage3_sft.pt")  # Stage 3 SFT モデル
ref_policy = deepcopy(policy)               # Reference policy（frozen）
ref_policy.eval()
for p in ref_policy.parameters():
    p.requires_grad = False

# LLM のみ trainable
freeze(policy.vision_encoder)
freeze(policy.adapter)
freeze(policy.flow_head)
unfreeze(policy.llm)

optimizer = AdamW(policy.llm.parameters(), lr=5e-6)

for epoch in range(n_epochs):
    for batch in dataloader:
        # 1. ロールアウト: K 個の応答をサンプリング
        with torch.no_grad():
            rollouts = []
            for k in range(K):
                output = policy.generate(batch, temperature=0.8)
                rollouts.append(output)

        # 2. 報酬計算
        rewards = []
        for output in rollouts:
            r_reason = reason_reward.compute(output)       # オフライン（事前計算済み）
            r_consistency = consistency_reward.compute(output)
            r_traj = trajectory_reward.compute(output, batch.expert_traj)
            R = w_reason * (r_reason / 5) + w_consistency * r_consistency + w_traj * r_traj
            rewards.append(R)

        # 3. Advantage 計算（std 正規化なし）
        rewards = torch.tensor(rewards)
        advantages = rewards - rewards.mean()  # A_i = r_i - r̄

        # 4. Softmax-weighted ポリシー更新（単一ステップ）
        softmax_weights = torch.softmax(beta_grpo * advantages, dim=0)

        total_loss = 0.0
        for k, output in enumerate(rollouts):
            # 現在のポリシーでのログ確率
            new_log_prob = compute_log_prob(policy, output)

            # Reference policy でのログ確率
            with torch.no_grad():
                ref_log_prob = compute_log_prob(ref_policy, output)

            # KL ペナルティ
            kl = (new_log_prob - ref_log_prob).mean()

            # Softmax-weighted GRPO loss
            total_loss -= softmax_weights[k] * (new_log_prob.sum() - lambda_kl * kl)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.llm.parameters(), 1.0)
        optimizer.step()
```

---

## 注意事項

- **API コスト**: r_reason のオフライン計算に外部 LLM API を使用するため、ロールアウト数 × エポック数に比例して API コストが発生する。キャッシュを活用し、同一応答の再計算を避ける
- **ロールアウトの多様性**: Temperature が低すぎると K 個の応答が類似し、advantage の分散が小さくなる。Temperature 0.7〜1.0 の範囲で調整
- **Reward hacking**: 合成報酬が増加しても個別報酬のバランスが崩れることがある。各報酬の推移を個別にモニタリングする
- **KL 発散**: β が小さすぎるとポリシーが SFT から大きく逸脱し、生成品質が崩壊するリスクがある。KL の値を常に監視し、急激な増加がないか確認する
- **計算時間**: RL は SFT に比べて K 倍のフォワードパスが必要。RTX 4090 単体では学習に時間がかかるため、データ量とエポック数を適切に設定する
