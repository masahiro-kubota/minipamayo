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
| アルゴリズム | GRPO | **同一** |
| 報酬: 推論品質 | LRM（DeepSeek-R1, Cosmos-Reason）| 外部 LLM API（デフォルト: gpt-4o-mini）|
| 報酬: CoC-Action 一貫性 | ルールベース（バイナリ）| **同一** |
| 報酬: 軌道品質 | L2 + 衝突 + ジャーク | **同一** |
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
3. **Advantage 計算**: グループ G = {o_1, ..., o_K} 内の相対的な advantage
   ```
   A_i = (R(o_i) - mean(G)) / std(G)
   ```
4. **ポリシー更新**: clipped surrogate objective + KL 正則化
   ```
   L_GRPO = -E[ min(r_t(θ) * A_i, clip(r_t(θ), 1-ε, 1+ε) * A_i) ] + β * KL(π_θ || π_ref)
   ```
   ここで `r_t(θ) = π_θ(o|q) / π_old(o|q)` は policy ratio

### Multi-step 更新（Cosmos Reason Mini の知見）

Cosmos Reason Mini での RL 検証で以下の知見が得られている:

- **μ=4（multi-step 更新）**: 同じロールアウトデータに対して μ 回の最適化ステップを実行
- Policy ratio `r_t(θ)` は **old policy**（ロールアウト生成時のポリシー）に対して計算
- μ > 1 のとき、ステップを重ねるごとにポリシーが変化するため clipping が有効に機能
- MCQ+CoT SFT 併用時: 92.5% → 93.3% への改善が確認されている

### KL 正則化

- Reference policy: Stage 3 SFT モデルを frozen で保持
- KL 項により、RL でポリシーが SFT モデルから過度に逸脱することを防ぐ
- β（KL 係数）でその強度を調整

---

## 報酬信号（3 要素）

### 1. 推論品質 (r_reason)

LRM（大規模推論モデル）による 0-5 スケールの採点。

**評価観点**:
- **行動一貫性**: 推論が正しい運転行動を記述しているか
- **因果推論品質**: 因果要因が正しく特定されているか

**MiniPamayo での実装**:
- 外部 LLM API（Claude API / GPT-4o）で代替
- **オフライン計算**: API レイテンシのためオンライン計算は非現実的
  1. ロールアウト生成後、全応答をバッチで API に送信
  2. 採点結果をキャッシュ
  3. ポリシー更新時にキャッシュから読み出し

**プロンプト例**:
```
以下は自動運転シーンに対する推論トレースです。
0-5 のスケールで評価してください。

評価基準:
- 5: 因果要因が正確に特定され、運転行動と一貫した推論
- 3: おおよそ妥当だが、因果要因の一部が不正確または欠落
- 1: 推論が視覚入力と矛盾、または行動と不整合
- 0: 無関係な推論

[推論トレース]
{reasoning_trace}

[予測された行動]
{predicted_action}

スコア (0-5):
```

### 2. CoC-Action 一貫性 (r_consistency)

バイナリ報酬。推論トレースの意図と実際の予測軌道が一致しているかを検証する。

**手順**:
1. 予測軌道（離散トークン → 連続制御入力 → waypoint 列）から **meta-action** を抽出
2. 推論トレースに含まれる Driving Decision（`go_straight`, `turn_left`, `stop` 等）を抽出
3. 両者を照合:
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

予測軌道の物理的品質を評価する。3 つのサブ報酬の重み付き和。

**3a. L2 距離 (r_l2)**:
- 予測軌道とエキスパート軌道の waypoint 間 L2 距離
- 正規化して 0-1 に変換: `r_l2 = exp(-alpha * mean_l2_distance)` (alpha=0.5)

**3b. 衝突ペナルティ (r_collision)**:
- 予測軌道上の各 waypoint と周囲障害物の距離を計算
- nuScenes annotation（bounding box）を利用
- 衝突判定: 最近接距離が閾値以下なら衝突
- `r_collision = 1 - n_collisions / n_waypoints`

**3c. ジャーク抑制 (r_jerk)**:
- 制御入力列の 2 次差分（ジャーク）を計算
- 急激な制御変化にペナルティ
- `r_jerk = exp(-gamma * mean_jerk)` (gamma=2.0)

**合成報酬**:
```
r_traj = w_l2 * r_l2 + w_col * r_collision + w_jerk * r_jerk
```

### 合成報酬

3 要素を重み付き和で合成:
```
R(o_i) = w_reason * r_reason + w_consistency * r_consistency + w_traj * r_traj
```

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

**`grpo.py`**:
```python
class GRPOTrainer:
    """GRPO によるポリシー最適化。"""

    def __init__(self, policy, ref_policy, reward_fn, config):
        self.policy = policy          # 学習対象（LLM のみ trainable）
        self.ref_policy = ref_policy  # SFT モデル（frozen）
        self.reward_fn = reward_fn
        self.config = config

    def rollout(self, prompts, K):
        """各プロンプトに対して K 個の応答をサンプリング。"""
        ...

    def compute_advantages(self, rewards):
        """グループ内相対 advantage を計算。"""
        # A_i = (R(o_i) - mean(G)) / std(G)
        ...

    def policy_step(self, rollout_data, advantages, mu=4):
        """Multi-step ポリシー更新（μ 回の最適化ステップ）。"""
        for step in range(mu):
            # ratio = pi_theta / pi_old
            # clipped surrogate + KL penalty
            ...

    def train_epoch(self):
        """ロールアウト → 報酬計算 → ポリシー更新のループ。"""
        ...
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
- [ ] グループ内相対 advantage の計算
- [ ] Clipped surrogate objective の実装
- [ ] KL 正則化の実装（SFT モデルとの KL divergence）
- [ ] Multi-step 更新 (μ=4) の実装
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

- [ ] Multi-step 更新 (μ=4) の実装と検証
- [ ] Policy ratio は old policy に対して計算（μ > 1 で clipping が有効に機能することを確認）
- [ ] Train/eval データ分割でリーク防止
- [ ] reward hacking の監視（報酬が上がるが実際の品質が低下していないか）

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
| μ（multi-step 更新） | 4 | Cosmos Reason Mini での知見 |
| ε（clipping 係数） | 0.2 | GRPO 標準値 |
| β（KL 係数） | 0.01〜0.1 | チューニング対象。小さすぎると SFT から逸脱、大きすぎると学習が進まない |
| 学習率 | 1e-6〜5e-6 | RL では SFT より小さい学習率を使う |
| Temperature（サンプリング） | 0.7〜1.0 | ロールアウト生成時 |
| w_reason（推論品質の重み） | 0.4 | チューニング対象 |
| w_consistency（一貫性の重み） | 0.3 | チューニング対象 |
| w_traj（軌道品質の重み） | 0.3 | チューニング対象 |
| w_l2 / w_col / w_jerk | 0.5 / 0.3 / 0.2 | r_traj 内のサブ重み |
| alpha（L2 報酬スケーリング） | 0.5 | r_l2 = exp(-alpha * mean_l2) |
| gamma（ジャーク報酬スケーリング） | 2.0 | r_jerk = exp(-gamma * mean_jerk) |
| r_reason モデル | gpt-4o-mini | 外部 LLM API（--reason_model で変更可） |
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
            R = w_reason * r_reason + w_consistency * r_consistency + w_traj * r_traj
            rewards.append(R)

        # 3. Advantage 計算
        rewards = torch.tensor(rewards)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 4. Multi-step ポリシー更新
        old_log_probs = compute_log_probs(policy, rollouts)  # ロールアウト時のログ確率
        for mu_step in range(mu):
            new_log_probs = compute_log_probs(policy, rollouts)
            ratio = (new_log_probs - old_log_probs).exp()

            # Clipped surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty
            kl = compute_kl(policy, ref_policy, rollouts)
            loss = policy_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 注意事項

- **API コスト**: r_reason のオフライン計算に外部 LLM API を使用するため、ロールアウト数 × エポック数に比例して API コストが発生する。キャッシュを活用し、同一応答の再計算を避ける
- **ロールアウトの多様性**: Temperature が低すぎると K 個の応答が類似し、advantage の分散が小さくなる。Temperature 0.7〜1.0 の範囲で調整
- **Reward hacking**: 合成報酬が増加しても個別報酬のバランスが崩れることがある。各報酬の推移を個別にモニタリングする
- **KL 発散**: β が小さすぎるとポリシーが SFT から大きく逸脱し、生成品質が崩壊するリスクがある。KL の値を常に監視し、急激な増加がないか確認する
- **計算時間**: RL は SFT に比べて K 倍のフォワードパスが必要。RTX 4090 単体では学習に時間がかかるため、データ量とエポック数を適切に設定する
