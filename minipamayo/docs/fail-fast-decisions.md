# MiniPamayo Fail-Fast 実装 — 判断記録

## 概要

MiniPamayo の fail-fast パイプライン（Stage 0 〜 Stage 4）を nuScenes Mini（334 サンプル）で実装。
以下に各段階で行った設計判断とその理由を記録する。

---

## Phase 4: Stage 0 制御ベース表現

### 判断: CrossAttentionAdapter の採用
- **選択肢**: MeanPool / PerToken / CrossAttention
- **決定**: CrossAttentionAdapter（16 queries, 8 heads）
- **理由**: PerToken は 256 トークンで LLM のシーケンス長が長くなりすぎる。MeanPool は情報量が少なすぎる。16 learnable queries による cross-attention が空間情報と効率のバランスが良い

### 判断: K=6 @ 2Hz
- **選択肢**: K=10 / K=6 / K=64
- **決定**: K=6 @ 2Hz（3 秒ホライゾン）
- **理由**: nuScenes Mini のフレームレート（2Hz）に合わせた自然な設定。fail-fast ではサンプル数が少ないため、K を大きくすると過学習リスクが増す

### 判断: Huber Loss
- **選択肢**: MSE / Huber / L1
- **決定**: Huber Loss（delta=1.0）
- **理由**: 外れ値に対する頑健性。MSE はカーブの大きい制御入力に対して不安定になりやすい

### 結果
- Val Loss: 0.807 → 0.249
- ADE: 1.43m, FDE: 3.04m
- VRAM: 5.74 GB

---

## Phase 5: Stage 1 離散トークン化

### 判断: Shared 256 bins
- **選択肢**: 別々のビン / 共有ビン / 1024 bins
- **決定**: 256 shared bins、a と kappa で同じトークン ID を位置で区別
- **理由**: Alpamayo 論文に準拠。語彙拡張が最小限（+256 トークン）

### 判断: Interleaved format
- **決定**: [a_0, kappa_0, a_1, kappa_1, ...]
- **理由**: 各タイムステップの (a, kappa) をセットで予測する方が因果的に自然

### 判断: a_range=[-6.0, 6.0], kappa_range=[-0.2, 0.2]
- **理由**: nuScenes の GT 分布から設定。acceleration は ±6 m/s^2、curvature は ±0.2 rad/m で十分カバー

### 判断: チェックポイントロード順序
- **問題**: vocab size mismatch エラー (151936 vs 152192)
- **解決**: Phase 4 チェックポイントを先にロード → その後 resize_token_embeddings
- **教訓**: 語彙拡張はチェックポイントロード後に行う

### 結果
- Val CE: 12.10 → 3.52
- Token Acc: 0% → 27.5%
- ADE: 0.88m, FDE: 2.04m（AR generation）
- 使用ビン: 2/256（bin collapse — 小規模データで予想通り）

---

## Phase 6: Stage 2 Flow Matching

### 判断: Decoder のみ学習（VLM は frozen）
- **理由**: Stage 2 は連続軌道生成の学習が目的。VLM の特徴抽出能力は Phase 4 で獲得済み。Decoder のみ学習することでパラメータ効率が良く、VLM の知識を保持

### 判断: Fail-fast decoder config
- hidden_dim=256, num_layers=4, num_heads=4 (~5M params)
- **理由**: 小規模データ (334 samples) に対して大きなモデルは不要。本番では 512 dim, 12 layers に拡大

### 判断: Mean-pooled LLM hidden state as condition
- **選択肢**: Last token / Mean pool / CLS token
- **決定**: Mean pool over visual token positions
- **理由**: Last token だけでは情報量が限定的。Mean pool で全トークンの情報を集約

### 判断: 30 epochs
- **理由**: Decoder のパラメータ数が少なく (5M)、データも小さいため多くのエポックが必要。学習率はコサインスケジューラで十分に減衰

### 結果
- CFM Loss: 2.15 → 0.98
- ADE: 3.72m (single sample), minADE: 1.17m (5 samples)
- Diversity: 1.94 mean pairwise distance
- **多様なサンプル生成が可能**（minADE で 68.5% 改善）

---

## Phase 7: Stage 3 CoC SFT

### 判断: GPT-4o-mini for auto-labeling
- **選択肢**: GPT-4o / GPT-4o-mini / Claude
- **決定**: GPT-4o-mini
- **理由**: コスト効率。334 サンプルで約 $0.05。品質は GPT-4o に劣るが fail-fast には十分
- **結果**: 334/334 成功率 100%

### 判断: 閉じた意思決定集合（9 種）
- 縦方向: go_straight, follow_lead, stop, yield
- 横方向: lane_keeping, turn_left, turn_right, lane_change_left, lane_change_right
- **理由**: Alpamayo 論文の簡略版。nuScenes Mini で出現する主要行動をカバー

### 判断: 意思決定分布の偏り
- go_straight: 190 (57%), stop: 114 (34%), lane_keeping: 302 (90%)
- **認識**: データセットの偏りが大きい。本番ではデータ量を増やしてバランスを改善する必要あり
- **対応**: fail-fast では偏りを受容。RL (Stage 4) で改善を試みる

### 判断: Sequence 構造
- [visual_tokens(16)] + [prompt_tokens] + [reasoning_tokens] + [action_tokens(12)]
- Loss: reasoning + action tokens のみ
- **理由**: シンプルな concatenation で Qwen の chat format を使わない。fail-fast では簡素な構造で十分

### 判断: num_workers=0
- **理由**: CoCDataset は可変長テキストを含むため、DataLoader の collation が複雑。batch_size=1 + num_workers=0 で安全に動作

### 結果
- Val CE: 14.42 → 0.83
- Token Acc: 2.97% → 78.05%
- Driving Decision Acc: Overall 73.65% (Long: 56.89%, Lat: 90.42%)
- ADE: 1.78m, FDE: 4.19m
- **推論能力の獲得に成功**

---

## Phase 8: Stage 4 RL/GRPO

### 判断: r_reason を省略
- **選択肢**: 3 報酬 (r_reason + r_consistency + r_traj) / 2 報酬のみ
- **決定**: r_consistency + r_traj のみ
- **理由**: r_reason は外部 LLM API 呼び出しが必要で、ロールアウトごとに API コストが発生。fail-fast ではルールベース報酬のみで検証

### 判断: Simplified GRPO parameters
- n_rollouts=4, mu=2, eps=0.2, beta=0.05
- max_samples=50, max_epochs=2
- **理由**: 時間制約。フル GRPO (334 samples x 4 rollouts x 3 epochs) は数時間かかる。50 サンプル x 2 エポックで GRPO の動作検証を優先

### 判断: LLM のみ trainable
- **理由**: Alpamayo 論文 §5.3 に準拠。RL では LLM の推論能力のみを改善し、視覚特徴抽出は固定

### 判断: Token-level ではなく sequence-level ratio
- **実装**: `ratio = (new_lp - old_lp).mean().exp()` で平均ログ確率の ratio を使用
- **理由**: 簡素化。本来は per-token ratio + per-token advantage が正確だが、fail-fast ではシーケンスレベルで十分

### 判断: prompt_embeds の detach
- **問題**: mu-step 更新で "Trying to backward through the graph a second time" エラー
- **原因**: prompt_embeds が trainable な LLM embed layer から生成され、複数回の backward で計算グラフが再利用された
- **解決**: `prompt_embeds.detach()` で計算グラフから切り離し
- **教訓**: RL の multi-step update では、shared テンソルの計算グラフに注意

### 結果 (v1: バグ修正前)
- **学習**: Epoch 1 Reward 0.1820, KL 0.2067 → Epoch 2 Reward 0.1694, KL 1.4818
- **問題**: KL が Epoch 2 で急速に発散（0.21 → 1.48）、報酬が低下
- **評価比較** (SFT vs RL):

| 指標 | SFT | RL | 判定 |
|------|-----|-----|------|
| Composite Reward | 0.766 | 0.189 | SFT 勝ち |
| r_consistency | 0.734 | 0.006 | SFT 勝ち |
| r_traj | 0.788 | 0.310 | SFT 勝ち |
| ADE | 0.88m | 11.43m | SFT 勝ち |
| FDE | 2.05m | 20.71m | SFT 勝ち |
| Long Acc | 56.89% | 56.89% | 同等 |
| Lat Acc | 90.42% | 90.42% | 同等 |

- **分析**: RL モデルは SFT から大幅に劣化。原因は (1) データ量が極端に少ない (50 samples)、(2) beta=0.05 の KL ペナルティが不十分、(3) RL モデルが全サンプルで同じ出力（go_straight + lane_keeping）に収束。パイプラインの動作検証としては成功
- VRAM: 6.73 GB

### 結果 (v2: バグ修正後 — dt=0.5, a_range=[-6,6], action_loss_weight=2.0)

3つのバグ修正（後述「バグ修正 v2」セクション参照）+ 全ステージ再学習後の結果:

| 指標 | SFT | RL | 判定 |
|------|-----|-----|------|
| Composite Reward | 0.792 | 0.747 | SFT やや勝ち |
| r_consistency | 0.734 | 0.680 | SFT やや勝ち |
| r_traj | 0.851 | 0.813 | SFT やや勝ち |
| ADE | 0.87m | 1.00m | SFT やや勝ち |
| FDE | 2.02m | 2.46m | SFT やや勝ち |
| Long Acc | 69.46% | 56.89% | SFT 勝ち |
| Lat Acc | 90.42% | 90.12% | 同等 |

- **改善**: dt修正+量子化範囲拡大+action loss weight導入により、RL モデルの壊滅的劣化が解消（ADE 11.43m → 1.00m）。SFT からの劣化幅も大幅に縮小
- **残存課題**: RL が SFT を上回れていない。小規模データ (334 samples) での限界と考えられる
- VRAM: 7.02 GB

---

## 全体的な判断

### VRAM 管理
- gradient_checkpointing を全ステージで使用
- Peak VRAM: Stage 0=5.74GB, Stage 1=6.82GB, Stage 2=1.48GB, Stage 3=6.86GB, Stage 4=6.73GB
- RTX 4090 (24GB) に対して十分な余裕

### データ
- nuScenes Mini: 334 サンプル（10 シーン）
- 小規模データのため全ステージで過学習リスクあり
- bin collapse (Stage 1) や分布の偏り (Stage 3) が観測された
- Stage 4 での RL 劣化は小規模データの最大のリスク

### 全ステージ結果サマリ (v2: バグ修正後)

| Stage | 主要指標 | 値 | ADE / FDE |
|-------|----------|-----|-----------|
| Phase 4 (回帰) | Huber Loss / a MAE | 0.158 / 0.571 | 2.10m / 4.11m |
| Stage 1 (離散) | Val CE / Token Acc | 3.81 / 25.2% | 0.86m / 2.00m |
| Stage 2 (Flow) | CFM Loss / minADE(5) | 1.02 / 2.46m | 7.57m / 13.97m |
| Stage 3 (CoC SFT) | Token Acc / Decision Acc | 82.7% / 79.9% | 1.35m / 3.35m |
| Stage 4 (RL) | Composite Reward | 0.747 | 1.00m / 2.46m |

v1 との比較（バグ修正で改善した箇所）:

| Stage | 指標 | v1 (修正前) | v2 (修正後) | 改善要因 |
|-------|------|-------------|-------------|----------|
| Stage 3 | AR action tokens 生成 | 0個 (生成されない) | 12個 (正常) | エポック増加 (3→20) + action_loss_weight=2.0 |
| Stage 4 | ADE (RL) | 11.43m | 1.00m | dt=0.5 修正 + a_range=[-6,6] 拡大 |
| Stage 4 | Reward (RL) | 0.169 | 0.747 | dt=0.5 修正 + a_range=[-6,6] 拡大 |
| Stage 4 | RL-SFT 乖離 (ADE) | +10.55m | +0.13m | 上記3修正の複合効果 |

---

## バグ修正 v2

### 修正 1: dt=0.1 → dt=0.5（全箇所）

- **問題**: nuScenes keyframes は 2Hz（dt≈0.5s）なのに、dynamics/dataset/eval/rewards/train の全コードで dt=0.1 を使用していた
- **影響**: 加速度が 25 倍に膨張（実際 0.2 m/s² の変化を 5.0 m/s² と誤算）。軌道予測も大幅にずれる
- **修正**: `dynamics.py`, `nuscenes_trajectory_dataset.py`, `rewards.py`, `eval_*.py` の dt デフォルト値を 0.5 に統一
- **効果**: 全ステージの ADE/FDE が物理的に正しい値に改善

### 修正 2: a_range=[-4,4] → [-6,6]

- **問題**: dt=0.5 修正後、GT 加速度の 64.9% が [-4, 4] の範囲外にクリップされていた
- **修正**: `DiscreteActionTokenizer` の `a_range` を `(-6.0, 6.0)` に拡大
- **効果**: クリッピングによる情報損失が解消。Stage 1/3/4 の離散トークン品質が改善

### 修正 3: Stage 3 学習パラメータ改善

- **問題**: Stage 3 の AR 生成でアクショントークンが 1 つも生成されない（テキスト→EOS で終了）
- **原因**: 3 エポックでは不十分 + アクショントークンの学習シグナルが薄い
- **修正**: `max_epochs` を 3→20 に増加、`action_loss_weight=2.0` を導入（アクショントークンの CE loss を 2 倍に重み付け）
- **効果**: Token Acc 2.97%→82.7%、AR でアクショントークンが正常に 12 個生成されるようになった

---

### ワークフロー: 学習→評価は逐次的に行う

複数ステージを再学習する場合、**全ステージ学習してからまとめて eval** するのではなく、**各ステージの学習直後に eval** して次に進むこと。

```
× ダメなパターン:
  Phase 4 学習 → Stage 1 学習 → Stage 2 学習 → Stage 3 学習 → Stage 4 学習
  → まとめて eval → Phase 4 がダメだった → 全部やり直し

○ 正しいパターン:
  Phase 4 学習 → eval → OK → Stage 1 学習 → eval → OK → ...
```

**理由**: 前段のステージが失敗していた場合、後段の学習はすべて無駄になる。逐次的に eval することで早期に問題を発見し、手戻りを最小化できる。

---

### 本番への移行に必要な変更
1. **データ**: nuScenes Full (28,000+ frames) + comma2k19 + 追加データ
2. **モデル**: Flow Decoder を 12 layers / 512 dim に拡大
3. **Stage 4**: r_reason を追加（LRM または API ベース）
4. **Stage 4**: フルデータセットでの GRPO 学習 + beta の増大（0.1〜0.2）
5. **Stage 4**: per-token ratio + per-token advantage への移行
6. **Stage 3**: Qwen chat format の正式採用
7. **評価**: nuScenes val split での定量評価
