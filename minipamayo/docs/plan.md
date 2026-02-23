# MiniPamayo 実装計画

## 全体方針

**fail-fast** で段階的に進める。各 Stage に明確な Exit 条件を設け、クリアしてから次へ進む。

設計書 v0.3（§3.7）に定義された 6 段階の学習パイプラインに対応:

```
ドメイン SFT → Stage 0（回帰）→ Stage 1（離散トークン）→ Stage 2（Flow）→ Stage 3（CoC SFT）→ Stage 4（RL）
```

---

## Phase 1: プロジェクト基盤

### 1.1 環境構築

- [ ] Python 3.11+ 環境（venv or conda）
- [ ] PyTorch 2.x + CUDA 12.x
- [ ] 依存パッケージ: transformers, accelerate, einops, wandb
- [ ] プロジェクト構成:

```
minipamayo/
├── docs/               # 設計・計画ドキュメント
├── src/
│   └── minipamayo/
│       ├── models/
│       │   ├── vision_encoder.py    # DINOv2 ViT-B/14 ラッパー
│       │   ├── adapter.py           # Vision→LLM Adapter
│       │   ├── llm.py               # Qwen2.5-0.5B ラッパー
│       │   ├── action_head.py       # MLP 回帰ヘッド (Stage 0)
│       │   ├── discrete_head.py     # 離散トークン化 (Stage 1)
│       │   ├── trajectory_decoder.py # Trajectory Decoder / Flow Matching (Stage 2)
│       │   ├── dynamics.py          # ユニサイクルダイナミクス・制御表現
│       │   └── minipamayo.py        # 統合モデル
│       ├── data/
│       │   ├── dataset.py           # データセットクラス
│       │   ├── transforms.py        # 画像前処理
│       │   ├── action_label.py      # GT 制御列の逆算 (ego pose → (a, κ))
│       │   └── coc_labeling.py      # CoC auto-labeling パイプライン
│       ├── training/
│       │   ├── trainer.py           # 学習ループ
│       │   ├── losses.py            # 損失関数
│       │   ├── grpo.py              # GRPO (RL ポストトレーニング)
│       │   └── rewards.py           # 報酬関数 (Stage 4)
│       └── utils/
│           └── config.py            # 設定管理
├── configs/
│   ├── domain_sft.yaml
│   ├── stage0.yaml
│   ├── stage1.yaml
│   ├── stage2.yaml
│   ├── stage3.yaml
│   └── stage4.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── visualize.py
│   └── auto_label.py              # VLM auto-labeling スクリプト
├── tests/
├── pyproject.toml
└── README.md
```

### 1.2 データセット準備

- [ ] データセット選定・ダウンロード（→ [datasets.md](datasets.md) 参照）
- [ ] DataLoader 実装（画像 + アクションラベル + egomotion）
- [ ] データ前処理パイプライン（リサイズ、正規化）
- [ ] GT 制御列の逆算実装（§3.5）:
  - ego pose 軌道 → 制御入力 (a, κ) への変換
  - ユニサイクルダイナミクスの Euler 積分
  - 最小二乗法 + Tikhonov 正則化による逆算
- [ ] 単体テスト: バッチが正しく取り出せることを確認

---

## Phase 2: VLM 構築 + ドメイン SFT（前段パイプライン）

### 目的

DINOv2 ViT-B/14 + Qwen2.5-0.5B から汎用 VLM を構築し、運転ドメインの知識を注入する（§3.4）。後段の全 Stage の土台。

### Phase 2a: Qwen2.5-VL Mini（汎用 VLM 構築）

**詳細は [Qwen2.5-VL Mini 設計書](qwen-vl-mini/design.md) / [計画書](qwen-vl-mini/plan.md) を参照。**

- [ ] Feature Alignment: Adapter のみ学習（画像キャプションデータ）
- [ ] Visual Instruction Tuning: Adapter + LLM 学習（視覚 QA データ）
- [ ] Exit: 画像に対する質問に妥当な回答が生成できる

### Phase 2b: Cosmos Reason Mini（運転ドメイン特化）

**詳細は [Cosmos Reason Mini 設計書](cosmos-reason-mini/design.md) / [計画書](cosmos-reason-mini/plan.md) を参照。**

- [ ] Physical AI SFT: 教師 VLM で運転シーンの QA データを作成し、SFT を実施
- [ ] Physical AI RL（任意）: QA を MCQ に変換し、GRPO で推論品質を改善
- [ ] 学習済み重み（Vision Encoder + Adapter + LLM）を保存

### Exit 条件

- [ ] 画像を入力して運転シーンの記述・推論が生成できる
- [ ] OOM しない（24 GB 以内）
- [ ] 重みを MiniPamayo Stage 0 にロードできる

---

## Phase 3: Stage 0 — パイプライン検証（最小回帰）

### 目的

勾配が全経路 **DINO → Adapter → LLM → Action Head** に流れ、学習が回ることを確認する。

### 3.1 各モジュール実装

- [x] **Vision Encoder**: DINOv2 ViT-B/14 ロード + forward
- [x] **Adapter**: 平均 Pool + Linear（最小実装）
  - DINOv2 出力 (256×768) → 平均Pool → (1×896)
  - Phase 4 で Cross-Attention Pooling (16×896) に置き換え
- [x] **LLM**: Qwen2.5-0.5B ロード + visual tokens 入力の forward（minipamayo.py に統合）
  - 視覚トークンを inputs_embeds 経由で注入
- [x] **Action Head**: MLP（LLM hidden → [steer, throttle]）— fail-fast 用の最小出力

### 3.2 統合・学習ループ

- [x] 統合モデル `MiniPamayo` クラス
- [x] ドメイン SFT の学習済み重みを初期値として使用（§3.7）
- [x] 学習ループ実装（train_stage0.py に統合）:
  - micro-batch=1, grad accumulation=16
  - bf16 mixed precision
  - gradient checkpointing（DINO + LLM）
  - AdamW optimizer
  - Huber loss
- [x] wandb ロギング（オプション、--use_wandb フラグ）
- [x] VRAM 使用量の実測・記録（5.69 GB）

### 3.3 評価スクリプト (`eval_stage0.py`)

- [x] チェックポイントロード → 全データで推論
- [x] Loss / Steer MAE / Throttle MAE の計算
- [x] サンプルごとの予測 vs GT 表示
- [x] 予測分布の統計（GT の分散をカバーしているか）
- [x] Worst 5 の特定（画像ファイル名付き）
- [x] Input dependency チェック

### 3.4 Exit 条件

- [x] 学習 loss が安定して下がる（0.991 → 0.018）
- [x] OOM しない（5.69 GB / 24 GB）
- [x] 推論で入力画像に応じて出力が変化する（input-dependent: YES）

### 想定所要時間

最小データセット（数千フレーム）で数時間〜半日の学習で loss 低下を確認できる想定。

---

## Phase 4: Stage 0 改善（制御ベース表現への移行）

### 4.1 制御ベースアクション表現の導入

- [ ] ユニサイクルダイナミクス実装（§3.5）
  - 制御入力 (a, κ) → Euler 積分 → (x, y, θ, v) 軌道
  - 逆算: ego pose 軌道 → (a, κ) の GT 制御列
- [ ] Action Head の出力を `[steer, throttle]` (2,) → `(a, κ)` (K, 2) に拡張
  - K=64（6.4秒 @ 10Hz）— Alpamayo と同一
- [ ] Loss: Huber loss on (a, κ) 制御入力列
- [ ] 制御入力 → 軌道変換の可視化（予測 vs GT を画像上にプロット）

### 4.2 Adapter 改善

- [ ] Cross-Attention Pooling に置き換え（§3.2 方式 1）
  - learnable query: 16 個
  - DINOv2 パッチ特徴を key/value に使用
- [ ] 性能比較（loss 低下速度、最終 loss）

### 4.3 評価指標の整備

- [ ] 制御入力 → waypoint 変換を経由して ADE / FDE を計算
- [ ] 可視化: 予測軌道 vs GT をカメラ画像上にプロット

### 4.4 評価スクリプト

- [ ] `eval_stage0.py` を制御ベース表現に対応させる
  - (a, κ) × 64 の予測 vs GT（サンプルごと）
  - ADE / FDE の計算
  - 予測分布の統計（GT の分散をどれだけカバーしているか）
  - forward_dynamics で軌道に変換して可視化

### 4.5 Exit 条件

- [ ] 制御ベース表現で loss が安定して下がる
- [ ] ADE / FDE が意味のある値を示す
- [ ] 可視化で軌道がおおよそ妥当

---

## Phase 5: Stage 1 — アクション離散化（Dual Representation の基盤）

### 目的

Alpamayo の Dual Representation 戦略の前半（§3.6 Stage B）。制御入力を離散トークン化し、LLM の自己回帰フレームワークで行動を生成する。Stage 3（CoC SFT）で推論トークンと行動トークンを同じシーケンスで扱うための前提。

### 5.1 離散トークン化

- [ ] 制御入力 (aᵢ, κᵢ) の均一量子化
  - 加速度 a: 所定範囲を N_bins で量子化
  - 曲率 κ: 所定範囲を N_bins で量子化
- [ ] LLM の語彙に離散アクショントークンを追加（128 トークン = 64 × 2）
- [ ] embedding layer と LM head の拡張

### 5.2 学習

- [ ] Stage 0 の学習済み重みを初期値として使用
- [ ] Vision Encoder + Adapter + LLM すべて trainable（§3.7）
- [ ] 入力: [visual_tokens]、出力: [action_tokens]
- [ ] Loss: cross-entropy（next-token prediction）
- [ ] Teacher forcing で学習

### 5.3 評価スクリプト (`eval_stage1.py`)

- [ ] チェックポイントロード → 全データで推論
- [ ] トークン精度: 離散トークンの top-1 accuracy
- [ ] 離散トークン → dequantize → 連続制御入力 → 軌道への decode
- [ ] ADE / FDE を回帰版（Stage 0）と比較
- [ ] 予測トークン分布の分析（特定ビンに偏っていないか）
- [ ] 量子化ビン数の影響調査

### 5.4 Exit 条件

- [ ] cross-entropy loss が下がる
- [ ] 離散化による性能劣化が許容範囲内
- [ ] decode した軌道が妥当

---

## Phase 6: Stage 2 — Flow Matching Expert

### 目的

VLM の KV-cache を条件として、Alpamayo 準拠の Flow Matching Expert で連続的かつ多様な軌道を生成する（§3.6 Stage C）。

### 6.1 Expert 実装（Alpamayo 準拠、~146M params、Expert/VLM≈30%）

- [x] Qwen2 Transformer ベースの Expert（`AutoModel.from_config(Qwen2Config(...))`）
- [x] KV-cache 互換性制約: `num_kv_heads=2`, `head_dim=64`, `num_hidden_layers=24`
- [x] Fourier Feature V2 + MLP 入力エンコーディング（各 action 次元別）
- [x] Non-causal attention, position_ids continuation, KV-cache crop
- [x] CFM loss（Gaussian OT path, Beta time schedule）
- [x] `cfm_sample`（Euler 積分、デフォルト 10 ステップ）
- [x] `load_decoder_from_checkpoint` ヘルパー関数

### 6.2 学習

- [x] Phase 4 の学習済み VLM 重み（Vision + Adapter + LLM）を frozen で使用
- [x] VLM → `use_cache=True` → KV-cache を Expert に直接渡す
- [x] Expert のみ trainable、CFM loss で学習
- [x] カーブシーンの `WeightedRandomSampler` オーバーサンプリング（3×）
- [x] trainval で再学習（旧 cross-attention デコーダーの checkpoint は非互換）
  - Val CFM: 2.638 → 2.083（5 epochs）、ADE=1.64m, FDE=3.58m

### 6.3 評価

- [x] `eval_stage2.py`: CFM loss, ADE/FDE, diversity（minADE）
- [x] `scripts/find_curve_scenes.py`: カーブシーン特化の可視化
- [x] `visualize.py --stage stage2`: BEV プロット
- [x] ADE/FDE/diversity メトリクスの検証（ADE=1.64m, FDE=3.58m, diversity=3.10, variance=0.415）

### 6.4 下流スクリプト（Stage 3/4 互換）

- [x] `train_stage4.py`: `extract_flow_trajectory` → KV-cache 方式に更新
- [x] `eval_stage4.py`: 同上
- [x] `visualize.py`: `_load_decoder`, `_flow_trajectory` → KV-cache 方式に更新
- [x] チェックポイントキー名統一（`decoder_state_dict`）

### 6.5 Exit 条件

- [x] CFM loss が安定して下がる（2.638 → 2.083、単調減少）
- [ ] カーブシーンで曲線軌道が生成される（未検証、デコーダー規模の制約）
- [x] ADE/FDE が正常範囲（ADE=1.64m, FDE=3.58m）

---

## Phase 7: Stage 3 — CoC 推論 SFT

### 目的

構造化された推論（Chain of Causation）を LLM に獲得させ、推論トークンとアクショントークンを同一シーケンスで自己回帰生成する（§3.8）。

### 7.1 CoC auto-labeling パイプライン

- [ ] 教師 VLM（GPT-4o 等）を使った CoC アノテーション生成
  - 閉じた意思決定集合の定義（データセットに応じたサブセット、2軸分類）
    - 縦方向: `{go_straight, follow_lead, stop, yield}`
    - 横方向: `{lane_keeping, turn_left, turn_right, lane_change_left, lane_change_right}`
  - Critical Components の特定
  - CoC Trace（因果推論テキスト）の生成
- [ ] 因果混乱の防止: 過去の観測のみから因果要因を特定するプロンプト設計
- [ ] 生成データの品質チェック

### 7.2 学習

- [ ] Stage 1/2 の学習済み重みを初期値として使用
- [ ] Vision Encoder + Adapter + LLM は trainable、Flow Head は frozen（§3.7）
- [ ] 入力: [visual_tokens, egomotion_tokens]
- [ ] 出力: [CoC_reasoning_tokens, meta_action_tokens, trajectory_tokens]
- [ ] Loss: cross-entropy（推論 + 離散アクションの joint next-token prediction）
- [ ] 推論トレースと離散軌道トークンを連結したシーケンスで学習

### 7.3 評価スクリプト (`eval_stage3.py`)

- [ ] チェックポイントロード → 全データで推論
- [ ] SFT loss の計算
- [ ] 生成された CoC 推論テキストのサンプル表示（N 件）
- [ ] Driving Decision の一致率（GT meta-action vs 予測 meta-action）
- [ ] 離散トークン精度（推論付き vs なし の比較）
- [ ] 推論テキストの定性的評価（サンプルを目視確認）
- [ ] 生成された推論の品質（外部 LLM による自動評価 — オプション）

### 7.4 Exit 条件

- [ ] SFT loss が下がる
- [ ] 推論テキストが入力画像に対しておおよそ妥当な内容
- [ ] 推論ありの方が軌道品質が改善（or 同等）

---

## Phase 8: Stage 4 — RL ポストトレーニング

### 目的

SFT だけでは残る推論品質の問題（データバイアス、推論-行動の不一致、ハルシネーション）を RL で改善する（§3.9）。

### 8.1 GRPO 実装

- [ ] GRPO（Group Relative Policy Optimization）の基本実装（Alpamayo §5.3.2 準拠）
  - K 個のロールアウトサンプリング（K=4〜8、VRAM 制約）
  - Advantage: `A_i = r_i - r̄`（グループ平均、std 正規化なし）
  - Softmax-weighted policy gradient: `L = -Σ softmax(β·A_i) × (log π_θ(τ_i) - λ_KL·KL)`
  - KL 正則化（SFT モデルを reference policy として保持）
- [ ] LLM のみ trainable、他は frozen（§3.7）

### 8.2 報酬関数

- [ ] **推論品質 (r_reason)**: 外部 LLM API による 0-5 スケール採点
  - オフライン計算（API レイテンシのためオンラインは困難）
- [ ] **CoC-Action 一貫性 (r_consistency)**: バイナリ報酬
  - 予測軌道 → meta-action 抽出 → 推論テキストの意図と照合
- [ ] **低レベル軌道品質 (r_traj)**（ペナルティ形式）:
  - `r_traj = -(λ_L2·||x_pred-x_expert||²_2 + λ_coll·I[collision] + λ_jerk·J(x_pred))`
  - バイナリ衝突指示関数（nuScenes の周囲物体情報を利用）

### 8.3 学習

- [ ] SFT モデル（Stage 3）を初期値として使用
- [ ] Reference policy を frozen で保持（KL 計算用）
- [ ] 推論品質報酬はオフラインで事前計算
- [ ] 軌道品質報酬はオンライン計算
- [ ] ロールアウト → 報酬計算 → ポリシー更新のループ実装

### 8.4 評価スクリプト (`eval_stage4.py`)

- [ ] チェックポイントロード → 全データで推論
- [ ] SFT モデル（Stage 3）との比較:
  - 推論品質スコアの改善
  - CoC-Action 一貫性の改善
  - 軌道品質（ADE / FDE）の改善
- [ ] 推論と行動の一貫性チェック（「止まる」と言って止まるか等）
- [ ] KL ダイバージェンスの計測（reference policy との距離）
- [ ] Stage 0〜4 の全 Stage 横断比較テーブルの出力

### 8.5 Exit 条件

- [ ] 3 つの報酬すべてで Stage 3 より改善
- [ ] 推論-行動の不一致が減少
- [ ] KL が発散しない（SFT モデルから逸脱しすぎない）

---

## 実装優先順位まとめ

```
Phase 1 (基盤)
    ↓
Phase 2 (ドメイン SFT)
    ↓
Phase 3 (Stage 0: 回帰 fail-fast)
    ↓
Phase 4 (Stage 0: 制御ベース表現)
    ↓
Phase 5 (Stage 1: 離散トークン) ──→ Phase 6 (Stage 2: Flow)
    ↓                                       ↓
Phase 7 (Stage 3: CoC SFT) ←───────────────┘
    ↓
Phase 8 (Stage 4: RL)
```

**最短経路**: Phase 1 → 2 → 3 → 4 → 5 → 7（Flow スキップ、CoC のみ）
**推奨経路**: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8（全 Stage 実施）

---

## リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| VRAM 不足 | 学習不可 | micro-batch=1, checkpointing, 解像度縮小 |
| LLM に視覚トークンが伝わらない | 学習が進まない | ドメイン SFT で Adapter を先に学習、frozen LLM で adapter だけ先に学習 |
| ドメイン SFT のデータ品質 | 後段の性能上限 | 教師 VLM のプロンプトを改善、生成データのフィルタリング |
| 制御ベース表現の GT 逆算精度 | 学習ターゲットの品質 | Tikhonov 正則化（有限差分行列 + 正則化項）で平滑化、逆算結果を可視化して確認 |
| 離散トークンの量子化誤差 | 軌道精度の劣化 | ビン数を調整、量子化範囲をデータ分布に合わせる |
| Flow の学習が不安定 | Stage 2 停滞 | Flow steps を減らす、条件付けを Option A（軽量）に |
| CoC auto-labeling の品質 | 推論学習の質 | プロンプトを反復改善、VLM の出力をフィルタリング |
| RL のロールアウトが VRAM に収まらない | Stage 4 実行不可 | K を減らす（K=2〜4）、推論品質報酬を完全オフライン化 |
| データセットの質 | 性能上限が低い | 複数データセットを試す、シミュレータ併用 |
