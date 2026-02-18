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

- [ ] **Vision Encoder**: DINOv2 ViT-B/14 ロード + forward
- [ ] **Adapter**: 平均 Pool + Linear（最小実装）
  - DINOv2 出力 (256×768) → (16×896)
- [ ] **LLM**: Qwen2.5-0.5B ロード + visual tokens 入力の forward
  - 視覚トークンを embedding 空間に注入する方法を確定
- [ ] **Action Head**: MLP（LLM hidden → [steer, throttle]）— fail-fast 用の最小出力

### 3.2 統合・学習ループ

- [ ] 統合モデル `MiniPamayo` クラス
- [ ] ドメイン SFT の学習済み重みを初期値として使用（§3.7）
- [ ] 学習ループ実装:
  - micro-batch=1, grad accumulation=16
  - bf16 mixed precision
  - gradient checkpointing（DINO + LLM）
  - AdamW optimizer
  - Huber loss
- [ ] wandb / tensorboard ロギング
- [ ] VRAM 使用量の実測・記録

### 3.3 Exit 条件

- [ ] 学習 loss が安定して下がる
- [ ] OOM しない（24 GB 以内）
- [ ] 推論で入力画像に応じて出力が変化する（overfitting でも可）

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

### 4.4 Exit 条件

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

### 5.3 評価

- [ ] 離散トークン → 連続制御入力 → 軌道への decode
- [ ] ADE / FDE を回帰版（Stage 0）と比較
- [ ] 量子化ビン数の影響調査

### 5.4 Exit 条件

- [ ] cross-entropy loss が下がる
- [ ] 離散化による性能劣化が許容範囲内
- [ ] decode した軌道が妥当

---

## Phase 6: Stage 2 — Flow Matching

### 目的

LLM 内部表現を条件として、Flow Matching で連続的かつ多様な軌道を生成する（§3.6 Stage C）。Dual Representation の推論側を完成させる。

### 6.1 Trajectory Decoder 実装（~150M params）

- [ ] Conditional Flow Matching (CFM) の基本実装
  - Gaussian OT path: aₜ = t·a + (1-t)·ε
  - Target field: u = a - ε
  - Flow network: 小さな Transformer
- [ ] 条件付け方式:
  - **Option A（軽量）**: LLM 最終層 hidden states → 条件ベクトル
  - **Option B（Alpamayo 寄り）**: LLM KV-cache → cross-attention
- [ ] タイムステップ embedding の実装

### 6.2 学習

- [ ] Stage 0/1 の学習済み重み（Vision + Adapter + LLM）を初期値として使用
- [ ] **stop-gradient**: Vision Encoder + Adapter + LLM は frozen（§3.7）
- [ ] Trajectory Decoder のみ trainable
- [ ] CFM loss の実装
- [ ] gradient checkpointing を Trajectory Decoder にも適用

### 6.3 評価

- [ ] 同一入力から複数の軌道をサンプリング → 多様性の確認
- [ ] ADE / FDE を回帰版・離散版と比較
- [ ] Flow steps 数の影響を調査（10, 20, 50）

### 6.4 Exit 条件

- [ ] CFM loss が下がる
- [ ] 回帰版より多様な軌道が出る（or ノイズ耐性が上がる）
- [ ] 推論速度が離散トークン自己回帰より速い

---

## Phase 7: Stage 3 — CoC 推論 SFT

### 目的

構造化された推論（Chain of Causation）を LLM に獲得させ、推論トークンとアクショントークンを同一シーケンスで自己回帰生成する（§3.8）。

### 7.1 CoC auto-labeling パイプライン

- [ ] 教師 VLM（GPT-4o 等）を使った CoC アノテーション生成
  - 閉じた意思決定集合の定義（データセットに応じたサブセット）
    - 例: `{go_straight, turn_left, turn_right, stop, follow_lead, lane_change_left, lane_change_right, yield}`
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

### 7.3 評価

- [ ] 生成された推論の品質（外部 LLM による自動評価）
- [ ] 推論の Driving Decision が正しいかの一致率
- [ ] 推論付き / なしでの軌道品質比較
- [ ] 推論テキストの定性的評価（サンプルを目視確認）

### 7.4 Exit 条件

- [ ] SFT loss が下がる
- [ ] 推論テキストが入力画像に対しておおよそ妥当な内容
- [ ] 推論ありの方が軌道品質が改善（or 同等）

---

## Phase 8: Stage 4 — RL ポストトレーニング

### 目的

SFT だけでは残る推論品質の問題（データバイアス、推論-行動の不一致、ハルシネーション）を RL で改善する（§3.9）。

### 8.1 GRPO 実装

- [ ] GRPO（Group Relative Policy Optimization）の基本実装
  - K 個のロールアウトサンプリング（K=4〜8、VRAM 制約）
  - グループ内相対 advantage 計算
  - KL 正則化（SFT モデルを reference policy として保持）
- [ ] LLM のみ trainable、他は frozen（§3.7）

### 8.2 報酬関数

- [ ] **推論品質 (r_reason)**: 外部 LLM API による 0-5 スケール採点
  - オフライン計算（API レイテンシのためオンラインは困難）
- [ ] **CoC-Action 一貫性 (r_consistency)**: バイナリ報酬
  - 予測軌道 → meta-action 抽出 → 推論テキストの意図と照合
- [ ] **低レベル軌道品質 (r_traj)**:
  - L2 距離（予測 vs エキスパート軌道）
  - ジャーク抑制ペナルティ
  - 衝突ペナルティ（nuScenes の周囲物体情報を利用）

### 8.3 学習

- [ ] SFT モデル（Stage 3）を初期値として使用
- [ ] Reference policy を frozen で保持（KL 計算用）
- [ ] 推論品質報酬はオフラインで事前計算
- [ ] 軌道品質報酬はオンライン計算
- [ ] ロールアウト → 報酬計算 → ポリシー更新のループ実装

### 8.4 評価

- [ ] SFT モデル（Stage 3）との比較:
  - 推論品質スコアの改善
  - CoC-Action 一貫性の改善
  - 軌道品質（ADE / FDE）の改善
- [ ] 推論と行動の一貫性チェック（「止まる」と言って止まるか等）
- [ ] KL 正則化の強度による影響調査

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
| 制御ベース表現の GT 逆算精度 | 学習ターゲットの品質 | Tikhonov 正則化で平滑化、逆算結果を可視化して確認 |
| 離散トークンの量子化誤差 | 軌道精度の劣化 | ビン数を調整、量子化範囲をデータ分布に合わせる |
| Flow の学習が不安定 | Stage 2 停滞 | Flow steps を減らす、条件付けを Option A（軽量）に |
| CoC auto-labeling の品質 | 推論学習の質 | プロンプトを反復改善、VLM の出力をフィルタリング |
| RL のロールアウトが VRAM に収まらない | Stage 4 実行不可 | K を減らす（K=2〜4）、推論品質報酬を完全オフライン化 |
| データセットの質 | 性能上限が低い | 複数データセットを試す、シミュレータ併用 |
