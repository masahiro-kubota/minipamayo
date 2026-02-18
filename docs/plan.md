# MiniPamayo 実装計画

## 全体方針

**fail-fast** で段階的に進める。各 Stage に明確な Exit 条件を設け、クリアしてから次へ進む。

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
│       │   ├── vision_encoder.py    # DINOv2 ViT-S/14 ラッパー
│       │   ├── adapter.py           # Vision→LLM Adapter
│       │   ├── llm.py               # SmolLM2-360M ラッパー
│       │   ├── action_head.py       # MLP 回帰ヘッド
│       │   ├── flow_head.py         # Flow Matching ヘッド (Stage 2)
│       │   └── minipamayo.py        # 統合モデル
│       ├── data/
│       │   ├── dataset.py           # データセットクラス
│       │   └── transforms.py        # 画像前処理
│       ├── training/
│       │   ├── trainer.py           # 学習ループ
│       │   └── losses.py            # 損失関数
│       └── utils/
│           └── config.py            # 設定管理
├── configs/
│   ├── stage0.yaml
│   ├── stage1.yaml
│   └── stage2.yaml
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── visualize.py
├── tests/
├── pyproject.toml
└── README.md
```

### 1.2 データセット準備

- [ ] データセット選定・ダウンロード（→ [datasets.md](datasets.md) 参照）
- [ ] DataLoader 実装（画像 + アクションラベル）
- [ ] データ前処理パイプライン（リサイズ、正規化）
- [ ] 単体テスト: バッチが正しく取り出せることを確認

---

## Phase 2: Stage 0 — パイプライン検証（最小回帰）

### 目的

勾配が全経路 **DINO → Adapter → LLM → Action Head** に流れ、学習が回ることを確認する。

### 2.1 各モジュール実装

- [ ] **Vision Encoder**: DINOv2 ViT-S/14 ロード + forward
- [ ] **Adapter**: 平均 Pool + Linear（最小実装）
  - DINOv2 出力 (256×384) → (16×960)
- [ ] **LLM**: SmolLM2-360M ロード + visual tokens 入力の forward
  - 視覚トークンを embedding 空間に注入する方法を確定
- [ ] **Action Head**: MLP（LLM hidden → [steer, throttle]）

### 2.2 統合・学習ループ

- [ ] 統合モデル `MiniPamayo` クラス
- [ ] 学習ループ実装:
  - micro-batch=1, grad accumulation=16
  - bf16 mixed precision
  - gradient checkpointing（DINO + LLM）
  - AdamW optimizer
  - Huber loss
- [ ] wandb / tensorboard ロギング
- [ ] VRAM 使用量の実測・記録

### 2.3 Exit 条件

- [ ] 学習 loss が安定して下がる
- [ ] OOM しない（24 GB 以内）
- [ ] 推論で入力画像に応じて出力が変化する（overfitting でも可）

### 想定所要時間

最小データセット（数千フレーム）で数時間〜半日の学習で loss 低下を確認できる想定。

---

## Phase 3: Stage 0 改善

### 3.1 Adapter 改善

- [ ] Cross-Attention Pooling に置き換え
  - learnable query: 16 個
  - DINOv2 パッチ特徴を key/value に使用
- [ ] 性能比較（loss 低下速度、最終 loss）

### 3.2 評価指標の整備

- [ ] Waypoint 予測に変更: `(K, 2)` 出力
- [ ] 評価指標: ADE (Average Displacement Error), FDE (Final Displacement Error)
- [ ] 可視化: 予測 waypoint vs GT をカメラ画像上にプロット

### 3.3 テキスト入力の追加（任意）

- [ ] 固定テキストプロンプト（例: "Drive forward"）を LLM に入力
- [ ] テキスト有無で学習挙動を比較

---

## Phase 4: Stage 1 — アクション離散化（任意・スキップ可）

### 目的

Alpamayo の「アクション離散トークン化」思想を試す。回帰が安定した後のみ実施。

- [ ] 連続アクション → VQ-VAE / k-means でコードブック学習
- [ ] 離散トークンを LLM の語彙に追加
- [ ] LLM の出力として離散アクショントークンを生成
- [ ] 連続値への decode

### Exit 条件

- [ ] 離散化による性能劣化が許容範囲内

---

## Phase 5: Stage 2 — Flow Matching

### 目的

LLM 内部表現を条件として、Flow Matching で多様な軌道を生成する。

### 5.1 Flow Head 実装

- [ ] Flow Matching の基本実装
  - Conditional Flow Matching (CFM) or Rectified Flow
  - Flow network: 小さな MLP or Transformer
  - Flow steps: 初期 10
- [ ] 条件付け方式の選択:
  - **Option A**: LLM 最終層 hidden states → 条件ベクトル
  - **Option B**: LLM KV-cache → cross-attention で条件付け

### 5.2 学習

- [ ] Flow loss (CFM loss) の実装
- [ ] Stage 0 の学習済み重み（Vision + Adapter + LLM）を初期値として使用
- [ ] Flow Head のみ or 全体を fine-tune
- [ ] gradient checkpointing を Flow Head にも適用

### 5.3 評価

- [ ] 同一入力から複数の軌道をサンプリング → 多様性の確認
- [ ] ADE / FDE を回帰版と比較
- [ ] Flow steps 数の影響を調査（10, 20, 50）

### Exit 条件

- [ ] Flow loss が下がる
- [ ] 回帰版より多様な軌道が出る（or ノイズ耐性が上がる）

---

## 実装優先順位まとめ

```
Phase 1 (基盤)  ──▶  Phase 2 (Stage 0: 回帰) ──▶  Phase 3 (改善)
                                                         │
                                                         ▼
                                                   Phase 5 (Stage 2: Flow)
                                                         ▲
                                                         │ (任意)
                                                   Phase 4 (Stage 1: 離散化)
```

**最短経路**: Phase 1 → Phase 2 → Phase 5（Stage 1 スキップ）

---

## リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| VRAM 不足 | 学習不可 | micro-batch=1, checkpointing, 解像度縮小 |
| LLM に視覚トークンが伝わらない | 学習が進まない | Adapter を段階的に複雑化、frozen LLM で adapter だけ先に学習 |
| Flow の学習が不安定 | Stage 2 停滞 | Flow steps を減らす、条件付けを Option A（軽量）に |
| データセットの質 | 性能上限が低い | 複数データセットを試す、シミュレータ併用 |
