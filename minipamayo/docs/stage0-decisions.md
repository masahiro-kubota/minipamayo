# Stage 0 Phase 3 実装 — 判断記録

## 概要

Stage 0 Phase 3（fail-fast パイプライン検証）の実装中に行った判断を記録する。

---

## 判断 1: パッケージ管理 — uv 採用

**選択肢**: uv / pip / conda
**決定**: uv
**理由**: qwen-vl-mini / cosmos-reason-mini と同じ管理ツールで統一。`uv run python -m minipamayo.xxx` で実行するパターンも踏襲。

---

## 判断 2: Adapter アーキテクチャ — MeanPoolAdapter（新規）

**選択肢**:
- A) チェックポイントの PerTokenAdapter をそのまま使用（256 視覚トークン）
- B) 設計書通り MeanPoolAdapter を新規作成（1 視覚トークン）

**決定**: B（MeanPoolAdapter）
**理由**:
- 設計書 §3.2 / stage0-regression.md の「Phase 3: 平均Pool + Linear → 1 token」に従う
- fail-fast の目的は「勾配が全経路に流れること」の検証であり、最小構成が適切
- 256 トークンの LLM forward は 1 トークンより遅く、fail-fast には過剰

**トレードオフ**:
- チェックポイントの Adapter 重み（PerTokenAdapter 用）は破棄される
- VisionEncoder と LLM の重みは引き続き使用されるため、視覚-言語アライメントの大部分は保持

---

## 判断 3: チェックポイント — rl-mini-merged/checkpoint-final.pt

**選択肢**:
- `cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt` (1.3 GB)
- `cosmos-reason-mini/checkpoints/sft-mini-merged/checkpoint-18.pt` (3.8 GB)
- `qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt` (3.8 GB)

**決定**: rl-mini-merged/checkpoint-final.pt
**理由**: Cosmos Reason Mini の RL 学習済み（最終段階）の重みであり、運転ドメインの知識が最も注入された状態。設計書の「Cosmos Reason Mini → MiniPamayo Stage 0」の流れに合致。

---

## 判断 4: nuScenes データ — 共有パスで参照

**選択肢**:
- A) symlink を作成（`minipamayo/data/nuscenes → cosmos-reason-mini/data/nuscenes`）
- B) コマンドライン引数でパスを指定

**決定**: B（引数指定、デフォルト値に相対パスを設定）
**理由**:
- symlink は `.gitignore` 管理が複雑になる
- 引数のデフォルト値 `../cosmos-reason-mini/data/nuscenes` で十分実用的
- 将来 nuScenes Trainval を別の場所に置く場合にも柔軟

---

## 判断 5: LLM 入力 — 視覚トークンのみ（テキストなし）

**選択肢**:
- A) 視覚トークンのみを LLM に入力
- B) 視覚トークン + テキストプロンプト（"Predict driving action."等）

**決定**: A（視覚トークンのみ）
**理由**:
- Phase 3 は fail-fast であり、LLM はここでは特徴抽出器として使用
- テキストプロンプトの追加は Stage 3（CoC SFT）で行う
- 最小構成で動作検証を優先

**注意**: 1 視覚トークンのみの場合、decoder-only LLM の self-attention は実質 MLP として機能する（自分自身にしか attend できない）。これは Phase 3 では許容される。Phase 4 で Cross-Attention Pooling（16 tokens）に置き換えるとこの問題は解消する。

---

## 判断 6: GT アクション — ego pose 差分から近似算出

**方式**: 3 フレーム連続を使用
- steer = yaw(frame i) - yaw(frame i-1)（ラジアン）
- speed = ||pos(frame i) - pos(frame i-1)|| / dt
- throttle = speed(frame i) - speed(frame i-1)（加速度近似）

**注意点**:
- nuScenes キーフレームは 2Hz なので dt ≈ 0.5s
- steer/throttle はフレーム間の差分であり、制御入力 (a, κ) とは異なる
- Phase 4 で制御ベース表現に移行する際に、Tikhonov 正則化付き逆ダイナミクスに置き換える

---

## 判断 7: Train/Val 分割

**方式**: ランダム分割 80/20（seed=42）
**理由**: nuScenes Mini は 10 シーンのみ。シーン単位で分割すると val が 2 シーンだけになり不安定。ランダム分割の方がサンプル数のバランスが良い。

**注意**: ランダム分割では同じシーンの異なるフレームが train/val に分かれるため、情報漏洩のリスクがある。Phase 4（Trainval 使用時）ではシーン単位分割に切り替えるべき。

---

## 判断 8: Qwen2.5-0.5B の実 vocab_size

チェックポイントの `embed_tokens.weight` は shape (151936, 896)。設計書 stage1-discrete-tokens.md では vocab_size=151,646 と記載しているが、実際のモデルは 151,936（64 の倍数にパディング）。

**影響**: Stage 1 で離散トークンを追加する際、`vocab_offset` は 151,936 にすべき。stage1-discrete-tokens.md の vocab_size_extended は要修正（151,902 → 152,192）。

**対応**: Stage 1 実装時に修正する。

---

## 実装結果サマリ

### パイプラインテスト

| 項目 | 結果 |
|---|---|
| Forward pass | 成功 — 出力 shape (B, 2) |
| Backward pass | 成功 — 全 4 モジュールに勾配 |
| Peak VRAM (単体テスト) | 2.45 GB |
| データセットサイズ | 384 サンプル（nuScenes Mini, 10 scenes） |
| Train/Val 分割 | 308 / 76 |

### 学習結果（10 epoch, nuScenes Mini）

| メトリクス | 学習前 | 学習後 | 改善 |
|---|---|---|---|
| Val Loss | 0.991 | 0.018 | -98.2% |
| Steer MAE | 1.372 | 0.037 | -97.3% |
| Throttle MAE | 1.510 | 0.195 | -87.1% |

| 項目 | 値 |
|---|---|
| Peak VRAM | 5.69 GB |
| パラメータ | 581.5M（全 trainable） |
| Input-dependent | YES（異なる画像に対し異なる出力） |
| 勾配フロー | 全 4 モジュール OK |
| Best checkpoint | checkpoints/stage0/best.pt |

### Phase 3 Exit Criteria

- [x] 勾配が全経路に流れる（VisionEncoder → Adapter → LLM → ActionHead）
- [x] 学習 loss が減少する
- [x] OOM なし（5.69 GB / 24 GB）
- [x] 入力依存の出力（collapse なし）

### 評価結果（全384サンプル、eval_stage0.py）

| メトリクス | 値 |
|---|---|
| Loss (Huber) | 0.019415 |
| Steer MAE | 0.034353 |
| Throttle MAE | 0.192706 |

#### 予測分布の分析

| チャネル | | mean | std | min | max |
|---|---|---|---|---|---|
| **Steer** | GT | +0.004 | 0.054 | -0.258 | +0.237 |
| | Pred | +0.019 | **0.007** | +0.004 | +0.040 |
| **Throttle** | GT | +0.024 | 0.291 | -1.301 | +0.855 |
| | Pred | +0.056 | **0.082** | -0.048 | +0.242 |

**所見**:
- 予測の分散が GT に比べて極端に小さい（steer: 0.007 vs 0.054、throttle: 0.082 vs 0.291）
- モデルは「平均値付近を出す」ことで Loss を下げている状態で、意味のある予測ではない
- 特に急減速（GT ≈ -1.2）を全く捉えられていない（Worst 5 は全て急減速シーン）
- Stage 0 の目的（勾配フロー検証）は達成済みなので想定内。フルデータでの学習後に再評価する

### 作成ファイル

```
minipamayo/
├── pyproject.toml
├── src/minipamayo/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vision_encoder.py    # DINOv2 ViT-B/14 wrapper
│   │   ├── adapter.py           # MeanPoolAdapter + PerTokenAdapter
│   │   ├── action_head.py       # MLP regression head
│   │   └── minipamayo.py        # 統合モデル + checkpoint loading
│   ├── data/
│   │   ├── __init__.py
│   │   └── nuscenes_dataset.py  # nuScenes dataset (steer/throttle GT)
│   ├── train_stage0.py          # 学習スクリプト
│   └── eval_stage0.py           # 評価スクリプト
└── docs/
    └── stage0-decisions.md      # 本ファイル
```
