# Stage 1: Feature Alignment — 学習結果レポート

## 概要

DINOv2 ViT-B/14 の視覚特徴を Qwen2.5-0.5B の入力空間にマッピングする Adapter（2層 MLP）を学習した。
VisionEncoder + LLM は frozen、**Adapter のみ学習**（5.1M パラメータ）。

## 学習設定

| パラメータ | 値 |
|---|---|
| データセット | LLaVA-CC3M-Pretrain-595K (595,375 サンプル) |
| Trainable params | 5,115,776 (Adapter only) |
| Optimizer | AdamW (betas=0.9/0.95) |
| Learning rate | 1e-3 (cosine with warmup) |
| Warmup steps | 69 (3% of total) |
| Batch size | micro=4, grad_accum=64, **global=256** |
| Epochs | 1 |
| Total steps | 2,325 |
| Max sequence length | 512 |
| Precision | bf16 (autocast) |
| GPU | NVIDIA RTX 4090 (24GB) |

## 学習結果

| 指標 | 値 |
|---|---|
| 初期 loss | 7.74 |
| 最終 loss | 3.29 |
| Loss 低下率 | 57.5% (7.74 → 3.29) |
| Loss 最低値 | ~2.55 (Step 680 付近) |
| NaN 発生 | なし |
| 最終 grad_norm | 0.40 |
| 学習時間 | 約 2.5 時間 |

## Loss カーブの特徴

1. **Step 0-70 (warmup)**: 7.74 → 4.57 (急激な低下、warmup 中)
2. **Step 70-300**: 4.57 → 3.5-5.0 で振動しながら低下
3. **Step 300-2325**: 3.0-5.5 の範囲で振動、平均は徐々に低下
4. **後半 (Step 1500+)**: cosine schedule で lr が低下し、3.0 未満の値も頻出

## T2 自動チェック結果

- ✓ Adapter weights are being updated (Step 1)
- ✓ Learning check passed: 7.7406 → 4.4128 (Step 100)
- ✓ NaN なし (全 2,325 ステップ)

## チェックポイント

| ファイル | ステップ | サイズ |
|---|---|---|
| checkpoint-499.pt | 499 | 59 MB |
| checkpoint-999.pt | 999 | 59 MB |
| checkpoint-1499.pt | 1499 | 59 MB |
| checkpoint-1999.pt | 1999 | 59 MB |
| **checkpoint-2325.pt** | **2325 (final)** | **59 MB** |

## wandb

- Run: https://wandb.ai/norikenpi-individual/qwen-vl-mini/runs/5ubxpdw3

## 判断・考察

### Loss プラトーについて

Loss が Step 300 あたりから ~4.0 前後でプラトーに見えたが、最後まで走らせた。理由:
1. **Cosine schedule の後半効果**: 学習率が0に向けて滑らかに下がるため、後半で微調整が入る
2. **LLaVA 論文準拠**: オリジナルも 1 epoch 完走
3. **アニーリング済みの重み**: 最終チェックポイントが最もlr-annealedされた重み

実際、後半で loss の下限が 3.0 → 2.8 → 2.6 と徐々に下がっていた。

### Stage 1 の妥当性

- Stage 1 (Adapter only) の目的は「DINOv2 特徴を LLM が読める形にマッピングする」こと
- Loss ~3.3 は次トークン予測の cross-entropy として妥当（vocab=151,936 のランダム予測は ~11.9）
- 完璧なキャプション生成は期待していない（それは Stage 2 の仕事）

## Stage 1 前後比較

同じ 5 枚の COCO val2014 画像（seed=42）に対して、学習前（ランダム初期化 Adapter）と学習後（checkpoint-2325.pt）の生成テキストを比較した。

```bash
# 前（ランダム Adapter）
uv run python -m qwen_vl_mini.eval_qualitative --image-dir data/coco/val2014 --seed 42
# 後（学習済み Adapter）
uv run python -m qwen_vl_mini.eval_qualitative --checkpoint checkpoints/stage1/checkpoint-2325.pt --stage 1 --image-dir data/coco/val2014 --seed 42
```

| 画像 | 内容 | 前（ランダム Adapter） | 後（checkpoint-2325.pt） |
|------|------|------------------------|--------------------------|
| 105156 | 馬車 | "Describe this image in detail.InSection InSection..." | "horse drawn cart on the street" |
| 022861 | ピザ | "Describe this image in detail.InSection InSection..." | "the best way to eat pizza is to have it with a side of pasta" |
| 258529 | 石造りの家 | "Describe this image in detail.InSection InSection..." | "the house is a small stone hut with a small garden" |
| 229840 | 犬 | "Describe this image in detail.InSection InSection..." | "a dog with a black and white dog on a fence" |
| 209747 | 猫 | "Describe this image in detail.InSection InSection..." | "a kitten in a bathroom with a sink and a toothbrush" |

**前（ランダム Adapter）**: 全画像で同一パターン — プロンプトのオウム返し + "InSection" トークンの無限繰り返し。画像の内容を一切反映していない。

**後（学習済み Adapter）**: 5/5 枚で画像の主要物体を正しく認識。画像ごとに異なるキャプションを生成。Adapter の学習が視覚-言語アライメントに成功したことを示す。

## 定性的評価（eval_qualitative.py）

COCO val2014 からランダム 5 枚を選択し、5 種類の質問で生成テスト（greedy decoding, max_new_tokens=200）。

### 結果サマリ

| 画像 | 内容 | 生成テキスト（冒頭） | 画像認識 |
|------|------|----------------------|----------|
| 105156 | 馬車 | "horse drawn cart on the street" | ✓ |
| 022861 | ピザ | "the best way to eat pizza is to have it with a side of pasta" | ✓ |
| 258529 | 石造りの家 | "the house is a small stone hut with a small garden" | ✓ |
| 229840 | 犬 | "a dog with a black and white dog on a fence" | ✓ |
| 209747 | 猫 | "a kitten in a bathroom with a sink and a toothbrush" | ✓ |

### 観察

**ポジティブ**:
- 5/5 枚で画像の主要な物体を正しく認識（馬車、ピザ、家、犬、猫）
- 画像ごとに異なる出力 — 縮退（全画像同じ出力）は発生していない
- Adapter による視覚-言語アライメントが機能している

**ネガティブ**:
- 全サンプルで繰り返しループに入る（EOS を生成できない）
- 質問の違いに対する応答の分化が弱い（ほぼ同じキャプションを返す）
- 多言語ゴミトークン混入（アラビア語、中国語、ロシア語等 — Qwen2.5 の多言語 vocab の影響）

### Exit 条件の判定

| 条件 | 結果 |
|------|------|
| Loss が安定して下がる | **✓** 7.74 → 3.29 |
| 画像に関連したテキストが生成される | **✓** 5/5 枚で関連キーワードを含む |
| 画像の内容に応じて出力が変化する | **✓** 画像ごとに異なる出力 |

繰り返しと多言語ゴミは Stage 1（Adapter のみ学習）の制約として想定内。Stage 2 での改善を期待。

### 定量ベンチマーク（POPE 等）を実施しない理由

Stage 1 は Adapter のみの学習で、質問応答能力を獲得していない。POPE（Yes/No 判定）等のベンチマークを測定しても、instruction following ができないため ~50%（ランダム相当）になることが予測される。SmolVLM-256M 等の instruction tuning 済みモデルとの比較も不公平。定量ベンチマークは Stage 2（全パラメータ解凍 + instruction tuning）完了後に実施する。

## 次のステップ

Stage 2: Visual Instruction Tuning に進む。
- Stage 1 の最終チェックポイント (`checkpoint-2325.pt`) の Adapter 重みを引き継ぎ
- LLaVA-Instruct-150K + COCO 2014 画像で学習
- DINOv2 + Adapter + LLM を全解凍して end-to-end 学習
