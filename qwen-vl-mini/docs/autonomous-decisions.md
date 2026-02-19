# 自律判断ログ

ユーザーの就寝中に行った判断の記録。

---

## 判断 1: Stage 1 学習を最後まで走らせる

**状況**: Step 860/2325 (37%) の時点で loss が ~4.0 付近でプラトーに見えた。ユーザーから「最後まで走らせるか、早期停止するか」の質問があった。

**判断**: 最後まで走らせる

**理由**:
1. Cosine schedule の後半で lr が下がり、微調整効果が期待できる
2. LLaVA 論文が 1 epoch 完走を採用
3. 残り時間が約 1 時間程度と短い
4. 最終チェックポイントが最も安定した重みになる

**結果**: 正解。最終 loss=3.29 で、後半で loss の下限が徐々に改善した。

---

## 判断 2: Stage 2 の実装に着手する

**状況**: Stage 1 学習が完了（2,325 ステップ、loss 7.74→3.29）。

**判断**: Phase 3（Stage 2: Visual Instruction Tuning）の実装に進む

**理由**:
1. Phase 2 の Exit 条件（loss が安定して下がる）をクリア
2. 設計書 (phase3-implementation.md) に詳細な実装プランが既にある
3. Stage 2 のデータ（COCO 2014 ~13GB）のダウンロードに時間がかかるため、先に開始すべき
4. ユーザーから「引き続き進めてください。自分で判断していい」との指示

**実装内容**:
- LLaVA-Instruct-150K + COCO 2014 画像のダウンロード
- InstructDataset + InstructCollator の実装
- train_stage2.py の実装
- パイプライン検証（fail-fast）

---

## 判断 3: データダウンロードを先行開始

**状況**: COCO 2014 train images は ~13 GB で、ダウンロードに時間がかかる。

**判断**: コード実装と並行してダウンロードを開始する

**理由**:
1. ダウンロードは I/O バウンドでコード実装と並行可能
2. データが揃ってからパイプライン検証できるよう準備しておく

---

## 判断 4: COCO 画像ファイル名の不一致に対応

**状況**: LLaVA-Instruct JSON では画像名が `000000033471.jpg` だが、COCO train2014 の実ファイル名は `COCO_train2014_000000033471.jpg`。

**判断**: InstructDataset に `_resolve_image_path()` メソッドを追加し、両方の命名規則に対応

**理由**:
1. ファイルのリネームは 82K ファイルで非現実的
2. dataset 側で透過的に解決するのが最もクリーン
3. bare name → COCO prefix の順で試行するフォールバック方式

---

## 判断 5: Stage 2 でまず VE frozen で開始する可能性

**状況**: 設計書では全パラメータ解凍を推奨しつつ、不安定な場合の対策も記載。

**判断**: `--ve_frozen` オプションを train_stage2.py に実装済み。まずは全解凍で試行し、発散したら `--ve_frozen` に切り替える方針。

**理由**:
1. 設計書の推奨ワークフロー（まず全解凍 → 発散なら対策）に準拠
2. Qwen2.5-VL 方式（Stage 2 で VE 解凍）と同方式
3. DINOv2 ViT-B/14 は 86M と小さいため VRAM への影響は限定的

---

## 判断 6: Python 出力バッファリング問題の対処

**状況**: Stage 2 学習をバックグラウンドで起動したが、30分経過してもログ出力が表示されなかった。GPU は 70% 利用率で学習自体は進行中（checkpoint-99, checkpoint-199 が保存済み）。

**判断**: 学習を停止し、`PYTHONUNBUFFERED=1` + `log()` ヘルパー関数（`flush=True`）を追加して再起動

**理由**:
1. Python はリダイレクト先が tty でない場合、stdout をブロックバッファリングする
2. print 出力がバッファに溜まり、ファイルに書き出されないため監視不能
3. `PYTHONUNBUFFERED=1` + `flush=True` で確実にリアルタイム出力

---

## 判断 7: train_stage2.py に `--resume` 機能を追加

**状況**: バッファリング修正のため学習を停止したが、checkpoint-199 まで進行済み。最初から再学習するのは無駄。

**判断**: `--resume` フラグを追加。全モデル重み + optimizer + scheduler の状態を復元し、途中から再開可能にする。

**理由**:
1. 200ステップ分の計算を無駄にしない
2. データのスキップ（microbatch_count ベース）で正確な再開点を再現
3. 将来的にも学習中断からの復帰に必要な機能

---

## 実装済みファイル (Stage 2)

| ファイル | 内容 |
|---|---|
| `data/instruct_dataset.py` | InstructDataset + InstructCollator |
| `train_stage2.py` | Stage 2 学習スクリプト（resume 対応） |
| `test_pipeline_stage2.py` | Stage 2 パイプライン検証 |
| `data/__init__.py` | InstructDataset をエクスポートに追加 |

## データ状況

| データ | 状況 |
|---|---|
| LLaVA-Instruct-150K (157K samples) | ✓ 完了 |
| COCO 2014 train images (~13 GB, 82,783枚) | ✓ 完了（展開済み） |

## Stage 2 パイプライン検証結果

全 6 テスト通過。Peak VRAM: 6,473 MB。

## Stage 2 学習状況

- 全ステップ数: 2,464（2 epochs, global batch=128）
- checkpoint-199 から再開、現在進行中
- 初期 loss: ~1.77、Step 245 時点で ~1.45
- 速度: ~9 steps/min → 推定残り ~4時間
