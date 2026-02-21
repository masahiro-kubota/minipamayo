# Stage 3: Chain of Causation (CoC) SFT — 実装計画

## 目的

構造化された推論（Chain of Causation）を LLM に獲得させ、推論トークンとアクショントークンを同一シーケンスで自己回帰生成する。Alpamayo-R1 の核心的貢献の一つであり、自由形式の CoT ではなく **構造化された因果推論** で推論と行動の因果整合性を確保する。

SFT で推論能力を模倣学習する。ただし SFT だけではデータバイアスによる不完全な推論が残るため、Stage 4 RL で改善する前提。

---

## Alpamayo 論文との対応

| 本ドキュメント | Alpamayo 論文 | 内容 |
|---|---|---|
| CoC の構造定義 | **§4.1** CoC Dataset Design | Driving Decision + Critical Components + CoC Trace の 3 要素 |
| 閉じた意思決定集合 | **§4.1** | 縦方向・横方向の離散的な運転意思決定カテゴリ |
| Auto-labeling パイプライン | **§4.3** Auto-labeling | VLM による CoC アノテーション自動生成 |
| 因果混乱の防止 | **§4.2** Causal Confusion | 過去の観測のみから因果要因を特定 |
| 推論 SFT | **§5.2** Eliciting Reasoning | SFT で推論能力を注入（RL の前段） |
| 勾配制御 | **§5.1** Action Expert Training | Stage ごとの freeze/unfreeze 戦略 |

---

## 前提条件

- [x] Qwen2.5-VL Mini Stage 1（Feature Alignment）完了 — Adapter が視覚-言語アライメントを獲得
- [x] Qwen2.5-VL Mini Stage 2（Visual Instruction Tuning）完了 — VLM 能力を獲得
- [ ] MiniPamayo Stage 0（回帰 / 制御ベース表現）完了 — 行動予測パイプラインの検証
- [ ] MiniPamayo Stage 1（離散トークン化）完了 — LLM 語彙に離散アクショントークン追加済み
- [ ] MiniPamayo Stage 2（Flow Matching）完了 — Trajectory Decoder 学習済み

Stage 3 は Stage 1（離散トークン）の上に構築される。推論トークンと離散アクショントークンを同一の自己回帰シーケンスで生成するため、離散トークン化が前提。

---

## CoC の構造

Alpamayo 論文 §4.1 に倣い、3 要素で構造化された推論を定義する。

### 1. Driving Decision（運転意思決定）

閉じた集合から選択する。Alpamayo の完全なリストから、MiniPamayo で使用するデータセットに関連するサブセットのみ定義。

**Alpamayo（完全版）**:
- 縦方向: set speed tracking, lead obstacle following, speed adaptation, gap-searching, acceleration for passing, yield, stop for static constraints
- 横方向: lane keeping, merge/split, out-of-lane nudge, in-lane nudge, lane change, pull-over, turn, lateral maneuver abort

**MiniPamayo（簡略版サブセット）**:

| カテゴリ | 意思決定ラベル | 説明 |
|---|---|---|
| 縦方向 | `go_straight` | 設定速度で直進 |
| 縦方向 | `follow_lead` | 先行車に追従 |
| 縦方向 | `stop` | 停止（信号、一時停止等） |
| 縦方向 | `yield` | 譲る（合流、歩行者等） |
| 横方向 | `lane_keeping` | 現在の車線を維持 |
| 横方向 | `turn_left` | 左折 |
| 横方向 | `turn_right` | 右折 |
| 横方向 | `lane_change_left` | 左車線変更 |
| 横方向 | `lane_change_right` | 右車線変更 |

計 9 種（縦方向 4 + 横方向 5）。nuScenes / comma2k19 等の公開データセットで出現する主要な運転行動をカバー。

### 2. Critical Components（重要な構成要素）

意思決定に直接影響する因果要因を特定する。

例:
- 信号状態（赤/黄/青）
- 先行車の存在と距離
- 歩行者の横断
- 車線構成（分岐、合流）
- ルート指示（直進、右折等）
- 速度制限標識
- 路面状態

### 3. Composed CoC Trace（推論トレース）

Driving Decision と Critical Components を結ぶ自然言語の因果推論。

**出力例**:
```
[Driving Decision]
longitudinal: follow_lead
lateral: lane_keeping

[Critical Components]
- Lead vehicle: sedan at approximately 15m ahead, decelerating
- Traffic signal: green
- Lane marking: solid white line on both sides

[CoC Trace]
The lead vehicle ahead is decelerating, requiring speed adaptation to maintain
a safe following distance. The traffic signal is green, so there is no need to
stop. Solid lane markings on both sides prevent lane changes. The appropriate
action is to follow the lead vehicle while maintaining the current lane.
```

---

## CoC データ作成

### Auto-labeling パイプライン設計

教師 VLM（GPT-4o）を使用して、学習データに CoC アノテーションを自動付与する。

#### パイプライン概要

```
運転シーン画像 + ego情報
       ↓
  GPT-4o API (auto-labeling)
       ↓
  CoC アノテーション（JSON）
       ↓
  品質フィルタリング
       ↓
  学習用シーケンス構築
```

#### 閉じた意思決定集合の定義

プロンプトに意思決定集合を明示的に含め、モデルがこの集合内から選択するよう制約する。

```python
DRIVING_DECISIONS = {
    "longitudinal": [
        "go_straight",
        "follow_lead",
        "stop",
        "yield",
    ],
    "lateral": [
        "lane_keeping",
        "turn_left",
        "turn_right",
        "lane_change_left",
        "lane_change_right",
    ],
}
```

#### Critical Components 特定のプロンプト設計

```
Given this driving scene image and the ego vehicle's current state:
- Speed: {speed} m/s
- Yaw rate: {yaw_rate} rad/s

Identify the critical components that directly influence the driving decision.
Focus ONLY on observable elements in the current image. Do NOT infer future
states or use information not visible in the image.

List each critical component with:
1. Type (e.g., lead_vehicle, traffic_signal, pedestrian, lane_marking, sign)
2. Description (position, state, relevance to ego vehicle)
3. Causal relevance (how it affects the driving decision)
```

#### CoC Trace 生成のプロンプト設計

```
Based on the identified critical components, generate a Chain of Causation
trace that explains the causal reasoning from observation to driving decision.

Requirements:
1. Reference ONLY the critical components identified above
2. Explain the causal chain: observation → reasoning → decision
3. The driving decision must be selected from the following closed set:
   - Longitudinal: {longitudinal_decisions}
   - Lateral: {lateral_decisions}
4. Keep the reasoning concise (2-4 sentences)
5. Ensure causal consistency: the reasoning must logically lead to the decision

Output format:
[Driving Decision]
longitudinal: <decision>
lateral: <decision>

[Critical Components]
- <component_type>: <description>
...

[CoC Trace]
<causal reasoning text>
```

#### 因果混乱防止プロンプト

Alpamayo 論文 §4.2 に倣い、未来の情報漏洩を防ぐ:

```
IMPORTANT: Causal confusion prevention rules:
- Use ONLY information observable in the CURRENT image
- Do NOT reference future events or outcomes
- Do NOT use hindsight knowledge about what will happen next
- Base all reasoning on currently visible objects, signals, and road geometry
- If the cause of a decision is ambiguous from the current frame alone,
  state the uncertainty explicitly
```

### データ品質チェック

auto-labeling の出力に対して以下のフィルタリングを適用:

1. **構造検証**: JSON パース可能か、必須フィールドが存在するか
2. **意思決定集合チェック**: Driving Decision が閉じた集合内か
3. **因果整合性**: CoC Trace が参照する Critical Components が実際にリストに存在するか
4. **長さ制約**: 推論テキストが極端に短い（<20 tokens）または長い（>200 tokens）ものを除外
5. **重複排除**: 同一テンプレートの機械的な繰り返しを検出・除外

```python
def validate_coc_annotation(annotation: dict) -> bool:
    """CoC アノテーションの品質チェック。"""
    # 1. 構造検証
    required_keys = ["driving_decision", "critical_components", "coc_trace"]
    if not all(k in annotation for k in required_keys):
        return False

    # 2. 意思決定集合チェック
    decision = annotation["driving_decision"]
    if decision["longitudinal"] not in DRIVING_DECISIONS["longitudinal"]:
        return False
    if decision["lateral"] not in DRIVING_DECISIONS["lateral"]:
        return False

    # 3. 長さ制約
    trace = annotation["coc_trace"]
    trace_tokens = len(trace.split())
    if trace_tokens < 20 or trace_tokens > 200:
        return False

    return True
```

### 学習シーケンス

Stage 3 の学習シーケンスは、推論トークン・意思決定トークン・軌道トークンを連結した自己回帰シーケンス:

```
入力:  [visual_tokens, egomotion_tokens]
出力:  [CoC_reasoning_tokens, meta_action_tokens, trajectory_tokens]
       ├── 自然言語推論 ──┤├── 意思決定 ──┤├── 制御入力列 ──┤
```

具体的なトークンシーケンス:

```
<|im_start|>system
You are an autonomous driving agent. Analyze the scene and provide
structured reasoning before deciding on actions.<|im_end|>
<|im_start|>user
[visual_tokens] [egomotion_tokens]
What driving action should be taken?<|im_end|>
<|im_start|>assistant
[Driving Decision]
longitudinal: follow_lead
lateral: lane_keeping

[Critical Components]
- Lead vehicle: sedan at 15m, decelerating
- Traffic signal: green

[CoC Trace]
The lead vehicle is decelerating ahead. Maintaining lane and adjusting
speed to follow safely.

[Action]
<action_0> <action_1> ... <action_127><|im_end|>
```

**Loss 計算**: assistant の出力部分全体（推論テキスト + 離散アクショントークン）に cross-entropy loss を適用。Joint next-token prediction で推論と行動を同時に学習する。推論トークンとアクショントークンの loss 重みは**均一（1:1）**とする（Alpamayo 論文 §5.2 準拠）。

---

## 勾配制御

設計書 §3.7 に従い、Stage 3 の勾配制御:

| モジュール | 状態 | 理由 |
|---|---|---|
| **Vision Encoder** | trainable | 推論に必要な視覚特徴の微調整 |
| **Adapter** | trainable | 推論コンテキストに合わせた適応 |
| **LLM** | trainable | CoC 推論能力の獲得（本 Stage の主目的） |
| **Action Head** | frozen | Stage 0 で学習済み。推論 SFT で壊さない |
| **Flow Head** | frozen | Stage 2 で学習済み。推論 SFT で壊さない |

```python
def set_stage3(model):
    """Stage 3: CoC SFT の勾配制御。"""
    model.vision_encoder.requires_grad_(True)
    model.adapter.requires_grad_(True)
    model.llm.requires_grad_(True)
    model.action_head.requires_grad_(False)
    if hasattr(model, "flow_head"):
        model.flow_head.requires_grad_(False)
```

---

## プロジェクト構成

Stage 3 で追加・変更するファイル:

```
minipamayo/src/minipamayo/
├── models/
│   └── minipamayo.py            # set_stage3() 追加
├── data/
│   ├── coc_labeling.py          # CoC auto-labeling パイプライン（GPT-4o API）
│   └── coc_dataset.py           # CoC 付き学習データセット
├── training/
│   └── trainer.py               # Stage 3 学習ループ追加
├── prompts/
│   ├── coc_system.txt           # システムプロンプト
│   ├── coc_critical_components.txt  # Critical Components 特定プロンプト
│   └── coc_trace_generation.txt     # CoC Trace 生成プロンプト
├── configs/
│   └── stage3.yaml              # ハイパーパラメータ設定
└── scripts/
    ├── auto_label_coc.py        # CoC auto-labeling 実行スクリプト
    └── eval_stage3.py           # Stage 3 評価スクリプト
```

---

## 実装ステップ

### Step 1: 意思決定集合の定義

- [ ] 縦方向・横方向の意思決定ラベルを定数として定義
- [ ] 使用するデータセット（nuScenes 等）で各ラベルの出現頻度を確認
- [ ] 必要に応じてラベルの追加・削除を調整

### Step 2: Auto-labeling プロンプト設計

- [ ] Critical Components 特定プロンプトの作成・テスト
- [ ] CoC Trace 生成プロンプトの作成・テスト
- [ ] 因果混乱防止の制約をプロンプトに組み込み
- [ ] 少数サンプル（10-20 枚）で出力品質を手動確認
- [ ] プロンプトの反復改善（3-5 回程度）

### Step 3: CoC データ生成（GPT-4o API）

- [ ] `coc_labeling.py` の実装
  - GPT-4o API 呼び出し（画像 + テキストプロンプト）
  - JSON パース + バリデーション
  - リトライロジック（API エラー、パース失敗時）
  - レート制限対応（バッチ処理 + スリープ）
- [ ] データセット全体への auto-labeling 実行
- [ ] 生成結果の保存（JSON Lines 形式）

### Step 4: データクリーニング・品質チェック

- [ ] `validate_coc_annotation()` による自動フィルタリング
- [ ] 統計レポート生成（意思決定ラベルの分布、棄却率等）
- [ ] ランダムサンプル 50 件の目視確認
- [ ] フィルタ後のデータ件数が学習に十分か確認

### Step 5: データセット・学習ループの実装

- [ ] `coc_dataset.py` の実装
  - CoC アノテーション + 画像 + egomotion → 学習シーケンス構築
  - 推論テキスト + 離散アクショントークンの連結
  - Loss マスク構築（assistant 出力部分のみ）
- [ ] Stage 3 学習ループの実装
  - Stage 1/2 の学習済み重みをロード
  - `set_stage3()` で勾配制御を設定
  - cross-entropy loss（推論 + 離散アクションの joint next-token prediction）
  - gradient checkpointing（LLM + Vision Encoder）

### Step 6: 学習実行と評価

- [ ] 推論テキストの定性的評価（サンプル目視確認）
- [ ] Driving Decision 一致率の計測
- [ ] 推論品質の外部 LLM 自動評価
- [ ] 推論付き/なしでの軌道品質比較

---

## 評価

### 推論品質（外部 LLM 自動評価）

外部 LLM（Claude API / GPT-4o）に推論テキストを採点させる。

```python
EVAL_PROMPT = """
Rate the following driving reasoning on a scale of 0-5:

Scene description: {scene_description}
Ground truth action: {gt_action}

Model's reasoning:
{model_reasoning}

Scoring criteria:
0: Completely irrelevant or nonsensical
1: Mentions driving but reasoning is wrong
2: Partially correct reasoning but major errors
3: Mostly correct, minor issues
4: Good reasoning with correct causal chain
5: Excellent reasoning, correct and comprehensive

Score (0-5):
"""
```

### Driving Decision 一致率

生成された Driving Decision が GT（auto-label で付与した正解）と一致する割合:

```python
def compute_decision_accuracy(predictions, ground_truths):
    """Driving Decision の一致率。"""
    longitudinal_correct = 0
    lateral_correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        if pred["longitudinal"] == gt["longitudinal"]:
            longitudinal_correct += 1
        if pred["lateral"] == gt["lateral"]:
            lateral_correct += 1

    return {
        "longitudinal_accuracy": longitudinal_correct / total,
        "lateral_accuracy": lateral_correct / total,
        "overall_accuracy": (longitudinal_correct + lateral_correct) / (2 * total),
    }
```

### 推論付き/なしでの軌道品質比較

Stage 1（離散トークンのみ、推論なし）と Stage 3（推論 + 離散トークン）の軌道品質を比較:

| 指標 | Stage 1（推論なし） | Stage 3（推論あり） |
|---|---|---|
| ADE (m) | — | — |
| FDE (m) | — | — |
| Driving Decision 一致率 | N/A | — |

推論を付加することで軌道品質が改善（または同等を維持）することを確認する。

### 推論テキストのサンプル目視確認

ランダムに 20-30 サンプルを抽出し、以下を手動確認:

- 推論が画像の内容と整合しているか
- Critical Components が実際に画像に存在するか
- 因果推論が論理的に Driving Decision につながっているか
- ハルシネーション（画像にない物体への言及）がないか

---

## ハイパーパラメータ

```yaml
# Stage 3: CoC SFT
stage: 3
description: "Chain of Causation SFT"

# モデル
stage1_2_checkpoint: "checkpoints/stage1-discrete/best.pt"  # 離散トークン学習済み
freeze_flow_head: true

# データ
coc_data_path: "data/coc_annotations.jsonl"
image_dir: "data/images/"
max_length: 2048        # 推論テキスト + アクショントークンで長くなるため

# 学習
lr: 2e-5               # LLM + Adapter
ve_lr: 1e-5             # Vision Encoder（メイン LR の半分）
batch_size: 1           # micro-batch（全パラメータ解凍 + 長いシーケンス）
grad_accum: 64          # → global batch = 64
epochs: 3               # 推論能力の獲得には複数エポック必要
warmup_ratio: 0.03
weight_decay: 0.1
max_grad_norm: 1.0

# スケジューラ
scheduler: cosine_with_warmup

# チェックポイント
save_steps: 100
logging_steps: 5
output_dir: "checkpoints/stage3"

# 精度
precision: bf16
gradient_checkpointing: true
```

### VRAM 見積もり

| コンポーネント | サイズ |
|---|---|
| 全パラメータ (582M x 12 bytes) | ~7.0 GB |
| Activation (checkpointing ON, 長いシーケンス) | ~3-4 GB |
| **合計** | **~11 GB** |

RTX 4090 (24 GB) で十分余裕あり。シーケンスが長くなるため Stage 2 (VLM) より activation が大きいが、問題ない範囲。

---

## Exit 条件

| 条件 | 確認方法 | 目標 |
|---|---|---|
| SFT loss が安定して下がる | wandb の loss curve | 初期値から有意に低下 |
| 推論テキストが画像に対しておおよそ妥当 | サンプル目視確認 (20-30件) | 8割以上で画像と整合 |
| Driving Decision 一致率 | 自動評価スクリプト | overall >60%（ランダム ~22.5% を大幅に上回る） |
| 推論品質スコア | 外部 LLM 自動評価 | 平均 >2.5 / 5.0 |
| 推論ありの軌道品質 | ADE / FDE 比較 | Stage 1 と同等以上 |
| OOM しない | VRAM モニタリング | 24 GB 以内 |

Exit 条件をクリアしたら Stage 4（RL ポストトレーニング）に進む。Stage 3 の学習済み重みが Stage 4 の初期値（SFT policy）および reference policy となる。
