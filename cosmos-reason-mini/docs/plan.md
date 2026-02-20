# Cosmos Reason Mini 実装計画

## 全体方針

**前提**: [Qwen2.5-VL Mini](../qwen-vl-mini/plan.md) で構築した汎用 VLM（Feature Alignment + Visual Instruction Tuning 済み）の重みを使用する。

Cosmos-Reason の 2 段階学習パイプライン（Physical AI SFT → Physical AI RL）を小規模再現する。**fail-fast** で進め、各 Phase に Exit 条件を設ける。

```
[Qwen2.5-VL Mini 完了] → Phase 1 (SFT データ作成) → Phase 2 (SFT 学習) → Phase 3 (RL データ作成) → Phase 4 (RL 学習)
```

完了後、学習済み重みを MiniPamayo の Stage 0 に引き継ぐ。

---

## Phase 1: SFT データ作成

**Mini 検証: ✅ 完了** — 50 フレーム → 201 QA → LLaVA 形式

**目標**: ~2.3M QA ペア（Cosmos-Reason1 の ~3.85M に対して ~1.7 倍差）。必要ディスク: ~250 GB。

### 1.1 Tier 1: 画像ベースの既存 QA（API コスト 0）

LLaVA 形式への変換のみで即利用可能。最優先で取得する。

- [ ] **NuScenes-QA** (~460K QA, ~100 MB annotation)
  - nuScenes keyframes 画像が必要（§1.1a で共通 DL）
  - テンプレートベースの空間推論 QA（DINOv2 の強みと合致）
  - LLaVA 形式に変換
- [ ] **DriveLM-nuScenes** (~300K QA, ~200 MB annotation)
  - nuScenes keyframes 画像を共有
  - フロントカメラ画像のみフィルタ（6カメラ → 1カメラ）
  - Graph VQA を LLaVA 形式に変換
- [ ] **Robo2VLM-1** (~684K QA → 200K 使用, ~30 GB)
  - 画像同梱の MCQ VQA。CC-BY-4.0
  - ロボット操作の空間推論・操作理解
  - 200K をランダムサンプリング → LLaVA 形式に変換
- [ ] **CLEVR** (~865K QA → 200K 使用, ~5 GB)
  - 合成画像の空間推論。CC-BY-4.0
  - 属性識別、カウント、比較、空間関係、論理操作
  - 200K をランダムサンプリング
- [ ] **PhysBench** (~10K QA, ~3.5 GB images)
  - 物理推論ベンチマーク（力学、流体、安定性等）
  - 画像エントリのみ抽出（動画エントリは除外）
- [ ] **CODA-LM** (~42K QA, ~99 MB annotation)
  - コーナーケース特化。CODA 画像は別途 DL
- [ ] **nuScenes keyframes** (~35 GB, 共通画像ベース)
  - NuScenes-QA, DriveLM, Cosmos-Reason1 AV で共有

**Tier 1 小計: ~1.2M QA, ~75 GB**

### 1.2 Tier 2: 動画→フレーム抽出 + 既存アノテーション

Cosmos-Reason1 SFT Dataset の公開アノテーション + フレーム抽出。

- [ ] **Cosmos-Reason1 SFT Dataset** (~84 GB) を HuggingFace から DL
  - `nvidia/Cosmos-Reason1-SFT-Dataset` (CC-BY-4.0)
  - 以下のサブセットを抽出:
    - [ ] BridgeData V2 分 (~258K → 200K 使用): フレーム抽出
    - [ ] RoboVQA 分 (~1.14M → 300K 使用): フレーム抽出（動画同梱）
    - [ ] HoloAssist 分 (~273K → 200K 使用): フレーム抽出（動画別途 DL）
    - [ ] AgiBot 分 (~38.9K → 全量使用): フレーム抽出
    - [ ] AV 分 (~12.4K → 全量使用): フレーム抽出
  - **動画→画像変換**: 各アノテーションの timestamp からキーフレームを抽出
  - **時間依存 QA のフィルタ**: 「次に何が起こるか」等の temporal QA は除外
- [ ] **SUTD-TrafficQA** (~62.5K MCQ, ~20.5 GB)
  - 交通シーン MCQ。SFT 用 MCQ+CoT + RL 用 MCQ
  - 動画からキーフレーム抽出
- [ ] **Reason2Drive** (~633K → 300K 使用, ~50 GB 推定)
  - nuScenes + Waymo + ONCE 横断
  - チェーン推論 QA（Perception / Prediction / Reasoning）
  - nuScenes 部分は画像共有、他は動画からフレーム抽出

**Tier 2 小計: ~1.1M QA, ~155 GB**

### 1.3 Tier 3: GPT-4o による追加 QA 生成（補完）

Tier 1+2 でカバーできない部分を API で補完:

- [x] nuScenes フレーム選定 + キャプション + QA 生成（Mini 検証済み）
- [ ] **Something-Something V2** (~220K clips, ~19.4 GB) からフレーム抽出 + QA 生成
  - 物体操作の因果関係に特化（~50K QA 目標）
- [ ] **BDD-X** (~7K 動画, annotation ~717 KB) から行動 + 理由説明 QA（~26K）
- [ ] nuScenes 画像から追加空間推論 QA（~50K QA 目標）
- [ ] **CoT は短く簡潔に**（SmolVLM 知見: 小型モデルでは過剰な CoT が有害）

**Tier 3 小計: ~126K QA, ~20 GB + API コスト**

### 1.4 データクリーニングと統合

- [ ] 各データセットを LLaVA 形式に統一変換
- [ ] 「画像の説明によると」「キャプションから」等の不要な参照を除去
- [ ] 品質の低い QA をフィルタリング
- [ ] 動画 QA → 画像 QA 変換時の temporal QA 除外
- [ ] **DINOv2 の強みを活かす QA 比率の確認**: 空間関係 QA を重視
- [ ] **ドメイン比率の確認**: 運転 ~48% / ロボット ~32% / 人間 ~11% / 物理 ~9%

### 1.5 データセット統計の目標

| ドメイン | QA 数 | 比率 | 主要データソース |
|---|---|---|---|
| 自動運転 | ~1.1M | ~48% | NuScenes-QA + DriveLM + Reason2Drive + CODA-LM + SUTD-TrafficQA |
| ロボット操作 | ~738K | ~32% | Robo2VLM-1 + Cosmos-Reason1 (BridgeData + RoboVQA + AgiBot) |
| 人間活動 | ~250K | ~11% | Cosmos-Reason1 (HoloAssist) + SSv2 |
| 一般物理推論 | ~210K | ~9% | CLEVR + PhysBench |
| **合計** | **~2.3M** | **100%** | **Cosmos-Reason1 の ~3.85M に対して ~1.7 倍差** |

### 1.6 ディスク容量見積もり

| データ | 容量 | 取得元 |
|---|---|---|
| nuScenes keyframes (共通) | ~35 GB | nuscenes.org |
| NuScenes-QA + DriveLM annotations | ~300 MB | GitHub / HuggingFace |
| Robo2VLM-1 (200K subset) | ~30 GB | HuggingFace |
| CLEVR (200K subset) | ~5 GB | Stanford |
| PhysBench (images) | ~3.5 GB | HuggingFace |
| CODA-LM | ~99 MB | HuggingFace |
| Cosmos-Reason1 SFT Dataset | ~84 GB | HuggingFace |
| SUTD-TrafficQA | ~20.5 GB | Zenodo |
| Reason2Drive (300K subset) | ~50 GB | GitHub |
| Something-Something V2 | ~19.4 GB | 公式サイト |
| BDD-X | ~717 KB | GitHub |
| **合計** | **~250 GB** | |

### 1.7 Exit 条件

- [ ] QA データが合計 ~2M 以上に達している
- [ ] 全 4 ドメインのデータが含まれている
- [ ] Cosmos-Reason1 との差が 2 倍以内
- [ ] サンプリングした QA の品質が妥当
- [ ] DataLoader でバッチが正しく取り出せる

---

## Phase 2: Physical AI SFT

**Mini 検証: ✅ 完了** — 201 QA で SFT、loss 1.32 → 0.76、MCQ ベースライン 64.4%

### 2.1 学習設定

- [x] **Qwen2.5-VL Mini の学習済み重みをロード**
- [ ] Vision Encoder + Adapter + LLM すべて trainable
- [ ] 入力: [visual_tokens] + [質問テキスト]
- [ ] 出力: [回答テキスト]（理解 QA）or [CoT 推論トレース + 回答]（推論 QA）
- [ ] **SFT データに MCQ + CoT を含める** (`convert_mcq_to_sft.py` で変換・統合)
  - Cosmos-Reason1 では SFT 段階で MCQ フォーマット (`<think>...<answer>`) を学習
  - MCQ + CoT を自由形式 QA と混合して SFT → RL でのフォーマット学習負荷を軽減
- [ ] Loss: cross-entropy（next-token prediction）
- [ ] ハイパーパラメータ:
  - 学習率: 2e-5 → 2e-6（cosine annealing）— VLM 構築済みなので小さめ
  - micro-batch=1, grad_accum=16
  - bf16 mixed precision
  - gradient checkpointing ON
  - AdamW (β1=0.9, β2=0.95), weight_decay=0.01
- [ ] wandb ロギング

### 2.2 学習実行

- [ ] 理解 QA + 推論 QA を混合して学習
- [ ] ~1,000〜3,000 イテレーション（データ規模に応じて調整）
- [ ] loss カーブの監視
- [ ] **チェックポイントを 25 ステップごとに保存**（SmolVLM 知見: 訓練を長く続けると一部指標が低下。最適点は途中にある）
- [ ] **訓練不安定時の対応**: 全パラメータ解凍で発散したら LoRA（rank=256）を検討（Idefics2, Imp 知見）

### 2.3 評価

- [ ] 学習データとは別の画像で推論を実行
- [ ] 生成テキストの定性的評価（運転シーンを正しく記述できているか）
- [ ] 推論 QA に対する回答の妥当性確認
- [ ] Qwen2.5-VL Mini（SFT 前）との比較: 運転ドメインの理解が改善しているか

### 2.4 Exit 条件

- [ ] SFT loss が安定して下がる
- [ ] 画像入力に対して運転シーンの記述が生成できる
- [ ] 推論 QA に対して（大雑把にでも）妥当な回答が返る

---

## Phase 3: RL データ作成

**Mini 検証: ✅ 完了 (v2)** — 101 推論 QA → 101 MCQ (正解シャッフル済み)

### 3.1 MCQ 変換

SFT データの QA を MCQ（多肢選択問題）形式に変換:

- [x] **MCQ 生成プロンプト**の設計
  - 正解選択肢 + 3 つの妥当だが不正解な選択肢
  - 選択肢は具体的で、ビジュアル推論なしでは回答できないようにする
- [x] **コード側で正解位置シャッフル** (GPT の JSON 例バイアス回避)
- [ ] **SUTD-TrafficQA** (~62.5K MCQ) を RL 用に変換
- [ ] SFT の推論 QA サブセットを MCQ に変換（GPT-4o API）
- [ ] 目標: ~5,000〜10,000 MCQ

### 3.2 回答形式の統一

- [ ] 回答タグ形式の定義: `<answer>A</answer>`
- [ ] 正規表現パターンマッチングによる自動採点の実装
- [ ] 検証: 正解判定が正しく動作するか

### 3.3 Exit 条件

- [ ] MCQ データが目標数に達している
- [ ] 自動採点が正しく動作する
- [ ] SFT モデルの MCQ 正解率をベースラインとして記録

---

## Phase 4: Physical AI RL

**Mini 検証: ✅ 完了 (v3: リーク修正)** — GRPO 20 iter、MCQ 正解率 38.3% → 86.7% (+48.4%)、MCQ+CoT SFT 併用時: 92.5% → 93.3% ※eval split (40 MCQ, 20 images) でリークなし評価、GRPO も train MCQ のみで学習

### 4.1 GRPO 実装

- [x] GRPO の基本実装
  - 各質問に対して K=4〜8 の応答をサンプリング
  - グループ内で報酬を正規化: `A_i = (R(o_i) - mean(G)) / std(G)`
  - KL 正則化（SFT モデルを reference policy として frozen で保持）
  - **Multi-step 更新 (μ=4)**: 同じロールアウトデータに μ 回の最適化ステップ
  - ratio は old policy（ロールアウト時）に対して計算 → μ > 1 で clipping が有効
- [ ] 報酬関数: MCQ 正解 → 1、不正解 → 0
- [ ] 回答抽出: `<answer>...</answer>` タグから正規表現で抽出

### 4.2 学習設定

- [ ] SFT モデルを初期値 + reference policy として使用
- [ ] LLM のみ trainable（Vision Encoder + Adapter は frozen）
- [ ] ハイパーパラメータ:
  - 学習率: 4e-6
  - KL 係数: 0.005
  - ロールアウト数: K=4〜8 / 質問
  - バッチサイズ: 4〜8 質問
  - 最大トークン長: 2,048
  - イテレーション: ~100〜300
- [ ] MCQ 選択肢の動的シャッフル（汎化促進、Cosmos-Reason と同様）

### 4.3 学習実行

- [ ] ロールアウト → 報酬計算 → ポリシー更新のループ
- [ ] MCQ 正解率の推移を監視
- [ ] KL の推移を監視（発散していないか）

### 4.4 評価

- [ ] RL 後の MCQ 正解率 vs SFT ベースライン
- [ ] 推論テキストの品質比較（RL 前後で定性的に改善しているか）
- [ ] KL が適切な範囲に収まっているか

### 4.5 Exit 条件

- [ ] MCQ 正解率が SFT ベースラインより改善
- [ ] KL が発散していない
- [ ] 推論テキストの品質が維持 or 改善

---

## MiniPamayo への引き継ぎ

### 重み引き継ぎ

- [ ] Cosmos Reason Mini の最終重み（Vision Encoder + Adapter + LLM）を保存
- [ ] MiniPamayo Stage 0 の初期値として使用可能な形式で export
- [ ] 引き継ぎ時の確認: MiniPamayo で重みをロードし、forward pass が通ること

### 知見の引き継ぎ

- [ ] Adapter の最適な方式（MLP vs Cross-Attention）の結論
- [ ] 学習率・バッチサイズ等の最適設定の記録
- [ ] データ品質に関する知見（どんなプロンプトが良い QA を生成するか）

---

## 実装優先順位まとめ

```
[Qwen2.5-VL Mini 完了]
    ↓
Phase 1 (SFT データ作成: auto-labeling)
    ↓
Phase 2 (Physical AI SFT)
    ↓
Phase 3 (RL データ作成: MCQ 変換)
    ↓
Phase 4 (Physical AI RL)
    ↓
→ MiniPamayo Stage 0 に引き継ぎ
```

**最短経路**: Phase 1 → 2 → MiniPamayo 引き継ぎ（RL スキップ）
**推奨経路**: Phase 1 → 2 → 3 → 4 → MiniPamayo 引き継ぎ（全 Stage 実施）

---

## リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| 教師 VLM の API コスト | データ作成費用 | 少量で始めて品質を確認後にスケール |
| auto-labeling の品質 | SFT の上限 | プロンプトを反復改善、生成データをフィルタリング |
| Qwen2.5-0.5B の表現力の限界 | 推論品質の限界 | 目的は技術理解なので許容、CoT を短くする |
| RL の報酬が sparse すぎる | 学習が進まない | MCQ を易しめにする、部分正解に中間報酬を付与 |
| VRAM 不足（RL 時） | RL 実行不可 | K を減らす（K=2〜4）、Vision Encoder を frozen に |
| 運転ドメインデータの多様性不足 | 汎化しない | 複数データセットを併用、CARLA でデータ補完 |

---

## 小規模 VLM 論文からの注意事項

詳細は [小規模 VLM 構築の知見まとめ](../small-vlm-research.md) を参照。実装時に特に注意すべき点:

### データ作成（Phase 1）

1. **CoT 推論トレースを短く保つ**: 0.5B モデルでは長い推論チェーンが容量を圧迫する。CoT は 2〜3 ステップ程度の簡潔なものに（SmolVLM: CoT は全体の 0.02〜0.05% が最適）
2. **データ品質を最優先**: 少量（~1,000）から始めて品質確認後にスケール。低品質データの大量投入は逆効果（TinyLLaVA: 1.1B でデータ増加が逆効果になるケースあり）
3. **空間関係 QA を重視**: DINOv2 は細粒度の空間情報に優れるため、この強みを活かす QA を多めに（COMM 論文）
4. **テキストのみの QA を混入しない**: LLM-SFT データの再利用は性能低下を招く（SmolVLM: 画像 -6.5%）

### SFT 学習（Phase 2）

5. **チェックポイントを頻繁に保存**: 25 ステップごと。最適点は訓練終了時とは限らない（SmolVLM）
6. **2 エポック以内に収める**: 3 エポック以上で過学習リスク（Imp）
7. **訓練発散時は LoRA を検討**: 全パラメータ解凍で不安定な場合、LoRA rank=256 が有効（Imp, Idefics2）

### RL 学習（Phase 4）

8. **MCQ の難易度調整**: 0.5B モデルが SFT 後に 25% 以上の正解率を出せるレベルに（ランダム正解率 25% を大幅に超えないと RL の学習信号が不足）
