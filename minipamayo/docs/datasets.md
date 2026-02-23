# MiniPamayo データセット調査

## 1. 問題: Alpamayo 0.5B との桁違いのデータ差

| 観点 | Alpamayo 0.5B | MiniPamayo（現行計画） | 差 |
|---|---|---|---|
| 学習データ量 | **80,000 時間** | ~40 時間（nuScenes 5.5h + comma2k19 33h） | **~2,000 倍** |
| 地理的多様性 | 25 か国 2,500 都市 | カリフォルニア 280 号線 + シンガポール/ボストン | 圧倒的に劣る |
| 車種多様性 | 不明（NVIDIA 自社フリート） | 限定的 | — |
| カメラ | 7 台サラウンド | 1 台フロント | 制約として許容 |
| 制御信号 | CAN bus（加速度・曲率等） | ego pose → 逆算 / CAN bus | — |

**0.5B のような小規模モデルでも、データが少なすぎると汎化性能が壊滅的に低下する**。技術理解が目的とはいえ、数十時間のデータでは overfitting して意味のある学習が困難になる可能性が高い。

---

## 2. 公開データセット一覧

行動予測（軌道予測・制御予測）に使える**制御信号 or 軌道情報付き**のデータセットを網羅的に調査した。

### 2.1 Tier S: 制御信号付き大規模データ（最優先）

#### commaSteeringControl
- **データ量**: **~12,500 時間**（openpilot engage 中のデータ）
- **アノテーション**: ステアリング制御値
- **車種**: 数百車種、10 以上のブランド
- **ライセンス**: 公開（HuggingFace）
- **URL**: https://huggingface.co/datasets/commaai/commaSteeringControl
- **MiniPamayo 適用性**: **高い**。世界最大の制御データセット。ただし映像データが含まれるかは要確認（制御値のみの可能性）。映像がなければ行動予測モデルの学習には直接使えないが、制御分布の事前知識やデコーダの事前学習には使える

#### commaCarSegments
- **データ量**: **145,595 セグメント（~2,500 時間）**
- **アノテーション**: 走行セグメント（映像 + CAN データ）
- **車種**: 3,677 ユーザー・223 車種
- **ライセンス**: 公開（HuggingFace）
- **URL**: https://huggingface.co/datasets/commaai/commaCarSegments
- **MiniPamayo 適用性**: **非常に高い**。映像 + 制御信号のペアが大量にある。comma2k19 の 33 時間から ~75 倍にスケールアップ可能

#### commaVQ
- **データ量**: 学習済みモデルは **3,000,000 分（50,000 時間）** のデータで学習。公開されているのは **100,000 分（~1,667 時間）** のトークン化済み動画
- **アノテーション**: VQ-VAE で各フレームを 128 トークンに圧縮済み
- **ライセンス**: MIT License
- **URL**: https://github.com/commaai/commavq
- **MiniPamayo 適用性**: **中〜高**。トークン化済みのため生画像ではないが、世界モデル的な事前学習に利用可能。制御信号との対応は不明

### 2.2 Tier A: 軌道・制御情報付き中規模データ

#### nuScenes Full (v1.0-trainval)
- **データ量**: 1,000 シーン（各 20 秒）= **~5.5 時間**。1.4M カメラ画像、34K キーフレーム（train 850 + val 150 シーン）
- **アノテーション**: 6 カメラ 360 度、ego pose（2Hz キーフレーム）、CAN bus（速度・加速度・操舵角）、23 クラス 3D bbox
- **ライセンス**: CC BY-NC-SA 4.0
- **ストレージ**: ~300 GB
- **URL**: https://www.nuscenes.org/
- **MiniPamayo 適用性**: **非常に高い**（既に計画に含まれている）。ego pose から制御入力 (a, κ) を逆算可能。3D bbox で衝突判定にも使える。ただしデータ量は小さい

#### comma2k19
- **データ量**: 2,019 セグメント（各 1 分）= **33 時間**
- **アノテーション**: CAN bus（steering_angle, wheel_speeds）、GPS、IMU、フロントカメラ動画
- **ライセンス**: MIT License
- **ストレージ**: ~100 GB（10 GB チャンクで分割）
- **URL**: https://github.com/commaai/comma2k19
- **MiniPamayo 適用性**: **高い**（既に計画に含まれている）。フロントカメラ + ステアリング角が直接使える。MIT ライセンスで制約なし。ただし高速道路通勤データに偏っている

#### Waymo Open Dataset — End-to-End (E2E)
- **データ量**: 5,000 セグメント（各 ~8 秒）= **~12 時間**
- **アノテーション**: 8 カメラ 360 度 + ルーティング情報 + ego 状態（過去 12 秒の履歴 → 将来 5 秒の waypoint 予測）
- **ライセンス**: Waymo Dataset License（非商用研究のみ）
- **ストレージ**: 大（TB 単位、全データの場合）
- **URL**: https://waymo.com/open/
- **MiniPamayo 適用性**: **高い**。E2E データセットは MiniPamayo の Stage 0〜1 に直接適合（waypoint 予測タスク）。ただし予測ホライズンが 5 秒（MiniPamayo は 6.4 秒）で若干短い。ダウンロードサイズが大きい

#### Waymo Open Dataset — Motion Forecasting
- **データ量**: **103,354 シナリオ**（各 9.1 秒、10Hz）
- **アノテーション**: 周囲エージェントの 2D 位置・速度・向き（10Hz）。過去 1 秒の履歴 → 将来 8 秒の軌道予測
- **ライセンス**: Waymo Dataset License（非商用研究のみ）
- **ストレージ**: ~200 GB
- **URL**: https://waymo.com/open/
- **MiniPamayo 適用性**: **中**。103K シナリオは大規模だが、画像データではなく agent 状態のみ。軌道デコーダの事前学習や評価には使えるが、Vision → Action の end-to-end 学習には不向き

#### Argoverse 2 — Motion Forecasting
- **データ量**: **250,000 シナリオ**（各 11 秒、10Hz）
- **アノテーション**: 2D 位置・速度・向き（10Hz）。過去 5 秒の履歴 → 将来 6 秒の予測。ローカルベクトルマップ付き
- **ライセンス**: CC BY-NC-SA 4.0
- **ストレージ**: 58 GB（Motion のみ）、Sensor データは ~1 TB
- **URL**: https://www.argoverse.org/av2.html
- **MiniPamayo 適用性**: **中**。250K シナリオは最大級の motion forecasting データだが、画像なしの場合は Waymo Motion と同様。Sensor データ（カメラ画像付き）なら有用だが 1 TB と大きい

#### A2D2 (Audi Autonomous Driving Dataset)
- **データ量**: 41,277 アノテーション付きフレーム + **392,556 連続フレーム**（非アノテーション）
- **アノテーション**: セマンティックセグメンテーション、3D bbox、6 カメラ、5 LiDAR、**CAN bus データ**
- **ライセンス**: CC BY-ND 4.0（商用利用可、ただし改変不可）
- **ストレージ**: ~2.3 TB
- **URL**: https://www.a2d2.audi/
- **MiniPamayo 適用性**: **中〜高**。CAN bus データ + フロントカメラが使える。392K 連続フレームは Stage 0 の学習データとして有用。ただし CC BY-ND のためデータ改変に制限あり

#### Honda HDD (Honda Driving Dataset)
- **データ量**: **104 時間** のサンフランシスコ湾岸エリア実走行データ
- **アノテーション**: 高レベル運転行動ラベル、GPS、IMU、CAN bus データ
- **ライセンス**: 大学研究者限定（データ共有契約が必要）
- **URL**: https://usa.honda-ri.com/hdd
- **MiniPamayo 適用性**: **中**。104 時間は中規模。CAN bus データが含まれるので制御信号活用可能。ただし大学限定アクセス

#### Lyft Level 5
- **データ量**: 55,000 以上の 3D アノテーションフレーム、**1,000 時間以上**の Perception + Motion データ
- **アノテーション**: 3D bbox、HD マップ、自車軌道
- **ライセンス**: 公開研究用
- **URL**: https://level5.lyft.com/dataset/
- **MiniPamayo 適用性**: **中**。nuScenes 形式に近い。1,000 時間は大規模だが、Woven Planet 買収後の継続的サポートは不明

#### KITTI
- **データ量**: Odometry: 22 シーケンス（~40 分）。7,481 訓練画像（3D bbox 付き）
- **アノテーション**: ステレオ画像、LiDAR、GPS/IMU、3D bbox
- **ライセンス**: CC BY-NC-SA 3.0
- **ストレージ**: ~180 GB
- **URL**: https://www.cvlibs.net/datasets/kitti/
- **MiniPamayo 適用性**: **低〜中**。古典的だが規模は現在の基準では小さい。GPS/IMU で自車運動推定は可能だが制御信号は限定的

### 2.3 Tier B: 大規模だが制御信号なし（事前学習・補完用）

#### OpenDV-YouTube
- **データ量**: **1,700 時間以上**、244 都市、40 か国
- **アノテーション**: VLM 生成のテキスト記述。制御信号なし
- **ライセンス**: YouTube 利用規約に準拠（再配布不可）
- **ストレージ**: フル ~3 TB（1080P）、mini 版 28 時間 ~44 GB
- **URL**: https://huggingface.co/datasets/OpenDriveLab/OpenDV-YouTube-Language
- **MiniPamayo 適用性**: **低〜中**（直接の行動予測には使えない）。制御信号がないため Stage 0〜1 の教師あり学習には不向き。ただし Vision Encoder の事前学習や、Cosmos Reason Mini の SFT データとしては有用。mini 版（28 時間）で試すのが現実的

#### ONCE Dataset
- **データ量**: **100 万 LiDAR シーン**、700 万カメラ画像、144 時間の走行
- **アノテーション**: 3D 物体検出。40 ビーム LiDAR + 7 カメラ
- **ライセンス**: 公開
- **URL**: https://once-for-auto-driving.github.io/
- **MiniPamayo 適用性**: **低**。物体検出特化で制御信号や waypoint 情報はなし

### 2.4 Tier C: シミュレータ・合成データ

#### CARLA シミュレータ
- **データ量**: **無制限**に生成可能
- **アノテーション**: 完全なセンサーデータ（カメラ・LiDAR・深度等）+ 制御信号 + 3D bbox + セグメンテーション。天候・照明・交通量を自由に変更
- **ライセンス**: MIT License
- **URL**: https://carla.org/
- **MiniPamayo 適用性**: **中〜高**。データ量の問題を根本的に解決できる可能性がある。ただし sim-to-real ギャップが課題。CARLA2Real（CARLAの出力をフォトリアリスティックに変換、13 FPS）で緩和可能。DriveLM-CARLA は CARLA の privileged 情報から QA ペアを自動生成する手法で、MiniPamayo の CoC データ作成にも使える

#### MetaDrive
- **データ量**: **無制限**に生成可能（300 FPS の軽量シミュレータ）
- **アノテーション**: マルチモーダル観測（LiDAR, RGB/深度カメラ, 鳥瞰図, スカラーデータ）
- **特徴**: Waymo / nuPlan / Lyft の実データをインポートして仮想環境を再構築可能
- **ライセンス**: Apache 2.0
- **URL**: https://github.com/metadriverse/metadrive
- **MiniPamayo 適用性**: **中**。RL 環境として Stage 4 に直接利用可能。ただし画像品質はCARLAより低い

#### RiskBench
- **データ量**: CARLA ベースの大規模リスクシナリオ
- **アノテーション**: リスク検出・位置特定、リスク予測、意思決定支援
- **ライセンス**: 公開
- **URL**: https://github.com/HCIS-Lab/RiskBench
- **MiniPamayo 適用性**: **低〜中**。コーナーケースの評価・学習データとして

### 2.5 NVIDIA 公式データ（Alpamayo 関連）

#### PhysicalAI-Autonomous-Vehicles Dataset
- **データ量**: **1,727 時間**、306,152 クリップ（各 20 秒）、25 か国 2,500 都市
- **アノテーション**: マルチカメラ 7 台（1080p/30fps）+ LiDAR + レーダー最大 10 基
- **ライセンス**: NVIDIA AV Dataset License（12 か月有効、**NVIDIA 技術を使用した AV 開発限定、再配布不可、派生物作成不可**）
- **ストレージ**: ~100 TB
- **URL**: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
- **MiniPamayo 適用性**: **低**（ライセンス制約）。派生物作成不可のため学習データとしては使えない。評価参考や設計のリファレンスとして

#### NAVSIM (nuPlan ベースの E2E 評価ベンチマーク)
- **データ量**: navtrain 14 GB（ログ）+ 445 GB（センサー）。navhard_two_stage 31 GB
- **アノテーション**: nuPlan のダウンサンプル版（2Hz）。E2E 運転評価用
- **ライセンス**: nuPlan ライセンスに準拠
- **URL**: https://github.com/autonomousvision/navsim
- **MiniPamayo 適用性**: **中〜高**。MiniPamayo の評価ベンチマークとして有用。navtrain は学習データとしても使える可能性あり

#### nuPlan
- **データ量**: **1,300 時間以上**、15,000 以上のログ、4 都市
- **アノテーション**: 自車軌道、HD マップ、周囲エージェント状態
- **ライセンス**: 学術利用は無料、商用利用は別途ライセンス
- **ストレージ**: 大（TB 単位）
- **URL**: https://www.nuplan.org/
- **MiniPamayo 適用性**: **中〜高**。1,300 時間は大規模。カメラ画像 + 自車軌道があれば Stage 0 に直接利用可能。ただしダウンロード・処理コストが大きい

---

## 3. データ規模比較

### 3.1 制御信号付きデータの規模感

```
Alpamayo 0.5B:        80,000 時間  ████████████████████████████████████████████████████████████████████
commaSteeringControl:  12,500 時間  ████████████  (映像なしの可能性)
commaCarSegments:       2,500 時間  ██▌
commaVQ:                1,667 時間  █▋  (トークン化済み)
nuPlan:                 1,300 時間  █▎
Lyft Level 5:           1,000 時間  █
Honda HDD:                104 時間  ▏
comma2k19:                 33 時間  ▏
Waymo E2E:                 12 時間  ▏
nuScenes:                 5.5 時間  ▏

MiniPamayo 現行計画:       ~40 時間  ▏
```

### 3.2 現実的に確保可能なデータ量

| データセット | 時間 | 映像 | 制御信号 | ストレージ | 入手難易度 |
|---|---|---|---|---|---|
| **commaCarSegments** | ~2,500h | あり | あり | 要確認 | 低（HuggingFace） |
| **nuPlan** (navtrain) | ~1,300h | あり | あり（ego軌道） | ~460 GB | 中（要登録） |
| **comma2k19** | 33h | あり | あり | ~100 GB | 低（GitHub） |
| **nuScenes Full** | 5.5h | あり | あり | ~300 GB | 低（要登録） |
| **Waymo E2E** | ~12h | あり | あり | 大 | 中（要登録） |
| **A2D2** | 不明（392K frames） | あり | あり（CAN） | ~2.3 TB | 低 |
| **CARLA 生成** | 無制限 | あり | あり | 自由 | 中（セットアップ） |
| **合計（CARLA除く）** | **~3,850h** | | | | |

---

## 4. 推奨データ戦略

### 4.1 段階的スケールアップ

MiniPamayo は fail-fast アプローチを取っているため、データも段階的にスケールアップする。

**Phase A: パイプライン検証（~40 時間）** — 現行計画
- nuScenes Mini (10 シーン) → nuScenes Full (850 train シーン)
- comma2k19 数セグメント → 全セグメント
- **目的**: Stage 0 の勾配伝播・loss 低下を確認

**Phase B: 中規模学習（~500-2,500 時間）** — データ追加
- **commaCarSegments** のサブセット（~500-2,500 時間）を追加
  - comma2k19 と同じデータ形式（comma.ai のデータパイプライン）
  - HuggingFace から直接ダウンロード可能
  - 多様な車種・ユーザーでデータの多様性が大幅に向上
- **目的**: overfitting を抑制し、汎化性能を改善

**Phase C: 大規模学習（~1,300-3,000+ 時間）** — さらなるスケール
- **nuPlan** (navtrain) を追加（~1,300 時間）
- **CARLA** でロングテール・コーナーケースを補完
- **目的**: Alpamayo 0.5B との差を ~20-60 倍に縮小（80,000h vs 1,300-3,800h）

### 4.2 データフォーマット統一

異なるデータセットを統一的に扱うため、以下の共通フォーマットに変換する:

```python
@dataclass
class DrivingSample:
    image: torch.Tensor          # (3, 224, 224) — フロントカメラ RGB
    ego_state: torch.Tensor      # (D,) — 速度, yaw rate 等
    gt_controls: torch.Tensor    # (6, 2) — (a, κ) 制御入力列 @ dt=0.5s, 3秒
    # または
    gt_trajectory: torch.Tensor  # (6, 2) — (x, y) waypoints @ dt=0.5s, 3秒
    metadata: dict               # データセット名、シーンID 等
```

各データセットからの変換:
- **nuScenes**: ego pose (2Hz keyframe) → inverse dynamics (dt=0.5s) → (a, κ)
- **comma2k19 / commaCarSegments**: CAN bus (steering_angle, speed) → (a, κ) 変換
- **nuPlan**: ego trajectory → 2Hz リサンプル → inverse dynamics (dt=0.5s) → (a, κ)
- **CARLA**: 直接 (a, κ) を取得可能（privileged 情報）

### 4.3 データ量の目安

小規模 VLM (~0.5B) の学習に必要なデータ量の目安:

| 参考モデル | パラメータ数 | 学習データ量 | 備考 |
|---|---|---|---|
| Alpamayo 0.5B | ~0.5B | 80,000 時間 | NVIDIA フルスケール |
| Alpamayo 0.5B (推定最小) | ~0.5B | ~1,000-5,000 時間 | 論文に明記なし。汎化に必要な最小規模の推定 |
| comma.ai openpilot | ~数十M | ~12,500+ 時間 | 制御モデル特化 |
| MiniPamayo（Phase B 目標） | ~730M | **~500-2,500 時間** | commaCarSegments 活用 |
| MiniPamayo（Phase C 目標） | ~730M | **~1,300-3,800 時間** | nuPlan + CARLA 追加 |

---

## 5. データ以外のスケールギャップ対策

データ量だけでなく、少ないデータで効率的に学習する手法も検討する。

### 5.1 知識蒸留（Knowledge Distillation）

- **DiMA (CVPR 2025)**: マルチモーダル LLM から視覚ベースのプランナーへ知識蒸留。L2 軌道エラー 37% 削減、衝突率 80% 削減
- **CoVLA (WACV 2025)**: 大規模 VLM を使って自動キャプション・QA 生成。10,000 動画クリップ・80 時間超のデータを自動構築
- **MiniPamayo への示唆**: Cosmos Reason Mini（VLM）の出力を使って、行動予測の学習データを enrichment できる（例: VLM が生成したシーン記述を追加入力として使う）

### 5.2 データ拡張

- **幾何学的拡張**: 左右反転（ステアリング角も反転）、クロップ、カラーjitter
- **時間的拡張**: 異なる時間オフセットでのサンプリング、速度のスケーリング
- **Diffusion ベースの合成**: GenDDS（Stable Diffusion XL で多様な運転シナリオを生成）、LTDA-Drive（LLM ガイドのロングテールデータ拡張）
- **実効データ量の増加**: 左右反転だけでも実質 2 倍。時間オフセットでさらに数倍

### 5.3 事前学習の活用

- **DINOv2 の事前学習**: 既に ImageNet で事前学習済みの視覚特徴を使うことで、少ないデータでも視覚理解が効く
- **Cosmos Reason Mini の事前学習**: 運転ドメインの VLM SFT/RL で獲得した重みが初期値。視覚理解の基盤が既にある
- **MiniPamayo の学習は「行動予測ヘッド + fine-tune」**: ゼロから学習するわけではない

### 5.4 世界モデル事前学習

- **commaVQ**: 50,000 時間分のトークン化済み動画で世界モデル（次フレーム予測）を事前学習。この表現を MiniPamayo の Vision Encoder に転移できる可能性
- **ViDAR (CVPR 2024)**: 過去の視覚入力から未来の点群を予測。3D 幾何と時間ダイナミクスを同時にモデル化

---

## 6. まとめ: データ確保の現実的プラン

### 最低限（Phase A: ~40 時間）
- nuScenes Full + comma2k19
- パイプライン検証には十分。汎化は期待できない

### 推奨（Phase B: ~500-2,500 時間）
- 上記 + **commaCarSegments**
- Alpamayo との差を ~2,000 倍 → ~30-160 倍に縮小
- 入手容易（HuggingFace）でデータパイプラインも comma2k19 と互換

### 理想（Phase C: ~3,000+ 時間）
- 上記 + **nuPlan** + **CARLA 生成**
- Alpamayo との差を ~20 倍程度に
- ストレージと処理コストが大きくなるが、汎化性能は大幅改善

### データ確保ロードマップ

```
Phase A (パイプライン検証)
├── nuScenes Mini (~4 GB) ← 既に使用中
├── nuScenes Full (~300 GB) ← ダウンロード待ち
└── comma2k19 (~100 GB)
    ↓
Phase B (中規模学習)
├── commaCarSegments (容量要確認) ← HuggingFace DL
└── データフォーマット統一パイプライン構築
    ↓
Phase C (大規模学習) — 任意
├── nuPlan navtrain (~460 GB) ← 要登録
├── CARLA 生成 ← シミュレータセットアップ
└── Waymo E2E ← 要登録
```
