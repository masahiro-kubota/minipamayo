# MiniPamayo データセット要件

## 1. 必要なデータの形式

MiniPamayo の学習には、最低限以下のペアデータが必要:

| データ | 形式 | 用途 |
|---|---|---|
| **フロントカメラ画像** | RGB（リサイズ→224×224） | Vision Encoder 入力 |
| **アクションラベル** | 連続値 `[steer, throttle]` or waypoints `[(x,y),...]` | Action Head の教師信号 |
| (任意) テキスト指示 | 自然言語文字列 | LLM への条件付け |
| (任意) 車両状態 | speed, yaw_rate 等 | 補助入力 |

**キーポイント**: 画像とアクション（制御値 or 軌跡）が時刻同期されたペアであることが必須。

---

## 2. 推奨データセット

### 2.1 最優先: nuScenes（推奨）

**自動運転向け。単一カメラ + ego trajectory で MiniPamayo に最適。**

| 項目 | 内容 |
|---|---|
| URL | https://www.nuscenes.org/ |
| サイズ | Full: ~300 GB / Mini (テスト用): ~4 GB |
| フレーム数 | Full: ~400K frames / Mini: ~3.5K frames |
| カメラ | 6 台（**FRONT のみ使用**で MiniPamayo に適合） |
| アクション | ego pose（位置・姿勢）→ waypoints に変換可 |
| ライセンス | CC BY-NC-SA 4.0（研究利用可） |

**使い方**:
- CAM_FRONT の画像のみ使用
- ego pose の差分から waypoints `[(dx1,dy1),...,(dxK,dyK)]` を計算
- steer/throttle は直接提供されないが、ego pose 差分から近似可能
- `nuscenes-devkit` で簡単にアクセス可

**開始手順**:
```bash
# Mini split（テスト用、~4 GB）
pip install nuscenes-devkit
# nuScenes サイトからダウンロード（要登録）
```

**メリット**:
- 豊富な都市走行シーン、天候・時刻のバリエーション
- 研究で広く使われておりベンチマーク比較が容易
- Mini split で素早く検証可能

---

### 2.2 comma2k19 / commaai データセット

**単一フロントカメラ + CAN バスデータ。steer/throttle が直接得られる。**

| 項目 | 内容 |
|---|---|
| URL | https://github.com/commaai/comma2k19 |
| サイズ | ~100 GB（全体）/ 一部だけ使用可 |
| フレーム数 | 33 時間分の走行（~2000 セグメント、各 1 分） |
| カメラ | **1 台（フロント）** ← MiniPamayo に完全一致 |
| アクション | CAN バス: **steering_angle, speed, accel** が直接得られる |
| ライセンス | MIT |

**使い方**:
- `steering_angle` → steer, `speed` の差分 → throttle として直接利用
- 画像は 1164×874 → 224×224 にリサイズ
- waypoints が必要な場合は speed + steering_angle から近似計算

**メリット**:
- **steer/throttle が直接得られる**（変換不要で Stage 0 に最適）
- 単一カメラで MiniPamayo の設定と完全一致
- MIT ライセンスで制約が少ない

**注意点**:
- 高速道路中心（シーン多様性は nuScenes より低い）
- ダウンロードが大きいので、数セグメントだけ取得して始めるのが良い

---

### 2.3 CARLA シミュレータ（自前データ生成）

**完全に制御可能な環境で任意のデータを生成できる。**

| 項目 | 内容 |
|---|---|
| URL | https://carla.org/ |
| サイズ | 自由（生成量による） |
| カメラ | 任意の台数・位置に設定可 |
| アクション | **steer, throttle, brake** が完全に取得可能 |
| ライセンス | MIT |

**使い方**:
- CARLA サーバーを起動し、autopilot モードで走行 → データ収集
- カメラ画像 + 制御値 + waypoints を同時記録
- テキスト指示（"Turn left at the intersection"）も自前で付与可能

**メリット**:
- 完全な ground truth（steer, throttle, waypoints すべて）
- データ量を自由に増減できる
- テキスト指示付きデータも作れる（Stage 2 以降で有用）

**注意点**:
- シミュレータのセットアップが必要（GPU リソースも消費）
- sim-to-real ギャップがある（本プロジェクトでは問題なし：目的は技術理解）

---

### 2.4 BDD100K

**大規模ドライブデータ。多様なシーンだがアクションラベルは限定的。**

| 項目 | 内容 |
|---|---|
| URL | https://bdd-data.berkeley.edu/ |
| サイズ | ~1.8 TB（動画）/ 画像のみ ~6 GB |
| フレーム数 | 100K ビデオクリップ |
| カメラ | **1 台（ダッシュカメラ）** |
| アクション | GPS 軌跡 → waypoints に変換可 |
| ライセンス | BSD-3（研究利用可） |

**注意点**:
- steer/throttle は直接提供されない
- GPS からの waypoint 変換は可能だが精度に限界あり
- 画像の多様性は高い（天候、時刻、都市/郊外）

---

## 3. データセット選定ガイド

### 段階別の推奨

| Stage | 最優先 | 次点 | 備考 |
|---|---|---|---|
| **Stage 0 (fail-fast)** | **comma2k19** (数セグメント) | nuScenes Mini | steer/throttle が直接得られる comma が最速 |
| **Stage 0 (本格学習)** | **nuScenes Full** | comma2k19 (全体) | waypoint 予測に移行するなら nuScenes が有利 |
| **Stage 2 (Flow)** | **nuScenes Full** | CARLA 生成データ | 多様な軌道が必要、CARLA なら量を増やせる |

### 推奨シナリオ

```
Step 1: comma2k19 の数セグメントで Stage 0 を fail-fast 検証
         ↓ 学習が回ることを確認
Step 2: nuScenes Mini (~4 GB) で waypoint 予測に移行
         ↓ パイプラインが安定
Step 3: nuScenes Full で本格学習 → Stage 2 (Flow)
```

---

## 4. データ前処理パイプライン

どのデータセットを使う場合でも、以下の統一フォーマットに変換する:

### 4.1 統一フォーマット

```python
{
    "image": torch.Tensor,       # (3, 224, 224) — RGB, normalized
    "action": torch.Tensor,      # (2,) — [steer, throttle]  (Stage 0)
                                 # or (K, 2) — waypoints      (Stage 0+)
    "text": str or None,         # テキスト指示（任意）
    "ego_state": torch.Tensor,   # (D,) — speed, yaw_rate 等（任意）
    "timestamp": float,          # タイムスタンプ
}
```

### 4.2 画像前処理

```python
transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 標準
              std=[0.229, 0.224, 0.225]),
])
```

### 4.3 アクション正規化

- steer: [-1, 1] にスケーリング（データセットの最大操舵角で割る）
- throttle: [0, 1] にスケーリング
- waypoints: 自車座標系で正規化（最大距離で割る）

---

## 5. データ量の目安

| 用途 | フレーム数 | ストレージ | 備考 |
|---|---|---|---|
| fail-fast 検証 | ~1,000 | ~数百 MB | overfitting でも OK |
| Stage 0 学習 | ~10,000–50,000 | ~数 GB | loss が下がるか確認 |
| Stage 2 (Flow) 学習 | ~50,000–200,000 | ~10–50 GB | 多様な軌道が必要 |

---

## 6. ダウンロード・準備チェックリスト

### 最速で始める場合（comma2k19）

- [ ] https://github.com/commaai/comma2k19 からセグメントを数個ダウンロード
- [ ] 画像フレーム抽出（動画 → JPEG/PNG）
- [ ] CAN データから steer, speed を抽出
- [ ] 統一フォーマットへ変換するスクリプト作成
- [ ] DataLoader のテスト

### nuScenes を使う場合

- [ ] https://www.nuscenes.org/ でアカウント作成
- [ ] nuScenes Mini をダウンロード（~4 GB）
- [ ] `pip install nuscenes-devkit`
- [ ] CAM_FRONT 画像 + ego pose 抽出スクリプト作成
- [ ] ego pose → waypoints 変換
- [ ] 統一フォーマットへ変換
- [ ] DataLoader のテスト
