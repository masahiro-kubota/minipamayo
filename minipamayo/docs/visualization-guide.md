# Visualization 出力の見方

`visualize.py` が各 stage ごとに出力する PNG 画像の読み方ガイド。

## 実行方法

```bash
cd minipamayo && uv run python -m minipamayo.visualize \
    --stage <phase4|stage1|stage2|stage3|stage4> \
    --checkpoint <path>
```

---

## Phase 4 (回帰軌道) — 2列構成

| 列 | 内容 |
|---|---|
| 左: 入力画像 | nuScenes のカメラ画像 |
| 右: BEV 軌道 | 青線=GT、赤破線=Pred。左上に ADE/FDE 表示 |

### 確認ポイント

- **赤(Pred)と青(GT)が重なっているか** — 重なっていれば予測精度が高い
- 直進シーンで大きく曲がっている → curvature 予測が壊れている
- 全サンプルで赤が同じ形 → mean collapse（入力を無視して定数出力）

---

## Stage 1 (離散トークン) — 2列構成

| 列 | 内容 |
|---|---|
| 左: 入力画像 | nuScenes のカメラ画像 |
| 右: BEV 軌道 | 青=GT、赤=Pred。タイトルに `Token Acc: X/12` |

### 確認ポイント

- **Token Acc** — 0/12 や 1/12 ならトークン予測がほぼ失敗
- 軌道がガタガタ → 離散化の量子化誤差 or AR 誤差伝播
- Phase 4 の結果と比較して大幅に劣化していないか

---

## Stage 2 (Flow Matching) — 2列構成

| 列 | 内容 |
|---|---|
| 左: 入力画像 | nuScenes のカメラ画像 |
| 右: BEV 軌道 | 青=GT、赤=Best sample、**薄赤=複数サンプル** |

### 確認ポイント

- **薄赤(Samples)の広がり方が最重要**
  - 扇状に広がっている → 多様性があり、確率的モデリングが機能
  - 全部重なって 1 本に見える → diversity が低い（deterministic に退化）
- Best sample(赤)が GT(青)に近い → minADE が良い
- タイトルの `minADE` が Phase 4 の ADE より小さければ multi-sample の恩恵あり

---

## Stage 3 (CoC SFT) — 3列構成

| 列 | 内容 |
|---|---|
| 左: 入力画像 | nuScenes のカメラ画像 |
| 中: BEV 軌道 | 青=GT、赤=Pred、灰色=障害物。タイトルに GT の decision |
| 右: CoC テキスト | 上半分=Generated、下半分=GT CoC |

### 確認ポイント

- **テキスト (右列)**
  - Generated が自然な文になっているか（崩壊した文字列でないか）
  - `longitudinal:` / `lateral:` の decision が GT と一致しているか
- **BEV (中列)**
  - 障害物(灰色)がある場合、予測軌道が避けているか
  - decision が `decelerate` なのに軌道が長い(=加速している) → 不整合

---

## Stage 4 (GRPO RL vs SFT) — 3列構成

| 列 | 内容 |
|---|---|
| 左: 入力画像 | nuScenes のカメラ画像 |
| 中: BEV 軌道 | 青=GT、**赤=RL**、**緑=SFT**、灰色=障害物 |
| 右: CoC テキスト | 上半分=RL 生成、下半分=SFT 生成 |

### 確認ポイント

- **赤(RL)が緑(SFT)より青(GT)に近いか** → RL で軌道品質が改善
- 障害物付近で赤が回避、緑が衝突 → RL の collision reward が効いている
- テキスト: RL の推論が SFT と同等の自然さを保っているか（catastrophic forgetting がないか）
- 全サンプルで赤と緑が同じ → RL が実質何も学んでいない

---

## 全 stage 共通の赤信号

| 症状 | 意味 |
|---|---|
| 全サンプルで同じ軌道 | mean collapse（入力を無視） |
| 軌道が原点付近で止まっている | action がほぼゼロ |
| ADE/FDE が二桁メートル | 予測が完全に破綻 |
| カーブで直進、直進でカーブ | 入力画像を見ていない |

## BEV プロットの読み方

### 座標系

- ego-centric 座標系（+x=前方, +y=左）
- BEV 表示: 上=前方, 右=右方向

### シンボル

| シンボル | 意味 |
|---|---|
| 黒丸 (原点) | 自車の現在位置 |
| 黒矢印 (上向き) | 自車の進行方向（前方） |
| 青線 + 丸マーカー | GT 軌道（正解） |
| 赤破線 + 三角マーカー | Pred 軌道（モデル予測） |
| 薄赤線 (Stage 2) | Flow Matching の複数サンプル |
| 緑破線 + 四角マーカー (Stage 4) | SFT モデルの軌道（比較用） |
| 灰色矩形 (Stage 3/4) | 障害物 |
| 左上テキスト | ADE / FDE の数値 |

軌道上の各マーカーは 0.5 秒間隔の waypoint を表す（K=6 なら 0.5s〜3.0s の 6 点）。
