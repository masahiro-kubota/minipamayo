# Stage2 Reasoning SFT Dataset 契約

この文書は、canonical な `Stage2 / reasoning_sft` が受け取る JSONL dataset 契約を固定するためのものです。

目的は次の 2 つです。

- Stage1A と同じ観測契約を Stage2 に引き継ぐ
- synthetic reasoning ではなく、dataset から与えられた `reasoning_text` を本流 supervision にする

## 位置づけ

canonical Stage2 は論文上の reasoning SFT に対応する。

- 入力観測は Stage1A と同じ
  - `o_image`
  - `o_egomotion`
- 追加 supervision として `reasoning_text` を持つ
- action token target も引き続き持つ

つまり Stage2 record は

- Stage1 record
- `+ reasoning_text`

であるべきで、Stage1 より観測を減らしてはいけない。

## 必須フィールド

各 JSONL record は少なくとも次を持つ。

- `sample_id: str`
- `image_path: str`
- `action: list[float]`
- `v0: float`
- `gt_waypoints: list[list[float]]`
- `dt: float`
- `ego_history_xyz: list[list[float]]`
- `ego_history_rot: list[list[list[float]]]`
- `reasoning_text: str`

意味は以下の通り。

- `image_path`
  - Stage1 と同じく front image への相対 path
- `action`
  - canonical では `64 waypoint × 2 = 128` scalar
  - layout は `[a1, k1, a2, k2, ...]`
- `v0`
  - current ego speed
- `gt_waypoints`
  - future trajectory supervision
- `dt`
  - canonical では `0.1`
- `ego_history_xyz`
  - canonical では ego-frame history `(history_steps, 3)`
- `ego_history_rot`
  - canonical では ego-frame rotation matrices `(history_steps, 3, 3)`
- `reasoning_text`
  - CoC / reasoning SFT の teacher text

## 任意フィールド

次は保持してよい。

- `command`
- `planner_state`
- `decision_longitudinal`
- `decision_lateral`

これらは analysis や experiment 用の補助であり、canonical Stage2 の必須契約ではない。

## canonical と experiment の分離

canonical Stage2 では、`reasoning_text` は dataset が明示的に持つ必要がある。

つまり canonical では次をしない。

- `command` と `planner_state` からその場で synthetic reasoning を生成する
- planner label だけから teacher text を即席で組み立てる

それらは `experiments/synthetic_reasoning/` に閉じ込める。

## prompt 契約

Stage2 prompt は Stage1A と同じ history placeholder を持つ。

- image token
- history placeholder token 列
- reasoning 指示 text

そして canonical path では、placeholder 位置に

- `ego_history_xyz`
- `ego_history_rot`

から得た continuous history embedding を `inputs_embeds` 上で fuse する。

## 生成 target 契約

teacher target は

1. `reasoning_text`
2. action section header
3. action token 列
4. `eos`

の順で連結する。

action 部分は loss weight を上げてもよいが、canonical target の系列順は固定する。

## smoke 用 dataset

smoke 検証では Stage1 smoke JSONL に `reasoning_text` を足した派生 JSONL を使ってよい。

ただしそれはあくまで contract 検証用であり、本番の canonical Stage2 dataset と混同しない。

## 実装対応

現在の canonical loader はこれを前提にしている。

- [dataset.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/dataset.py)
- [runner.py](/home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/stage2/reasoning_sft/train/runner.py)
