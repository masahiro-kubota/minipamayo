# MCAP Telemetry Specification

このドキュメントは、`carla_alpamayo` が route-loop 実行時に出力する MCAP telemetry の現行仕様をまとめたものです。

目的:

- 現行の MCAP 出力を別プロジェクトでも再現しやすくする
- Foxglove で見ている topic / frame / payload 形状を明文化する
- 実装の source of truth を追わなくても payload 仕様を把握できるようにする

対象実装:

- `libs/schemas/mcap_route_log.py`
- `ad_stack/run.py`

この仕様書では、schema 定義 dict だけでなく、実際に emit している payload も基準にします。schema 定義と実 payload が食い違う箇所は、`write_frame()` / `write_static_scene()` の出力内容を優先して記述します。

## 1. Output Layout

route-loop 実行の MCAP 出力は、evaluation 出力ディレクトリ配下の `telemetry/` に置かれます。

```text
outputs/evaluate/<run_id>/
  telemetry/
    index.json
    segment_0000.mcap
    segment_0001.mcap
    ...
```

各ファイルの役割:

- `index.json`
  - segment の一覧と時間範囲を持つ manifest
  - schema 本体はここには置かない
- `segment_XXXX.mcap`
  - 実データ本体
  - channel, schema, metadata は各 MCAP ファイル内に埋め込まれる

segment は `segment_seconds` 単位でローテーションされます。default は `600.0` 秒です。

## 2. MCAP Container Settings

MCAP writer の設定は次の通りです。

- profile: `carla_alpamayo.route_loop`
- library: `carla_alpamayo`
- compression: `zstd`
- chunking: enabled
- schema encoding: `jsonschema`
- message encoding: `json`

file metadata には `episode` キーで次の情報を入れます。

- `episode_id`
- `route_name`
- `town`
- `weather`
- `compression`

`compression` の metadata 値は `zstd_chunked` です。

## 3. Coordinate And Frame Conventions

Foxglove 向けに、CARLA 座標系から次の変換を入れています。

- 位置:
  - `x -> x`
  - `y -> -y`
  - `z -> z`
- 姿勢:
  - `roll_deg -> roll_deg`
  - `pitch_deg -> -pitch_deg`
  - `yaw_deg -> -yaw_deg`

frame は次の 3 つを使います。

- `map`
- `ego/base_link`
- `ego/front_camera`

front camera の固定外部パラメータは次の通りです。

- parent: `ego/base_link`
- child: `ego/front_camera`
- translation: `{ "x": 1.5, "y": 0.0, "z": 2.4 }`
- rotation: identity quaternion

別プロジェクトで再利用する場合、CARLA 由来でない座標系ならこの変換は差し替える必要があります。

## 4. Common Time Representation

各 message 内の time は Foxglove 互換の object で表現します。

```json
{
  "sec": 1710000000,
  "nsec": 123456789
}
```

仕様:

- `sec`: integer
- `nsec`: integer

MCAP の `log_time` / `publish_time` はナノ秒整数です。message 内の `timestamp` は上記 object です。

## 5. Topic Overview

| topic | schema name | kind | notes |
| --- | --- | --- | --- |
| `/camera/front/compressed` | `foxglove.CompressedImage` | dynamic | front RGB を JPEG + base64 で格納 |
| `/map/scene` | `foxglove.SceneUpdate` | static | lane centerline と planned route |
| `/tf` | `foxglove.FrameTransforms` | dynamic | `map -> ego/base_link`, `ego/base_link -> ego/front_camera` |
| `/ego/state` | `carla_alpamayo.EgoState` | dynamic | ego の状態量 |
| `/ego/control` | `carla_alpamayo.EgoControl` | dynamic | 制御値を分離 |
| `/ego/planning` | `carla_alpamayo.EgoPlanning` | dynamic | planner / behavior 系状態 |

dynamic topic は artifact 記録タイミングごとに出力されます。default の artifact 記録レートは `10Hz` です。

## 6. Topic Specifications

### 6.1 `/camera/front/compressed`

channel metadata:

- `frame_id`: `ego/front_camera`
- `camera_width`: stringified integer
- `camera_height`: stringified integer
- `jpeg_quality`: stringified integer

payload:

```json
{
  "timestamp": { "sec": 1710000000, "nsec": 123456789 },
  "frame_id": "ego/front_camera",
  "data": "<base64-encoded-jpeg>",
  "format": "jpeg"
}
```

備考:

- `current_rgb` は JPEG へ圧縮した後、base64 文字列にして JSON に詰める
- raw binary channel ではなく JSON channel を使う

### 6.2 `/map/scene`

channel metadata:

- `frame_id`: `map`

この topic は static scene 用です。現在は 2 entity を出します。

- `lane_centerlines`
- `planned_route`

payload の骨格:

```json
{
  "deletions": [],
  "entities": [
    {
      "timestamp": { "sec": 1710000000, "nsec": 0 },
      "frame_id": "map",
      "id": "lane_centerlines",
      "lifetime": { "sec": 0, "nsec": 0 },
      "frame_locked": true,
      "metadata": [
        { "key": "kind", "value": "lane_centerlines" },
        { "key": "route_name", "value": "<route_name>" }
      ],
      "lines": []
    },
    {
      "timestamp": { "sec": 1710000000, "nsec": 0 },
      "frame_id": "map",
      "id": "planned_route",
      "lifetime": { "sec": 0, "nsec": 0 },
      "frame_locked": true,
      "metadata": [
        { "key": "kind", "value": "planned_route" },
        { "key": "route_name", "value": "<route_name>" }
      ],
      "lines": []
    }
  ]
}
```

line object の主なフィールド:

- `type`: `0`
- `pose`: identity
- `thickness`
- `scale_invariant`: `false`
- `points`: `{x, y, z}` の配列
- `color`: `{r, g, b, a}`
- `indices`: 現状は空配列

描画色:

- lane centerlines:
  - `r=0.55`, `g=0.55`, `b=0.55`, `a=0.65`
  - `thickness=0.12`
- planned route:
  - `r=0.08`, `g=0.72`, `b=1.0`, `a=0.95`
  - `thickness=0.3`

備考:

- lane centerlines の範囲は caller 側の `mcap_map_scope` に依存する
- segment をまたいだ場合、新しい segment の先頭で static scene を再送する

### 6.3 `/tf`

payload:

```json
{
  "transforms": [
    {
      "timestamp": { "sec": 1710000000, "nsec": 123456789 },
      "parent_frame_id": "map",
      "child_frame_id": "ego/base_link",
      "translation": { "x": 1.0, "y": 2.0, "z": 0.0 },
      "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
    },
    {
      "timestamp": { "sec": 1710000000, "nsec": 123456789 },
      "parent_frame_id": "ego/base_link",
      "child_frame_id": "ego/front_camera",
      "translation": { "x": 1.5, "y": 0.0, "z": 2.4 },
      "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
    }
  ]
}
```

備考:

- `map -> ego/base_link` は毎 frame の ego pose から作る
- `ego/base_link -> ego/front_camera` は固定 transform
- 回転は quaternion

### 6.4 `/ego/state`

payload:

```json
{
  "timestamp": { "sec": 1710000000, "nsec": 123456789 },
  "episode_id": "<episode_id>",
  "frame_id": 123,
  "elapsed_seconds": 12.3,
  "speed_mps": 4.5,
  "route_completion_ratio": 0.42,
  "distance_to_goal_m": 37.8,
  "pose": {
    "x": 1.0,
    "y": 2.0,
    "z": 0.0,
    "yaw_deg": -90.0,
    "pitch_deg": 0.0,
    "roll_deg": 0.0
  }
}
```

意味:

- `episode_id`: episode 識別子
- `frame_id`: 記録フレーム番号
- `elapsed_seconds`: route-loop 開始からの経過秒
- `speed_mps`: ego 速度
- `route_completion_ratio`: route 進捗
- `distance_to_goal_m`: goal までの距離
- `pose`: Foxglove 向けに変換済みの位置と Euler 角

### 6.5 `/ego/control`

payload:

```json
{
  "timestamp": { "sec": 1710000000, "nsec": 123456789 },
  "episode_id": "<episode_id>",
  "frame_id": 123,
  "elapsed_seconds": 12.3,
  "steer": 0.1,
  "throttle": 0.3,
  "brake": 0.0
}
```

### 6.6 `/ego/planning`

payload:

```json
{
  "timestamp": { "sec": 1710000000, "nsec": 123456789 },
  "episode_id": "<episode_id>",
  "frame_id": 123,
  "elapsed_seconds": 12.3,
  "behavior": "lane_follow",
  "planner_state": "cruise",
  "traffic_light_state": "green",
  "overtake_state": null,
  "target_lane_id": null,
  "min_ttc": 4.8
}
```

nullable field:

- `behavior`
- `planner_state`
- `traffic_light_state`
- `overtake_state`
- `target_lane_id`
- `min_ttc`

## 7. Segment Index Specification

`telemetry/index.json` は次の shape を持ちます。

```json
{
  "episode_id": "<episode_id>",
  "route_name": "<route_name>",
  "town": "Town01",
  "weather": "ClearNoon",
  "segment_seconds": 600.0,
  "segments": [
    {
      "segment_index": 0,
      "path": "segment_0000.mcap",
      "start_elapsed_seconds": 0.1,
      "end_elapsed_seconds": 599.9,
      "frame_count": 6000
    }
  ]
}
```

`segments[]` 各要素の意味:

- `segment_index`: 0-origin の segment 番号
- `path`: telemetry directory からの相対ファイル名
- `start_elapsed_seconds`: その segment に最初に入った frame の elapsed time
- `end_elapsed_seconds`: その segment で最後に書いた frame の elapsed time
- `frame_count`: その segment に含まれる dynamic frame 数

segment index は frame 書き込みに合わせて更新され、close 時にも最終更新されます。

## 8. Rotation Behavior

segment の切り替えは `elapsed_seconds // segment_seconds` で決まります。

動作:

1. 次の frame が別 segment に入ったら現在の writer を close する
2. 新しい `segment_XXXX.mcap` を開く
3. static scene が保持されていれば、新 segment の先頭で `/map/scene` を再送する
4. その後に dynamic frame を書く
5. `index.json` を更新する

`segment_seconds <= 0.0` の場合は single-file mode になり、常に `segment_0000.mcap` を使います。

## 9. Producer Inputs

writer に渡す入力の最小セットは `EgoStateSample` と `current_rgb` です。

`EgoStateSample` の入力項目:

- `episode_id`
- `frame_id`
- `timestamp_s`
- `elapsed_seconds`
- `speed_mps`
- `behavior`
- `route_completion_ratio`
- `distance_to_goal_m`
- `planner_state`
- `traffic_light_state`
- `lead_vehicle_distance_m`
- `overtake_state`
- `target_lane_id`
- `min_ttc`
- `pose`
- `control`

現状、`lead_vehicle_distance_m` は input として保持していますが、MCAP topic には出していません。

## 10. Implementation Notes And Current Gaps

現行実装で再利用時に気をつける点:

- `/ego/state` の schema 定義には `control` が required として残っている
- ただし実際の `/ego/state` payload には `control` を含めない
- `control` は `/ego/control` に分離済み

つまり、現行仕様としては次の理解が正です。

- schema dict の source: 実装ファイル内の JSON Schema
- 実 payload の source: `write_frame()` が実際に書く JSON
- 外部再利用時の互換対象: 実 payload

このズレを解消して shared module 化するなら、`_EGO_STATE_JSON_SCHEMA` から `control` を外すのが自然です。

## 11. Reuse Guidance For Other Projects

同じ形で他プロジェクトに持っていく場合の最小方針:

1. topic 名はそのまま維持する
2. time object は `{sec, nsec}` 形式を使う
3. `map`, `ego/base_link`, `ego/front_camera` の frame 構成を維持する
4. static scene と dynamic frame を分ける
5. segment + `index.json` 方式を踏襲する

差し替え対象:

- 座標変換
- camera extrinsic
- `planned_route` / `lane_centerlines` の生成元
- `behavior` / `planner_state` / `traffic_light_state` などの planner 由来 field

Foxglove 互換性より転送効率を優先する場合は、将来的に message encoding や schema 形式を `protobuf` へ切り替える余地があります。ただし、現行仕様の互換再現が目的なら `jsonschema` + `json` のまま合わせるのが最も簡単です。

## 12. Source Of Truth

現行実装の source of truth:

- schema 登録: `libs/schemas/mcap_route_log.py`
- payload 生成: `libs/schemas/mcap_route_log.py`
- writer 呼び出しと記録タイミング: `ad_stack/run.py`

この仕様書を更新するときは、上記実装との差分がないことを優先して確認します。
