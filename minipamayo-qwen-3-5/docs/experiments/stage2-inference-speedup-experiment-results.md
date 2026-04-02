# Stage2 Inference Speedup Experiment Results

## Scope

このメモは [stage2-inference-speedup-experiment-plan.md](/home/masa/minipamayo/minipamayo-qwen-3-5/docs/experiments/stage2-inference-speedup-experiment-plan.md) の結果記録用。

まずは **精度を無視した speed-only benchmark** として、

- `reasoning token`
- `flow_steps`
- `image token`
- `history token`

を順に切り分ける。

## Environment

### Local

- GPU: `NVIDIA GeForce RTX 4070 Ti`
- code path: official `Stage2` wrapper core
- checkpoints: scratch `1 epoch` の `Stage1A -> Stage1B -> Stage2`
- sample: single sample, warm run

### Remote

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- code path: official `Stage2` wrapper core
- checkpoints: same scratch `1 epoch` 系列
- sample: same single sample, warm run

## Baseline

### Stage2 scratch budget sweep

scratch `Stage2` は `<|traj_future_start|>` を安定して出せず、reasoning token budget をかなり使い切る。

| reasoning budget | 4070 Ti mean (s) | RTX6000 mean (s) |
| --- | ---: | ---: |
| 16 | 0.290 | 0.282 |
| 32 | 0.501 | 0.485 |
| 64 | 0.919 | 0.887 |
| 128 | 1.764 | 1.745 |
| 256 | 3.463 | 3.305 |

読み取り:

- どちらの GPU でも `<|traj_future_start|>` を出せず、token budget 依存でほぼ線形に伸びる
- RTX6000 にしても大幅には縮まらない
- scratch `Stage2` の current path は、GPU の raw throughput よりも batch=1 の逐次 reasoning rollout に強く支配されている

## Phase 0

### Current input breakdown

same sample の current `Stage2` input prefix は約 `280 tokens`。

- image tokens: `180`
- history placeholder: `48`
- history start/end: `2`
- reasoning instruction text: `18`
- system prompt text: `12`
- assistant prefill `<|cot_start|>`: `1`
- chat template overhead: 約 `19`

以降の image token 数は、`image_grid_thw` の積を `4` で割った実効 token 数で表記する。

### Prefill-only ablation

`Stage2` end-to-end だけではなく、official wrapper path の最初の `VLM prefill` だけを CUDA event で直接計測した。

| config | image tokens | total input tokens | prefill mean (ms) | prefill median (ms) |
| --- | ---: | ---: | ---: | ---: |
| full image + history + reasoning | 180 | 280 | 31.30 | 31.31 |
| small image + history + reasoning | 60 | 160 | 34.81 | 30.30 |
| no image + history + reasoning | 0 | 98 | 23.80 | 23.68 |
| no image + no history + reasoning | 0 | 48 | 23.42 | 23.38 |
| no image + minimal text | 0 | 30 | 23.37 | 23.32 |

読み取り:

- `image 180 -> 60` では prefill median は `31.31ms -> 30.30ms` と小さい差しか出ない
- 一方で image 自体を外すと `~23.7ms` まで下がる
- つまり input token 数は効いているが、**image token 数そのもの**よりも、multimodal prefill の固定コストが大きい
- `98 -> 48 -> 30 tokens` にしても `23.8ms -> 23.4ms -> 23.3ms` なので、text/history 側にもかなり明確な floor がある

## Phase 1A

### Reasoning token sweep

#### 4070 Ti

| reasoning budget | mean (s) | median (s) | notes |
| --- | ---: | ---: | --- |
| 1 | 0.0999 | 0.0995 | no `<|traj_future_start|>` |
| 2 | 0.1183 | 0.1180 | no `<|traj_future_start|>` |
| 4 | 0.1369 | 0.1320 | no `<|traj_future_start|>` |
| 8 | 0.1858 | 0.1851 | no `<|traj_future_start|>` |
| 16 | 0.2877 | 0.2852 | no `<|traj_future_start|>` |

読み取り:

- `reasoning budget = 1` まで詰めると、4070 Ti でも `~0.10s` まで落ちる
- つまり current scratch `Stage2` は、想定どおり reasoning token 数が最も強い knob
- 16 token では `~3.5 Hz` だが、1 token まで削ると `~10 Hz` 近辺まで一気に下がる

#### RTX6000

- baseline budget sweep は取得済み
- `1/2/4/8` の短 budget sweep は未取得

## Phase 1B

### Flow step sweep

#### 条件 A: reasoning budget = 16

| flow steps | mean (s) | median (s) | delta vs step10 |
| --- | ---: | ---: | ---: |
| 2 | 0.2746 | 0.2750 | -5.8% |
| 4 | 0.2526 | 0.2526 | -13.4% |
| 6 | 0.2651 | 0.2646 | -9.1% |
| 10 | 0.2915 | 0.2910 | baseline |

読み取り:

- `reasoning budget = 16` では `flow_steps=4` が最速
- ただし効き幅は `~13%` 程度で、reasoning rollout 支配という見立ては変わらない

#### 条件 B: reasoning budget = 1

| flow steps | mean (s) | median (s) | delta vs step10 |
| --- | ---: | ---: | ---: |
| 2 | 0.0410 | 0.0411 | -58.8% |
| 4 | 0.0533 | 0.0532 | -46.4% |
| 6 | 0.0688 | 0.0680 | -30.8% |
| 10 | 0.0995 | 0.0993 | baseline |

読み取り:

- `reasoning budget = 1` まで削った状態では、`flow_steps` が支配的になる
- `flow_steps=2` にすると `~0.041s` まで下がり、4070 Ti でも `~24 Hz`
- したがって、`10 Hz` 到達の条件は「reasoning token をほぼ出さない」ことに加えて「flow_steps も 2-4 程度まで削る」こと

## Phase 2

### Image token compression

条件:

- `reasoning budget = 1`
- `flow_steps = 2`
- history placeholder = `48`

| image budget | image tokens | total input tokens | mean (s) | median (s) |
| --- | ---: | ---: | ---: | ---: |
| `163840:196608` | 180 | 280 | 0.0408 | 0.0407 |
| `131072:131072` | 120 | 220 | 0.0408 | 0.0408 |
| `98304:98304` | 91 | 191 | 0.0399 | 0.0398 |
| `65536:65536` | 60 | 160 | 0.0401 | 0.0402 |

読み取り:

- `reasoning=1 + flow_steps=2` の regime では、image token を `180 -> 60` まで削っても差はごく小さい
- この条件では image prefill はもはや支配項ではなく、diffusion / wrapper 側の fixed cost が支配している

### History token compression

条件:

- `reasoning budget = 1`
- `flow_steps = 2`
- benchmark-only で prompt の history placeholder 数を短くする

#### image tokens = 180

| history tokens | total input tokens | mean (s) | median (s) |
| --- | ---: | ---: | ---: |
| 48 | 280 | 0.0424 | 0.0424 |
| 16 | 248 | 0.0430 | 0.0430 |
| 8 | 240 | 0.0442 | 0.0444 |

#### image tokens = 60

| history tokens | total input tokens | mean (s) | median (s) |
| --- | ---: | ---: | ---: |
| 48 | 160 | 0.0424 | 0.0419 |
| 16 | 128 | 0.0433 | 0.0431 |
| 8 | 120 | 0.0421 | 0.0421 |

benchmark 実装メモ:

- speed-only ablation として、prompt 側の history placeholder 数だけを短くした
- `AlpamayoR1.fuse_traj_tokens(...)` の `masked_scatter` は、48-token quantizer 出力の先頭 `N` 個だけを使用して shorter placeholder に埋め込めるので、benchmark path はこれで通った
- 精度保証のある設計ではない

読み取り:

- `reasoning=1 + flow_steps=2` の regime では、history token 数もほぼ効かない
- `48 -> 8` にしても latency は誤差級
- current speed bottleneck は history prefix ではない

## Phase 3

### Combined best-case

現時点での best-case は、少なくとも local 4070 Ti では次の条件。

- `reasoning budget = 1`
- `flow_steps = 2`
- image tokens / history tokens は baseline のままでもよい

代表値:

- baseline prefix (`image=180`, `history=48`): `0.0424s`
- compressed prefix (`image=60`, `history=8`): `0.0421s`

読み取り:

- best-case latency は `~0.04s`
- これは `~24 Hz`
- current scratch `Stage2` でも、**reasoning token と flow_steps を極端に削れば 10 Hz は超える**
- 逆に、prefix compression 単独ではほとんど効かない

## Conclusion

### 1. current scratch `Stage2` の main knob は reasoning token 数

- `<|traj_future_start|>` を早く出せない current scratch path では、latency は reasoning token budget にほぼ比例する
- `16 token` では `~0.29s`
- `1 token` まで詰めると `~0.10s`

### 2. 10 Hz 到達には flow_steps も一緒に削る必要がある

- `reasoning=1, flow=10` では `~0.10s`
- `reasoning=1, flow=2` では `~0.04s`

つまり、10 Hz は

- reasoning token をほぼ出さない
- expert diffusion も 2-4 steps に削る

のセットで達成される。

### 3. image/history compression は、この regime では二次効果

- `image 180 -> 60`
- `history 48 -> 8`

まで削っても、`reasoning=1, flow=2` では差がごく小さい。

したがって current scratch `Stage2` の current bottleneck は、

- prefix 長

ではなく

- VLM reasoning rollout と expert diffusion steps

である。

## Practical Implication

current scratch model で online speed を最優先するなら、優先順位はこうなる。

1. `<|traj_future_start|>` をほぼ即時に出す
2. `flow_steps` を `2-4` に削る
3. その上で image/history token を詰めるか検討する

prefix compression を先にやるより、reasoning / diffusion 側を切る方がはるかに効く。
