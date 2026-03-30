# Alpamayo Core Alignment Plan

このメモは、`minipamayo-qwen-3-5` を `/home/masa/minipamayo/related_repos/alpamayo` に段階的に寄せるための実装順を整理したものです。

前提:
- 目的は「今の repo 都合の abstraction を守ること」ではなく、「Alpamayo 公開実装との差分を減らすこと」。
- `config.py` / `models/alpamayo_r1.py` はすでに持ち込んでおり、`stage2` inference は wrapper 経由に寄せた。
- `helper.py` / `test_inference.py` も top-level に追加済みで、Alpamayo-style の推論入口は一通り揃った。
- `record_adapter.py` のような repo 固有 glue は残してよいが、pure core に混ぜない。

## 目標

最終的に top-level に以下の shared core を揃える。

- `action_space/`
- `diffusion/`
- `models/`
- `geometry/`

各 stage はそれらを import するだけに寄せる。

## 現状の diff 状況

2026-03-30 時点で、以下の `diff -ru` を使って Alpamayo 側と比較した。

```bash
diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/action_space \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/action_space

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/diffusion \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/diffusion

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/models \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/models

diff -ru \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/geometry \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/geometry

diff -u \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/helper.py \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/helper.py

diff -u \
  /home/masa/minipamayo/related_repos/alpamayo/src/alpamayo_r1/test_inference.py \
  /home/masa/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/test_inference.py
```

再確認結果:

- 同名 file を持つ比較対象については、未記載の non-import-path 差分は新たに見つかっていない
- same-name file で non-import-path 差分が残っているのは、現時点では
  - `models/base_model.py`
  - `models/alpamayo_r1.py`
  - `test_inference.py`
  のみ
- top-level shared core 配下で Alpamayo 側に同名 file が無い extra file は、現時点では
  - `contract/record_adapter.py`
  - `models/expert_cache_utils.py`
- `Stage1B` 固有の extra file は、現時点では
  - `stage1/stage1b_action_expert.py`
  - `stage1/stage1b_diffusion_adapter.py`
- `Stage1B` 専用 diffusion adapter は mirror を汚さないため
  `stage1/stage1b_diffusion_adapter.py` へ退避した
- したがって、`helper.py`, `config.py`, `geometry/rotation.py`, `models/action_in_proj.py`,
  `models/delta_tokenizer.py`, `models/token_utils.py`, `action_space/*`, `diffusion/flow_matching.py`
  については、このメモの分類以外の hidden diff は確認できていない

### 1. 差分なし

以下は Alpamayo 側と一致している。

- `action_space/action_space.py`
- `action_space/__init__.py`
- `diffusion/base.py`
- `diffusion/__init__.py`
- `geometry/rotation.py`
- `helper.py`
- `models/action_in_proj.py`
- `models/delta_tokenizer.py`
- `models/token_utils.py`

### 2. import path 差だけ

以下は実質 import path 差だけで、ロジック差は残っていない。

- `action_space/discrete_action_space.py`
- `action_space/unicycle_accel_curvature.py`
- `action_space/utils.py`
- `config.py`
- `diffusion/flow_matching.py`

### 3. runtime shim / entrypoint 差分

以下は Alpamayo 実装をほぼ持ち込んでいるが、`minipamayo-qwen-3-5` の runtime 差や入力契約差を吸収するための差分が残っている。

- `models/base_model.py`
  - `tie_weights(self, *args, **kwargs)` の互換 shim がある
  - これは `transformers==5.4.0` 側の追加引数を受けるため
- `models/alpamayo_r1.py`
  - pure import path 差に加えて以下が残っている
  - Alpamayo の inline 4D additive mask ではなく、inline の 2D padding mask を使う
  - これは `transformers==5.4.0 + Qwen3.5-0.8B + flash_attention_2` の expert path が 2D padding mask を期待するため
  - `transformers==4.57.1` の `Qwen3-VL` には 4D mask を padding mask に落とす互換分岐があるが、`Qwen3.5` の expert path には同等の分岐が無い
  - Alpamayo の in-place `prompt_cache.crop(...)` ではなく、step ごとに cache clone を使う
    - これは `Qwen3_5DynamicCache` が `crop()` を持たないため
- `test_inference.py`
  - Alpamayo は `load_physical_aiavdataset.py` から直接 tensor を作る
  - こちらは `samples_reasoning_sft.jsonl` と `record_adapter` 系の契約を使う
  - wrapper 呼び出し自体は揃えたが、入力 loader は別物
  - これは runtime shim ではなく、demo entrypoint の入力契約差分

### 4. repo 固有として残している差分

以下は Alpamayo には無いが、`minipamayo-qwen-3-5` 側の stage 分割や JSONL record 契約のために残している。

ラベルの意味:

- `[DATA-CONTRACT]`
  - `physical_ai_av` / `load_physical_aiavdataset.py` 前提ではなく、
    `jsonl + images` の canonical record 契約を受けるための差分
- `[STAGE-SPLIT]`
  - Alpamayo 本家の integrated `AlpamayoR1` を、
    `Stage1A` / `Stage1B` に分割して再利用するための差分
- `[QWEN3.5-RUNTIME]`
  - Qwen3.5 系 runtime / cache 挙動に合わせるための差分
- `[NOT-NO-PAST-IMAGE]`
  - 「過去 image を使わない」ことが主因ではない、という明示ラベル

これら 4 ファイルについては、主因は `過去 image なし` ではない。
`過去 image なし` の差分は主に prompt / input loader / demo inference entrypoint 側にある。

整理すると、repo 固有差分は次の 2 群に分かれる。

- top-level shared utility
  - `contract/record_adapter.py`
  - `models/expert_cache_utils.py`
- `Stage1B` 専用実装
  - `stage1/stage1b_action_expert.py`
  - `stage1/stage1b_diffusion_adapter.py`

#### 4.1 top-level shared utility

- `contract/record_adapter.py` `[DATA-CONTRACT]` `[NOT-NO-PAST-IMAGE]`
  - record から tensor を読む
  - shape adapter
  - canonical action / waypoint payload の橋渡し
  - これは Alpamayo の `load_physical_aiavdataset.py` 直読みに対して、
    この repo が `jsonl` record から canonical tensor を復元するための差分
  - `Qwen3.5` 由来ではない
  - `過去 image なし` 由来でもない
- `models/expert_cache_utils.py` `[QWEN3.5-RUNTIME]` `[NOT-NO-PAST-IMAGE]`
  - `clone_prompt_cache_for_expert` など、`AlpamayoR1` と detached `Stage1B` の両方が使う cache utility
  - `models/` に残しているのは、この最小 shared utility を `alpamayo_r1.py` から直接使うため
  - 主因は `Qwen3_5DynamicCache` の `crop()` 非対応を吸う runtime 差分
  - `過去 image なし` はこのファイルの主因ではない

#### 4.2 `Stage1B` 専用実装

- `stage1/stage1b_action_expert.py` `[STAGE-SPLIT]` `[QWEN3.5-RUNTIME]` `[NOT-NO-PAST-IMAGE]`
  - `Stage1B` 用 action expert 本体
  - ファイルが存在する主因は、Alpamayo 本家では `AlpamayoR1` の中に内包されている expert path を、
    `Stage1B` 単体の checkpoint / train / eval / inference から使えるように外出ししていること
  - したがって主ラベルは `[STAGE-SPLIT]`
  - ただし実装の一部には `Qwen3_5DynamicCache` の `crop()` 非対応に合わせた conditioning / cache clone 処理が入っており、
    そこは `[QWEN3.5-RUNTIME]` 差分
  - `過去 image なし` はこのファイルの主因ではない
- `stage1/stage1b_diffusion_adapter.py` `[STAGE-SPLIT]` `[NOT-NO-PAST-IMAGE]`
  - `Stage1B` の expert decoding に必要な diffusion adapter
  - detached path だが、現在は `FlowMatching(x_dims=expert.action_dims)` を使い、
    timestep も expert 側 shared helper で整形している
  - つまり action-space shape / timestep rank の契約は Alpamayo 本家の integrated path に揃えている
  - `loss/sample` 本体は `stage1/stage1b_action_expert.py` の shared helper に寄せており、
    `stage1b_diffusion_adapter.py` 自体は thin wrapper に留めている
  - detached runtime 側の diffusion 組み立ても `diffusion_cfg` 相当の dict から instantiate する
  - 差分の主因は「integrated `AlpamayoR1.diffusion.sample(step_fn=...)` を、
    `Stage1B` 単体 runner から呼べるように切り出していること」であり、
    shape 契約そのものではない
  - `Qwen3.5` 由来ではない
  - `過去 image なし` 由来でもない

### 5. 差分ソース抜粋

以下は `diff -u` で確認した、import path 差以外の実ロジック差分の抜粋。

#### `models/base_model.py`

```diff
@@
-    def tie_weights(self) -> None:
+    def tie_weights(self, *args, **kwargs) -> None:
         """Delegate weight tying to the nested VLM model."""
+        # transformers 5.4 calls tie_weights() from shared PreTrainedModel flows
+        # such as init_weights() and from_pretrained(), sometimes with arguments
+        # like `missing_keys` / `recompute_mapping`.
         if hasattr(self.vlm, "tie_weights"):
-            self.vlm.tie_weights()
+            try:
+                self.vlm.tie_weights(*args, **kwargs)
+            except TypeError:
+                self.vlm.tie_weights()
```

#### `models/alpamayo_r1.py`

```diff
@@
-        attention_mask = torch.zeros(
-            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
-            dtype=torch.float32,
+        attention_mask = torch.zeros(
+            (b_star, prefill_seq_len + n_diffusion_tokens),
+            dtype=torch.long,
             device=device,
         )
         for i in range(b_star):
-            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
-                attention_mask.dtype
-            ).min
+            attention_mask[i, : offset[i]] = 1
+        attention_mask[:, prefill_seq_len:] = 1
@@
-                past_key_values=prompt_cache,
+                past_key_values=expert_prompt_cache,
                 attention_mask=attention_mask,
                 use_cache=True,
                 **forward_kwargs,
             )
-            prompt_cache.crop(prefill_seq_len)
```

#### `test_inference.py`

```diff
@@
-data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
-messages = helper.create_message(data["image_frames"].flatten(0, 1))
+sample = load_reasoning_sample(args.sample_jsonl, args.sample_index)
+wrapper_inputs = build_wrapper_inputs_for_sample(
+    processor=bundle["processor"],
+    sample=sample,
+    history_token_count=int(bundle["history_quantizer"].token_count),
+    device=device,
+)
@@
-pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(...)
+pred_xyz, pred_rot, extra = bundle["wrapper"].sample_trajectories_from_data_with_vlm_rollout(
+    data=wrapper_inputs,
+    ...
+)
```

#### upstream に対応 file が無い repo 固有 core

以下は Alpamayo 側に同名 file が無いので `diff -u` 対象ではない。代わりに、差分の本体になっている source excerpt を残す。

`contract/record_adapter.py`

```python
def canonical_action_tensor_from_record(record: dict) -> torch.Tensor:
    ...
    if "ego_future_xyz" in record and "ego_future_rot" in record:
        future_xyz, future_rot = canonicalize_future_sample_tensors(...)
    else:
        future_xyz, future_rot = derive_future_tensors_from_global_poses(record)
    return canonical_action_tensor_from_tensors(...)
```

`stage1/stage1b_diffusion_adapter.py`

```python
conditioning = expert.prepare_conditioning(
    prompt_cache=prompt_cache,
    prompt_attention_mask=prompt_attention_mask,
)
return flow_matching_sample(
    expert=expert,
    conditioning=conditioning,
    n_steps=self.n_steps,
)
```

`models/expert_cache_utils.py`

```python
def clone_prompt_cache_for_expert(prompt_cache, num_layers: int):
    if hasattr(prompt_cache, "key_cache") and hasattr(prompt_cache, "value_cache"):
        cloned = copy.deepcopy(prompt_cache)
        cloned.key_cache = list(cloned.key_cache[:num_layers])
        cloned.value_cache = list(cloned.value_cache[:num_layers])
        return cloned
```

## 確定した方針

### 1. repo 固有 core / adapter は適切なレイヤに置く

対象:
- `contract/record_adapter.py`
- `models/expert_cache_utils.py`
- `stage1/stage1b_action_expert.py`
- `stage1/stage1b_diffusion_adapter.py`

方針:
- `contract/record_adapter.py` と `models/expert_cache_utils.py` は
  「Alpamayo に無い repo 固有 shared utility」として top-level に残す
- `stage1/stage1b_action_expert.py` と `stage1/stage1b_diffusion_adapter.py` は
  Stage1B 専用実装として `stage1/` に置く
- pure Alpamayo mirror を置く `diffusion/` には戻さない

注意:
- `record_adapter.py` は Alpamayo loader 不在を埋める層であり、`contract/` に置く
- `expert_cache_utils.py` は `alpamayo_r1.py` から直接使う最小 shared utility だけを残す
- `stage1b_diffusion_adapter.py` は Stage1B train/eval/inference が使う stage-specific shared 実装
  - detached adapter ではあるが、action-space shape / timestep rank の契約は
    Alpamayo 本家の integrated path に揃えている
  - 今後比較対象になるのは「shape 契約」ではなく、
    detached adapter という配置そのものが必要かどうか

### 2. wrapper builder は当面 runner に残す

対象:
- `stage2/reasoning_sft/inference/runner.py` の wrapper 組み立て部分

方針:
- Stage1A / Stage1B checkpoint 契約を読んで `AlpamayoR1` を構成する builder は、当面 `runner.py` に残す
- 2つ目の明確な caller が出た時点で shared helper へ切り出す

### 3. runtime shim は Qwen3.5 runtime 差として維持する

対象:
- `models/base_model.py`
- `models/alpamayo_r1.py`

方針:
- `Qwen3.5-0.8B` を使う前提を維持する限り、Alpamayo 純正との差分は runtime shim として受容する
- 差分ゼロは目指さず、「なぜ必要かが説明できる shim だけ残す」を採用する

## 優先順位

優先度の高い残件は、このメモの範囲では基本的に解消済み。
以後は以下を必要に応じて進める。

- `load_physical_aiavdataset.py` 相当を追加して、Alpamayo と同じ demo 入力経路も持つかどうか
- wrapper builder を shared helper に切り出すだけの再利用需要が出るかどうか
- `Qwen3.5` runtime shim が将来の `transformers` / model stack 更新で不要になるかどうか

## 各段階の確認

各段階で最低限確認する。

- `py_compile`
- `stage1.vlm_ce.train --help`
- `stage1.expert_cfm.train --help`
- `stage2.reasoning_sft.inference --help`
- `test_inference.py --help`
- 必要なら single-sample inference smoke

## 完了条件

- pure shared core は top-level `action_space/`, `diffusion/`, `models/`, `geometry/` にある
- stage 配下には train/eval/inference と stage 固有 glue だけが残る
- Alpamayo の同名 file と diff を見たとき、差分は
  - import path
  - repo 固有 glue
  - Qwen3.5 runtime shim
  に限定される
- `stage2` inference の handoff は wrapper 経由で実行される
