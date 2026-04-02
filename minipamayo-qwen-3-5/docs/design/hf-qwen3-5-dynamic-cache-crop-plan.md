# Qwen3_5DynamicCache `crop()` Contribution Plan

このメモは、`huggingface/transformers` の `Qwen3_5DynamicCache` で `crop()` を使えるようにするための実装方針を整理したものです。

## 動機

- `minipamayo-qwen-3-5` では、`stage2` wrapper の expert path が diffusion の各 step ごとに `past_key_values` を再利用する。
- Alpamayo 本家は `prompt_cache.crop(prefill_seq_len)` を前提に、各 step の後で future token 分を切り戻している。
- しかし `transformers==5.4.0` の `Qwen3_5DynamicCache` は `crop()` を持たない。
- そのため現在の `minipamayo-qwen-3-5` では、`hasattr(prompt_cache, "crop")` で guard する shim が残っている。
- これは `Qwen3.5-0.8B` 固有の runtime 差であり、Alpamayo と差分を生む。

## 実装対象

modular 側では `Qwen3_5DynamicCache` は `Qwen3NextDynamicCache` の subclass なので、実装自体は親の `Qwen3NextDynamicCache` に入れる。

つまり:

- 実装点:
  - `src/transformers/models/qwen3_next/modular_qwen3_next.py`
- 効果の対象:
  - `Qwen3NextDynamicCache`
  - `Qwen3_5DynamicCache`

## 目標

- `Qwen3NextDynamicCache` に `crop(max_length: int)` を追加し、`Qwen3_5DynamicCache` でも使えるようにする。
- `DynamicCache.crop()` と同様に、「full-attention token cache の seq_len を切り戻す」という API を提供する。
- `minipamayo-qwen-3-5` 側では `hasattr(prompt_cache, "crop")` guard を不要にできる状態を目指す。

## 実装方針

### 1. full-attention cache だけを crop 対象にする

`Qwen3_5DynamicCache` は 2 系統の state を持つ。

- full attention:
  - `key_cache`
  - `value_cache`
- linear attention:
  - `conv_states`
  - `recurrent_states`

今回 `crop()` で切り戻したいのは、`seq_len` を持つ full-attention 側だけ。

具体的には:

- `key_cache[layer_idx] = key_cache[layer_idx][..., :max_length, :]`
- `value_cache[layer_idx] = value_cache[layer_idx][..., :max_length, :]`

を full-attention layer に対して行う。

### 2. linear-attention state はそのままにする

`conv_states` / `recurrent_states` は token length をそのまま持つ構造ではない。
そのため今回の `crop()` では変更しない。

理由:

- `prompt_cache.crop(prefill_seq_len)` の用途は、future token を expert に追加した結果伸びた full-attention cache を元に戻すこと
- linear-attention state はこの `seq_len` 切り戻し API の責務ではない
- まずは Alpamayo が期待する token-cache の意味を満たす最小実装に留める

### 3. API は `DynamicCache.crop()` に寄せる

`Qwen3_5DynamicCache` は `DynamicCache` の subclass ではないが、呼び出し側の期待は同じにしたい。

そのため:

- メソッド名は `crop`
- 引数は `max_length: int`
- `max_length` まで truncate する

という互換 API にする。

### 4. 負の長さや未初期化 layer は安全に扱う

実装時の扱い:

- `max_length < 0` は `ValueError`
- `key_cache[layer_idx] is None` の layer は skip
- full-attention 以外の layer も skip

## 期待する効果

- `minipamayo-qwen-3-5` の `models/alpamayo_r1.py` にある `hasattr(prompt_cache, "crop")` guard を外せる
- `Qwen3.5-0.8B` でも Alpamayo の cache-handling 前提に近づく
- `past_key_values` を使う expert / diffusion 系の実装で、Qwen3.5 系 cache API の一貫性が上がる

## テスト方針

最低限これを確認する。

1. `Qwen3_5DynamicCache` を作る
2. full-attention layer に対して `update()` で token を追加する
3. `get_seq_length()` が伸びることを確認する
4. `crop(max_length)` を呼ぶ
5. `get_seq_length()` が `max_length` に戻ることを確認する
6. `conv_states` / `recurrent_states` は変化しないことを確認する

必要なら追加で:

- `max_length == current_length`
- `max_length > current_length`
- `max_length < 0`

も見る。

## PR のスコープ

この contribution では以下に限定する。

- `Qwen3_5DynamicCache.crop()` の追加
- それに対応する unit test の追加

この PR ではやらない:

- `flash_attention_2` の 4D additive mask 対応
- `minipamayo-qwen-3-5` 側の shim 削除
- `Qwen3.5` の cache 全体の設計変更

## `minipamayo-qwen-3-5` 側の後続作業

upstream で `crop()` が使えるようになったら、こちらでやることはこれです。

1. `transformers` fork を使って `stage2` wrapper inference を再確認する
2. `hasattr(prompt_cache, "crop")` guard を削除する
3. `alpamayo-core-alignment-plan.md` の runtime shim を更新する
