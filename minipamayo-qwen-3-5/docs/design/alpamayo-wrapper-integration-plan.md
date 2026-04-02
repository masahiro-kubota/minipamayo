# Alpamayo Wrapper Integration Plan

## 目的

- `related_repos/alpamayo` を実質の正本として扱う。
- `config.py` と `models/alpamayo_r1.py` を `minipamayo-qwen-3-5` に持ち込み、`stage2` の handoff 実装を wrapper 側へ寄せる。
- `record_adapter.py` は吸収せず、repo 固有の record/JSONL adapter として残す。
- `Qwen3.5-0.8B` 差分は許容するが、差分は wrapper 周辺に閉じ込める。

## ここまでで解消したこと

- `config.py` は移植済み
- `models/alpamayo_r1.py` は移植済み
- `stage2/reasoning_sft/inference/runner.py` は manual handoff ではなく wrapper を組み立てて呼ぶ形に変更済み
- `Qwen3.5-0.8B` 差分は `pretrained_modules["vlm"]` 経路で吸収する実装にした
- Stage1B metadata には `action_space_cfg` 経由で action normalization stats を保存するようにした

## いま残っている論点

### 1. wrapper builder の最終配置

- いまは `stage2/reasoning_sft/inference/runner.py` の中で
  - Stage1A checkpoint から tokenizer/history/token registry を復元
  - Stage1B checkpoint から expert/action-space config を復元
  - `AlpamayoR1Config` を組み立てる
  形になっている。
- これを今後 shared helper に出すか、runner 固有の glue のままにするかを決める必要がある。

### 2. GPU が空いた時の single-sample smoke

- いま通っているのは CPU-only の
  - `py_compile`
  - import
  - `--help`
  まで。
- 実 checkpoint を読んで
  - wrapper 構築
  - reasoning rollout
  - trajectory sampling
  まで通るかは GPU 空き後に 1 sample で確認する。

### 3. `record_adapter.py` は残す

- これは JSONL record 契約を扱う repo 固有層なので、wrapper に吸収しない。
- 今後も
  - record から history/future tensor を読む
  - single-traj-group shape を整える
  - eval/inference payload に戻す
  役割に限定する。

## ここではまだやらない

- `models/alpamayo_r1.py` を train path まで全面導入すること
- `stage1/expert_cfm/train` を wrapper ベースに組み直すこと
- `record_adapter.py` の吸収
- `Qwen3.5-0.8B` 差を完全に消すこと

## 完了条件

- `config.py` と `models/alpamayo_r1.py` が repo に存在する
- `stage2` inference が manual handoff ではなく wrapper 経由になる
- `record_adapter.py` は残す
- 新しい Stage1B checkpoint metadata だけで wrapper sampling に必要な action-space stats が復元できる
- GPU で single-sample wrapper inference smoke が通る
