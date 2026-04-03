# Ignore Rule Run Entrypoints

このメモは、`ignore_rule_data` の `completion_001` 系 run で、次回どの入口を使うかを固定する。

## Canonical Entrypoints

学習込みで最初から最後まで回す:

- [ignore_rule_run.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/ops/ignore_rule_run.py)
  - `uv run python -m minipamayo_qwen35.ops.ignore_rule_run full`

curve-only eval を回し直す:

- [ignore_rule_run.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/ops/ignore_rule_run.py)
  - `uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage ...`

completion run の監視:

- [ignore_rule_run.py](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/src/minipamayo_qwen35/ops/ignore_rule_run.py)
  - `uv run python -m minipamayo_qwen35.ops.ignore_rule_run watch --attempt-name ...`

eval artifact の閲覧 UI:

- [run_eval_inspector.sh](/media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5/scripts/run_eval_inspector.sh)

## 使い分け

最初から全部やる:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run full
```

`Stage1A` curve eval だけやり直す:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage stage1a
```

`Stage1B` curve eval だけやり直す:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage stage1b
```

`Stage2` curve eval と curve sample inference だけやり直す:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage stage2
```

`Stage1A -> Stage1B -> Stage2` の curve-only eval 一式だけやり直す:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage all
```

`Stage1B` から切り直して full run を再開する:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run full \
  --start-stage stage1b \
  --attempt-name completion_ignore_rule_stage1b_restart_001 \
  --session-name ignore-rule-completion-stage1b-001
```

監視する:

```bash
cd /media/masa/ssd_data/minipamayo/minipamayo-qwen-3-5
uv run python -m minipamayo_qwen35.ops.ignore_rule_run watch \
  --attempt-name completion_ignore_rule_full_001
```

## 役割の切り分け

- `minipamayo_qwen35.ops.ignore_rule_run full` は train chain の正本。`Stage2 train preprocess`, `Stage1A train`, `Stage1B train`, `Stage2 train` を責務に持つ
- 同 subcommand は必要なタイミングで内部的に `eval` subcommand を subprocess で呼び、`Stage1A/Stage1B/Stage2` の curve-only eval attempt を独立 log root に残す
- `minipamayo_qwen35.ops.ignore_rule_run eval` は eval chain の正本。`Stage1A curve eval`, `Stage1B curve eval`, `Stage2 curve eval preprocess`, `Stage2 curve eval`, `Stage2 curve sample inference` を責務に持つ
- `minipamayo_qwen35.ops.ignore_rule_run watch` は `run.status.json`, `monitor.alert`, `master.log tail` を表示するだけで、学習や評価は起動しない

## 過去の経路

- 初期の completion run では `scripts/` 配下の attempt 固定 shell が正本だった
- `Stage1A` curve eval の 1 回目 rerun は module entrypoint 直叩きで回した
- これらは過去 run の経路であり、今後の canonical entrypoint ではない

## 判断ルール

- train を含むなら `uv run python -m minipamayo_qwen35.ops.ignore_rule_run full`
- eval だけなら `uv run python -m minipamayo_qwen35.ops.ignore_rule_run eval --target-stage ...`
- watch だけなら `uv run python -m minipamayo_qwen35.ops.ignore_rule_run watch --attempt-name ...`
- `python -m minipamayo_qwen35.stage...` を直接叩くのは debugging 時だけにする
