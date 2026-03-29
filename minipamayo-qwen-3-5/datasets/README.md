# Datasets Layout

This repo keeps dataset files out of `src/` so code and data stay separate.

Recommended layout:

```text
datasets/
  raw/
    <collection_name>/
      <episode_id>/
        summary.json
        telemetry/
          index.json
          segment_0000.mcap
          segment_0001.mcap

  processed/
    stage1/
      <dataset_name>/
        samples.jsonl
        extract_summary.json
        run_config.json
        images/

  splits/
    stage1/
      train.jsonl
      val.jsonl
```

Usage:

- Place raw CARLA telemetry episodes under `datasets/raw/<collection_name>/`.
- Extract Stage 1 training data under `datasets/processed/stage1/`.
- Keep split JSONL files under `datasets/splits/stage1/` if you generate them.
- Keep extraction configs under `configs/stage1/data/`.
- Stage 1 extraction is config-only. Pass `--config-json` and keep all extraction settings inside the config.

Batch extraction example:

```bash
uv run python -m minipamayo_qwen35.stage1.preprocess \
  --config-json configs/stage1/data/ignore_rule_data.json
```

Progress logging:

- progress events are emitted to `stderr`
- the final extraction summary remains JSON on `stdout`
- set `"log_every": 0` in the config `extract` block to disable periodic progress logs

Extraction config format:

- top-level `path_base` defaults to `project_root`
- top-level `extract` holds shared extraction settings such as `k`, `dt`, `sample_stride`, `max_samples`, and `log_every`
- `episode_dir + output_dir`
- or `mcap_paths + output_dir`
- `summary_path` is optional when `mcap_paths` all live under the same `telemetry/` directory
- CLI overrides are intentionally disabled so the config remains the full execution record

Available `path_base` values:

- `project_root`
- `datasets_root`
- `config_dir`

See `configs/stage1/data/ignore_rule_data.json` for a concrete example.
