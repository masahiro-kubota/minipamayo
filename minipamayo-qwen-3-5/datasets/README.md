# Datasets Layout

This repo keeps dataset files out of `src/` so code and data stay separate.

Recommended layout:

```text
datasets/
  raw/
    mcap/
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
        images/

  splits/
    stage1/
      train.jsonl
      val.jsonl

  manifests/
    stage1/
      ignore_rule_data.json
```

Usage:

- Place raw CARLA telemetry episodes under `datasets/raw/mcap/`.
- Extract Stage 1 training data under `datasets/processed/stage1/`.
- Keep split JSONL files under `datasets/splits/stage1/` if you generate them.

Single episode example:

```bash
PYTHONPATH=src python -m minipamayo_qwen35.data.mcap_stage1_extractor \
  --episode-dir datasets/raw/mcap/<episode_id> \
  --output-dir datasets/processed/stage1/<dataset_name>
```

Batch manifest example:

```bash
PYTHONPATH=src python -m minipamayo_qwen35.data.mcap_stage1_extractor \
  --manifest-json datasets/manifests/stage1/ignore_rule_data.json
```

Manifest format:

- top-level `path_base` defaults to `datasets_root`
- `episode_dir + output_dir`
- or `mcap_paths + output_dir`
- `summary_path` is optional when `mcap_paths` all live under the same `telemetry/` directory

Available `path_base` values:

- `datasets_root`
- `project_root`
- `manifest_dir`

See `datasets/manifests/stage1/ignore_rule_data.json` for a concrete example.
