"""Shared JSONL path normalization and loading helpers."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_jsonl_paths(
    jsonl_path: str | Path | list[str] | list[Path],
    *,
    dataset_name: str,
) -> list[Path]:
    if isinstance(jsonl_path, str | Path):
        raw_paths = [Path(jsonl_path)]
    elif isinstance(jsonl_path, list) and jsonl_path:
        raw_paths = [Path(path) for path in jsonl_path]
    else:
        raise RuntimeError(f"{dataset_name} requires one or more JSONL paths.")

    normalized_paths: list[Path] = []
    for path in raw_paths:
        resolved_path = path.resolve()
        if not resolved_path.exists():
            raise RuntimeError(f"{dataset_name} JSONL does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise RuntimeError(f"{dataset_name} JSONL path must be a file: {resolved_path}")
        normalized_paths.append(resolved_path)
    return normalized_paths


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

