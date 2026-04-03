"""Load one canonical reasoning JSONL sample for top-level smoke inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .contract.history_tokens import canonicalize_history_sample_tensors
from .contract.record_adapter import (
    canonicalize_future_sample_tensors,
    derive_future_tensors_from_global_poses,
)
from .utils.jsonl import normalize_jsonl_paths, read_jsonl


def _read_sample_record(
    sample_jsonl: str | Path | list[str] | list[Path],
    sample_index: int,
) -> tuple[dict[str, Any], Path]:
    jsonl_paths = normalize_jsonl_paths(
        sample_jsonl,
        dataset_name="load_reasoning_jsonl_sample",
    )
    records: list[dict[str, Any]] = []
    root_dirs: list[Path] = []
    for path in jsonl_paths:
        source_records = read_jsonl(path)
        records.extend(source_records)
        root_dirs.extend([path.parent] * len(source_records))
    if sample_index < 0 or sample_index >= len(records):
        raise RuntimeError(
            f"`sample_index` {sample_index} is out of range for dataset size {len(records)}."
        )
    return records[sample_index], root_dirs[sample_index]


def load_reasoning_jsonl_sample(
    sample_jsonl: str | Path | list[str] | list[Path],
    sample_index: int = 0,
) -> dict[str, Any]:
    record, root_dir = _read_sample_record(sample_jsonl, sample_index)
    required_keys = ["sample_id", "image_path", "ego_history_xyz", "ego_history_rot"]
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise RuntimeError(
            "Reasoning JSONL sample is missing canonical fields:\n" + "\n".join(missing_keys)
        )

    image_path = root_dir / str(record["image_path"])
    with Image.open(image_path) as raw_image:
        rgb_image = raw_image.convert("RGB")
        image_frames = torch.from_numpy(np.array(rgb_image, copy=True)).permute(2, 0, 1).unsqueeze(0)

    ego_history_xyz, ego_history_rot = canonicalize_history_sample_tensors(
        torch.tensor(record["ego_history_xyz"], dtype=torch.float32),
        torch.tensor(record["ego_history_rot"], dtype=torch.float32),
    )
    if "ego_future_xyz" in record and "ego_future_rot" in record:
        ego_future_xyz, ego_future_rot = canonicalize_future_sample_tensors(
            torch.tensor(record["ego_future_xyz"], dtype=torch.float32),
            torch.tensor(record["ego_future_rot"], dtype=torch.float32),
        )
    else:
        ego_future_xyz, ego_future_rot = derive_future_tensors_from_global_poses(record)

    return {
        "sample_id": str(record["sample_id"]),
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz.unsqueeze(0),
        "ego_history_rot": ego_history_rot.unsqueeze(0),
        "ego_future_xyz": ego_future_xyz.unsqueeze(0),
        "ego_future_rot": ego_future_rot.unsqueeze(0),
    }
