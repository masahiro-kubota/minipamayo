"""Small JSONL dataset reader for Stage 1 saved records."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...contract.history_tokens import canonicalize_history_sample_tensors
from ...action_space.record_adapter import (
    derive_future_tensors_from_global_poses,
    saved_action_tensor_from_record,
)


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


class Stage1JsonlDataset(Dataset):
    """Returns metadata and saved canonical labels; image decoding is done at training time."""

    def __init__(self, jsonl_path: str | Path | list[str] | list[Path], max_samples: int = 0):
        self.jsonl_paths = normalize_jsonl_paths(jsonl_path, dataset_name="Stage1JsonlDataset")
        if len(self.jsonl_paths) == 1:
            self.jsonl_path = self.jsonl_paths[0]

        records: list[dict] = []
        record_root_dirs: list[Path] = []
        for path in self.jsonl_paths:
            source_records = read_jsonl(path)
            records.extend(source_records)
            record_root_dirs.extend([path.parent] * len(source_records))

        if max_samples > 0:
            records = records[:max_samples]
            record_root_dirs = record_root_dirs[:max_samples]

        self.records = records
        self.record_root_dirs = record_root_dirs

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        root_dir = self.record_root_dirs[index]
        required_keys = [
            "sample_id",
            "image_path",
            "action",
            "v0",
            "dt",
            "gt_waypoints",
            "command",
            "ego_history_xyz",
            "ego_history_rot",
        ]
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            raise RuntimeError(
                "Stage 1 dataset record is missing canonical fields:\n" + "\n".join(missing_keys)
            )
        ego_history_xyz, ego_history_rot = canonicalize_history_sample_tensors(
            torch.tensor(record["ego_history_xyz"], dtype=torch.float32),
            torch.tensor(record["ego_history_rot"], dtype=torch.float32),
        )
        if "ego_future_xyz" in record and "ego_future_rot" in record:
            ego_future_xyz, ego_future_rot = canonicalize_history_sample_tensors(
                torch.tensor(record["ego_future_xyz"], dtype=torch.float32),
                torch.tensor(record["ego_future_rot"], dtype=torch.float32),
            )
        else:
            ego_future_xyz, ego_future_rot = derive_future_tensors_from_global_poses(record)
        return {
            "sample_id": record["sample_id"],
            "image_path": str(root_dir / record["image_path"]),
            "action": saved_action_tensor_from_record(record),
            "v0": torch.tensor(record["v0"], dtype=torch.float32),
            "dt": torch.tensor(record["dt"], dtype=torch.float32),
            "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "command": record["command"],
        }
