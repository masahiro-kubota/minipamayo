"""Stage 1 dataset contracts plus JSONL helper compatibility exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..contract.history_tokens import canonicalize_history_sample_tensors
from ..contract.record_adapter import (
    derive_future_tensors_from_global_poses,
    saved_action_tensor_from_record,
)
from ..utils.jsonl import normalize_jsonl_paths, read_jsonl

CANONICAL_SHARED_RECORD_KEYS = (
    "sample_id",
    "image_path",
    "action",
    "v0",
    "gt_waypoints",
    "ego_history_xyz",
    "ego_history_rot",
)


def load_jsonl_record_bundle(
    jsonl_path: str | Path | list[str] | list[Path],
    *,
    dataset_name: str,
    max_samples: int = 0,
) -> tuple[list[Path], list[dict[str, Any]], list[Path]]:
    jsonl_paths = normalize_jsonl_paths(jsonl_path, dataset_name=dataset_name)
    records: list[dict[str, Any]] = []
    record_root_dirs: list[Path] = []
    for path in jsonl_paths:
        source_records = read_jsonl(path)
        records.extend(source_records)
        record_root_dirs.extend([path.parent] * len(source_records))

    if max_samples > 0:
        records = records[:max_samples]
        record_root_dirs = record_root_dirs[:max_samples]

    return jsonl_paths, records, record_root_dirs


def require_record_keys(
    record: dict[str, Any],
    required_keys: list[str] | tuple[str, ...],
    *,
    error_message: str,
) -> None:
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise RuntimeError(error_message + "\n" + "\n".join(missing_keys))


def build_canonical_record_common_sample(
    record: dict[str, Any],
    *,
    root_dir: Path,
) -> dict[str, Any]:
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
        "image_path": str(root_dir / str(record["image_path"])),
        "action": saved_action_tensor_from_record(record),
        "v0": torch.tensor(record["v0"], dtype=torch.float32),
        "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": ego_future_rot,
    }


class Stage1JsonlDataset(Dataset):
    """Returns metadata and saved canonical labels; image decoding is done at training time."""

    def __init__(self, jsonl_path: str | Path | list[str] | list[Path], max_samples: int = 0):
        self.jsonl_paths, self.records, self.record_root_dirs = load_jsonl_record_bundle(
            jsonl_path,
            dataset_name="Stage1JsonlDataset",
            max_samples=max_samples,
        )
        if len(self.jsonl_paths) == 1:
            self.jsonl_path = self.jsonl_paths[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        root_dir = self.record_root_dirs[index]
        require_record_keys(
            record,
            [*CANONICAL_SHARED_RECORD_KEYS, "dt", "command"],
            error_message="Stage 1 dataset record is missing canonical fields:",
        )
        sample = build_canonical_record_common_sample(record, root_dir=root_dir)
        sample["dt"] = torch.tensor(record["dt"], dtype=torch.float32)
        sample["command"] = record["command"]
        return sample


def stage1_collate(samples: list[dict]) -> dict:
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "action": torch.stack([sample["action"] for sample in samples], dim=0),
        "v0": torch.stack([sample["v0"] for sample in samples], dim=0),
        "dt": torch.stack([sample["dt"] for sample in samples], dim=0),
        "gt_waypoints": torch.stack([sample["gt_waypoints"] for sample in samples], dim=0),
        "ego_history_xyz": torch.stack([sample["ego_history_xyz"] for sample in samples], dim=0),
        "ego_history_rot": torch.stack([sample["ego_history_rot"] for sample in samples], dim=0),
        "command": [sample["command"] for sample in samples],
    }
