"""Canonical reasoning-SFT dataset contract.

This dataset is intentionally separate from the old synthetic reasoning path.
Canonical Stage 2 and Stage 3 expect reasoning supervision to be provided by
the dataset itself via `reasoning_text`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from ...stage1.data.dataset import normalize_jsonl_paths, read_jsonl
from ...stage1.data.canonical_action import (
    canonical_action_tensor_from_tensors,
    derive_future_tensors_from_global_poses,
)
from ...stage1.tokenization.history import canonicalize_history_sample_tensors

if TYPE_CHECKING:
    from pathlib import Path


class ReasoningSftJsonlDataset(Dataset):
    """Stage 1 JSONL records plus provided reasoning supervision."""

    def __init__(self, jsonl_path: str | Path | list[str] | list[Path], max_samples: int = 0):
        self.jsonl_paths = normalize_jsonl_paths(
            jsonl_path,
            dataset_name="ReasoningSftJsonlDataset",
        )
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
            "v0",
            "gt_waypoints",
            "dt",
            "ego_history_xyz",
            "ego_history_rot",
            "reasoning_text",
        ]
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            raise RuntimeError(
                "Reasoning SFT dataset record is missing canonical fields:\n"
                + "\n".join(missing_keys)
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
        canonical_action = canonical_action_tensor_from_tensors(
            history_xyz=ego_history_xyz,
            history_rot=ego_history_rot,
            future_xyz=ego_future_xyz,
            future_rot=ego_future_rot,
            dt=float(record["dt"]),
        )
        sample = {
            "sample_id": str(record["sample_id"]),
            "image_path": str(root_dir / str(record["image_path"])),
            "action": canonical_action,
            "v0": torch.tensor(record["v0"], dtype=torch.float32),
            "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
            "dt": float(record["dt"]),
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "reasoning_text": str(record["reasoning_text"]),
        }
        if "command" in record:
            sample["command"] = str(record["command"])
        if "planner_state" in record:
            sample["planner_state"] = str(record["planner_state"])
        if "decision_longitudinal" in record:
            sample["decision_longitudinal"] = str(record["decision_longitudinal"])
        if "decision_lateral" in record:
            sample["decision_lateral"] = str(record["decision_lateral"])
        return sample


def reasoning_sft_collate(samples: list[dict]) -> dict:
    batch = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "action": torch.stack([sample["action"] for sample in samples], dim=0),
        "v0": torch.stack([sample["v0"] for sample in samples], dim=0),
        "gt_waypoints": torch.stack([sample["gt_waypoints"] for sample in samples], dim=0),
        "ego_history_xyz": torch.stack([sample["ego_history_xyz"] for sample in samples], dim=0),
        "ego_history_rot": torch.stack([sample["ego_history_rot"] for sample in samples], dim=0),
        "ego_future_xyz": torch.stack([sample["ego_future_xyz"] for sample in samples], dim=0),
        "ego_future_rot": torch.stack([sample["ego_future_rot"] for sample in samples], dim=0),
        "dt": [sample["dt"] for sample in samples],
        "reasoning_text": [sample["reasoning_text"] for sample in samples],
    }
    optional_keys = [
        "command",
        "planner_state",
        "decision_longitudinal",
        "decision_lateral",
    ]
    for key in optional_keys:
        if any(key in sample for sample in samples):
            batch[key] = [sample.get(key, "") for sample in samples]
    return batch
