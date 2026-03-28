"""Small JSONL dataset reader for Stage 1."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..tokenization.history import canonicalize_history_sample_tensors


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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


def _rotation_matrix_from_yaw(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _derive_future_tensors_from_global_poses(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "ego_pose" not in record or "future_poses_global" not in record:
        raise RuntimeError(
            "Stage 1 record is missing `ego_pose` or `future_poses_global`, "
            "which are required to derive canonical future trajectory tensors."
        )
    ego_pose = record["ego_pose"]
    future_poses = record["future_poses_global"]
    if not isinstance(future_poses, list) or not future_poses:
        raise RuntimeError("Stage 1 record has invalid `future_poses_global`.")
    if "gt_waypoints" not in record or not isinstance(record["gt_waypoints"], list):
        raise RuntimeError("Stage 1 record is missing canonical `gt_waypoints` for future alignment.")
    target_steps = len(record["gt_waypoints"])
    if target_steps <= 0:
        raise RuntimeError("Stage 1 record has empty `gt_waypoints`.")
    future_poses = future_poses[:target_steps]

    origin_x = float(ego_pose["x"])
    origin_y = float(ego_pose["y"])
    origin_yaw = math.radians(float(ego_pose["yaw_deg"]))
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)

    future_xyz = np.zeros((len(future_poses), 3), dtype=np.float32)
    future_rot = np.zeros((len(future_poses), 3, 3), dtype=np.float32)
    for pose_idx, pose in enumerate(future_poses):
        dx = float(pose["x"]) - origin_x
        dy = float(pose["y"]) - origin_y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        future_xyz[pose_idx, 0] = local_x
        future_xyz[pose_idx, 1] = local_y

        yaw_rad = math.radians(float(pose["yaw_deg"]))
        local_yaw = math.atan2(math.sin(yaw_rad - origin_yaw), math.cos(yaw_rad - origin_yaw))
        future_rot[pose_idx] = _rotation_matrix_from_yaw(local_yaw)

    return (
        torch.from_numpy(future_xyz).unsqueeze(0),
        torch.from_numpy(future_rot).unsqueeze(0),
    )


class Stage1JsonlDataset(Dataset):
    """Returns metadata and labels; image decoding is done at training time."""

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
            ego_future_xyz, ego_future_rot = _derive_future_tensors_from_global_poses(record)
        return {
            "sample_id": record["sample_id"],
            "image_path": str(root_dir / record["image_path"]),
            "action": torch.tensor(record["action"], dtype=torch.float32),
            "v0": torch.tensor(record["v0"], dtype=torch.float32),
            "dt": torch.tensor(record["dt"], dtype=torch.float32),
            "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
            "command": record["command"],
        }
