"""Synthetic-reasoning JSONL dataset shared by Stage 2 and Stage 3 experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..stage1.dataset import normalize_jsonl_paths, read_jsonl
from .synthetic import build_reasoning_text, infer_driving_decision


class SyntheticReasoningJsonlDataset(Dataset):
    """Stage 1 JSONL records augmented with deterministic synthetic reasoning."""

    def __init__(self, jsonl_path: str | Path | list[str] | list[Path], max_samples: int = 0):
        self.jsonl_paths = normalize_jsonl_paths(
            jsonl_path,
            dataset_name="SyntheticReasoningJsonlDataset",
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
            "action",
            "v0",
            "gt_waypoints",
            "command",
            "planner_state",
            "dt",
        ]
        missing_keys = [key for key in required_keys if key not in record]
        if missing_keys:
            raise RuntimeError(
                "Synthetic reasoning dataset record is missing canonical fields:\n"
                + "\n".join(missing_keys)
            )

        command = str(record["command"])
        planner_state = str(record["planner_state"])
        decision = infer_driving_decision(command, planner_state)
        reasoning_text = build_reasoning_text(
            command=command,
            planner_state=planner_state,
            decision=decision,
        )
        return {
            "sample_id": str(record["sample_id"]),
            "image_path": str(root_dir / str(record["image_path"])),
            "action": torch.tensor(record["action"], dtype=torch.float32),
            "v0": torch.tensor(record["v0"], dtype=torch.float32),
            "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
            "command": command,
            "planner_state": planner_state,
            "dt": float(record["dt"]),
            "reasoning_text": reasoning_text,
            "decision_longitudinal": decision["longitudinal"],
            "decision_lateral": decision["lateral"],
        }


def synthetic_reasoning_collate(samples: list[dict]) -> dict:
    """Collate synthetic reasoning records for Stage 2/3 experiments."""

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "action": torch.stack([sample["action"] for sample in samples], dim=0),
        "v0": torch.stack([sample["v0"] for sample in samples], dim=0),
        "gt_waypoints": torch.stack([sample["gt_waypoints"] for sample in samples], dim=0),
        "command": [sample["command"] for sample in samples],
        "planner_state": [sample["planner_state"] for sample in samples],
        "dt": [sample["dt"] for sample in samples],
        "reasoning_text": [sample["reasoning_text"] for sample in samples],
        "decision_longitudinal": [sample["decision_longitudinal"] for sample in samples],
        "decision_lateral": [sample["decision_lateral"] for sample in samples],
    }
