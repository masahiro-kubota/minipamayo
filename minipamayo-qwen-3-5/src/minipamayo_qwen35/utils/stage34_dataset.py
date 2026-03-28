"""Shared JSONL dataset for the Qwen3.5 Stage 2-4 path."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..sequence.stage3_builder import build_reasoning_text, infer_driving_decision
from ..data.stage1_dataset import read_jsonl


class Stage34JsonlDataset(Dataset):
    """Stage 1 JSONL records with synthetic reasoning targets."""

    def __init__(self, jsonl_path: str | Path, max_samples: int = 0):
        records = read_jsonl(jsonl_path)
        if max_samples > 0:
            records = records[:max_samples]
        self.jsonl_path = Path(jsonl_path)
        self.root_dir = self.jsonl_path.parent
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
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
                "Stage 2-4 dataset record is missing canonical fields:\n"
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
            "image_path": str(self.root_dir / str(record["image_path"])),
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


def stage34_collate(samples: list[dict]) -> dict:
    """Collate function shared across Stage 2-4."""

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
