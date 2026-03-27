"""Small JSONL dataset reader for Stage 1."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


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
    """Returns metadata and labels; image decoding is done at training time."""

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
        return {
            "sample_id": record["sample_id"],
            "image_path": str(self.root_dir / record["image_path"]),
            "action": torch.tensor(record["action"], dtype=torch.float32),
            "v0": torch.tensor(record["v0"], dtype=torch.float32),
            "gt_waypoints": torch.tensor(record["gt_waypoints"], dtype=torch.float32),
            "command": record.get("command", "lanefollow"),
        }
