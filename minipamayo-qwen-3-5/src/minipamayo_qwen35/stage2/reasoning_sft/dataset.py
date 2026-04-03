"""Canonical reasoning-SFT dataset contract.

This dataset is intentionally separate from the old synthetic reasoning path.
Canonical Stage 2 and Stage 3 expect reasoning supervision to be provided by
the dataset itself via `reasoning_text`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from ...stage1.dataset import (
    CANONICAL_SHARED_RECORD_KEYS,
    build_canonical_record_common_sample,
    load_jsonl_record_bundle,
    require_record_keys,
)
from ...stage1.stage1_train_data import build_jsonl_dataloader, build_jsonl_train_val_dataloaders

if TYPE_CHECKING:
    from pathlib import Path


class ReasoningSftJsonlDataset(Dataset):
    """Stage 1 JSONL records with saved canonical actions plus reasoning supervision."""

    def __init__(self, jsonl_path: str | Path | list[str] | list[Path], max_samples: int = 0):
        self.jsonl_paths, self.records, self.record_root_dirs = load_jsonl_record_bundle(
            jsonl_path,
            dataset_name="ReasoningSftJsonlDataset",
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
            [*CANONICAL_SHARED_RECORD_KEYS, "dt", "reasoning_text"],
            error_message="Reasoning SFT dataset record is missing canonical fields:",
        )
        sample = build_canonical_record_common_sample(record, root_dir=root_dir)
        sample["sample_id"] = str(sample["sample_id"])
        sample["dt"] = float(record["dt"])
        sample["reasoning_text"] = str(record["reasoning_text"])
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


def build_reasoning_sft_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return build_jsonl_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=reasoning_sft_collate,
    )


def build_stage2_train_val_dataloaders(
    *,
    train_jsonl: str | Path | list[str] | list[Path],
    val_jsonl: str | Path | list[str] | list[Path] | None,
    max_samples: int,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    return build_jsonl_train_val_dataloaders(
        dataset_ctor=ReasoningSftJsonlDataset,
        collate_fn=reasoning_sft_collate,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        train_max_samples=max_samples,
        val_max_samples=0,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        require_validation_split=False,
    )


def build_stage2_handoff_probe_dataset(
    *,
    handoff_probe_jsonl: list[str] | None,
    handoff_probe_max_per_jsonl: int,
):
    if not handoff_probe_jsonl:
        return None
    if handoff_probe_max_per_jsonl <= 0:
        raise RuntimeError(
            "`handoff_probe_max_per_jsonl` must be > 0 when `handoff_probe_jsonl` is set."
        )
    probe_samples = []
    for jsonl_path in handoff_probe_jsonl:
        source_dataset = ReasoningSftJsonlDataset(jsonl_path)
        source_count = min(handoff_probe_max_per_jsonl, len(source_dataset))
        for idx in range(source_count):
            probe_samples.append(source_dataset[idx])
    if not probe_samples:
        raise RuntimeError("Explicit handoff probe dataset resolved to zero samples.")
    return probe_samples


def load_reasoning_sample(sample_jsonl: str | Path, sample_index: int) -> dict:
    dataset = ReasoningSftJsonlDataset(sample_jsonl)
    if sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index` {sample_index} is out of range for dataset size {len(dataset)}."
        )
    return dataset[sample_index]
