"""Stage 2 dataset adapters built on top-level reasoning dataset contracts."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ...reasoning.dataset import ReasoningSftJsonlDataset, reasoning_sft_collate


def build_reasoning_sft_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
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
    train_dataset = ReasoningSftJsonlDataset(train_jsonl, max_samples=max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if val_jsonl:
        val_dataset = ReasoningSftJsonlDataset(val_jsonl)
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty.")
    elif len(train_dataset) >= 2 and val_fraction > 0:
        val_size = max(1, int(round(len(train_dataset) * val_fraction)))
        val_size = min(val_size, len(train_dataset) - 1)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        val_dataset = None

    train_loader = build_reasoning_sft_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_reasoning_sft_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
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
