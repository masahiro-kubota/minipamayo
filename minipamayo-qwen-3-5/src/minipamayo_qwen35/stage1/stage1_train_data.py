from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import Stage1JsonlDataset, stage1_collate


def build_stage1_train_val_dataloaders(
    *,
    train_jsonl: list[str],
    val_jsonl: list[str] | None,
    max_samples: int,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    require_validation_split: bool,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = Stage1JsonlDataset(train_jsonl, max_samples=max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if val_jsonl:
        val_dataset = Stage1JsonlDataset(val_jsonl, max_samples=max_samples)
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty.")
    else:
        if not (0.0 < val_fraction < 1.0):
            if require_validation_split:
                raise RuntimeError(
                    "`val_fraction` must be in (0, 1) when `val_jsonl` is not provided."
                )
            val_dataset = None
        elif len(train_dataset) < 2:
            if require_validation_split:
                raise RuntimeError(
                    "Validation split left no training samples. Reduce `val_fraction` or use more data."
                )
            val_dataset = None
        else:
            val_size = max(1, int(round(len(train_dataset) * val_fraction)))
            val_size = min(val_size, len(train_dataset) - 1)
            train_size = len(train_dataset) - val_size
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=generator,
            )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "collate_fn": stage1_collate,
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
    )
