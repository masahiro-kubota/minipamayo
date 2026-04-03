from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import Stage1JsonlDataset, stage1_collate


def build_jsonl_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def build_jsonl_train_val_dataloaders(
    *,
    dataset_ctor,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    train_jsonl,
    val_jsonl,
    train_max_samples: int,
    val_max_samples: int,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    require_validation_split: bool,
    dataset_ctor_kwargs_train: dict[str, Any] | None = None,
    dataset_ctor_kwargs_val: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = dataset_ctor(
        train_jsonl,
        max_samples=train_max_samples,
        **(dataset_ctor_kwargs_train or {}),
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if val_jsonl:
        val_dataset = dataset_ctor(
            val_jsonl,
            max_samples=val_max_samples,
            **(dataset_ctor_kwargs_val or {}),
        )
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

    train_loader = build_jsonl_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_jsonl_dataloader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
    )


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
    return build_jsonl_train_val_dataloaders(
        dataset_ctor=Stage1JsonlDataset,
        collate_fn=stage1_collate,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        train_max_samples=max_samples,
        val_max_samples=max_samples,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        require_validation_split=require_validation_split,
    )
