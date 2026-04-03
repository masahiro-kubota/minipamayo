"""Stage 2 dataset adapters built on top-level reasoning dataset contracts."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from ...reasoning.dataset import ReasoningSftJsonlDataset, reasoning_sft_collate
from ...stage1.stage1_train_data import build_jsonl_dataloader, build_jsonl_train_val_dataloaders


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
