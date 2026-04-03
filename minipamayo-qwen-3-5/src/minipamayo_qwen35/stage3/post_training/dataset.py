"""Canonical Stage 3 dataset contract and dataloader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ...reasoning.dataset import ReasoningSftJsonlDataset, reasoning_sft_collate
from ...stage1.stage1_train_data import build_jsonl_dataloader, build_jsonl_train_val_dataloaders
from ...utils.jsonl import read_jsonl


@dataclass(frozen=True)
class Stage3ManifestEntry:
    """Lightweight curation manifest row."""

    sample_id: str
    weight: float = 1.0
    disagreement_score: float | None = None


def _load_manifest(path: str | Path) -> dict[str, Stage3ManifestEntry]:
    records = read_jsonl(path)
    manifest: dict[str, Stage3ManifestEntry] = {}
    for record in records:
        if "sample_id" not in record:
            raise RuntimeError("Stage 3 manifest rows must define `sample_id`.")
        sample_id = str(record["sample_id"])
        if sample_id in manifest:
            raise RuntimeError(f"Stage 3 manifest contains duplicate sample_id: {sample_id}")
        manifest[sample_id] = Stage3ManifestEntry(
            sample_id=sample_id,
            weight=float(record.get("weight", 1.0)),
            disagreement_score=(
                float(record["disagreement_score"])
                if "disagreement_score" in record and record["disagreement_score"] is not None
                else None
            ),
        )
    return manifest


class Stage3PostTrainingDataset(Dataset):
    """Reasoning dataset with optional Stage 3 curation weights."""

    def __init__(
        self,
        jsonl_path: str | Path | list[str] | list[Path],
        *,
        manifest_jsonl: str | Path | None = None,
        max_samples: int = 0,
    ) -> None:
        base_dataset = ReasoningSftJsonlDataset(jsonl_path, max_samples=max_samples)
        manifest = _load_manifest(manifest_jsonl) if manifest_jsonl else None

        indices: list[int] = []
        weights: list[float] = []
        disagreement_scores: list[float | None] = []
        if manifest is None:
            indices = list(range(len(base_dataset)))
            weights = [1.0] * len(base_dataset)
            disagreement_scores = [None] * len(base_dataset)
        else:
            sample_id_to_index = {
                str(base_dataset.records[idx]["sample_id"]): idx for idx in range(len(base_dataset))
            }
            missing_ids = sorted(set(manifest) - set(sample_id_to_index))
            if missing_ids:
                raise RuntimeError(
                    "Stage 3 manifest references sample_ids that are absent from the dataset:\n"
                    + "\n".join(missing_ids[:20])
                )
            for sample_id, entry in manifest.items():
                indices.append(sample_id_to_index[sample_id])
                weights.append(float(entry.weight))
                disagreement_scores.append(entry.disagreement_score)

        self.base_dataset = base_dataset
        self.indices = indices
        self.weights = weights
        self.disagreement_scores = disagreement_scores
        self.manifest_jsonl = str(Path(manifest_jsonl).resolve()) if manifest_jsonl else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        sample = dict(self.base_dataset[self.indices[index]])
        sample["sample_weight"] = torch.tensor(self.weights[index], dtype=torch.float32)
        disagreement_score = self.disagreement_scores[index]
        if disagreement_score is not None:
            sample["disagreement_score"] = float(disagreement_score)
        return sample


def stage3_post_training_collate(samples: list[dict]) -> dict:
    batch = reasoning_sft_collate(samples)
    batch["sample_weight"] = torch.stack([sample["sample_weight"] for sample in samples], dim=0)
    if any("disagreement_score" in sample for sample in samples):
        batch["disagreement_score"] = torch.tensor(
            [float(sample.get("disagreement_score", 0.0)) for sample in samples],
            dtype=torch.float32,
        )
    return batch


def build_stage3_dataloader(
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
        collate_fn=stage3_post_training_collate,
    )


def build_stage3_train_val_dataloaders(
    *,
    train_jsonl: str | Path | list[str] | list[Path],
    val_jsonl: str | Path | list[str] | list[Path] | None,
    manifest_jsonl: str | Path | None,
    max_samples: int,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None, int, int]:
    return build_jsonl_train_val_dataloaders(
        dataset_ctor=Stage3PostTrainingDataset,
        collate_fn=stage3_post_training_collate,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        train_max_samples=max_samples,
        val_max_samples=max_samples,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        require_validation_split=False,
        dataset_ctor_kwargs_train={"manifest_jsonl": manifest_jsonl},
    )
