from __future__ import annotations

import json
from pathlib import Path

import pytest

from minipamayo_qwen35.stage3.post_training.dataset import (
    Stage3PostTrainingDataset,
    build_stage3_train_val_dataloaders,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _record(sample_id: str) -> dict:
    rot0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rot1 = [[0.99, -0.1, 0.0], [0.1, 0.99, 0.0], [0.0, 0.0, 1.0]]
    return {
        "sample_id": sample_id,
        "image_path": f"{sample_id}.png",
        "action": [[0.1, 0.2], [0.3, 0.4]],
        "v0": 3.0,
        "gt_waypoints": [[0.0, 0.0], [1.0, 0.5]],
        "dt": 0.1,
        "ego_history_xyz": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "ego_history_rot": [rot0, rot1],
        "ego_future_xyz": [[1.5, 0.1, 0.0], [2.0, 0.2, 0.0]],
        "ego_future_rot": [rot0, rot1],
        "reasoning_text": f"reason {sample_id}",
        "command": "go",
    }


def test_stage3_dataset_applies_manifest_weights_and_disagreement_scores(tmp_path: Path) -> None:
    train_jsonl = _write_jsonl(tmp_path / "train.jsonl", [_record("a"), _record("b"), _record("c")])
    manifest_jsonl = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"sample_id": "b", "weight": 1.5, "disagreement_score": 0.2},
            {"sample_id": "a", "weight": 0.7, "disagreement_score": 0.9},
        ],
    )

    dataset = Stage3PostTrainingDataset(train_jsonl, manifest_jsonl=manifest_jsonl)

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["sample_id"] == "b"
    assert float(first["sample_weight"]) == pytest.approx(1.5)
    assert first["disagreement_score"] == 0.2
    assert second["sample_id"] == "a"
    assert float(second["sample_weight"]) == pytest.approx(0.7)
    assert second["disagreement_score"] == 0.9


def test_stage3_train_val_dataloaders_keep_manifest_filtered_train_and_split(tmp_path: Path) -> None:
    train_jsonl = _write_jsonl(tmp_path / "train.jsonl", [_record("a"), _record("b"), _record("c")])
    manifest_jsonl = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"sample_id": "a", "weight": 1.1, "disagreement_score": 0.4},
            {"sample_id": "c", "weight": 0.9, "disagreement_score": 0.8},
        ],
    )

    train_loader, val_loader, train_size, val_size = build_stage3_train_val_dataloaders(
        train_jsonl=str(train_jsonl),
        val_jsonl=None,
        manifest_jsonl=str(manifest_jsonl),
        max_samples=0,
        val_fraction=0.5,
        batch_size=1,
        num_workers=0,
        seed=7,
    )

    assert train_size == 1
    assert val_size == 1
    assert val_loader is not None
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch["sample_weight"].shape[0] == 1
    assert val_batch["sample_weight"].shape[0] == 1
    assert "disagreement_score" in train_batch
    assert "disagreement_score" in val_batch


def test_stage3_train_val_dataloaders_do_not_apply_manifest_to_explicit_val_jsonl(tmp_path: Path) -> None:
    train_jsonl = _write_jsonl(tmp_path / "train.jsonl", [_record("a"), _record("b"), _record("c")])
    val_jsonl = _write_jsonl(tmp_path / "val.jsonl", [_record("val_only")])
    manifest_jsonl = _write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"sample_id": "a", "weight": 1.1, "disagreement_score": 0.4},
            {"sample_id": "c", "weight": 0.9, "disagreement_score": 0.8},
        ],
    )

    train_loader, val_loader, train_size, val_size = build_stage3_train_val_dataloaders(
        train_jsonl=str(train_jsonl),
        val_jsonl=str(val_jsonl),
        manifest_jsonl=str(manifest_jsonl),
        max_samples=0,
        val_fraction=0.5,
        batch_size=1,
        num_workers=0,
        seed=7,
    )

    assert train_size == 2
    assert val_size == 1
    assert val_loader is not None
    val_batch = next(iter(val_loader))
    assert float(val_batch["sample_weight"][0]) == pytest.approx(1.0)
    assert "disagreement_score" not in val_batch
