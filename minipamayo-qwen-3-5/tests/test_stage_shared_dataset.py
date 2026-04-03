from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from minipamayo_qwen35.stage1.dataset import Stage1JsonlDataset
from minipamayo_qwen35.stage1.stage1_train_data import (
    build_jsonl_train_val_dataloaders,
    build_stage1_train_val_dataloaders,
)
from minipamayo_qwen35.stage1.dataset import stage1_collate
from minipamayo_qwen35.stage2.reasoning_sft.dataset import (
    ReasoningSftJsonlDataset,
    build_stage2_train_val_dataloaders,
)
from minipamayo_qwen35.utils import preflight


def _identity_rot() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _make_record(
    sample_id: str,
    *,
    include_future: bool = True,
    include_command: bool = True,
    include_reasoning: bool = True,
) -> dict:
    gt_waypoints = [
        [1.0, 0.0],
        [2.0, 1.0],
    ]
    record = {
        "sample_id": sample_id,
        "image_path": "images/frame.jpg",
        "action": [0.1, 0.0, 0.2, 0.05],
        "v0": 1.5,
        "dt": 0.5,
        "gt_waypoints": gt_waypoints,
        "ego_history_xyz": [[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        "ego_history_rot": [[_identity_rot(), _identity_rot()]],
        "planner_state": "nominal",
        "decision_longitudinal": "maintain",
        "decision_lateral": "keep_lane",
    }
    if include_future:
        record["ego_future_xyz"] = [[[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]]
        record["ego_future_rot"] = [[_identity_rot(), _identity_rot()]]
    else:
        record["ego_pose"] = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        record["future_poses_global"] = [
            {"x": 1.0, "y": 0.0, "yaw_deg": 0.0},
            {"x": 2.0, "y": 1.0, "yaw_deg": 0.0},
        ]
    if include_command:
        record["command"] = "go_straight"
    if include_reasoning:
        record["reasoning_text"] = "keep lane and maintain speed"
    return record


def _write_jsonl(tmp_path: Path, file_name: str, records: list[dict]) -> Path:
    image_path = tmp_path / "images" / "frame.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not-an-image-but-not-opened-here")

    jsonl_path = tmp_path / file_name
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    return jsonl_path


def test_stage1_and_stage2_share_common_canonical_fields(tmp_path: Path) -> None:
    jsonl_path = _write_jsonl(tmp_path, "shared.jsonl", [_make_record("sample-0")])

    stage1_sample = Stage1JsonlDataset(jsonl_path)[0]
    stage2_sample = ReasoningSftJsonlDataset(jsonl_path)[0]

    assert str(stage1_sample["sample_id"]) == stage2_sample["sample_id"]
    assert stage1_sample["image_path"] == stage2_sample["image_path"]
    assert float(stage1_sample["dt"].item()) == stage2_sample["dt"]
    torch.testing.assert_close(stage1_sample["action"], stage2_sample["action"])
    torch.testing.assert_close(stage1_sample["v0"], stage2_sample["v0"])
    torch.testing.assert_close(stage1_sample["gt_waypoints"], stage2_sample["gt_waypoints"])
    torch.testing.assert_close(stage1_sample["ego_history_xyz"], stage2_sample["ego_history_xyz"])
    torch.testing.assert_close(stage1_sample["ego_history_rot"], stage2_sample["ego_history_rot"])
    torch.testing.assert_close(stage1_sample["ego_future_xyz"], stage2_sample["ego_future_xyz"])
    torch.testing.assert_close(stage1_sample["ego_future_rot"], stage2_sample["ego_future_rot"])


def test_stage_specific_required_keys_are_preserved(tmp_path: Path) -> None:
    stage1_jsonl = _write_jsonl(
        tmp_path,
        "missing_command.jsonl",
        [_make_record("sample-0", include_command=False)],
    )
    stage2_jsonl = _write_jsonl(
        tmp_path,
        "missing_reasoning.jsonl",
        [_make_record("sample-1", include_reasoning=False)],
    )

    with pytest.raises(RuntimeError, match="command"):
        Stage1JsonlDataset(stage1_jsonl)[0]
    with pytest.raises(RuntimeError, match="reasoning_text"):
        ReasoningSftJsonlDataset(stage2_jsonl)[0]


def test_future_tensor_fallback_matches_explicit_future(tmp_path: Path) -> None:
    explicit_jsonl = _write_jsonl(tmp_path, "explicit.jsonl", [_make_record("explicit")])
    derived_jsonl = _write_jsonl(
        tmp_path,
        "derived.jsonl",
        [_make_record("derived", include_future=False)],
    )

    explicit_sample = ReasoningSftJsonlDataset(explicit_jsonl)[0]
    derived_sample = ReasoningSftJsonlDataset(derived_jsonl)[0]

    torch.testing.assert_close(explicit_sample["ego_future_xyz"], derived_sample["ego_future_xyz"])
    torch.testing.assert_close(explicit_sample["ego_future_rot"], derived_sample["ego_future_rot"])


def test_generic_train_val_builder_split_is_seeded(tmp_path: Path) -> None:
    train_jsonl = _write_jsonl(
        tmp_path,
        "train.jsonl",
        [_make_record(f"sample-{idx}") for idx in range(6)],
    )

    first_train_loader, first_val_loader, _, _ = build_jsonl_train_val_dataloaders(
        dataset_ctor=Stage1JsonlDataset,
        collate_fn=stage1_collate,
        train_jsonl=[str(train_jsonl)],
        val_jsonl=None,
        train_max_samples=0,
        val_max_samples=0,
        val_fraction=0.34,
        batch_size=2,
        num_workers=0,
        seed=13,
        require_validation_split=True,
    )
    second_train_loader, second_val_loader, _, _ = build_jsonl_train_val_dataloaders(
        dataset_ctor=Stage1JsonlDataset,
        collate_fn=stage1_collate,
        train_jsonl=[str(train_jsonl)],
        val_jsonl=None,
        train_max_samples=0,
        val_max_samples=0,
        val_fraction=0.34,
        batch_size=2,
        num_workers=0,
        seed=13,
        require_validation_split=True,
    )

    assert list(first_train_loader.dataset.indices) == list(second_train_loader.dataset.indices)
    assert list(first_val_loader.dataset.indices) == list(second_val_loader.dataset.indices)


def test_stage_wrappers_preserve_explicit_val_max_samples_behavior(tmp_path: Path) -> None:
    train_jsonl = _write_jsonl(
        tmp_path,
        "train_wrapper.jsonl",
        [_make_record(f"train-{idx}") for idx in range(3)],
    )
    val_jsonl = _write_jsonl(
        tmp_path,
        "val_wrapper.jsonl",
        [_make_record(f"val-{idx}") for idx in range(3)],
    )

    _, _, stage1_train_size, stage1_val_size = build_stage1_train_val_dataloaders(
        train_jsonl=[str(train_jsonl)],
        val_jsonl=[str(val_jsonl)],
        max_samples=1,
        val_fraction=0.2,
        batch_size=1,
        num_workers=0,
        seed=7,
        require_validation_split=True,
    )
    _, _, stage2_train_size, stage2_val_size = build_stage2_train_val_dataloaders(
        train_jsonl=[str(train_jsonl)],
        val_jsonl=[str(val_jsonl)],
        max_samples=1,
        val_fraction=0.2,
        batch_size=1,
        num_workers=0,
        seed=7,
    )

    assert stage1_train_size == 1
    assert stage1_val_size == 1
    assert stage2_train_size == 1
    assert stage2_val_size == 3


def test_runtime_device_helpers_resolve_and_require_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)
    assert preflight.resolve_runtime_device("auto").type == "cpu"
    assert preflight.resolve_runtime_device("cpu").type == "cpu"
    assert preflight.resolve_runtime_device("cuda").type == "cuda"

    with pytest.raises(RuntimeError, match="CUDA required"):
        preflight.require_cuda_device(
            device_name="auto",
            git_cwd=Path.cwd(),
            error_message="CUDA required",
        )

    captured_git_cwds: list[Path] = []

    def _fake_enforce_runtime_prerequisites(*, git_cwd):
        captured_git_cwds.append(Path(git_cwd))
        return Path(git_cwd)

    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(preflight, "enforce_runtime_prerequisites", _fake_enforce_runtime_prerequisites)

    device = preflight.require_cuda_device(
        device_name="auto",
        git_cwd=Path.cwd(),
        error_message="CUDA required",
    )

    assert device.type == "cuda"
    assert captured_git_cwds == [Path.cwd()]
