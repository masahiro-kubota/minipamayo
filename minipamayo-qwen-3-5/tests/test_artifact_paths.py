from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from minipamayo_qwen35.utils.artifact_paths import (
    ArtifactScope,
    owner_json_path,
    reporting_paths_for_output,
    run_logs_root,
    scope_from_owner_json_path,
    validate_generic_artifact_path,
)
from minipamayo_qwen35.utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path, file_name: str, args: dict[str, Any]) -> Path:
    config_path = tmp_path / file_name
    config_path.write_text(json.dumps({"args": args}, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def _iter_artifact_strings(payload: Any) -> list[str]:
    out: list[str] = []
    if isinstance(payload, dict):
        for value in payload.values():
            out.extend(_iter_artifact_strings(value))
    elif isinstance(payload, list):
        for value in payload:
            out.extend(_iter_artifact_strings(value))
    elif isinstance(payload, str) and payload.startswith("artifacts/"):
        out.append(payload)
    return out


def test_reporting_paths_follow_owner_stem() -> None:
    output_json = owner_json_path(
        ArtifactScope(kind="eval", stage="stage1", component="vlm_ce", track="canonical"),
        "run_name",
    )
    paths = reporting_paths_for_output(output_json, include_per_sample_jsonl=True)

    assert paths.output_json == output_json
    assert paths.progress_json == output_json.with_name("run_name.progress.json")
    assert paths.per_sample_jsonl == output_json.with_name("run_name.per_sample.jsonl")
    assert paths.manifest_json == output_json.with_name("run_name.manifest.json")
    assert paths.plots_dir == output_json.parent / "run_name_plots"
    assert paths.output_mcap == output_json.with_suffix(".mcap")


def test_validate_generic_artifact_path_rejects_old_preprocess_layout() -> None:
    old_path = PROJECT_ROOT / "artifacts" / "stage1" / "preprocess" / "curve_thresholds" / "run.json"
    with pytest.raises(RuntimeError, match="Artifact path kind must be one of"):
        validate_generic_artifact_path(old_path)


def test_run_logs_root_includes_workflow_segment() -> None:
    assert run_logs_root("ignore_rule_completion", "attempt_001") == (
        PROJECT_ROOT / "artifacts" / "run_logs" / "ignore_rule_completion" / "attempt_001"
    )


def test_scope_from_owner_json_path_reuses_track_for_new_component() -> None:
    scope = scope_from_owner_json_path(
        PROJECT_ROOT
        / "artifacts"
        / "preprocess"
        / "stage1"
        / "curve_thresholds"
        / "experiments"
        / "pid_tuning"
        / "run.json",
        kind="preprocess",
        stage="stage1",
        component="curve_thresholds",
        target_component="curve_splits",
    )

    assert scope == ArtifactScope(
        kind="preprocess",
        stage="stage1",
        component="curve_splits",
        track="experiments/pid_tuning",
    )


def test_stage1_eval_parse_args_rejects_noncanonical_owner_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("minipamayo_qwen35.stage1.vlm_ce.eval")
    config_path = _write_config(
        tmp_path,
        "config.json",
        {
            "checkpoint": "checkpoints/stage1.pt",
            "test_jsonl": "datasets/test.jsonl",
            "output_json": "artifacts/output.json",
            "progress_json": "",
            "per_sample_jsonl": "",
            "progress_every_samples": 1,
            "progress_every_seconds": 1.0,
            "wandb_project": "smoke-project",
            "wandb_run_name": "smoke-run",
            "image_min_pixels": CANONICAL_IMAGE_MIN_PIXELS,
            "image_max_pixels": CANONICAL_IMAGE_MAX_PIXELS,
        },
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config-json", str(config_path)])
    with pytest.raises(RuntimeError, match="Artifact path kind must be one of"):
        module.parse_args()


def test_stage1_visualize_parse_args_derives_plots_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("minipamayo_qwen35.stage1.vlm_ce.visualize_eval")
    config_path = _write_config(
        tmp_path,
        "config.json",
        {
            "summary_json": "artifacts/eval/stage1/vlm_ce/canonical/run.json",
            "per_sample_jsonl": "artifacts/eval/stage1/vlm_ce/canonical/run.per_sample.jsonl",
            "wandb_project": "smoke-project",
            "wandb_run_name": "smoke-run",
            "overlay_count": 4,
            "worst_table_count": 4,
            "dpi": 120,
        },
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--config-json", str(config_path)])
    args = module.parse_args()

    assert Path(args.output_dir) == PROJECT_ROOT / "artifacts" / "eval" / "stage1" / "vlm_ce" / "canonical" / "run_plots"


def test_config_artifact_paths_follow_generic_layout() -> None:
    config_paths = sorted((PROJECT_ROOT / "configs").rglob("*.json"))
    assert config_paths

    for config_path in config_paths:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        for artifact_path in _iter_artifact_strings(payload):
            validate_generic_artifact_path(PROJECT_ROOT / artifact_path)
