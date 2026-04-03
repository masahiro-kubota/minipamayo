from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from minipamayo_qwen35.utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
)


def _write_config(tmp_path: Path, file_name: str, args: dict) -> Path:
    config_path = tmp_path / file_name
    config_path.write_text(
        json.dumps({"args": args}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def _reporting_args(*, include_per_sample_jsonl: bool) -> dict:
    args = {
        "progress_json": "",
        "progress_every_samples": 1,
        "progress_every_seconds": 1.0,
        "wandb_project": "smoke-project",
        "wandb_run_name": "smoke-run",
    }
    if include_per_sample_jsonl:
        args["per_sample_jsonl"] = ""
    return args


@pytest.mark.parametrize(
    ("module_name", "config_args", "expected_checks"),
    [
        (
            "minipamayo_qwen35.stage1.vlm_ce.inference",
            {
                "checkpoint": "checkpoints/stage1.pt",
                "test_jsonl": "datasets/test.jsonl",
                "output_json": "artifacts/inference/stage1/vlm_ce/canonical/stage1_infer.json",
            },
            {
                "checkpoint": "stage1.pt",
                "test_jsonl": "test.jsonl",
                "output_json": "artifacts/inference/stage1/vlm_ce/canonical/stage1_infer.json",
            },
        ),
        (
            "minipamayo_qwen35.stage1.vlm_ce.eval",
            {
                "checkpoint": "checkpoints/stage1.pt",
                "test_jsonl": "datasets/test.jsonl",
                "output_json": "artifacts/eval/stage1/vlm_ce/canonical/output.json",
                "image_min_pixels": CANONICAL_IMAGE_MIN_PIXELS,
                "image_max_pixels": CANONICAL_IMAGE_MAX_PIXELS,
                **_reporting_args(include_per_sample_jsonl=True),
            },
            {
                "checkpoint": "stage1.pt",
                "test_jsonl": "test.jsonl",
                "output_json": "artifacts/eval/stage1/vlm_ce/canonical/output.json",
                "progress_json": "artifacts/eval/stage1/vlm_ce/canonical/output.progress.json",
                "per_sample_jsonl": "artifacts/eval/stage1/vlm_ce/canonical/output.per_sample.jsonl",
            },
        ),
        (
            "minipamayo_qwen35.stage1.expert_cfm.inference",
            {
                "checkpoint": "checkpoints/stage1b.pt",
                "stage1_checkpoint": "checkpoints/stage1.pt",
                "sample_jsonl": "datasets/sample.jsonl",
                "output_json": "artifacts/inference/stage1/expert_cfm/canonical/output.json",
                "flow_steps": 10,
            },
            {
                "checkpoint": "stage1b.pt",
                "stage1_checkpoint": "stage1.pt",
                "sample_jsonl": "sample.jsonl",
                "output_json": "artifacts/inference/stage1/expert_cfm/canonical/output.json",
            },
        ),
        (
            "minipamayo_qwen35.stage1.expert_cfm.eval",
            {
                "checkpoint": "checkpoints/stage1b.pt",
                "stage1_checkpoint": "checkpoints/stage1.pt",
                "eval_jsonl": ["datasets/eval_a.jsonl", "datasets/eval_b.jsonl"],
                "output_json": "artifacts/eval/stage1/expert_cfm/canonical/output.json",
                "flow_steps": 10,
                **_reporting_args(include_per_sample_jsonl=True),
            },
            {
                "checkpoint": "stage1b.pt",
                "stage1_checkpoint": "stage1.pt",
                "output_json": "artifacts/eval/stage1/expert_cfm/canonical/output.json",
                "progress_json": "artifacts/eval/stage1/expert_cfm/canonical/output.progress.json",
                "per_sample_jsonl": "artifacts/eval/stage1/expert_cfm/canonical/output.per_sample.jsonl",
            },
        ),
        (
            "minipamayo_qwen35.stage2.reasoning_sft.train",
            {
                "stage1a_checkpoint": "checkpoints/stage1a.pt",
                "train_jsonl": "datasets/train.jsonl",
            },
            {"stage1a_checkpoint": "stage1a.pt", "train_jsonl": "train.jsonl"},
        ),
        (
            "minipamayo_qwen35.stage2.reasoning_sft.eval",
            {
                "checkpoint": "checkpoints/stage2.pt",
                "eval_jsonl": "datasets/eval.jsonl",
                "output_json": "artifacts/eval/stage2/reasoning_sft/canonical/output.json",
                "image_min_pixels": CANONICAL_IMAGE_MIN_PIXELS,
                "image_max_pixels": CANONICAL_IMAGE_MAX_PIXELS,
                **_reporting_args(include_per_sample_jsonl=True),
            },
            {
                "checkpoint": "stage2.pt",
                "eval_jsonl": "eval.jsonl",
                "output_json": "artifacts/eval/stage2/reasoning_sft/canonical/output.json",
                "progress_json": "artifacts/eval/stage2/reasoning_sft/canonical/output.progress.json",
                "per_sample_jsonl": "artifacts/eval/stage2/reasoning_sft/canonical/output.per_sample.jsonl",
            },
        ),
        (
            "minipamayo_qwen35.stage2.reasoning_sft.inference",
            {
                "checkpoint": "checkpoints/stage2.pt",
                "stage1b_checkpoint": "checkpoints/stage1b.pt",
                "sample_jsonl": "datasets/sample.jsonl",
                "sample_index": 0,
                "output_json": "artifacts/inference/stage2/reasoning_sft/canonical/output.json",
                "image_min_pixels": CANONICAL_IMAGE_MIN_PIXELS,
                "image_max_pixels": CANONICAL_IMAGE_MAX_PIXELS,
                "max_reasoning_tokens": 256,
                "flow_steps": 10,
                "temperature": 0.6,
                "top_p": 0.98,
                "top_k": 0,
                **_reporting_args(include_per_sample_jsonl=False),
            },
            {
                "checkpoint": "stage2.pt",
                "stage1b_checkpoint": "stage1b.pt",
                "sample_jsonl": "sample.jsonl",
                "output_json": "artifacts/inference/stage2/reasoning_sft/canonical/output.json",
                "progress_json": "artifacts/inference/stage2/reasoning_sft/canonical/output.progress.json",
            },
        ),
    ],
)
def test_entrypoint_parse_args_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    config_args: dict,
    expected_checks: dict[str, str],
) -> None:
    module = importlib.import_module(module_name)
    config_path = _write_config(tmp_path, "config.json", config_args)

    monkeypatch.setattr(sys, "argv", ["prog", "--config-json", str(config_path)])
    args = module.parse_args()

    assert Path(args.config_json) == config_path.resolve()
    for attr_name, expected_suffix in expected_checks.items():
        value = getattr(args, attr_name)
        if isinstance(value, list):
            assert value
            resolved_value = str(Path(value[0]).resolve())
        else:
            resolved_value = str(Path(value).resolve())
        assert resolved_value.endswith(expected_suffix)
