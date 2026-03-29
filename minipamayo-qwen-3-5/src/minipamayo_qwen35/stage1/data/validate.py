"""Validate saved canonical Stage 1 actions against recomputed actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ...utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ...action_space.unicycle_accel_curvature import (
    canonical_action_tensor_from_record,
    saved_action_tensor_from_record,
)
from .dataset import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate saved Stage 1 actions against recomputed canonical actions."
    )
    parser.add_argument("--config-json", type=str, default="")
    return parser


def build_validation_settings_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--report-limit", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    return parser


def _resolve_job_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def parse_args() -> tuple[argparse.Namespace, list[dict], Path]:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_parser().parse_args(), [], PROJECT_ROOT

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Stage 1 action validation accepts only --config-json. Put all settings in the JSON file."
        )

    config_path, payload = load_json_payload(pre_args.config_json)
    if not isinstance(payload, dict):
        raise RuntimeError("Config JSON must be an object.")

    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("Config JSON must contain a non-empty `jobs` list.")

    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={"project_root": PROJECT_ROOT, "config_dir": config_path.parent},
    )

    settings_parser = build_validation_settings_parser()
    raw_settings = payload.get("validation", {})
    if not isinstance(raw_settings, dict):
        raise RuntimeError("Config `validation` must be a JSON object.")
    validation_args = normalize_arg_config(
        raw_settings,
        settings_parser,
        exclude_dests={"help"},
    )

    parser = build_parser()
    parser.set_defaults(config_json=str(config_path))
    args = parser.parse_args([f"--config-json={config_path}"])
    args.config_json = str(config_path)
    args.config_payload = payload
    args.validation = settings_parser.parse_args([])
    for key, value in validation_args.items():
        setattr(args.validation, key, value)
    return args, jobs, base_dir


def _validate_job(job: dict, *, base_dir: Path, settings: argparse.Namespace) -> dict:
    if not isinstance(job, dict):
        raise RuntimeError("Each job must be an object.")
    input_jsonl = job.get("input_jsonl")
    if not isinstance(input_jsonl, str) or not input_jsonl:
        raise RuntimeError("Each job requires non-empty `input_jsonl`.")

    input_path = _resolve_job_path(base_dir, input_jsonl)
    records = read_jsonl(input_path)
    if settings.max_samples > 0:
        records = records[: settings.max_samples]

    checked_records = 0
    mismatches = 0
    max_abs_diff = 0.0
    mismatch_examples: list[dict] = []

    for record in records:
        checked_records += 1
        saved_action = saved_action_tensor_from_record(record)
        recomputed_action = canonical_action_tensor_from_record(record)
        if saved_action.shape != recomputed_action.shape:
            mismatch = {
                "sample_id": str(record.get("sample_id", checked_records - 1)),
                "reason": "shape_mismatch",
                "saved_shape": list(saved_action.shape),
                "recomputed_shape": list(recomputed_action.shape),
            }
            mismatches += 1
            if len(mismatch_examples) < settings.report_limit:
                mismatch_examples.append(mismatch)
            if settings.fail_fast:
                break
            continue

        abs_diff = (saved_action - recomputed_action).abs()
        max_abs_diff = max(max_abs_diff, float(abs_diff.max().item()))
        if not torch.allclose(saved_action, recomputed_action, atol=settings.atol, rtol=settings.rtol):
            mismatches += 1
            if len(mismatch_examples) < settings.report_limit:
                mismatch_examples.append(
                    {
                        "sample_id": str(record.get("sample_id", checked_records - 1)),
                        "max_abs_diff": float(abs_diff.max().item()),
                    }
                )
            if settings.fail_fast:
                break

    return {
        "input_jsonl": str(input_path),
        "checked_records": checked_records,
        "mismatches": mismatches,
        "max_abs_diff": max_abs_diff,
        "mismatch_examples": mismatch_examples,
        "atol": float(settings.atol),
        "rtol": float(settings.rtol),
    }


def main() -> None:
    args, jobs, base_dir = parse_args()
    summaries = []
    total_mismatches = 0
    for job_index, job in enumerate(jobs, start=1):
        summary = _validate_job(job, base_dir=base_dir, settings=args.validation)
        total_mismatches += int(summary["mismatches"])
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "event": "stage1_action_validation",
                    "job_index": job_index,
                    "total_jobs": len(jobs),
                    **summary,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps({"jobs": summaries}, ensure_ascii=False, indent=2))
    if total_mismatches > 0:
        raise RuntimeError(f"Stage 1 action validation found {total_mismatches} mismatched records.")


if __name__ == "__main__":
    main()
