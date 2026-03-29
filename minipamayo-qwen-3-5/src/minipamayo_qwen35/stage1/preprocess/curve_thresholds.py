"""Summarize curve-threshold candidates from canonical Stage 1 samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ...utils.json_config import load_json_payload, resolve_path_base

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "stage1" / "preprocess"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize curve-threshold candidates from extracted Stage 1 samples."
    )
    parser.add_argument("--config-json", type=str, required=True)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output JSON path. Defaults under artifacts/stage1/preprocess/curve_thresholds/.",
    )
    parser.add_argument("--percentiles", type=str, default="50,75,90,95,99")
    parser.add_argument("--kappa-thresholds", type=str, default="0.05,0.08,0.10,0.12,0.15")
    parser.add_argument("--yaw-thresholds", type=str, default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument(
        "--block-anchor-mode",
        type=str,
        choices=("or", "and", "kappa", "yaw"),
        default="or",
        help="How curve-block anchor samples are selected from kappa/yaw thresholds.",
    )
    parser.add_argument(
        "--block-kappa-threshold",
        type=float,
        default=0.08,
        help="Anchor threshold for max |kappa_gt| when building curve blocks.",
    )
    parser.add_argument(
        "--block-yaw-threshold",
        type=float,
        default=0.5,
        help="Anchor threshold for |yaw change| when building curve blocks.",
    )
    parser.add_argument(
        "--block-pre-seconds",
        type=float,
        default=1.0,
        help="Seconds to prepend before each anchor block.",
    )
    parser.add_argument(
        "--block-post-seconds",
        type=float,
        default=1.0,
        help="Seconds to append after each anchor block.",
    )
    return parser


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_config_base_dir(config_path: Path, payload) -> Path:
    return resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={
            "project_root": PROJECT_ROOT,
            "datasets_root": DATASETS_ROOT,
            "config_dir": config_path.parent,
        },
    )


def _parse_float_csv(csv: str) -> list[float]:
    values = [entry.strip() for entry in csv.split(",") if entry.strip()]
    if not values:
        raise RuntimeError("Expected at least one comma-separated numeric value.")
    return [float(value) for value in values]


def _slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _default_output_json(
    *,
    config_json: str,
    anchor_mode: str,
    block_kappa_threshold: float,
    block_yaw_threshold: float,
    block_pre_seconds: float,
    block_post_seconds: float,
) -> Path:
    config_stem = Path(config_json).resolve().stem
    filename = (
        f"{config_stem}"
        f"__mode-{anchor_mode}"
        f"__kappa-{_slug_float(block_kappa_threshold)}"
        f"__yaw-{_slug_float(block_yaw_threshold)}"
        f"__pre-{_slug_float(block_pre_seconds)}"
        f"__post-{_slug_float(block_post_seconds)}.json"
    )
    return ARTIFACTS_ROOT / "curve_thresholds" / filename


def _load_jsonl_paths(config_json: str) -> list[Path]:
    config_path, payload = load_json_payload(config_json)
    base_dir = _resolve_config_base_dir(config_path, payload)
    raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RuntimeError("Config JSON must define a non-empty jobs list.")

    jsonl_paths: list[Path] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise RuntimeError("Each config job must be an object.")
        output_dir_value = raw_job.get("output_dir")
        if not output_dir_value:
            raise RuntimeError("Each config job must define output_dir.")
        output_dir = _resolve_path(str(output_dir_value), base_dir)
        jsonl_path = output_dir / "samples.jsonl"
        if not jsonl_path.exists():
            raise RuntimeError(f"Missing extracted samples.jsonl: {jsonl_path}")
        jsonl_paths.append(jsonl_path)
    return jsonl_paths


def _load_extract_summary(jsonl_path: Path) -> dict:
    summary_path = jsonl_path.parent / "extract_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing extract_summary.json next to samples.jsonl: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise RuntimeError("Cannot compute a percentile from an empty list.")
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError(f"Percentile must be in [0, 100], got {percentile}.")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    alpha = position - lower
    return float(sorted_values[lower] * (1.0 - alpha) + sorted_values[upper] * alpha)


def _compute_sample_stats(record: dict) -> tuple[float, float]:
    action = record.get("action")
    if not isinstance(action, list) or not action or len(action) % 2 != 0:
        raise RuntimeError("Record is missing canonical interleaved `action`.")
    kappas = [float(value) for value in action[1::2]]
    max_abs_kappa = max(abs(value) for value in kappas)

    future_rot = record.get("ego_future_rot")
    if (
        not isinstance(future_rot, list)
        or not future_rot
        or not isinstance(future_rot[0], list)
        or not future_rot[0]
    ):
        raise RuntimeError("Record is missing canonical `ego_future_rot`.")
    final_rot = future_rot[0][-1]
    if not (
        isinstance(final_rot, list)
        and len(final_rot) == 3
        and isinstance(final_rot[0], list)
        and len(final_rot[0]) == 3
        and isinstance(final_rot[1], list)
        and len(final_rot[1]) == 3
    ):
        raise RuntimeError("Record has invalid canonical `ego_future_rot` layout.")
    yaw_change = abs(math.atan2(float(final_rot[1][0]), float(final_rot[0][0])))
    return max_abs_kappa, yaw_change


def _compute_sample_entry(record: dict, ordinal_index: int) -> dict:
    max_abs_kappa, abs_yaw_change = _compute_sample_stats(record)
    sample_index = int(record.get("sample_index", ordinal_index))
    source_frame_id = int(record.get("source_frame_id", sample_index))
    return {
        "sample_id": str(record.get("sample_id", f"{sample_index:06d}")),
        "sample_index": sample_index,
        "source_frame_id": source_frame_id,
        "max_abs_kappa": max_abs_kappa,
        "abs_yaw_change": abs_yaw_change,
    }


def _coverage(values: list[float], thresholds: list[float]) -> dict[str, dict[str, float]]:
    total = len(values)
    if total == 0:
        raise RuntimeError("Cannot compute threshold coverage from an empty list.")
    coverage: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        count = sum(1 for value in values if value >= threshold)
        coverage[f"{threshold:g}"] = {
            "count": count,
            "fraction": count / total,
        }
    return coverage


def _is_anchor_sample(
    sample: dict,
    *,
    anchor_mode: str,
    kappa_threshold: float,
    yaw_threshold: float,
) -> bool:
    kappa_hit = sample["max_abs_kappa"] >= kappa_threshold
    yaw_hit = sample["abs_yaw_change"] >= yaw_threshold
    if anchor_mode == "kappa":
        return kappa_hit
    if anchor_mode == "yaw":
        return yaw_hit
    if anchor_mode == "and":
        return kappa_hit and yaw_hit
    return kappa_hit or yaw_hit


def _build_curve_block_summary(
    samples: list[dict],
    *,
    extract_summary: dict,
    anchor_mode: str,
    block_kappa_threshold: float,
    block_yaw_threshold: float,
    block_pre_seconds: float,
    block_post_seconds: float,
) -> dict:
    record_hz = float(extract_summary["record_hz"])
    sample_stride_frames = int(extract_summary["sample_stride_frames"])
    sample_interval_seconds = sample_stride_frames / record_hz
    pre_expand_samples = int(math.ceil(block_pre_seconds / sample_interval_seconds))
    post_expand_samples = int(math.ceil(block_post_seconds / sample_interval_seconds))

    anchor_ranges: list[list[int]] = []
    for idx, sample in enumerate(samples):
        if not _is_anchor_sample(
            sample,
            anchor_mode=anchor_mode,
            kappa_threshold=block_kappa_threshold,
            yaw_threshold=block_yaw_threshold,
        ):
            continue
        if not anchor_ranges or sample["sample_index"] != samples[anchor_ranges[-1][1]]["sample_index"] + 1:
            anchor_ranges.append([idx, idx])
        else:
            anchor_ranges[-1][1] = idx

    expanded_ranges: list[list[int]] = []
    for start_idx, end_idx in anchor_ranges:
        expanded_start = max(0, start_idx - pre_expand_samples)
        expanded_end = min(len(samples) - 1, end_idx + post_expand_samples)
        if not expanded_ranges or expanded_start > expanded_ranges[-1][1] + 1:
            expanded_ranges.append([expanded_start, expanded_end])
        else:
            expanded_ranges[-1][1] = max(expanded_ranges[-1][1], expanded_end)

    blocks: list[dict] = []
    anchor_sample_count = sum(1 for sample in samples if _is_anchor_sample(
        sample,
        anchor_mode=anchor_mode,
        kappa_threshold=block_kappa_threshold,
        yaw_threshold=block_yaw_threshold,
    ))
    expanded_sample_count = 0
    for block_index, (start_idx, end_idx) in enumerate(expanded_ranges):
        block_samples = samples[start_idx : end_idx + 1]
        expanded_sample_count += len(block_samples)
        blocks.append(
            {
                "block_index": block_index,
                "start_sample_id": block_samples[0]["sample_id"],
                "end_sample_id": block_samples[-1]["sample_id"],
                "start_sample_index": block_samples[0]["sample_index"],
                "end_sample_index": block_samples[-1]["sample_index"],
                "start_source_frame_id": block_samples[0]["source_frame_id"],
                "end_source_frame_id": block_samples[-1]["source_frame_id"],
                "num_samples": len(block_samples),
                "duration_seconds": max(0.0, (len(block_samples) - 1) * sample_interval_seconds),
                "max_abs_kappa": max(sample["max_abs_kappa"] for sample in block_samples),
                "max_abs_yaw_change": max(sample["abs_yaw_change"] for sample in block_samples),
            }
        )

    return {
        "anchor_mode": anchor_mode,
        "anchor_thresholds": {
            "max_abs_kappa_gte": block_kappa_threshold,
            "abs_yaw_change_gte": block_yaw_threshold,
        },
        "expansion": {
            "pre_seconds": block_pre_seconds,
            "post_seconds": block_post_seconds,
            "pre_samples": pre_expand_samples,
            "post_samples": post_expand_samples,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "anchor_sample_count": anchor_sample_count,
        "anchor_sample_fraction": anchor_sample_count / len(samples),
        "block_sample_count": expanded_sample_count,
        "block_sample_fraction": expanded_sample_count / len(samples),
        "num_blocks": len(blocks),
        "blocks": blocks,
    }


def _summarize_run(
    jsonl_path: Path,
    *,
    percentiles: list[float],
    kappa_thresholds: list[float],
    yaw_thresholds: list[float],
    anchor_mode: str,
    block_kappa_threshold: float,
    block_yaw_threshold: float,
    block_pre_seconds: float,
    block_post_seconds: float,
) -> tuple[dict, list[float], list[float]]:
    max_abs_kappa_values: list[float] = []
    abs_yaw_change_values: list[float] = []
    samples: list[dict] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for ordinal_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = _compute_sample_entry(record, ordinal_index)
            samples.append(sample)
            max_abs_kappa_values.append(sample["max_abs_kappa"])
            abs_yaw_change_values.append(sample["abs_yaw_change"])

    samples.sort(key=lambda sample: sample["sample_index"])

    max_abs_kappa_sorted = sorted(max_abs_kappa_values)
    abs_yaw_change_sorted = sorted(abs_yaw_change_values)
    extract_summary = _load_extract_summary(jsonl_path)

    summary = {
        "run_name": jsonl_path.parent.name,
        "jsonl_path": str(jsonl_path),
        "num_samples": len(max_abs_kappa_values),
        "max_abs_kappa": {
            "percentiles": {
                f"p{int(percentile) if percentile.is_integer() else percentile:g}": _percentile(
                    max_abs_kappa_sorted, percentile
                )
                for percentile in percentiles
            },
            "max": max_abs_kappa_sorted[-1],
            "coverage": _coverage(max_abs_kappa_values, kappa_thresholds),
        },
        "abs_yaw_change": {
            "percentiles": {
                f"p{int(percentile) if percentile.is_integer() else percentile:g}": _percentile(
                    abs_yaw_change_sorted, percentile
                )
                for percentile in percentiles
            },
            "max": abs_yaw_change_sorted[-1],
            "coverage": _coverage(abs_yaw_change_values, yaw_thresholds),
        },
        "curve_blocks": _build_curve_block_summary(
            samples,
            extract_summary=extract_summary,
            anchor_mode=anchor_mode,
            block_kappa_threshold=block_kappa_threshold,
            block_yaw_threshold=block_yaw_threshold,
            block_pre_seconds=block_pre_seconds,
            block_post_seconds=block_post_seconds,
        ),
    }
    return summary, max_abs_kappa_values, abs_yaw_change_values


def main() -> None:
    args = build_parser().parse_args()
    percentiles = _parse_float_csv(args.percentiles)
    kappa_thresholds = _parse_float_csv(args.kappa_thresholds)
    yaw_thresholds = _parse_float_csv(args.yaw_thresholds)
    jsonl_paths = _load_jsonl_paths(args.config_json)

    run_summaries: list[dict] = []
    all_kappa_values: list[float] = []
    all_yaw_values: list[float] = []
    for jsonl_path in jsonl_paths:
        summary, run_kappa_values, run_yaw_values = _summarize_run(
            jsonl_path,
            percentiles=percentiles,
            kappa_thresholds=kappa_thresholds,
            yaw_thresholds=yaw_thresholds,
            anchor_mode=args.block_anchor_mode,
            block_kappa_threshold=args.block_kappa_threshold,
            block_yaw_threshold=args.block_yaw_threshold,
            block_pre_seconds=args.block_pre_seconds,
            block_post_seconds=args.block_post_seconds,
        )
        run_summaries.append(summary)
        all_kappa_values.extend(run_kappa_values)
        all_yaw_values.extend(run_yaw_values)

    overall_summary = {
        "run_name": "__overall__",
        "num_samples": len(all_kappa_values),
        "max_abs_kappa": {
            "percentiles": {
                f"p{int(percentile) if percentile.is_integer() else percentile:g}": _percentile(
                    sorted(all_kappa_values), percentile
                )
                for percentile in percentiles
            },
            "max": max(all_kappa_values),
            "coverage": _coverage(all_kappa_values, kappa_thresholds),
        },
        "abs_yaw_change": {
            "percentiles": {
                f"p{int(percentile) if percentile.is_integer() else percentile:g}": _percentile(
                    sorted(all_yaw_values), percentile
                )
                for percentile in percentiles
            },
            "max": max(all_yaw_values),
            "coverage": _coverage(all_yaw_values, yaw_thresholds),
        },
    }

    payload = {
        "config_json": str(Path(args.config_json).resolve()),
        "percentiles": percentiles,
        "kappa_thresholds": kappa_thresholds,
        "yaw_thresholds": yaw_thresholds,
        "curve_block_config": {
            "anchor_mode": args.block_anchor_mode,
            "block_kappa_threshold": args.block_kappa_threshold,
            "block_yaw_threshold": args.block_yaw_threshold,
            "block_pre_seconds": args.block_pre_seconds,
            "block_post_seconds": args.block_post_seconds,
        },
        "runs": run_summaries,
        "overall": overall_summary,
    }

    output_path = (
        Path(args.output_json).resolve()
        if args.output_json
        else _default_output_json(
            config_json=args.config_json,
            anchor_mode=args.block_anchor_mode,
            block_kappa_threshold=args.block_kappa_threshold,
            block_yaw_threshold=args.block_yaw_threshold,
            block_pre_seconds=args.block_pre_seconds,
            block_post_seconds=args.block_post_seconds,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output_json": str(output_path),
                **payload,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
