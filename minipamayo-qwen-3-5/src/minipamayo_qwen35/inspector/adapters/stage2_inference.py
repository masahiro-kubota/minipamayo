"""Normalize Stage 2 inference artifacts for the Streamlit inspector."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..cache import read_json, read_jsonl
from ..models import ArtifactManifest, NormalizedRun, NormalizedSample


def _duplicate_sample_reason(samples: list[NormalizedSample]) -> str | None:
    counts = Counter(sample.sample_id for sample in samples)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if not duplicates:
        return None
    return "Duplicate sample_id values: " + ", ".join(duplicates[:10])


def _resolve_source_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_jsonl = row.get("sample_jsonl")
    sample_index = row.get("sample_index")
    if not isinstance(sample_jsonl, str) or not sample_jsonl:
        return {}
    if not isinstance(sample_index, int):
        try:
            sample_index = int(sample_index)
        except (TypeError, ValueError):
            return {}
    sample_jsonl_path = sample_jsonl.split(",")[0].strip()
    if not sample_jsonl_path:
        return {}
    path = Path(sample_jsonl_path).resolve()
    if not path.exists():
        return {}
    source_rows = read_jsonl(str(path))
    if sample_index < 0 or sample_index >= len(source_rows):
        return {}
    return dict(source_rows[sample_index])


def _coalesce_image_path(row: dict[str, Any], source_row: dict[str, Any]) -> str:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path:
        return image_path
    source_image_path = source_row.get("image_path")
    if not isinstance(source_image_path, str) or not source_image_path:
        return ""
    sample_jsonl = row.get("sample_jsonl")
    if isinstance(sample_jsonl, str) and sample_jsonl:
        sample_jsonl_path = sample_jsonl.split(",")[0].strip()
        if sample_jsonl_path:
            candidate = (Path(sample_jsonl_path).resolve().parent / source_image_path).resolve()
            if candidate.exists():
                return str(candidate)
    return source_image_path


def _normalize_row(manifest: ArtifactManifest, row: dict) -> NormalizedSample:
    source_row = _resolve_source_row(row)
    prediction = dict(row.get("prediction", {}))
    ground_truth = dict(row.get("ground_truth", {}))
    metrics = dict(row.get("metrics", {}))
    reasoning = dict(row.get("reasoning", {}))
    command = row.get("command", source_row.get("command", ""))
    if "reasoning_text" in source_row and "reasoning_text" not in ground_truth:
        ground_truth["reasoning_text"] = source_row["reasoning_text"]
    if "gt_waypoints" in source_row and "waypoints" not in ground_truth:
        ground_truth["waypoints"] = source_row["gt_waypoints"]
    return NormalizedSample(
        stage=manifest.stage,
        run_name=manifest.run_name,
        sample_id=str(row["sample_id"]),
        sample_index=int(row["sample_index"]),
        image_path=_coalesce_image_path(row, source_row),
        command=str(command),
        gt_waypoints=[[float(x), float(y)] for x, y in ground_truth.get("waypoints", [])],
        pred_waypoints=[[float(x), float(y)] for x, y in prediction.get("waypoints", [])],
        ade_m=float(metrics["ade_m"]) if "ade_m" in metrics else None,
        fde_m=float(metrics["fde_m"]) if "fde_m" in metrics else None,
        reasoning_text_gt=str(ground_truth.get("reasoning_text", "")),
        reasoning_text_pred=str(reasoning.get("text", "")),
        raw=dict(row),
    )


def load_stage2_inference_run(manifest: ArtifactManifest) -> NormalizedRun:
    summary = read_json(manifest.summary_json)
    samples: list[NormalizedSample] = []
    if manifest.per_sample_jsonl:
        samples = [_normalize_row(manifest, row) for row in read_jsonl(manifest.per_sample_jsonl)]
    elif isinstance(summary.get("sample_id"), str):
        summary_row = dict(summary)
        if "sample_index" not in summary_row:
            summary_row["sample_index"] = 0
        samples = [_normalize_row(manifest, summary_row)]
    return NormalizedRun(
        manifest=manifest,
        summary=summary,
        samples=samples,
        invalid_reason=_duplicate_sample_reason(samples),
    )
