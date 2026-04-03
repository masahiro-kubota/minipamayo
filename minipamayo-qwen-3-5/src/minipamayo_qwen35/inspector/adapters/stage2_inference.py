"""Normalize Stage 2 inference artifacts for the Streamlit inspector."""

from __future__ import annotations

from collections import Counter

from ..cache import read_json, read_jsonl
from ..models import ArtifactManifest, NormalizedRun, NormalizedSample


def _duplicate_sample_reason(samples: list[NormalizedSample]) -> str | None:
    counts = Counter(sample.sample_id for sample in samples)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if not duplicates:
        return None
    return "Duplicate sample_id values: " + ", ".join(duplicates[:10])


def _normalize_row(manifest: ArtifactManifest, row: dict) -> NormalizedSample:
    prediction = dict(row.get("prediction", {}))
    ground_truth = dict(row.get("ground_truth", {}))
    metrics = dict(row.get("metrics", {}))
    reasoning = dict(row.get("reasoning", {}))
    return NormalizedSample(
        stage=manifest.stage,
        run_name=manifest.run_name,
        sample_id=str(row["sample_id"]),
        sample_index=int(row["sample_index"]),
        image_path=str(row["image_path"]),
        command=str(row.get("command", "")),
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
        if "image_path" not in summary_row and isinstance(summary.get("raw"), dict):
            summary_row["image_path"] = summary["raw"].get("image_path", "")
        samples = [_normalize_row(manifest, summary_row)]
    return NormalizedRun(
        manifest=manifest,
        summary=summary,
        samples=samples,
        invalid_reason=_duplicate_sample_reason(samples),
    )
