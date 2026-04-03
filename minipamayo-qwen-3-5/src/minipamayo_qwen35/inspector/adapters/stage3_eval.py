"""Normalize Stage 3 eval artifacts for the Streamlit inspector."""

from __future__ import annotations

from collections import Counter

from ..cache import read_json, read_jsonl
from ..grouping import derive_groups
from ..models import ArtifactManifest, NormalizedRun, NormalizedSample


def _duplicate_sample_reason(samples: list[NormalizedSample]) -> str | None:
    counts = Counter(sample.sample_id for sample in samples)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if not duplicates:
        return None
    return "Duplicate sample_id values: " + ", ".join(duplicates[:10])


def _normalize_waypoints(rows: list[list[float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in rows]


def load_stage3_eval_run(manifest: ArtifactManifest) -> NormalizedRun:
    summary = read_json(manifest.summary_json)
    samples: list[NormalizedSample] = []
    if manifest.per_sample_jsonl:
        for row in read_jsonl(manifest.per_sample_jsonl):
            metrics = dict(row.get("metrics", {}))
            samples.append(
                NormalizedSample(
                    stage=manifest.stage,
                    run_name=manifest.run_name,
                    sample_id=str(row["sample_id"]),
                    sample_index=int(row.get("record_sample_index", row["sample_index"])),
                    image_path=str(row["image_path"]),
                    source_frame_id=str(row.get("source_frame_id", "")),
                    command=str(row.get("command", "")),
                    gt_waypoints=_normalize_waypoints(row.get("gt_waypoints", [])),
                    pred_waypoints=_normalize_waypoints(row.get("pred_waypoints", [])),
                    ade_m=(
                        float(metrics["ade_m"])
                        if "ade_m" in metrics
                        else (float(row["ade_m"]) if "ade_m" in row else None)
                    ),
                    fde_m=(
                        float(metrics["fde_m"])
                        if "fde_m" in metrics
                        else (float(row["fde_m"]) if "fde_m" in row else None)
                    ),
                    reasoning_text_gt=str(row.get("reasoning_text", "")),
                    reasoning_text_pred=str(row.get("reasoning_text_pred", "")),
                    raw=dict(row),
                )
            )
    samples, groups = derive_groups(samples)
    return NormalizedRun(
        manifest=manifest,
        summary=summary,
        samples=samples,
        groups=groups,
        invalid_reason=_duplicate_sample_reason(samples),
    )
