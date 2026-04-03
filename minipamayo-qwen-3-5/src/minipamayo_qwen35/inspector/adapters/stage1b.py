"""Normalize Stage 1B eval artifacts for the Streamlit inspector."""

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


def load_stage1b_run(manifest: ArtifactManifest) -> NormalizedRun:
    summary = read_json(manifest.summary_json)
    samples: list[NormalizedSample] = []
    if manifest.per_sample_jsonl:
        for row in read_jsonl(manifest.per_sample_jsonl):
            metrics = dict(row.get("metrics", {}))
            pid_override = row.get("pid_override")
            pid_pred_waypoints = None
            if isinstance(pid_override, dict) and "pred_waypoints" in pid_override:
                pid_pred_waypoints = [
                    [float(x), float(y)] for x, y in pid_override.get("pred_waypoints", [])
                ]
            samples.append(
                NormalizedSample(
                    stage=manifest.stage,
                    run_name=manifest.run_name,
                    sample_id=str(row["sample_id"]),
                    sample_index=int(row.get("record_sample_index", row["sample_index"])),
                    image_path=str(row["image_path"]),
                    source_frame_id=str(row.get("source_frame_id", "")),
                    command=str(row.get("command", "")),
                    gt_waypoints=[[float(x), float(y)] for x, y in row.get("gt_waypoints", [])],
                    pred_waypoints=[[float(x), float(y)] for x, y in row.get("pred_waypoints", [])],
                    pid_pred_waypoints=pid_pred_waypoints,
                    ade_m=float(row["ade_m"]) if "ade_m" in row else None,
                    fde_m=float(row["fde_m"]) if "fde_m" in row else None,
                    max_lateral_error_m=(
                        float(row["max_lateral_error_m"]) if "max_lateral_error_m" in row else None
                    ),
                    action_mae_kappa=(
                        float(metrics["action_mae_kappa"]) if "action_mae_kappa" in metrics else None
                    ),
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
