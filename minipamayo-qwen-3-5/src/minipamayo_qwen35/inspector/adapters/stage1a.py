"""Normalize Stage 1A eval artifacts for the Streamlit inspector."""

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


def load_stage1a_run(manifest: ArtifactManifest) -> NormalizedRun:
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
                    sample_index=int(row["sample_index"]),
                    image_path=str(row["image_path"]),
                    command=str(row.get("command", "")),
                    gt_waypoints=[[float(x), float(y)] for x, y in row.get("gt_waypoints", [])],
                    pred_waypoints=[[float(x), float(y)] for x, y in row.get("pred_waypoints", [])],
                    ade_m=float(row["ade_m"]) if "ade_m" in row else None,
                    fde_m=float(row["fde_m"]) if "fde_m" in row else None,
                    token_accuracy=(
                        float(metrics["autoregressive_token_accuracy"])
                        if "autoregressive_token_accuracy" in metrics
                        else None
                    ),
                    action_mae_kappa=(
                        float(metrics["action_mae_kappa"]) if "action_mae_kappa" in metrics else None
                    ),
                    raw=dict(row),
                )
            )
    return NormalizedRun(
        manifest=manifest,
        summary=summary,
        samples=samples,
        invalid_reason=_duplicate_sample_reason(samples),
    )
