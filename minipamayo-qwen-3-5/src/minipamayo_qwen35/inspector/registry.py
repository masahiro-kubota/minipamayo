"""Discover and load inspector manifests from local artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import (
    load_stage1a_run,
    load_stage1b_run,
    load_stage2_eval_run,
    load_stage2_inference_run,
)
from .manifests import load_manifest
from .models import ArtifactManifest, NormalizedRun, Stage2RunBundle
from ..utils.artifact_paths import default_artifact_root

DEFAULT_ARTIFACT_ROOT = default_artifact_root()


@dataclass(frozen=True)
class ManifestRegistry:
    artifact_root: Path
    manifests: tuple[ArtifactManifest, ...]
    stage1a_manifests: tuple[ArtifactManifest, ...]
    stage1b_manifests: tuple[ArtifactManifest, ...]
    stage2_eval_manifests: tuple[ArtifactManifest, ...]
    stage2_inference_manifests: tuple[ArtifactManifest, ...]
    stage2_bundles: tuple[Stage2RunBundle, ...]


def default_artifact_root() -> Path:
    return DEFAULT_ARTIFACT_ROOT


def sample_lookup(run: NormalizedRun | None) -> dict[str, Any]:
    if run is None:
        return {}
    return {sample.sample_id: sample for sample in run.samples}


def group_lookup(run: NormalizedRun | None) -> dict[str, Any]:
    if run is None:
        return {}
    return {group.group_id: group for group in run.groups}


def select_matching_sample(run: NormalizedRun | None, sample_id: str) -> Any:
    return sample_lookup(run).get(sample_id)


def discover_manifest_paths(artifact_root: str | Path) -> list[Path]:
    root = Path(artifact_root).resolve()
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.manifest.json") if path.is_file())


def _sort_stage_manifests(manifests: tuple[ArtifactManifest, ...]) -> tuple[ArtifactManifest, ...]:
    return tuple(
        sorted(
            manifests,
            key=lambda manifest: (
                manifest.per_sample_jsonl is None,
                manifest.run_name,
                manifest.summary_json,
            ),
        )
    )


def load_manifest_registry(artifact_root: str | Path) -> ManifestRegistry:
    root = Path(artifact_root).resolve()
    manifests = tuple(load_manifest(path) for path in discover_manifest_paths(root))
    stage1a_manifests = _sort_stage_manifests(
        tuple(manifest for manifest in manifests if manifest.stage == "stage1a_eval")
    )
    stage1b_manifests = _sort_stage_manifests(
        tuple(manifest for manifest in manifests if manifest.stage == "stage1b_eval")
    )
    stage2_eval_manifests = _sort_stage_manifests(
        tuple(manifest for manifest in manifests if manifest.stage == "stage2_eval")
    )
    stage2_inference_manifests = _sort_stage_manifests(
        tuple(manifest for manifest in manifests if manifest.stage == "stage2_inference")
    )

    stage2_eval_map = {manifest.run_name: manifest for manifest in stage2_eval_manifests}
    stage2_inference_map = {manifest.run_name: manifest for manifest in stage2_inference_manifests}
    stage2_run_names = sorted(set(stage2_eval_map) | set(stage2_inference_map))
    stage2_bundles = tuple(
        Stage2RunBundle(
            run_name=run_name,
            eval_manifest=stage2_eval_map.get(run_name),
            inference_manifest=stage2_inference_map.get(run_name),
        )
        for run_name in stage2_run_names
    )
    return ManifestRegistry(
        artifact_root=root,
        manifests=manifests,
        stage1a_manifests=stage1a_manifests,
        stage1b_manifests=stage1b_manifests,
        stage2_eval_manifests=stage2_eval_manifests,
        stage2_inference_manifests=stage2_inference_manifests,
        stage2_bundles=stage2_bundles,
    )


def load_normalized_run(manifest: ArtifactManifest) -> NormalizedRun:
    if manifest.stage == "stage1a_eval":
        return load_stage1a_run(manifest)
    if manifest.stage == "stage1b_eval":
        return load_stage1b_run(manifest)
    if manifest.stage == "stage2_eval":
        return load_stage2_eval_run(manifest)
    if manifest.stage == "stage2_inference":
        return load_stage2_inference_run(manifest)
    raise RuntimeError(f"Unsupported manifest stage for inspector: {manifest.stage}")


def compare_runs_by_sample_id(
    *,
    stage1a_run: NormalizedRun | None = None,
    stage1b_run: NormalizedRun | None = None,
    stage2_eval_run: NormalizedRun | None = None,
    stage2_inference_run: NormalizedRun | None = None,
) -> list[dict[str, Any]]:
    stage1a_lookup = sample_lookup(stage1a_run)
    stage1b_lookup = sample_lookup(stage1b_run)
    stage2_eval_lookup = sample_lookup(stage2_eval_run)
    stage2_inference_lookup = sample_lookup(stage2_inference_run)
    sample_ids = sorted(
        set(stage1a_lookup) | set(stage1b_lookup) | set(stage2_eval_lookup) | set(stage2_inference_lookup)
    )
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        rows.append(
            {
                "sample_id": sample_id,
                "stage1a": select_matching_sample(stage1a_run, sample_id),
                "stage1b": select_matching_sample(stage1b_run, sample_id),
                "stage2_eval": select_matching_sample(stage2_eval_run, sample_id),
                "stage2_inference": select_matching_sample(stage2_inference_run, sample_id),
            }
        )
    return rows
