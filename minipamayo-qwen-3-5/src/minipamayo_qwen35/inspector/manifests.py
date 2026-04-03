"""Manifest helpers for inspector-friendly eval and inference artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ArtifactManifest

IGNORED_SUMMARY_SUFFIXES = (
    ".progress.json",
    ".manifest.json",
    ".per_sample.jsonl",
    ".visualization_manifest.json",
    ".worst_samples.json",
)
IGNORED_SUMMARY_FILENAMES = {
    "visualization_manifest.json",
    "worst_samples.json",
}


def manifest_path_for_summary(summary_json: str | Path) -> Path:
    summary_path = Path(summary_json).resolve()
    return summary_path.with_name(f"{summary_path.stem}.manifest.json")


def load_manifest(path: str | Path) -> ArtifactManifest:
    return ArtifactManifest.from_dict(json.loads(Path(path).resolve().read_text(encoding="utf-8")))


def write_manifest(manifest: ArtifactManifest) -> Path:
    output_path = manifest.manifest_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def upsert_manifest(
    *,
    artifact_kind: str,
    stage: str,
    run_name: str,
    summary_json: str | Path,
    checkpoint: str | None = None,
    dataset_path: str | None = None,
    progress_json: str | None = None,
    per_sample_jsonl: str | None = None,
    plots_dir: str | None = None,
    plots: dict[str, str] | None = None,
    wandb_run_url: str | None = None,
) -> ArtifactManifest:
    summary_path = Path(summary_json).resolve()
    manifest_path = manifest_path_for_summary(summary_path)
    current: ArtifactManifest | None = None
    if manifest_path.exists():
        current = load_manifest(manifest_path)
    manifest = ArtifactManifest(
        artifact_kind=artifact_kind,
        stage=stage,
        run_name=run_name,
        summary_json=str(summary_path),
        checkpoint=str(Path(checkpoint).resolve()) if checkpoint else (current.checkpoint if current is not None else None),
        dataset_path=dataset_path or (current.dataset_path if current is not None else None),
        progress_json=(
            str(Path(progress_json).resolve())
            if progress_json
            else (current.progress_json if current is not None else None)
        ),
        per_sample_jsonl=(
            str(Path(per_sample_jsonl).resolve())
            if per_sample_jsonl
            else (current.per_sample_jsonl if current is not None else None)
        ),
        plots_dir=(
            str(Path(plots_dir).resolve())
            if plots_dir
            else (current.plots_dir if current is not None else None)
        ),
        plots=(
            {str(key): str(Path(value).resolve()) for key, value in plots.items()}
            if plots is not None
            else (dict(current.plots) if current is not None else {})
        ),
        wandb_run_url=wandb_run_url or (current.wandb_run_url if current is not None else None),
    )
    write_manifest(manifest)
    return manifest


def update_manifest_plots(
    *,
    summary_json: str | Path,
    plots_dir: str | Path,
    plots: dict[str, str],
    wandb_run_url: str | None = None,
) -> ArtifactManifest:
    manifest_path = manifest_path_for_summary(summary_json)
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest does not exist for summary JSON: {Path(summary_json).resolve()}")
    manifest = load_manifest(manifest_path)
    updated = ArtifactManifest(
        artifact_kind=manifest.artifact_kind,
        stage=manifest.stage,
        run_name=manifest.run_name,
        summary_json=manifest.summary_json,
        checkpoint=manifest.checkpoint,
        dataset_path=manifest.dataset_path,
        progress_json=manifest.progress_json,
        per_sample_jsonl=manifest.per_sample_jsonl,
        plots_dir=str(Path(plots_dir).resolve()),
        plots={str(key): str(Path(value).resolve()) for key, value in plots.items()},
        wandb_run_url=wandb_run_url or manifest.wandb_run_url,
    )
    write_manifest(updated)
    return updated


def infer_stage_from_summary_path(summary_path: str | Path) -> str | None:
    path = Path(summary_path).resolve()
    parts = path.parts
    if "artifacts" not in parts:
        return None
    try:
        artifacts_index = parts.index("artifacts")
        artifact_kind = parts[artifacts_index + 1]
        major_stage = parts[artifacts_index + 2]
        sub_stage = parts[artifacts_index + 3]
    except IndexError:
        return None
    if artifact_kind == "eval" and major_stage == "stage1" and sub_stage == "vlm_ce":
        return "stage1a_eval"
    if artifact_kind == "eval" and major_stage == "stage1" and sub_stage == "expert_cfm":
        return "stage1b_eval"
    if artifact_kind == "eval" and major_stage == "stage2" and sub_stage == "reasoning_sft":
        return "stage2_eval"
    if artifact_kind == "inference" and major_stage == "stage2" and sub_stage == "reasoning_sft":
        return "stage2_inference"
    if artifact_kind == "inference" and major_stage == "stage1" and sub_stage == "expert_cfm":
        return "stage1b_inference"
    return None


def infer_artifact_kind_from_summary_path(summary_path: str | Path) -> str | None:
    path = Path(summary_path).resolve()
    parts = path.parts
    if "artifacts" not in parts:
        return None
    try:
        return str(parts[parts.index("artifacts") + 1])
    except IndexError:
        return None


def infer_dataset_path_from_summary(summary: dict[str, Any]) -> str | None:
    for key in ["test_jsonl", "eval_jsonl", "sample_jsonl", "input_jsonl", "dataset_path"]:
        if key in summary:
            value = summary[key]
            if isinstance(value, list):
                return ",".join(str(item) for item in value)
            if isinstance(value, str):
                return value
    run_args = summary.get("run_args")
    if isinstance(run_args, dict):
        for key in ["eval_jsonl", "sample_jsonl", "input_jsonl"]:
            if key in run_args and run_args[key]:
                value = run_args[key]
                if isinstance(value, list):
                    return ",".join(str(item) for item in value)
                if isinstance(value, str):
                    return value
    return None


def infer_checkpoint_from_summary(summary: dict[str, Any]) -> str | None:
    checkpoint = summary.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint:
        return checkpoint
    run_args = summary.get("run_args")
    if isinstance(run_args, dict):
        checkpoint = run_args.get("checkpoint")
        if isinstance(checkpoint, str) and checkpoint:
            return checkpoint
    return None


def collect_plot_paths(output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir).resolve()
    if not output_path.exists():
        return {}
    plots: dict[str, str] = {}
    for path in sorted(output_path.iterdir()):
        if path.is_dir():
            continue
        if path.name == "visualization_manifest.json":
            continue
        if path.suffix.lower() not in {".png", ".json"}:
            continue
        plots[path.stem] = str(path.resolve())
    return plots


def is_summary_candidate(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if path.parent.name.endswith("_plots"):
        return False
    name = path.name
    if name in IGNORED_SUMMARY_FILENAMES:
        return False
    if "_bak_" in name:
        return False
    for suffix in IGNORED_SUMMARY_SUFFIXES:
        if name.endswith(suffix):
            return False
    return True
