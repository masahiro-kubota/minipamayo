"""Backfill inspector manifests from existing local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifests import (
    collect_plot_paths,
    infer_artifact_kind_from_summary_path,
    infer_checkpoint_from_summary,
    infer_dataset_path_from_summary,
    infer_stage_from_summary_path,
    is_summary_candidate,
    manifest_path_for_summary,
    upsert_manifest,
)
from .registry import default_artifact_root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_visualization_wandb_url(output_dir: Path) -> str | None:
    manifest_path = output_dir / "visualization_manifest.json"
    if not manifest_path.exists():
        return None
    payload = _load_json(manifest_path)
    wandb_payload = payload.get("wandb")
    if isinstance(wandb_payload, dict):
        run_url = wandb_payload.get("run_url")
        if isinstance(run_url, str) and run_url:
            return run_url
    return None


def backfill_artifact_manifests(artifact_root: str | Path) -> list[Path]:
    root = Path(artifact_root).resolve()
    written_paths: list[Path] = []
    for summary_path in sorted(root.rglob("*.json")):
        if not is_summary_candidate(summary_path):
            continue
        stage = infer_stage_from_summary_path(summary_path)
        artifact_kind = infer_artifact_kind_from_summary_path(summary_path)
        if stage is None or artifact_kind is None:
            continue
        summary = _load_json(summary_path)
        per_sample_path = summary_path.with_name(f"{summary_path.stem}.per_sample.jsonl")
        progress_path = summary_path.with_name(f"{summary_path.stem}.progress.json")
        plots_dir = summary_path.parent / f"{summary_path.stem}_plots"
        manifest = upsert_manifest(
            artifact_kind=artifact_kind,
            stage=stage,
            run_name=summary_path.stem,
            summary_json=summary_path,
            checkpoint=infer_checkpoint_from_summary(summary),
            dataset_path=infer_dataset_path_from_summary(summary),
            progress_json=str(progress_path.resolve()) if progress_path.exists() else None,
            per_sample_jsonl=str(per_sample_path.resolve()) if per_sample_path.exists() else None,
            plots_dir=str(plots_dir.resolve()) if plots_dir.exists() else None,
            plots=collect_plot_paths(plots_dir) if plots_dir.exists() else None,
            wandb_run_url=_maybe_visualization_wandb_url(plots_dir) if plots_dir.exists() else None,
        )
        written_paths.append(manifest.manifest_path)
    return written_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill inspector manifests from local artifacts.")
    parser.add_argument("--artifact-root", type=str, default=str(default_artifact_root()))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written_paths = backfill_artifact_manifests(args.artifact_root)
    payload = {
        "artifact_root": str(Path(args.artifact_root).resolve()),
        "manifests_written": [str(path) for path in written_paths],
        "count": len(written_paths),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
