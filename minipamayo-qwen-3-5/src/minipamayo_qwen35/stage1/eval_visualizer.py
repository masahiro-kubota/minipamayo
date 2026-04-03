"""Shared orchestration helpers for Stage 1 eval visualization entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..inspector.manifests import update_manifest_plots
from ..utils.artifact_paths import ArtifactScope, apply_visualization_artifact_policy
from ..utils.preflight import init_required_online_wandb
from .eval_viz_common import load_json, load_jsonl, write_json

CONFIG_PATH_KEYS = {
    "summary_json",
    "per_sample_jsonl",
    "output_dir",
}


@dataclass(frozen=True)
class Stage1VisualizationContext:
    args: argparse.Namespace
    summary_path: Path
    per_sample_path: Path
    output_dir: Path
    overlays_dir: Path
    summary: dict[str, Any]
    samples: list[dict[str, Any]]
    worst_table_count: int


def build_stage1_visualize_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--per-sample-jsonl", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--overlay-count", type=int, default=16)
    parser.add_argument("--worst-table-count", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def parse_stage1_visualize_args(
    *,
    parser: argparse.ArgumentParser,
    parse_json_only_args: Callable[[argparse.ArgumentParser], argparse.Namespace],
    scope_for_config: Callable[[str], ArtifactScope],
) -> argparse.Namespace:
    args = parse_json_only_args(parser)
    apply_visualization_artifact_policy(args, scope=scope_for_config(str(args.config_json)))
    for key in ["summary_json", "per_sample_jsonl", "wandb_project", "wandb_run_name"]:
        if not str(getattr(args, key, "")):
            raise RuntimeError(f"`{key}` must be defined in the config JSON.")
    if int(args.overlay_count) <= 0:
        raise RuntimeError("`overlay_count` must be > 0.")
    if int(args.worst_table_count) <= 0:
        raise RuntimeError("`worst_table_count` must be > 0.")
    if int(args.dpi) <= 0:
        raise RuntimeError("`dpi` must be > 0.")
    return args


def load_visualization_context(
    args: argparse.Namespace,
    *,
    ensure_plot_fields: Callable[[list[dict[str, Any]]], None],
) -> Stage1VisualizationContext:
    summary_path = Path(args.summary_json).resolve()
    per_sample_path = Path(args.per_sample_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "worst_sample_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    samples = load_jsonl(per_sample_path)
    ensure_plot_fields(samples)

    expected_samples = int(summary.get("num_samples", 0))
    if expected_samples > 0 and expected_samples != len(samples):
        raise RuntimeError(
            f"Per-sample JSONL count {len(samples)} does not match summary num_samples {expected_samples}."
        )

    return Stage1VisualizationContext(
        args=args,
        summary_path=summary_path,
        per_sample_path=per_sample_path,
        output_dir=output_dir,
        overlays_dir=overlays_dir,
        summary=summary,
        samples=samples,
        worst_table_count=min(int(args.worst_table_count), len(samples)),
    )


def finalize_stage1_visualization(
    *,
    context: Stage1VisualizationContext,
    entrypoint: str,
    plot_paths: dict[str, Path],
    worst_manifest_rows: list[dict[str, Any]],
    populate_run_summary: Callable[[Any, Stage1VisualizationContext], None],
    log_wandb_artifacts: Callable[[Any, Any, Stage1VisualizationContext, dict[str, Path], list[dict[str, Any]]], None],
    extra_wandb_config: dict[str, Any] | None = None,
) -> None:
    run = init_required_online_wandb(
        project=str(context.args.wandb_project),
        entity=str(getattr(context.args, "wandb_entity", "")),
        name=str(context.args.wandb_run_name),
        config={
            "entrypoint": entrypoint,
            "config_json": str(context.args.config_json),
            "summary_json": str(context.summary_path),
            "per_sample_jsonl": str(context.per_sample_path),
            "output_dir": str(context.output_dir),
            "num_samples": len(context.samples),
            "overlay_count": int(context.args.overlay_count),
            "worst_table_count": context.worst_table_count,
            **(extra_wandb_config or {}),
        },
    )
    try:
        import wandb

        populate_run_summary(run, context)
        log_wandb_artifacts(run, wandb, context, plot_paths, worst_manifest_rows)
        manifest = {
            "summary_json": str(context.summary_path),
            "per_sample_jsonl": str(context.per_sample_path),
            "output_dir": str(context.output_dir),
            "plots": {key: str(path) for key, path in plot_paths.items() if key != "manifest"},
            "worst_samples": worst_manifest_rows,
            "wandb": {
                "project": context.args.wandb_project,
                "run_name": context.args.wandb_run_name,
                "run_url": getattr(run, "url", ""),
            },
        }
        write_json(plot_paths["manifest"], manifest)
        update_manifest_plots(
            summary_json=context.summary_path,
            plots_dir=context.output_dir,
            plots={key: str(path) for key, path in plot_paths.items() if key != "manifest"},
            wandb_run_url=getattr(run, "url", ""),
        )
        run.summary["manifest_path"] = str(plot_paths["manifest"])
        run.summary["output_dir"] = str(context.output_dir)
    finally:
        run.finish()
