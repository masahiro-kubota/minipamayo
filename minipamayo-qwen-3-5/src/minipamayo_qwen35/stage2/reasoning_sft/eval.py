"""Canonical Stage 2 reasoning SFT evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ...utils.eval_reporting import (
    EvalReporter,
    add_eval_reporting_args,
    reporting_path_keys,
    validate_eval_reporting_args,
)
from ...utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ...utils.run_metadata import collect_dataset_view_fingerprint, collect_processor_settings
from .cli import parse_stage2_json_only_args, require_stage2_cuda_device

CONFIG_PATH_KEYS = {"checkpoint", "eval_jsonl", "output_json"} | reporting_path_keys(
    include_per_sample_jsonl=False
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical Stage 2 reasoning SFT.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    add_eval_reporting_args(parser, include_per_sample_jsonl=False)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage2_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        json_only_error="Stage 2 evaluation accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.eval_jsonl:
        raise RuntimeError("`eval_jsonl` must be defined in the config JSON.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    validate_eval_reporting_args(args)
    return args


def main() -> None:
    args = parse_args()
    from ...models.checkpoint_loader import load_stage2_checkpoint_bundle
    from .dataset import ReasoningSftJsonlDataset, build_reasoning_sft_dataloader
    from .runtime import evaluate_stage2

    device = require_stage2_cuda_device(
        device_name=args.device,
        git_cwd=Path(__file__).resolve().parent,
        error_message="This Stage 2 evaluator is intended to run on CUDA.",
    )

    eval_dataset = ReasoningSftJsonlDataset(args.eval_jsonl, max_samples=args.max_samples)
    if len(eval_dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")
    eval_loader = build_reasoning_sft_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    bundle = load_stage2_checkpoint_bundle(
        checkpoint_path=args.checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        device=device,
        use_cache=False,
    )
    checkpoint = bundle["checkpoint"]
    checkpoint_args = bundle["checkpoint_args"]
    stage1_checkpoint = bundle["stage1_checkpoint"]
    model = bundle["model"]
    processor = bundle["processor"]
    history_registry = bundle["history_registry"]
    history_quantizer = bundle["history_quantizer"]
    model_dtype = bundle["model_dtype"]
    model.train()
    reporter = EvalReporter.from_args(
        args=args,
        stage="stage2_eval",
        total_samples=len(eval_dataset),
        checkpoint=args.checkpoint,
        dataset_path=args.eval_jsonl,
        extra_wandb_config={
            "entrypoint": "stage2.reasoning_sft.eval",
            "batch_size": int(args.batch_size),
            "max_samples": int(args.max_samples),
        },
    )
    reporter.emit_setup(
        "stage2_eval_setup",
        {
            "checkpoint": args.checkpoint,
            "base_stage1_checkpoint": str(checkpoint_args["stage1a_checkpoint"]),
            "eval_jsonl": args.eval_jsonl,
            "eval_size": len(eval_dataset),
            "batch_size": args.batch_size,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
        },
    )

    try:
        metrics = evaluate_stage2(
            model=model,
            dataloader=eval_loader,
            processor=processor,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            device=device,
            model_dtype=model_dtype,
            handoff_loss_weight=float(checkpoint_args["handoff_loss_weight"]),
            progress_callback=lambda processed, running_metrics: reporter.emit_progress(
                processed_samples=processed,
                running_metrics=running_metrics,
            ),
        )

        summary = {
            "config_json": args.config_json,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "checkpoint": args.checkpoint,
            "base_stage1_checkpoint": str(checkpoint_args["stage1a_checkpoint"]),
            "eval_jsonl": args.eval_jsonl,
            "eval_size": len(eval_dataset),
            "metrics": metrics,
            "dataset_fingerprint": collect_dataset_view_fingerprint(eval_dataset),
            "processor_settings": collect_processor_settings(
                processor,
                requested_min_pixels=args.image_min_pixels or None,
                requested_max_pixels=args.image_max_pixels or None,
            ),
            "base_stage1_metadata": stage1_checkpoint.get("stage1_metadata"),
            "stage2_metadata": checkpoint.get("stage2_metadata"),
        }
        reporter.emit_summary("stage2_eval_summary", summary)
    except Exception as exc:
        reporter.emit_failure("stage2_eval_failure", exc)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
