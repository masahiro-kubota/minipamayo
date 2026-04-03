"""Canonical Stage 2 batch inference over a JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...inspector.manifests import upsert_manifest
from ...utils.eval_reporting import (
    EvalReporter,
    add_eval_reporting_args,
    apply_eval_reporting_artifact_policy,
    reporting_path_keys,
    validate_eval_reporting_args,
)
from ...utils.image_budget import validate_canonical_image_budget
from ...utils.json_config import normalize_required_string_list
from .cli import artifact_scope_for_config, parse_stage2_json_only_args, require_stage2_cuda_device
from .dataset import ReasoningSftJsonlDataset
from .inference import build_stage2_inference_payload

CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage1b_checkpoint",
    "input_jsonl",
    "output_json",
} | reporting_path_keys(include_per_sample_jsonl=True)
MULTI_VALUE_CONFIG_KEYS = {"input_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical Stage 2 inference for every sample in a JSONL dataset.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--input-jsonl", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--max-reasoning-tokens", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    add_eval_reporting_args(parser, include_per_sample_jsonl=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage2_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        list_keys=MULTI_VALUE_CONFIG_KEYS,
        json_only_error="Stage 2 batch inference accepts only --config-json. Put all settings in the JSON file.",
    )
    args.input_jsonl = normalize_required_string_list(args.input_jsonl, key_name="input_jsonl")
    if not args.checkpoint or not args.stage1b_checkpoint:
        raise RuntimeError("`checkpoint` and `stage1b_checkpoint` must be defined in config JSON.")
    if args.max_reasoning_tokens <= 0:
        raise RuntimeError("`max_reasoning_tokens` must be > 0.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.temperature <= 0.0:
        raise RuntimeError("`temperature` must be > 0.")
    if not (0.0 < args.top_p <= 1.0):
        raise RuntimeError("`top_p` must be in (0, 1].")
    if args.top_k < 0:
        raise RuntimeError("`top_k` must be >= 0.")
    apply_eval_reporting_artifact_policy(
        args,
        scope=artifact_scope_for_config(args.config_json, kind="inference"),
        include_per_sample_jsonl=True,
    )
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    validate_eval_reporting_args(args, require_per_sample_jsonl=True)
    return args


def main() -> None:
    args = parse_args()
    from ...models.checkpoint_loader import load_stage2_inference_bundle

    device = require_stage2_cuda_device(
        device_name=args.device,
        git_cwd=Path(__file__).resolve().parent,
        error_message="Canonical Stage 2 batch inference currently expects CUDA.",
    )
    dataset = ReasoningSftJsonlDataset(args.input_jsonl, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Batch inference dataset is empty.")

    bundle = load_stage2_inference_bundle(
        stage2_checkpoint_path=args.checkpoint,
        stage1b_checkpoint_path=args.stage1b_checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        flow_steps=args.flow_steps,
        device=device,
    )
    bundle["device"] = device

    reporter = EvalReporter.from_args(
        args=args,
        stage="stage2_inference",
        total_samples=len(dataset),
        checkpoint=args.checkpoint,
        dataset_path=",".join(args.input_jsonl),
        extra_wandb_config={
            "entrypoint": "stage2.reasoning_sft.batch_inference",
            "flow_steps": int(args.flow_steps),
            "max_reasoning_tokens": int(args.max_reasoning_tokens),
            "max_samples": int(args.max_samples),
        },
    )
    wandb_run_url = str(getattr(reporter.wandb_run, "url", ""))
    reporter.emit_setup(
        "stage2_batch_inference_setup",
        {
            "checkpoint": args.checkpoint,
            "stage1b_checkpoint": args.stage1b_checkpoint,
            "input_jsonl": args.input_jsonl,
            "dataset_size": len(dataset),
            "max_reasoning_tokens": args.max_reasoning_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
    )

    total_ade = 0.0
    total_fde = 0.0
    try:
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            payload = build_stage2_inference_payload(
                bundle=bundle,
                sample=sample,
                checkpoint_path=args.checkpoint,
                stage1b_checkpoint=args.stage1b_checkpoint,
                sample_jsonl=args.input_jsonl[0] if len(args.input_jsonl) == 1 else ",".join(args.input_jsonl),
                sample_index=sample_index,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
                max_reasoning_tokens=args.max_reasoning_tokens,
                flow_steps=args.flow_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            total_ade += float(payload["metrics"]["ade_m"])
            total_fde += float(payload["metrics"]["fde_m"])
            reporter.emit_sample(
                {
                    "event": "sample",
                    **payload,
                },
                print_to_stdout=sample_index < 3,
            )
            reporter.emit_progress(
                processed_samples=sample_index + 1,
                running_metrics={
                    "ade_m": total_ade / float(sample_index + 1),
                    "fde_m": total_fde / float(sample_index + 1),
                },
            )

        summary = {
            "config_json": args.config_json,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "stage1b_checkpoint": str(Path(args.stage1b_checkpoint).resolve()),
            "input_jsonl": args.input_jsonl,
            "num_samples": len(dataset),
            "metrics": {
                "ade_m": total_ade / float(len(dataset)),
                "fde_m": total_fde / float(len(dataset)),
            },
            "run_config": {
                "max_reasoning_tokens": int(args.max_reasoning_tokens),
                "flow_steps": int(args.flow_steps),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
            },
        }
        reporter.emit_summary("stage2_batch_inference_summary", summary)
        upsert_manifest(
            artifact_kind="inference",
            stage="stage2_inference",
            run_name=Path(args.output_json).resolve().stem,
            summary_json=args.output_json,
            checkpoint=args.checkpoint,
            dataset_path=",".join(args.input_jsonl),
            progress_json=str(args.progress_json),
            per_sample_jsonl=str(args.per_sample_jsonl),
            wandb_run_url=wandb_run_url,
        )
    except Exception as exc:
        reporter.emit_failure("stage2_batch_inference_failure", exc)
        raise
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
