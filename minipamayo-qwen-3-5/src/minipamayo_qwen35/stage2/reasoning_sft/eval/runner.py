"""Canonical Stage 2 reasoning SFT evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from ....stage1.eval import load_components
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.run_metadata import collect_dataset_view_fingerprint, collect_processor_settings
from ..dataset import ReasoningSftJsonlDataset, reasoning_sft_collate
from ..train.runner import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {"checkpoint", "eval_jsonl", "output_json"}


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
    return parser


def _load_config_args(config_json: str, parser: argparse.ArgumentParser) -> tuple[str, dict, dict]:
    config_path, payload = load_json_payload(config_json)
    raw_config = payload.get("args") if isinstance(payload, dict) and "args" in payload else payload
    if not isinstance(raw_config, dict):
        raise RuntimeError("Config JSON must be an object or an object with an `args` object.")

    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={
            "project_root": PROJECT_ROOT,
            "config_dir": config_path.parent,
        },
    )
    config_args = normalize_arg_config(
        raw_config,
        parser,
        exclude_dests={"help", "config_json"},
        path_keys=CONFIG_PATH_KEYS,
        base_dir=base_dir,
    )
    return str(config_path), payload, config_args


def parse_args() -> argparse.Namespace:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_parser().parse_args()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError("Stage 2 evaluation accepts only --config-json. Put all settings in the JSON file.")

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.eval_jsonl:
        raise RuntimeError("`eval_jsonl` must be defined in the config JSON.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("This Stage 2 evaluator is intended to run on CUDA.")

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict) or "stage1a_checkpoint" not in checkpoint_args:
        raise RuntimeError("Stage 2 checkpoint is missing canonical `stage1a_checkpoint` args metadata.")
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Stage 2 checkpoint is missing canonical `model_state_dict`.")

    eval_dataset = ReasoningSftJsonlDataset(args.eval_jsonl, max_samples=args.max_samples)
    if len(eval_dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty.")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=reasoning_sft_collate,
    )

    stage1_args = SimpleNamespace(
        checkpoint=str(checkpoint_args["stage1a_checkpoint"]),
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    (
        stage1_checkpoint,
        model,
        processor,
        registry,
        _history_registry,
        _history_quantizer,
        quantizer,
        model_dtype,
    ) = load_components(stage1_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.config.use_cache = False
    model.to(device)
    model.train()

    metrics = evaluate(
        model=model,
        dataloader=eval_loader,
        processor=processor,
        registry=registry,
        quantizer=quantizer,
        device=device,
        model_dtype=model_dtype,
        action_loss_weight=float(checkpoint_args.get("action_loss_weight", 2.0)),
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
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"event": "stage2_eval_summary", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
