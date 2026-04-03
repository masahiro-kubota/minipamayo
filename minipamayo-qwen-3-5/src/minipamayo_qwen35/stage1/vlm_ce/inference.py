"""Alpamayo-style Stage 1 sample inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...contract.prompt import DEFAULT_SYSTEM_PROMPT, build_stage1_question_user_text
from ...utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
)
from ...utils.preflight import require_cuda_device
from ...utils.run_metadata import collect_processor_settings
from ..checkpoint_completion import require_completed_training_run
from ..dataset import Stage1JsonlDataset, stage1_collate
from ..stage1a_runtime import load_stage1a_runtime, run_stage1a_rollout_batch
from .cli import parse_config_json_only_args
from .metrics import infer_vision_tokens, require_record_field

CONFIG_PATH_KEYS = {
    "checkpoint",
    "test_jsonl",
    "output_json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-style Stage 1 inference on one sample."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--test-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_config_json_only_args(
        parser,
        path_keys=CONFIG_PATH_KEYS,
        error_message="Stage 1 inference accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.test_jsonl:
        raise RuntimeError("`test_jsonl` must be defined in the config JSON.")
    if not args.output_json:
        raise RuntimeError("`output_json` must be defined in the config JSON.")
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    require_completed_training_run(
        args.checkpoint,
        checkpoint_label="Stage 1A checkpoint",
        required_summary_keys=["completed_epochs", "best_epoch", "stop_reason"],
        allowed_stop_reasons={"max_epochs", "early_stopping"},
    )
    device = require_cuda_device(
        device_name=args.device,
        git_cwd=Path(__file__).resolve().parent,
        error_message="Stage 1 Alpamayo-style inference currently expects CUDA.",
    )

    runtime = load_stage1a_runtime(args, device=device)
    checkpoint_path = Path(args.checkpoint)
    dataset = Stage1JsonlDataset(args.test_jsonl)
    if args.sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index`={args.sample_index} is out of range for test dataset size {len(dataset)}."
        )
    sample = dataset[args.sample_index]
    record = dataset.records[args.sample_index]
    batch = stage1_collate([sample])
    rollout = run_stage1a_rollout_batch(runtime, batch, device=device)
    prompt_inputs = rollout["prompt_inputs"]
    decoded = rollout["decoded_rows"][0]
    pred_token_ids = decoded["pred_token_ids"]
    pred_bin_ids = decoded["pred_bin_ids"]
    pred_target_tensor = decoded["pred_target_tensor"]
    pred_action_tensor = decoded["pred_action_tensor"]
    pred_waypoints = decoded["pred_waypoints"]
    gt_target_tensor = decoded["gt_target_tensor"]
    gt_action_tensor = decoded["gt_action_tensor"]
    gt_waypoints = decoded["gt_waypoints"]

    image_grid_thw, vision_tokens = infer_vision_tokens(prompt_inputs)
    user_text = build_stage1_question_user_text(runtime.question, runtime.history_token_count)
    processor_settings = collect_processor_settings(
        runtime.processor,
        requested_min_pixels=args.image_min_pixels,
        requested_max_pixels=args.image_max_pixels,
    )

    output_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_kind": runtime.checkpoint["checkpoint_kind"],
        "test_jsonl": str(Path(args.test_jsonl).resolve()),
        "sample_index": args.sample_index,
        "sample_id": str(require_record_field(record, "sample_id")),
        "source_frame_id": int(require_record_field(record, "source_frame_id")),
        "image_path": str(sample["image_path"]),
        "message_style": "alpamayo_like_stage1",
        "message": {
            "system_text": DEFAULT_SYSTEM_PROMPT,
            "user_text": user_text,
            "num_images": 1,
            "history_steps": int(runtime.stage1_metadata["history_steps"]),
            "history_token_count": runtime.history_token_count,
            "add_generation_prompt": True,
        },
        "prompt_stats": {
            "input_ids_len": int(prompt_inputs["input_ids"].shape[-1]),
            "image_grid_thw": image_grid_thw,
            "vision_tokens": vision_tokens,
            "min_pixels": args.image_min_pixels,
            "max_pixels": args.image_max_pixels,
        },
        "processor_settings": processor_settings,
        "action_representation": str(runtime.stage1_metadata["action_representation"]),
        "question": runtime.question,
        "prediction": {
            "token_ids": pred_token_ids,
            "bin_ids": pred_bin_ids,
            "target": pred_target_tensor.tolist(),
            "full_action": pred_action_tensor.tolist(),
            "waypoints": pred_waypoints.tolist(),
        },
        "ground_truth": {
            "target": gt_target_tensor.tolist(),
            "full_action": gt_action_tensor.tolist(),
            "waypoints": gt_waypoints.tolist(),
        },
        "metrics": {
            "target_mae": decoded["target_mae"],
            "ade_m": decoded["ade_m"],
            "fde_m": decoded["fde_m"],
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "stage1_inference_complete",
                "checkpoint": str(checkpoint_path),
                "sample_id": output_payload["sample_id"],
                "output_json": str(output_path),
                "action_representation": output_payload["action_representation"],
                "vision_tokens": vision_tokens,
                "input_ids_len": output_payload["prompt_stats"]["input_ids_len"],
                "ade_m": decoded["ade_m"],
                "fde_m": decoded["fde_m"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
