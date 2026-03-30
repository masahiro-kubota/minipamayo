"""Canonical Stage 2 inference with Alpamayo-style handoff to Stage 1B expert."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ....contract.prompt import (
    COT_START_TOKEN,
    TRAJ_FUTURE_START_TOKEN,
    build_multimodal_messages,
    build_reasoning_user_text,
)
from ....contract.record_adapter import (
    canonicalize_history_batch_for_action_space,
)
from ....helper import to_device
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.run_metadata import collect_processor_settings
from ..bundle import load_stage2_inference_bundle
from ..cli import parse_stage2_json_only_args, require_stage2_cuda_device
from ..dataset import load_reasoning_sample

CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage1b_checkpoint",
    "sample_jsonl",
    "output_json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical Stage 2 reasoning rollout and hand off to Stage 1B expert."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--sample-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--max-reasoning-tokens", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage2_json_only_args(
        parser=parser,
        path_keys=CONFIG_PATH_KEYS,
        json_only_error="Stage 2 inference accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.checkpoint or not args.stage1b_checkpoint or not args.sample_jsonl:
        raise RuntimeError(
            "`checkpoint`, `stage1b_checkpoint`, and `sample_jsonl` must be defined in config JSON."
        )
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
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
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def build_wrapper_inputs_for_sample(
    *,
    processor,
    sample: dict[str, Any],
    history_token_count: int,
    device: torch.device,
) -> dict[str, Any]:
    image_path = Path(sample["image_path"])
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        frame_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
    messages = build_multimodal_messages(
        frames=frame_tensor,
        user_text=build_reasoning_user_text(history_token_count),
        assistant_prefill=COT_START_TOKEN,
    )
    tokenized_data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    wrapper_inputs = {
        "ego_history_xyz": sample["ego_history_xyz"].unsqueeze(0).to(dtype=torch.float32),
        "ego_history_rot": sample["ego_history_rot"].unsqueeze(0).to(dtype=torch.float32),
        "tokenized_data": tokenized_data,
    }
    return to_device(wrapper_inputs, device=device)


def main() -> None:
    args = parse_args()
    device = require_stage2_cuda_device(
        device_name=args.device,
        git_cwd=Path(__file__).resolve().parent,
        error_message="Canonical Stage 2 inference currently expects CUDA.",
    )

    bundle = load_stage2_inference_bundle(
        stage2_checkpoint_path=args.checkpoint,
        stage1b_checkpoint_path=args.stage1b_checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        flow_steps=args.flow_steps,
        device=device,
    )
    checkpoint = bundle["checkpoint"]
    checkpoint_args = bundle["checkpoint_args"]
    processor = bundle["processor"]
    history_quantizer = bundle["history_quantizer"]
    wrapper = bundle["wrapper"]
    sample = load_reasoning_sample(args.sample_jsonl, args.sample_index)
    wrapper_inputs = build_wrapper_inputs_for_sample(
        processor=processor,
        sample=sample,
        history_token_count=int(history_quantizer.token_count),
        device=device,
    )
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with amp_context:
        pred_xyz, pred_rot, extra = wrapper.sample_trajectories_from_data_with_vlm_rollout(
            wrapper_inputs,
            top_p=args.top_p,
            top_k=None if args.top_k <= 0 else int(args.top_k),
            temperature=args.temperature,
            num_traj_samples=1,
            num_traj_sets=1,
            return_extra=True,
            max_generation_length=args.max_reasoning_tokens,
        )
    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN))
    if stop_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|traj_future_start|>`.")
    reasoning_text = str(extra["cot"][0, 0, 0])

    history_xyz, history_rot = canonicalize_history_batch_for_action_space(
        sample["ego_history_xyz"].unsqueeze(0).to(device=device, dtype=torch.float32),
        sample["ego_history_rot"].unsqueeze(0).to(device=device, dtype=torch.float32),
    )
    pred_action = wrapper.action_space.traj_to_action(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        traj_future_xyz=pred_xyz[:, 0, 0],
        traj_future_rot=pred_rot[:, 0, 0],
    )
    pred_waypoints = pred_xyz[0, 0, 0, :, :2].detach().cpu()
    gt_waypoints = sample["gt_waypoints"].to(dtype=torch.float32)
    errors = torch.norm(pred_waypoints - gt_waypoints, dim=1)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1a_checkpoint": str(Path(checkpoint_args["stage1a_checkpoint"]).resolve()),
        "stage1b_checkpoint": str(Path(args.stage1b_checkpoint).resolve()),
        "sample_jsonl": str(Path(args.sample_jsonl).resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": sample["sample_id"],
        "prompt_style": "alpamayo_r1_wrapper",
        "reasoning": {
            "text": reasoning_text,
            "token_ids": None,
            "stop_token_id": stop_token_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "prediction": {
            "action": pred_action[0].reshape(-1).detach().cpu().tolist(),
            "waypoints": pred_waypoints.tolist(),
            "traj_xyz": pred_xyz[0, 0, 0].detach().cpu().tolist(),
            "traj_rot": pred_rot[0, 0, 0].detach().cpu().tolist(),
        },
        "ground_truth": {
            "waypoints": gt_waypoints.tolist(),
            "reasoning_text": sample["reasoning_text"],
        },
        "metrics": {
            "ade_m": float(errors.mean().item()),
            "fde_m": float(errors[-1].item()),
        },
        "processor_settings": collect_processor_settings(
            processor,
            requested_min_pixels=args.image_min_pixels,
            requested_max_pixels=args.image_max_pixels,
        ),
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
