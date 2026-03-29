"""Alpamayo-style end-to-end inference smoke script for minipamayo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .stage2.reasoning_sft.inference.runner import (
    build_wrapper_inputs_for_sample,
    load_reasoning_sample,
    load_stage2_inference_bundle,
)
from .utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from .utils.preflight import require_expected_cuda_toolkit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-style end-to-end wrapper inference on one JSONL sample."
    )
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--stage1b-checkpoint", type=str, required=True)
    parser.add_argument("--sample-jsonl", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--num-traj-sets", type=int, default=1)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--output-json", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    if args.num_traj_samples <= 0:
        raise RuntimeError("`num_traj_samples` must be > 0.")
    if args.num_traj_sets <= 0:
        raise RuntimeError("`num_traj_sets` must be > 0.")
    if args.max_generation_length <= 0:
        raise RuntimeError("`max_generation_length` must be > 0.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("`test_inference.py` currently expects CUDA.")
    require_expected_cuda_toolkit()

    bundle = load_stage2_inference_bundle(
        stage2_checkpoint_path=args.stage2_checkpoint,
        stage1b_checkpoint_path=args.stage1b_checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        flow_steps=args.flow_steps,
        device=device,
    )
    sample = load_reasoning_sample(args.sample_jsonl, args.sample_index)
    wrapper_inputs = build_wrapper_inputs_for_sample(
        processor=bundle["processor"],
        sample=sample,
        history_token_count=int(bundle["history_quantizer"].token_count),
        device=device,
    )

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = bundle["wrapper"].sample_trajectories_from_data_with_vlm_rollout(
            data=wrapper_inputs,
            top_p=args.top_p,
            top_k=None if args.top_k <= 0 else int(args.top_k),
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            num_traj_sets=args.num_traj_sets,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = sample["ego_future_xyz"].cpu().numpy()[0, :, :2]
    pred_xy = pred_xyz.detach().cpu().numpy()[0, :, :, :, :2].reshape(-1, gt_xy.shape[0], 2)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=-1).mean(-1)
    min_ade = float(diff.min())
    print("minADE:", min_ade, "meters")

    payload = {
        "stage2_checkpoint": str(Path(args.stage2_checkpoint).resolve()),
        "stage1b_checkpoint": str(Path(args.stage1b_checkpoint).resolve()),
        "sample_jsonl": str(Path(args.sample_jsonl).resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": str(sample["sample_id"]),
        "min_ade_m": min_ade,
        "cot": extra["cot"][0].tolist(),
        "prediction_shape": list(pred_xyz.shape),
        "prediction_xyz": pred_xyz.detach().cpu().tolist(),
        "prediction_rot": pred_rot.detach().cpu().tolist(),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
