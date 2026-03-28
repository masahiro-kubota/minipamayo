"""Canonical Stage 1B single-sample inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ....stage1.data.dataset import Stage1JsonlDataset
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import require_expected_cuda_toolkit
from ..common import (
    extract_prompt_cache,
    freeze_module,
    infer_prompt_text,
    load_stage1_condition_components,
    prepare_condition_inputs,
)
from ..action_space import UnicycleAccelCurvatureActionSpace
from ..diffusion import FlowMatchingDiffusion
from ..model import load_action_expert_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {"checkpoint", "stage1_checkpoint", "sample_jsonl", "output_json"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run canonical Stage 1B inference on one sample.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--sample-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--flow-steps", type=int, default=10)
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
        base_dirs={"project_root": PROJECT_ROOT, "config_dir": config_path.parent},
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
        raise RuntimeError(
            "Stage 1B inference accepts only --config-json. Put all settings in the JSON file."
        )
    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    if not args.checkpoint or not args.stage1_checkpoint or not args.sample_jsonl:
        raise RuntimeError("`checkpoint`, `stage1_checkpoint`, and `sample_jsonl` must be defined.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Canonical Stage 1B inference currently expects CUDA.")
    require_expected_cuda_toolkit()
    dataset = Stage1JsonlDataset(args.sample_jsonl)
    if args.sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index` {args.sample_index} is out of range for dataset size {len(dataset)}."
        )
    sample = dataset[args.sample_index]
    batch = {
        "sample_id": [sample["sample_id"]],
        "image_path": [sample["image_path"]],
        "action": sample["action"].unsqueeze(0),
        "v0": sample["v0"].unsqueeze(0),
        "dt": sample["dt"].unsqueeze(0),
        "gt_waypoints": sample["gt_waypoints"].unsqueeze(0),
        "ego_history_xyz": sample["ego_history_xyz"].unsqueeze(0),
        "ego_history_rot": sample["ego_history_rot"].unsqueeze(0),
        "command": [sample["command"]],
    }

    (
        stage1_checkpoint,
        model,
        processor,
        _registry,
        history_registry,
        history_quantizer,
        _quantizer,
        _model_dtype,
    ) = load_stage1_condition_components(args)
    freeze_module(model)
    prompt_text = infer_prompt_text(stage1_checkpoint, processor)
    prompt_inputs = prepare_condition_inputs(
        model=model,
        batch=batch,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        prompt_text=prompt_text,
        device=device,
    )
    prompt_cache, prompt_attention_mask = extract_prompt_cache(model, prompt_inputs)
    expert, checkpoint = load_action_expert_from_checkpoint(args.checkpoint, device)
    if "stage1b_metadata" not in checkpoint or not isinstance(checkpoint["stage1b_metadata"], dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `stage1b_metadata`.")
    stage1b_metadata = checkpoint["stage1b_metadata"]
    if "dt" not in stage1b_metadata:
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `dt`.")
    dt = float(stage1b_metadata["dt"])
    diffusion = FlowMatchingDiffusion(n_steps=args.flow_steps)
    action_space = UnicycleAccelCurvatureActionSpace(
        k=int(stage1b_metadata["k"]) if "k" in stage1b_metadata else int(sample["action"].shape[0] // 2),
        dt=dt,
    )
    pred_action = diffusion.sample(
        expert=expert,
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    ).reshape(1, -1, 2)
    pred_xyz, _pred_rot = action_space.action_to_traj(
        traj_history_xyz=batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
        traj_history_rot=batch["ego_history_rot"].to(device=device, dtype=torch.float32),
        action=pred_action,
    )
    pred_waypoints = pred_xyz[0, 0, :, :2].detach().cpu()
    gt_waypoints = batch["gt_waypoints"][0].to(dtype=torch.float32)
    errors = torch.norm(pred_waypoints.cpu() - gt_waypoints, dim=1)
    payload = {
        "sample_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "flow_steps": args.flow_steps,
        "dt": dt,
        "pred_action": pred_action[0].detach().cpu().tolist(),
        "pred_waypoints": pred_waypoints.detach().cpu().tolist(),
        "gt_waypoints": gt_waypoints.tolist(),
        "ade_m": float(errors.mean().item()),
        "fde_m": float(errors[-1].item()),
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
