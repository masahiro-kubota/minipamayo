"""Canonical Stage 1B single-sample inference."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import torch

from ....action_space.record_adapter import (
    canonicalize_future_batch_from_action_space,
    canonicalize_history_batch_for_action_space,
)
from ....stage1.dataset import Stage1JsonlDataset
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import require_expected_cuda_toolkit
from ....diffusion.action_expert import FlowMatchingDiffusion
from ....models.action_expert import load_action_expert_from_checkpoint
from ..common import (
    apply_longitudinal_pid_override,
    extract_prompt_cache,
    freeze_module,
    infer_prompt_text,
    load_stage1_condition_components,
    prepare_condition_inputs,
)
from ....action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

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
    parser.add_argument("--include-pid-override", action="store_true")
    parser.add_argument("--pid-target-speed-kmh", type=float, default=24.0)
    parser.add_argument("--pid-kp", type=float, default=1.0)
    parser.add_argument("--pid-ki", type=float, default=0.05)
    parser.add_argument("--pid-kd", type=float, default=0.0)
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
    if args.pid_target_speed_kmh <= 0.0:
        raise RuntimeError("`pid_target_speed_kmh` must be > 0.")
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
    if "action_space_cfg" not in stage1b_metadata or not isinstance(
        stage1b_metadata["action_space_cfg"], dict
    ):
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `action_space_cfg`.")
    action_space_cfg = dict(stage1b_metadata["action_space_cfg"])
    action_space_cfg.pop("_target_", None)
    action_space = UnicycleAccelCurvatureActionSpace(**action_space_cfg)
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with amp_context:
        pred_action = diffusion.sample(
            expert=expert,
            prompt_cache=prompt_cache,
            prompt_attention_mask=prompt_attention_mask,
        ).reshape(1, -1, 2)
    history_xyz, history_rot = canonicalize_history_batch_for_action_space(
        batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
        batch["ego_history_rot"].to(device=device, dtype=torch.float32),
    )
    pred_xyz, _pred_rot = action_space.action_to_traj(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        action=pred_action,
    )
    pred_xyz, _pred_rot = canonicalize_future_batch_from_action_space(pred_xyz, _pred_rot)
    pred_waypoints = pred_xyz[0, :, :2].detach().cpu()
    gt_waypoints = batch["gt_waypoints"][0].to(dtype=torch.float32)
    errors = torch.norm(pred_waypoints.cpu() - gt_waypoints, dim=1)
    lateral_error = (pred_waypoints[:, 1] - gt_waypoints[:, 1]).abs()
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
        "max_lateral_error_m": float(lateral_error.max().item()),
    }
    if args.include_pid_override:
        pid_action = apply_longitudinal_pid_override(
            pred_action=pred_action,
            v0=batch["v0"].to(device=device, dtype=torch.float32),
            dt=dt,
            target_speed_kmh=args.pid_target_speed_kmh,
            kp=args.pid_kp,
            ki=args.pid_ki,
            kd=args.pid_kd,
            accel_bounds=action_space.accel_bounds,
        )
        pid_xyz, pid_rot = action_space.action_to_traj(
            traj_history_xyz=history_xyz,
            traj_history_rot=history_rot,
            action=pid_action,
        )
        pid_xyz, pid_rot = canonicalize_future_batch_from_action_space(pid_xyz, pid_rot)
        pid_waypoints = pid_xyz[0, :, :2].detach().cpu()
        pid_errors = torch.norm(pid_waypoints - gt_waypoints, dim=1)
        pid_lateral_error = (pid_waypoints[:, 1] - gt_waypoints[:, 1]).abs()
        payload["pid_override"] = {
            "target_speed_kmh": float(args.pid_target_speed_kmh),
            "pid_gains": {
                "kp": float(args.pid_kp),
                "ki": float(args.pid_ki),
                "kd": float(args.pid_kd),
            },
            "pred_action": pid_action[0].detach().cpu().tolist(),
            "pred_waypoints": pid_waypoints.tolist(),
            "ade_m": float(pid_errors.mean().item()),
            "fde_m": float(pid_errors[-1].item()),
            "max_lateral_error_m": float(pid_lateral_error.max().item()),
        }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
