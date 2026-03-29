"""Canonical Stage 1B expert-CFM evaluation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ....stage1.dataset import Stage1JsonlDataset
from ....stage1.checkpoint_completion import require_completed_training_run
from ....stage1.vlm_ce.train import stage1_collate
from ....action_space.record_adapter import (
    canonicalize_future_batch_from_action_space,
    canonicalize_history_batch_for_action_space,
)
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import (
    load_json_payload,
    normalize_arg_config,
    normalize_required_string_list,
    resolve_path_base,
)
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
CONFIG_PATH_KEYS = {"checkpoint", "stage1_checkpoint", "eval_jsonl", "output_json"}
MULTI_VALUE_CONFIG_KEYS = {"eval_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the canonical Stage 1B expert CFM path.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--eval-jsonl", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
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
        list_keys=MULTI_VALUE_CONFIG_KEYS,
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
            "Stage 1B evaluation accepts only --config-json. Put all settings in the JSON file."
        )
    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    args.eval_jsonl = normalize_required_string_list(args.eval_jsonl, key_name="eval_jsonl")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.stage1_checkpoint:
        raise RuntimeError("`stage1_checkpoint` must be defined in the config JSON.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.pid_target_speed_kmh <= 0.0:
        raise RuntimeError("`pid_target_speed_kmh` must be > 0.")
    return args


def main() -> None:
    args = parse_args()
    require_completed_training_run(
        args.stage1_checkpoint,
        checkpoint_label="Stage 1A checkpoint",
        required_summary_keys=["completed_epochs", "best_epoch", "stop_reason"],
        allowed_stop_reasons={"max_epochs", "early_stopping"},
    )
    require_completed_training_run(
        args.checkpoint,
        checkpoint_label="Stage 1B checkpoint",
        required_summary_keys=["best_epoch", "history_length"],
    )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Canonical Stage 1B evaluation currently expects CUDA.")
    require_expected_cuda_toolkit()

    dataset = Stage1JsonlDataset(args.eval_jsonl)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=stage1_collate,
        persistent_workers=args.num_workers > 0,
    )

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
    expert, checkpoint = load_action_expert_from_checkpoint(args.checkpoint, device)
    if "stage1b_metadata" not in checkpoint or not isinstance(checkpoint["stage1b_metadata"], dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `stage1b_metadata`.")
    stage1b_metadata = checkpoint["stage1b_metadata"]
    required_stage1b_keys = ["dt", "condition_source"]
    missing_stage1b_keys = [key for key in required_stage1b_keys if key not in stage1b_metadata]
    if missing_stage1b_keys:
        raise RuntimeError(
            "Stage 1B checkpoint metadata is missing canonical fields:\n"
            + "\n".join(missing_stage1b_keys)
        )
    dt = float(stage1b_metadata["dt"])
    diffusion = FlowMatchingDiffusion(n_steps=args.flow_steps)
    if "action_space_cfg" not in stage1b_metadata or not isinstance(
        stage1b_metadata["action_space_cfg"], dict
    ):
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `action_space_cfg`.")
    action_space_cfg = dict(stage1b_metadata["action_space_cfg"])
    action_space_cfg.pop("_target_", None)
    action_space = UnicycleAccelCurvatureActionSpace(**action_space_cfg)

    total_loss = 0.0
    total_batches = 0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    total_action_steps = 0
    canonical_total_mean_max_lateral = 0.0
    canonical_global_max_lateral = 0.0
    pid_total_ade = 0.0
    pid_total_fde = 0.0
    pid_total_mean_max_lateral = 0.0
    pid_global_max_lateral = 0.0
    pid_total_action_mae_accel = 0.0
    pid_total_action_mae_kappa = 0.0
    canonical_total_action_mae_accel = 0.0
    canonical_total_action_mae_kappa = 0.0
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )

    with torch.no_grad():
        for batch in dataloader:
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
            gt_action = batch["action"].to(device=device, dtype=torch.float32)
            gt_action_seq = gt_action.reshape(gt_action.shape[0], -1, 2)
            with amp_context:
                loss = diffusion.loss(
                    expert=expert,
                    gt_action=gt_action,
                    prompt_cache=prompt_cache,
                    prompt_attention_mask=prompt_attention_mask,
                )
                pred_action = diffusion.sample(
                    expert=expert,
                    prompt_cache=prompt_cache,
                    prompt_attention_mask=prompt_attention_mask,
                ).reshape(gt_action.shape[0], -1, 2)
            pid_action = None
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
            gt_waypoints = batch["gt_waypoints"].to(device=device, dtype=torch.float32)
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
            pred_waypoints = pred_xyz[:, :, :2]
            displacement = torch.norm(pred_waypoints - gt_waypoints, dim=2)
            lateral_error = (pred_waypoints[:, :, 1] - gt_waypoints[:, :, 1]).abs()
            if pid_action is not None:
                pid_xyz, pid_rot = action_space.action_to_traj(
                    traj_history_xyz=history_xyz,
                    traj_history_rot=history_rot,
                    action=pid_action,
                )
                pid_xyz, pid_rot = canonicalize_future_batch_from_action_space(pid_xyz, pid_rot)
                pid_waypoints = pid_xyz[:, :, :2]
                pid_displacement = torch.norm(pid_waypoints - gt_waypoints, dim=2)
                pid_lateral_error = (pid_waypoints[:, :, 1] - gt_waypoints[:, :, 1]).abs()

            batch_size = gt_action.shape[0]
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            total_ade += float(displacement.mean(dim=1).sum().item())
            total_fde += float(displacement[:, -1].sum().item())
            total_samples += batch_size
            total_action_steps += batch_size * int(gt_action_seq.shape[1])
            canonical_total_mean_max_lateral += float(lateral_error.max(dim=1).values.sum().item())
            canonical_global_max_lateral = max(
                canonical_global_max_lateral, float(lateral_error.max().item())
            )
            canonical_total_action_mae_accel += float(
                (pred_action[:, :, 0] - gt_action_seq[:, :, 0]).abs().sum().item()
            )
            canonical_total_action_mae_kappa += float(
                (pred_action[:, :, 1] - gt_action_seq[:, :, 1]).abs().sum().item()
            )
            if pid_action is not None:
                pid_total_ade += float(pid_displacement.mean(dim=1).sum().item())
                pid_total_fde += float(pid_displacement[:, -1].sum().item())
                pid_total_mean_max_lateral += float(pid_lateral_error.max(dim=1).values.sum().item())
                pid_global_max_lateral = max(pid_global_max_lateral, float(pid_lateral_error.max().item()))
                pid_total_action_mae_accel += float(
                    (pid_action[:, :, 0] - gt_action_seq[:, :, 0]).abs().sum().item()
                )
                pid_total_action_mae_kappa += float(
                    (pid_action[:, :, 1] - gt_action_seq[:, :, 1]).abs().sum().item()
                )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "num_samples": total_samples,
        "cfm_loss": total_loss / max(1, total_batches),
        "ade_m": total_ade / max(1, total_samples),
        "fde_m": total_fde / max(1, total_samples),
        "mean_max_lateral_error_m": canonical_total_mean_max_lateral / max(1, total_samples),
        "global_max_lateral_error_m": canonical_global_max_lateral,
        "flow_steps": args.flow_steps,
        "condition_source": stage1b_metadata["condition_source"],
        "pid_override_enabled": bool(args.include_pid_override),
        "action_mae_accel": canonical_total_action_mae_accel / max(1, total_action_steps),
        "action_mae_kappa": canonical_total_action_mae_kappa / max(1, total_action_steps),
    }
    if args.include_pid_override:
        summary["pid_override"] = {
            "target_speed_kmh": float(args.pid_target_speed_kmh),
            "pid_gains": {
                "kp": float(args.pid_kp),
                "ki": float(args.pid_ki),
                "kd": float(args.pid_kd),
            },
            "ade_m": pid_total_ade / max(1, total_samples),
            "fde_m": pid_total_fde / max(1, total_samples),
            "mean_max_lateral_error_m": pid_total_mean_max_lateral / max(1, total_samples),
            "global_max_lateral_error_m": pid_global_max_lateral,
            "action_mae_accel": pid_total_action_mae_accel / max(1, total_action_steps),
            "action_mae_kappa": pid_total_action_mae_kappa / max(1, total_action_steps),
        }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
