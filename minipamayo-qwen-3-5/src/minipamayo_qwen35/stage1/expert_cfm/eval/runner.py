"""Canonical Stage 1B expert-CFM evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ....stage1.data.dataset import Stage1JsonlDataset
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
from ..core.common import (
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
    return args


def main() -> None:
    args = parse_args()
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

            batch_size = gt_action.shape[0]
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            total_ade += float(displacement.mean(dim=1).sum().item())
            total_fde += float(displacement[:, -1].sum().item())
            total_samples += batch_size

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
        "num_samples": total_samples,
        "cfm_loss": total_loss / max(1, total_batches),
        "ade_m": total_ade / max(1, total_samples),
        "fde_m": total_fde / max(1, total_samples),
        "flow_steps": args.flow_steps,
        "condition_source": stage1b_metadata["condition_source"],
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
