"""Shared helpers for canonical Stage 1B expert CFM training/evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from ...contract.task_spec import CanonicalStage1Spec
from ...contract.prompt import DEFAULT_QUESTION, build_prompt_text
from ..vlm_ce.eval import load_components
from ..vlm_ce.train import model_forward_inputs, prepare_prompt_inputs_with_history


def freeze_module(module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def infer_prompt_text(checkpoint: dict, processor) -> str:
    if "stage1_metadata" not in checkpoint or not isinstance(checkpoint["stage1_metadata"], dict):
        raise RuntimeError(
            "Stage 1B requires canonical `stage1_metadata` in the Stage 1A checkpoint."
        )
    stage1_metadata = checkpoint["stage1_metadata"]
    if "question" not in stage1_metadata:
        raise RuntimeError("Stage 1A checkpoint metadata is missing canonical `question`.")
    if "history_token_count" not in stage1_metadata:
        raise RuntimeError(
            "Stage 1A checkpoint metadata is missing canonical `history_token_count`."
        )
    question = str(stage1_metadata["question"])
    if not question:
        question = DEFAULT_QUESTION
    history_token_count = int(stage1_metadata["history_token_count"])
    return build_prompt_text(
        processor=processor,
        question=question,
        history_token_count=history_token_count,
    )


def load_stage1_condition_components(args):
    stage1_args = type(
        "Stage1ExpertArgs",
        (),
        {
            "checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "image_min_pixels": int(args.image_min_pixels),
            "image_max_pixels": int(args.image_max_pixels),
        },
    )()
    (
        checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    ) = load_components(stage1_args, task_spec=CanonicalStage1Spec())
    device_name = getattr(args, "device", None)
    if device_name:
        device = torch.device(
            device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device)
    return (
        checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    )


def prepare_condition_inputs(
    model,
    batch: dict,
    processor,
    history_registry,
    history_quantizer,
    prompt_text: str,
    device: torch.device,
) -> dict:
    return prepare_prompt_inputs_with_history(
        model=model,
        batch=batch,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        prompt_text=prompt_text,
        device=device,
    )


def extract_prompt_cache(
    model,
    prompt_inputs: dict,
) -> tuple[object, torch.Tensor]:
    outputs = model(
        **model_forward_inputs(prompt_inputs),
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    if not past_key_values:
        raise RuntimeError("Frozen Stage 1 VLM did not return `past_key_values` for Stage 1B.")
    attention_mask = prompt_inputs["attention_mask"].detach()
    return past_key_values, attention_mask


def compute_action_stats(dataset) -> dict[str, float]:
    accel_rows = []
    kappa_rows = []
    for index in range(len(dataset)):
        action = dataset[index]["action"].cpu().numpy()
        accel_rows.append(action[0::2])
        kappa_rows.append(action[1::2])
    accel = torch.tensor(accel_rows, dtype=torch.float32).flatten()
    kappa = torch.tensor(kappa_rows, dtype=torch.float32).flatten()
    accel_std = float(torch.std(accel, unbiased=False).item())
    kappa_std = float(torch.std(kappa, unbiased=False).item())
    if accel_std <= 0.0 or kappa_std <= 0.0:
        raise RuntimeError("Stage 1B action normalization requires non-zero accel and kappa std.")
    return {
        "accel_mean": float(torch.mean(accel).item()),
        "accel_std": accel_std,
        "kappa_mean": float(torch.mean(kappa).item()),
        "kappa_std": kappa_std,
    }


def build_stage1b_metadata(dataset, args, expert_config: dict, action_stats: dict) -> dict:
    record = dataset[0]
    action = record["action"]
    gt_waypoints = record["gt_waypoints"]
    dt_value = float(record["dt"].item()) if hasattr(record["dt"], "item") else float(record["dt"])
    return {
        "stage1_checkpoint": args.stage1_checkpoint,
        "train_jsonl": list(args.train_jsonl),
        "val_jsonl": list(args.val_jsonl) if args.val_jsonl is not None else None,
        "sample_format": "jsonl+images",
        "condition_source": "prompt_past_key_values",
        "conditioning_contract": "detached_prompt_cache_from_stage1a_prompt",
        "k": len(gt_waypoints),
        "action_dim": int(action.shape[0]),
        "dt": dt_value,
        "expert_architecture": "alpamayo_style_action_expert",
        "diffusion_architecture": "flow_matching",
        "action_space_contract": "alpamayo_unicycle_accel_curvature_single_traj_group",
        "action_space_cfg": {
            "_target_": "minipamayo_qwen35.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace",
            "dt": dt_value,
            "n_waypoints": len(gt_waypoints),
            "accel_mean": float(action_stats["accel_mean"]),
            "accel_std": float(action_stats["accel_std"]),
            "curvature_mean": float(action_stats["kappa_mean"]),
            "curvature_std": float(action_stats["kappa_std"]),
            "theta_lambda": 1e-6,
            "theta_ridge": 1e-8,
            "v_lambda": 1e-6,
            "v_ridge": 1e-4,
            "a_lambda": 1e-4,
            "a_ridge": 1e-4,
            "kappa_lambda": 1e-4,
            "kappa_ridge": 1e-4,
        },
        "expert_config": expert_config,
    }


def build_longitudinal_pid_accel_sequence(
    *,
    v0: torch.Tensor,
    target_speed_mps: float,
    dt: float,
    horizon: int,
    kp: float,
    ki: float,
    kd: float,
    accel_bounds: tuple[float, float],
) -> torch.Tensor:
    """Build a fixed-target longitudinal PID acceleration sequence.

    This is intentionally simple and is only used for lateral-isolation experiments:
    keep `kappa` from Stage 1B, replace the longitudinal channel with a controller that
    tries to hold a fixed target speed.
    """
    if horizon <= 0:
        raise RuntimeError("`horizon` must be > 0.")
    v = v0.to(dtype=torch.float32).reshape(-1)
    batch_size = int(v.shape[0])
    device = v.device
    target = torch.full((batch_size,), float(target_speed_mps), device=device, dtype=torch.float32)
    accel_min, accel_max = float(accel_bounds[0]), float(accel_bounds[1])
    integral = torch.zeros_like(v)
    prev_error = torch.zeros_like(v)
    accel_rows: list[torch.Tensor] = []

    for step_idx in range(horizon):
        error = target - v
        integral = integral + error * float(dt)
        if step_idx == 0:
            derivative = torch.zeros_like(error)
        else:
            derivative = (error - prev_error) / float(dt)
        accel = kp * error + ki * integral + kd * derivative
        accel = accel.clamp(min=accel_min, max=accel_max)
        accel_rows.append(accel)
        v = (v + float(dt) * accel).clamp(min=0.0)
        prev_error = error

    return torch.stack(accel_rows, dim=1)


def apply_longitudinal_pid_override(
    *,
    pred_action: torch.Tensor,
    v0: torch.Tensor,
    dt: float,
    target_speed_kmh: float,
    kp: float,
    ki: float,
    kd: float,
    accel_bounds: tuple[float, float],
) -> torch.Tensor:
    """Replace the longitudinal channel of a canonical `(accel, kappa)` action."""
    if pred_action.dim() != 3 or pred_action.shape[-1] != 2:
        raise RuntimeError(
            "Expected `pred_action` shaped (batch, k, 2) for PID override.\n"
            f"found={tuple(pred_action.shape)!r}"
        )
    pid_action = pred_action.to(dtype=torch.float32).clone()
    pid_accel = build_longitudinal_pid_accel_sequence(
        v0=v0,
        target_speed_mps=float(target_speed_kmh) / 3.6,
        dt=float(dt),
        horizon=int(pred_action.shape[1]),
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
        accel_bounds=accel_bounds,
    )
    pid_action[:, :, 0] = pid_accel
    return pid_action
