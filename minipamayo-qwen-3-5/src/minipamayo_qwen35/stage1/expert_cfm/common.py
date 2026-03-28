"""Shared helpers for canonical Stage 1B expert CFM training/evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from .. import CanonicalStage1Spec
from ..prompt import DEFAULT_QUESTION, build_prompt_text
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


def build_stage1b_metadata(dataset, args, expert_config: dict) -> dict:
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
            "_target_": "minipamayo_qwen35.stage1.expert_cfm.action_space.UnicycleAccelCurvatureActionSpace",
            "dt": dt_value,
            "n_waypoints": len(gt_waypoints),
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
