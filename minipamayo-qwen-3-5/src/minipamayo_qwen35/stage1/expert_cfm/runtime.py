"""Shared runtime helpers for canonical Stage 1B inference/evaluation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...contract.record_adapter import (
    canonicalize_future_batch_from_action_space,
    canonicalize_history_batch_for_action_space,
)
from ...action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from ...contract.history_tokens import HistoryTokenRegistry, HistoryTrajectoryQuantizer
from ...models.action_expert import Stage1ActionExpert, load_action_expert_from_checkpoint
from ..stage1b_diffusion import (
    BaseDiffusion,
    build_stage1b_diffusion_cfg,
    instantiate_stage1b_diffusion,
)
from ..stage1a_conditioning import (
    extract_prompt_cache,
    freeze_module,
    infer_prompt_text,
    load_stage1_condition_components,
    prepare_condition_inputs,
)
from .pid import apply_longitudinal_pid_override


@dataclass(frozen=True)
class Stage1BRuntime:
    """Frozen runtime components shared by Stage 1B inference and evaluation."""

    stage1_checkpoint: dict[str, Any]
    stage1b_checkpoint: dict[str, Any]
    model: Any
    processor: Any
    history_registry: HistoryTokenRegistry
    history_quantizer: HistoryTrajectoryQuantizer
    expert: Stage1ActionExpert
    action_space: UnicycleAccelCurvatureActionSpace
    diffusion: BaseDiffusion
    prompt_text: str
    dt: float
    stage1b_metadata: dict[str, Any]
    device: torch.device


@dataclass(frozen=True)
class Stage1BCondition:
    """Prompt-cache conditioning extracted from the frozen Stage 1A VLM."""

    prompt_cache: object
    prompt_attention_mask: torch.Tensor


@dataclass(frozen=True)
class Stage1BBatchResult:
    """Shared Stage 1B inference outputs for one batch."""

    loss: torch.Tensor | None
    pred_action: torch.Tensor
    pred_waypoints: torch.Tensor
    gt_action: torch.Tensor | None
    gt_action_seq: torch.Tensor | None
    gt_waypoints: torch.Tensor | None
    pid_action: torch.Tensor | None
    pid_waypoints: torch.Tensor | None


def _stage1b_amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _resolve_stage1b_metadata(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    if "stage1b_metadata" not in checkpoint or not isinstance(checkpoint["stage1b_metadata"], dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `stage1b_metadata`.")
    stage1b_metadata = dict(checkpoint["stage1b_metadata"])
    required_keys = ["dt", "condition_source", "action_space_cfg"]
    missing_keys = [key for key in required_keys if key not in stage1b_metadata]
    if missing_keys:
        raise RuntimeError(
            "Stage 1B checkpoint metadata is missing canonical fields:\n" + "\n".join(missing_keys)
        )
    if str(stage1b_metadata["condition_source"]) != "prompt_past_key_values":
        raise RuntimeError(
            "This Stage 1B runtime expects canonical prompt-cache conditioning.\n"
            f"found={stage1b_metadata['condition_source']!r}"
        )
    action_space_cfg = stage1b_metadata["action_space_cfg"]
    if not isinstance(action_space_cfg, dict):
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `action_space_cfg`.")
    diffusion_cfg = stage1b_metadata.get("diffusion_cfg")
    if diffusion_cfg is None:
        diffusion_cfg = build_stage1b_diffusion_cfg()
    if not isinstance(diffusion_cfg, dict):
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `diffusion_cfg`.")
    return (
        stage1b_metadata,
        dict(action_space_cfg),
        dict(diffusion_cfg),
        float(stage1b_metadata["dt"]),
    )


def load_stage1b_runtime(
    *,
    stage1_checkpoint: str | Path,
    checkpoint: str | Path,
    flow_steps: int,
    device: torch.device,
) -> Stage1BRuntime:
    """Load the frozen Stage 1A condition path and the Stage 1B expert runtime."""

    stage1_args = argparse.Namespace(
        stage1_checkpoint=str(Path(stage1_checkpoint).resolve()),
        device=str(device),
    )
    (
        loaded_stage1_checkpoint,
        model,
        processor,
        _registry,
        history_registry,
        history_quantizer,
        _quantizer,
        _model_dtype,
    ) = load_stage1_condition_components(stage1_args)
    freeze_module(model)
    prompt_text = infer_prompt_text(loaded_stage1_checkpoint, processor)

    expert, loaded_stage1b_checkpoint = load_action_expert_from_checkpoint(str(checkpoint), device)
    stage1b_metadata, action_space_cfg, diffusion_cfg, dt = _resolve_stage1b_metadata(
        loaded_stage1b_checkpoint
    )
    action_space_cfg.pop("_target_", None)
    action_space = UnicycleAccelCurvatureActionSpace(**action_space_cfg)
    diffusion = instantiate_stage1b_diffusion(diffusion_cfg=diffusion_cfg, n_steps=int(flow_steps))
    return Stage1BRuntime(
        stage1_checkpoint=loaded_stage1_checkpoint,
        stage1b_checkpoint=loaded_stage1b_checkpoint,
        model=model,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        expert=expert,
        action_space=action_space,
        diffusion=diffusion,
        prompt_text=prompt_text,
        dt=dt,
        stage1b_metadata=stage1b_metadata,
        device=device,
    )


def build_stage1b_prompt_inputs(*, runtime: Stage1BRuntime, batch: dict[str, Any]) -> dict[str, Any]:
    """Prepare canonical Stage 1 prompt inputs with history tokens injected."""

    return prepare_condition_inputs(
        model=runtime.model,
        batch=batch,
        processor=runtime.processor,
        history_registry=runtime.history_registry,
        history_quantizer=runtime.history_quantizer,
        prompt_text=runtime.prompt_text,
        device=runtime.device,
    )


def extract_stage1b_condition(
    *,
    runtime: Stage1BRuntime,
    prompt_inputs: dict[str, Any],
) -> Stage1BCondition:
    """Extract the prompt cache and attention mask used to condition the Stage 1B expert."""

    prompt_cache, prompt_attention_mask = extract_prompt_cache(runtime.model, prompt_inputs)
    return Stage1BCondition(
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    )


def sample_stage1b_action(
    *,
    runtime: Stage1BRuntime,
    condition: Stage1BCondition,
    gt_action: torch.Tensor | None = None,
    compute_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample a canonical `(accel, kappa)` action sequence from the Stage 1B expert."""

    if compute_loss and gt_action is None:
        raise RuntimeError("`gt_action` is required when `compute_loss=True`.")
    with _stage1b_amp_context(runtime.device):
        loss = None
        if compute_loss and gt_action is not None:
            loss = runtime.diffusion.loss(
                expert=runtime.expert,
                gt_action=gt_action,
                prompt_cache=condition.prompt_cache,
                prompt_attention_mask=condition.prompt_attention_mask,
            )
        pred_action = runtime.diffusion.sample(
            expert=runtime.expert,
            prompt_cache=condition.prompt_cache,
            prompt_attention_mask=condition.prompt_attention_mask,
        )
    return pred_action, loss


def decode_stage1b_trajectory(
    *,
    runtime: Stage1BRuntime,
    pred_action: torch.Tensor,
    batch: dict[str, Any],
) -> torch.Tensor:
    """Convert Stage 1B action samples into canonical future waypoints."""

    history_xyz, history_rot = canonicalize_history_batch_for_action_space(
        batch["ego_history_xyz"].to(device=runtime.device, dtype=torch.float32),
        batch["ego_history_rot"].to(device=runtime.device, dtype=torch.float32),
    )
    pred_xyz, pred_rot = runtime.action_space.action_to_traj(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        action=pred_action,
    )
    pred_xyz, pred_rot = canonicalize_future_batch_from_action_space(pred_xyz, pred_rot)
    return pred_xyz[:, :, :2]


def apply_stage1b_pid_override(
    *,
    runtime: Stage1BRuntime,
    pred_action: torch.Tensor,
    v0: torch.Tensor,
    target_speed_kmh: float,
    kp: float,
    ki: float,
    kd: float,
) -> torch.Tensor:
    """Apply the canonical longitudinal PID override to Stage 1B predictions."""

    return apply_longitudinal_pid_override(
        pred_action=pred_action,
        v0=v0,
        dt=runtime.dt,
        target_speed_kmh=target_speed_kmh,
        kp=kp,
        ki=ki,
        kd=kd,
        accel_bounds=runtime.action_space.accel_bounds,
    )


@torch.no_grad()
def run_stage1b_inference_batch(
    *,
    runtime: Stage1BRuntime,
    batch: dict[str, Any],
    include_pid_override: bool = False,
    pid_target_speed_kmh: float = 24.0,
    pid_kp: float = 1.0,
    pid_ki: float = 0.05,
    pid_kd: float = 0.0,
    compute_loss: bool = False,
) -> Stage1BBatchResult:
    """Run the shared Stage 1B inference path for one batch."""

    prompt_inputs = build_stage1b_prompt_inputs(runtime=runtime, batch=batch)
    condition = extract_stage1b_condition(runtime=runtime, prompt_inputs=prompt_inputs)

    gt_action = None
    gt_action_seq = None
    if "action" in batch and batch["action"] is not None:
        gt_action = batch["action"].to(device=runtime.device, dtype=torch.float32)
        gt_action_seq = gt_action.reshape(gt_action.shape[0], -1, 2)

    gt_waypoints = None
    if "gt_waypoints" in batch and batch["gt_waypoints"] is not None:
        gt_waypoints = batch["gt_waypoints"].to(device=runtime.device, dtype=torch.float32)

    pred_action, loss = sample_stage1b_action(
        runtime=runtime,
        condition=condition,
        gt_action=gt_action,
        compute_loss=compute_loss,
    )
    pred_waypoints = decode_stage1b_trajectory(
        runtime=runtime,
        pred_action=pred_action,
        batch=batch,
    )

    pid_action = None
    pid_waypoints = None
    if include_pid_override:
        if "v0" not in batch or batch["v0"] is None:
            raise RuntimeError("`v0` is required when `include_pid_override=True`.")
        pid_action = apply_stage1b_pid_override(
            runtime=runtime,
            pred_action=pred_action,
            v0=batch["v0"].to(device=runtime.device, dtype=torch.float32),
            target_speed_kmh=pid_target_speed_kmh,
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
        )
        pid_waypoints = decode_stage1b_trajectory(
            runtime=runtime,
            pred_action=pid_action,
            batch=batch,
        )

    return Stage1BBatchResult(
        loss=loss,
        pred_action=pred_action,
        pred_waypoints=pred_waypoints,
        gt_action=gt_action,
        gt_action_seq=gt_action_seq,
        gt_waypoints=gt_waypoints,
        pid_action=pid_action,
        pid_waypoints=pid_waypoints,
    )
