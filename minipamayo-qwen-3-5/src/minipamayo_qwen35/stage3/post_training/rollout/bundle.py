"""Policy/reference/expert bundle assembly for Stage 3 rollouts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ....stage1.stage1a_components import load_checkpoint
from ....stage2.reasoning_sft.bundle import load_stage2_inference_bundle
from ..common import freeze_model, validate_stage2_policy_contract


@dataclass
class Stage3RolloutBundle:
    """Everything Stage 3 needs to sample grouped rollouts."""

    checkpoint: dict[str, Any]
    stage2_metadata: dict[str, Any]
    policy_model: Any
    reference_model: Any
    processor: Any
    registry: Any
    history_registry: Any
    history_quantizer: Any
    quantizer: Any
    wrapper: Any
    model_dtype: torch.dtype
    device: torch.device

    @property
    def expected_action_token_count(self) -> int:
        return int(self.wrapper.config.tokens_per_future_traj)


def _infer_model_dtype(model) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration as exc:
        raise RuntimeError("Stage 3 policy model has no parameters.") from exc


def load_stage3_rollout_bundle(
    *,
    stage2_checkpoint_path: str | Path,
    stage1b_checkpoint_path: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
    flow_steps: int,
    device: torch.device,
) -> Stage3RolloutBundle:
    checkpoint = load_checkpoint(Path(stage2_checkpoint_path))
    stage2_metadata = validate_stage2_policy_contract(checkpoint)

    loaded = load_stage2_inference_bundle(
        stage2_checkpoint_path=stage2_checkpoint_path,
        stage1b_checkpoint_path=stage1b_checkpoint_path,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        flow_steps=flow_steps,
        device=device,
    )
    policy_model = loaded["model"]
    policy_model.to(device)
    reference_model = copy.deepcopy(policy_model)
    freeze_model(reference_model)

    wrapper = loaded["wrapper"]
    freeze_model(wrapper)
    return Stage3RolloutBundle(
        checkpoint=checkpoint,
        stage2_metadata=stage2_metadata,
        policy_model=policy_model,
        reference_model=reference_model,
        processor=loaded["processor"],
        registry=loaded["registry"],
        history_registry=loaded["history_registry"],
        history_quantizer=loaded["history_quantizer"],
        quantizer=loaded["quantizer"],
        wrapper=wrapper,
        model_dtype=_infer_model_dtype(policy_model),
        device=device,
    )
