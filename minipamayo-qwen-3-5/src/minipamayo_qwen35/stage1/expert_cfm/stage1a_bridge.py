"""Frozen Stage 1A bridge helpers for canonical Stage 1B."""

from __future__ import annotations

from pathlib import Path

import torch

from ...contract.task_spec import CanonicalStage1Spec
from ...contract.prompt import DEFAULT_QUESTION, build_prompt_text
from ...utils.image_budget import CANONICAL_IMAGE_MAX_PIXELS, CANONICAL_IMAGE_MIN_PIXELS
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
            "image_min_pixels": CANONICAL_IMAGE_MIN_PIXELS,
            "image_max_pixels": CANONICAL_IMAGE_MAX_PIXELS,
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
