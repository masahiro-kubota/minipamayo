"""Shared bundle loaders for canonical Stage 2 reasoning SFT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ...helper import get_processor
from ...stage1.stage1a_components import load_checkpoint, load_components
from .wrapper import build_alpamayo_wrapper


def _load_stage1a_components(
    *,
    stage1a_checkpoint: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
):
    stage1_args = argparse.Namespace(
        checkpoint=str(Path(stage1a_checkpoint).resolve()),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    return load_components(stage1_args)


def load_stage2_training_bundle(
    *,
    stage1a_checkpoint: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
    device: torch.device,
) -> dict[str, Any]:
    (
        stage1_checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    ) = _load_stage1a_components(
        stage1a_checkpoint=stage1a_checkpoint,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    model.to(device)
    return {
        "stage1_checkpoint": stage1_checkpoint,
        "stage1a_checkpoint_path": Path(stage1a_checkpoint).resolve(),
        "model": model,
        "processor": processor,
        "registry": registry,
        "history_registry": history_registry,
        "history_quantizer": history_quantizer,
        "quantizer": quantizer,
        "model_dtype": model_dtype,
    }


def _restore_stage2_model_state(model, checkpoint: dict[str, Any]) -> None:
    if "model_state_dict" not in checkpoint or not isinstance(checkpoint["model_state_dict"], dict):
        raise RuntimeError("Stage 2 checkpoint is missing canonical `model_state_dict`.")
    state_dict = checkpoint["model_state_dict"]
    embed_key = "model.language_model.embed_tokens.weight"
    if embed_key not in state_dict:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical embedding weights required for resize."
        )
    stage2_embed_rows = int(state_dict[embed_key].shape[0])
    target_embed_rows = model.get_input_embeddings().weight.shape[0]
    if stage2_embed_rows != target_embed_rows:
        model.resize_token_embeddings(stage2_embed_rows)
    model.load_state_dict(state_dict)
    if stage2_embed_rows != target_embed_rows:
        model.resize_token_embeddings(target_embed_rows)


def load_stage2_checkpoint_bundle(
    *,
    checkpoint_path: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
    device: torch.device,
    use_cache: bool,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(Path(checkpoint_path))
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict) or "stage1a_checkpoint" not in checkpoint_args:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical `stage1a_checkpoint` args metadata."
        )
    bundle = load_stage2_training_bundle(
        stage1a_checkpoint=str(checkpoint_args["stage1a_checkpoint"]),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        device=device,
    )
    model = bundle["model"]
    _restore_stage2_model_state(model, checkpoint)
    model.config.use_cache = use_cache
    return {
        "checkpoint": checkpoint,
        "checkpoint_args": checkpoint_args,
        **bundle,
    }


def load_stage2_inference_bundle(
    *,
    stage2_checkpoint_path: str | Path,
    stage1b_checkpoint_path: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
    flow_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    bundle = load_stage2_checkpoint_bundle(
        checkpoint_path=stage2_checkpoint_path,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        device=device,
        use_cache=True,
    )
    processor = get_processor(bundle["processor"].tokenizer)
    stage1b_checkpoint = torch.load(
        Path(stage1b_checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    wrapper = build_alpamayo_wrapper(
        stage1_checkpoint=bundle["stage1_checkpoint"],
        stage1_model=bundle["model"],
        processor=processor,
        registry=bundle["registry"],
        history_registry=bundle["history_registry"],
        history_quantizer=bundle["history_quantizer"],
        quantizer=bundle["quantizer"],
        stage1b_checkpoint=stage1b_checkpoint,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        flow_steps=flow_steps,
        device=device,
    )
    return {
        **bundle,
        "stage1b_checkpoint": stage1b_checkpoint,
        "processor": processor,
        "wrapper": wrapper,
    }
