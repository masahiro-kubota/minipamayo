"""Checkpoint loaders for canonical minipamayo inference bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_modeling

from ..config import AlpamayoR1Config
from ..models.alpamayo_r1 import AlpamayoR1
from ..models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN
from ..stage1.stage1a_components import load_checkpoint, load_components


def _resolve_reasoning_vla_dtype(dtype_name: str) -> str:
    if dtype_name == "bf16":
        return "bfloat16"
    if dtype_name == "fp16":
        return "float16"
    raise RuntimeError(f"Unsupported Stage 1A dtype for AlpamayoR1Config: {dtype_name!r}")


def _build_traj_tokenizer_cfg(stage1_checkpoint: dict) -> dict[str, Any]:
    quantizer_payload = stage1_checkpoint.get("quantizer")
    if not isinstance(quantizer_payload, dict):
        raise RuntimeError("Stage 1A checkpoint is missing canonical `quantizer` metadata.")
    required_keys = ["action_space_cfg", "dims_min", "dims_max", "num_bins"]
    missing_keys = [key for key in required_keys if key not in quantizer_payload]
    if missing_keys:
        raise RuntimeError(
            "Stage 1A checkpoint quantizer is missing canonical fields:\n" + "\n".join(missing_keys)
        )
    return {
        "_target_": "minipamayo_qwen35.action_space.discrete_action_space.DiscreteTrajectoryTokenizer",
        "action_space_cfg": dict(quantizer_payload["action_space_cfg"]),
        "dims_min": list(quantizer_payload["dims_min"]),
        "dims_max": list(quantizer_payload["dims_max"]),
        "num_bins": int(quantizer_payload["num_bins"]),
    }


def _build_history_tokenizer_cfg(stage1_checkpoint: dict) -> dict[str, Any]:
    history_quantizer_payload = stage1_checkpoint.get("history_quantizer")
    if not isinstance(history_quantizer_payload, dict):
        raise RuntimeError("Stage 1A checkpoint is missing canonical `history_quantizer` metadata.")
    required_keys = [
        "history_steps",
        "n_bins",
        "x_range",
        "y_range",
        "z_range",
        "yaw_range",
        "quantization_mode",
    ]
    missing_keys = [key for key in required_keys if key not in history_quantizer_payload]
    if missing_keys:
        raise RuntimeError(
            "Stage 1A checkpoint history_quantizer is missing canonical fields:\n"
            + "\n".join(missing_keys)
        )
    return {
        "_target_": "minipamayo_qwen35.contract.history_tokens.HistoryTrajectoryQuantizer",
        "history_steps": int(history_quantizer_payload["history_steps"]),
        "n_bins": int(history_quantizer_payload["n_bins"]),
        "x_range": list(history_quantizer_payload["x_range"]),
        "y_range": list(history_quantizer_payload["y_range"]),
        "z_range": list(history_quantizer_payload["z_range"]),
        "yaw_range": (
            list(history_quantizer_payload["yaw_range"])
            if history_quantizer_payload["yaw_range"] is not None
            else None
        ),
        "quantization_mode": str(history_quantizer_payload["quantization_mode"]),
    }


def _split_stage1b_state_dict(expert_state_dict: dict[str, torch.Tensor]) -> tuple[dict, dict, dict]:
    expert_weights: dict[str, torch.Tensor] = {}
    action_in_proj_weights: dict[str, torch.Tensor] = {}
    action_out_proj_weights: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    for key, value in expert_state_dict.items():
        if key.startswith("expert."):
            expert_weights[key.removeprefix("expert.")] = value
        elif key.startswith("action_in_proj."):
            action_in_proj_weights[key.removeprefix("action_in_proj.")] = value
        elif key.startswith("action_out_proj."):
            action_out_proj_weights[key.removeprefix("action_out_proj.")] = value
        else:
            unexpected_keys.append(key)
    allowed_unexpected = {"accel_mean", "accel_std", "kappa_mean", "kappa_std"}
    unknown_unexpected = sorted(set(unexpected_keys) - allowed_unexpected)
    if unknown_unexpected:
        raise RuntimeError(
            "Stage 1B expert_state_dict has unsupported top-level keys:\n"
            + "\n".join(unknown_unexpected)
        )
    return expert_weights, action_in_proj_weights, action_out_proj_weights


def _coerce_torch_dtype(dtype: object) -> torch.dtype | object:
    if not isinstance(dtype, str):
        return dtype
    normalized = dtype.removeprefix("torch.")
    if hasattr(torch, normalized):
        return getattr(torch, normalized)
    return dtype


def _coerce_torch_device(device: object) -> torch.device | object:
    if isinstance(device, int):
        return torch.device("cuda", device)
    if isinstance(device, str):
        return torch.device(device)
    return device


def _install_qwen35_fused_norm_gate_compat() -> None:
    original_cls = getattr(qwen35_modeling, "FusedRMSNormGated", None)
    if original_cls is None:
        return
    if getattr(original_cls, "__name__", "") == "_Stage2CompatFusedRMSNormGated":
        return

    class _Stage2CompatFusedRMSNormGated(original_cls):
        def __init__(self, *args, device=None, dtype=None, **kwargs):
            super().__init__(
                *args,
                device=_coerce_torch_device(device),
                dtype=_coerce_torch_dtype(dtype),
                **kwargs,
            )

    qwen35_modeling.FusedRMSNormGated = _Stage2CompatFusedRMSNormGated


def _patch_wrapper_token_contract(
    *,
    wrapper: AlpamayoR1,
    processor,
    registry,
    history_registry,
    history_quantizer,
    quantizer,
    stage1_metadata: dict,
) -> None:
    tokenizer = processor.tokenizer
    traj_token_start_id = int(tokenizer.convert_tokens_to_ids(registry.token_strings[0]))
    traj_token_end_id = int(tokenizer.convert_tokens_to_ids(registry.token_strings[-1]))
    history_token_start_id = int(tokenizer.convert_tokens_to_ids(history_registry.token_strings[0]))
    traj_token_ids = {
        key: int(token_id)
        for key, value in TRAJ_TOKEN.items()
        if (token_id := tokenizer.convert_tokens_to_ids(value)) is not None
    }
    special_token_ids = {
        key: int(token_id)
        for key, value in SPECIAL_TOKENS.items()
        if (token_id := tokenizer.convert_tokens_to_ids(value)) is not None
    }
    required_traj_keys = {"history", "future_start", "future_end"}
    missing_traj_keys = sorted(required_traj_keys - set(traj_token_ids))
    if missing_traj_keys:
        raise RuntimeError(
            "Tokenizer is missing required trajectory special tokens for Alpamayo wrapper:\n"
            + "\n".join(missing_traj_keys)
        )
    required_special_keys = {"traj_future_start", "traj_future_end"}
    missing_special_keys = sorted(required_special_keys - set(special_token_ids))
    if missing_special_keys:
        raise RuntimeError(
            "Tokenizer is missing required special tokens for Alpamayo wrapper:\n"
            + "\n".join(missing_special_keys)
        )

    wrapper.tokenizer = tokenizer
    wrapper.processor = processor
    wrapper.traj_tokenizer = quantizer
    wrapper.hist_traj_tokenizer = history_quantizer
    wrapper.future_token_start_idx = traj_token_start_id
    wrapper.hist_token_start_idx = history_token_start_id
    wrapper.special_token_ids = special_token_ids

    wrapper.config.vocab_size = len(tokenizer)
    wrapper.config.traj_vocab_size = int(registry.n_bins)
    wrapper.config.tokens_per_history_traj = int(history_quantizer.token_count)
    wrapper.config.tokens_per_future_traj = int(stage1_metadata["target_dim"])
    wrapper.config.traj_token_start_idx = traj_token_start_id
    wrapper.config.traj_token_ids = traj_token_ids

    tokenizer.traj_token_start_idx = traj_token_start_id
    tokenizer.traj_token_end_idx = traj_token_end_id
    tokenizer.traj_token_ids = dict(traj_token_ids)
    tokenizer._minipamayo_saved_processor = processor

    wrapper.vlm.config.vocab_size = len(tokenizer)
    if hasattr(wrapper.vlm.config, "text_config"):
        wrapper.vlm.config.text_config.vocab_size = len(tokenizer)
    if hasattr(wrapper.vlm, "generation_config"):
        wrapper.vlm.generation_config.pad_token_id = tokenizer.pad_token_id


def build_alpamayo_wrapper(
    *,
    stage1_checkpoint: dict,
    stage1_model,
    processor,
    registry,
    history_registry,
    history_quantizer,
    quantizer,
    stage1b_checkpoint: dict,
    image_min_pixels: int,
    image_max_pixels: int,
    flow_steps: int,
    device: torch.device,
) -> AlpamayoR1:
    stage1_args = stage1_checkpoint.get("args")
    if not isinstance(stage1_args, dict):
        raise RuntimeError("Stage 1A checkpoint is missing canonical `args` metadata.")
    required_stage1_arg_keys = ["model_path", "dtype"]
    missing_stage1_arg_keys = [key for key in required_stage1_arg_keys if key not in stage1_args]
    if missing_stage1_arg_keys:
        raise RuntimeError(
            "Stage 1A checkpoint args are missing canonical fields:\n"
            + "\n".join(missing_stage1_arg_keys)
        )
    stage1_metadata = stage1_checkpoint.get("stage1_metadata")
    if not isinstance(stage1_metadata, dict):
        raise RuntimeError("Stage 1A checkpoint is missing canonical `stage1_metadata`.")
    if "target_dim" not in stage1_metadata:
        raise RuntimeError("Stage 1A checkpoint metadata is missing canonical `target_dim`.")

    expert_config = stage1b_checkpoint.get("expert_config")
    if not isinstance(expert_config, dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `expert_config`.")
    stage1b_metadata = stage1b_checkpoint.get("stage1b_metadata")
    if not isinstance(stage1b_metadata, dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `stage1b_metadata`.")
    action_space_cfg = stage1b_metadata.get("action_space_cfg")
    if not isinstance(action_space_cfg, dict):
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `action_space_cfg`.")
    required_expert_cfg_keys = [
        "expert_cfg",
        "action_in_proj_cfg",
        "action_out_proj_cfg",
        "keep_same_dtype",
        "expert_non_causal_attention",
    ]
    missing_expert_cfg_keys = [key for key in required_expert_cfg_keys if key not in expert_config]
    if missing_expert_cfg_keys:
        raise RuntimeError(
            "Stage 1B expert_config is missing canonical fields:\n"
            + "\n".join(missing_expert_cfg_keys)
        )
    expert_state_dict = stage1b_checkpoint.get("expert_state_dict")
    if not isinstance(expert_state_dict, dict):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `expert_state_dict`.")
    expert_cfg = dict(expert_config["expert_cfg"])

    wrapper_config = AlpamayoR1Config(
        vlm_name_or_path=str(stage1_args["model_path"]),
        vlm_backend="qwenvl3",
        traj_tokenizer_cfg=_build_traj_tokenizer_cfg(stage1_checkpoint),
        hist_traj_tokenizer_cfg=_build_history_tokenizer_cfg(stage1_checkpoint),
        traj_vocab_size=int(registry.n_bins),
        tokens_per_history_traj=int(history_quantizer.token_count),
        tokens_per_future_traj=int(stage1_metadata["target_dim"]),
        model_dtype=_resolve_reasoning_vla_dtype(str(stage1_args["dtype"])),
        attn_implementation=getattr(stage1_model.config, "_attn_implementation", "flash_attention_2"),
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
        add_special_tokens=False,
        diffusion_cfg={
            "_target_": "minipamayo_qwen35.diffusion.flow_matching.FlowMatching",
            "num_inference_steps": int(flow_steps),
        },
        action_space_cfg=dict(action_space_cfg),
        action_in_proj_cfg=dict(expert_config["action_in_proj_cfg"]),
        action_out_proj_cfg=dict(expert_config["action_out_proj_cfg"]),
        expert_cfg=expert_cfg,
        keep_same_dtype=bool(expert_config["keep_same_dtype"]),
        expert_non_causal_attention=bool(expert_config["expert_non_causal_attention"]),
    )
    _install_qwen35_fused_norm_gate_compat()
    wrapper = AlpamayoR1(
        wrapper_config,
        pretrained_modules={"vlm": stage1_model, "traj_tokenizer": quantizer},
        original_vocab_size=stage1_model.get_input_embeddings().weight.shape[0],
    )
    _patch_wrapper_token_contract(
        wrapper=wrapper,
        processor=processor,
        registry=registry,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        quantizer=quantizer,
        stage1_metadata=stage1_metadata,
    )
    wrapper = wrapper.to(device)

    expert_weights, action_in_proj_weights, action_out_proj_weights = _split_stage1b_state_dict(
        expert_state_dict
    )
    wrapper.expert.load_state_dict(expert_weights)
    wrapper.action_in_proj.load_state_dict(action_in_proj_weights)
    wrapper.action_out_proj.load_state_dict(action_out_proj_weights)
    wrapper.eval()
    return wrapper


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
    processor = bundle["processor"]
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
