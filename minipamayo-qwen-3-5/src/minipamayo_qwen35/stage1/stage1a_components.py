from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..contract.history_tokens import HistoryTokenRegistry, HistoryTrajectoryQuantizer
from ..contract.prompt import PROMPT_SPECIAL_TOKENS, add_prompt_special_tokens
from ..contract.task_spec import CanonicalStage1Spec, KappaOnlyStage1Spec, Stage1TaskSpec
from ..contract.trajectory_tokens import Stage1TokenRegistry

CHECKPOINT_KIND_FULL = "full"
CHECKPOINT_KIND_MODEL_ONLY = "model_only"
CANONICAL_ATTN_IMPLEMENTATION = "flash_attention_2"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    return torch.bfloat16


def resolve_checkpoint_kind(checkpoint: dict) -> str:
    if "checkpoint_kind" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `checkpoint_kind`.")
    checkpoint_kind = checkpoint["checkpoint_kind"]
    if checkpoint_kind not in {CHECKPOINT_KIND_FULL, CHECKPOINT_KIND_MODEL_ONLY}:
        raise RuntimeError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")
    return checkpoint_kind


def resolve_checkpoint_args(checkpoint: dict) -> dict:
    resolve_checkpoint_kind(checkpoint)
    if "args" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `args` metadata.")
    checkpoint_args = checkpoint["args"]
    if not isinstance(checkpoint_args, dict):
        raise RuntimeError("Checkpoint is missing canonical `args` metadata.")
    required_keys = ["model_path", "dtype"]
    missing_keys = [key for key in required_keys if key not in checkpoint_args]
    if missing_keys:
        raise RuntimeError(
            "Checkpoint is missing canonical `args` fields:\n" + "\n".join(missing_keys)
        )
    return checkpoint_args


def resolve_processor_path(checkpoint_path: Path) -> str:
    saved_processor = checkpoint_path.parent / "processor"
    if not saved_processor.exists():
        raise RuntimeError(
            f"Checkpoint is missing the canonical saved processor directory: {saved_processor}"
        )
    return str(saved_processor)


def resolve_task_spec_from_checkpoint(checkpoint: dict) -> Stage1TaskSpec:
    if "stage1_metadata" not in checkpoint or not isinstance(checkpoint["stage1_metadata"], dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    action_representation = checkpoint["stage1_metadata"].get("action_representation")
    if action_representation == "accel_kappa":
        return CanonicalStage1Spec()
    if action_representation == "kappa_only":
        return KappaOnlyStage1Spec()
    raise RuntimeError(f"Unsupported Stage 1 action representation: {action_representation!r}")


def build_processor_kwargs(image_min_pixels: int, image_max_pixels: int) -> dict:
    kwargs = {}
    if image_min_pixels > 0:
        kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels > 0:
        kwargs["max_pixels"] = image_max_pixels
    return kwargs


def build_model_load_kwargs(model_dtype: torch.dtype) -> dict:
    return {
        "dtype": model_dtype,
        "trust_remote_code": True,
        "attn_implementation": CANONICAL_ATTN_IMPLEMENTATION,
    }


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must deserialize to a dict: {path}")
    return checkpoint


def build_training_token_contract(dataset, processor, task_spec: Stage1TaskSpec):
    history_quantizer = HistoryTrajectoryQuantizer()
    add_prompt_special_tokens(processor.tokenizer)
    quantizer = task_spec.build_quantizer(dataset)
    quantizer_n_bins = int(
        quantizer.num_bins if hasattr(quantizer, "num_bins") else quantizer.n_bins
    )
    registry = Stage1TokenRegistry(n_bins=quantizer_n_bins, start_index=0)
    added_action_tokens = registry.add_to_tokenizer(processor.tokenizer)
    history_registry = HistoryTokenRegistry(
        n_bins=history_quantizer.n_bins,
        start_index=registry.start_index + registry.n_bins,
    )
    added_history_tokens = history_registry.add_to_tokenizer(processor.tokenizer)
    return (
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        added_action_tokens,
        added_history_tokens,
    )


def build_stage1_metadata(
    dataset,
    *,
    train_jsonl: list[str],
    val_jsonl: list[str] | None,
    registry: Stage1TokenRegistry,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    quantizer,
    task_spec: Stage1TaskSpec,
    question: str,
) -> dict:
    sample = dataset[0]
    gt_waypoints = sample["gt_waypoints"].detach().cpu().reshape(-1, 2)
    full_action = sample["action"].detach().cpu().reshape(-1).numpy()
    ego_history_xyz = sample["ego_history_xyz"].detach().cpu().numpy()
    dt_value = float(sample["dt"].item())
    target = task_spec.target_from_action_array(full_action)
    return {
        "train_jsonl": list(train_jsonl),
        "val_jsonl": list(val_jsonl) if val_jsonl is not None else None,
        "sample_format": "jsonl+images",
        "k": int(gt_waypoints.shape[0]),
        "target_dim": int(target.shape[0]),
        "full_action_dim": int(full_action.shape[0]),
        "dt": dt_value,
        "action_token_scheme": "alpamayo_like_discrete_tokens",
        "token_prefix": registry.token_prefix,
        "token_start_index": registry.start_index,
        "history_token_scheme": "placeholder_input_ids_discrete_tokens",
        "history_token_prefix": history_registry.token_prefix,
        "history_token_start_index": history_registry.start_index,
        "history_steps": int(ego_history_xyz.shape[-2]),
        "history_token_count": history_quantizer.token_count,
        "history_layout": "ego_frame_xyz_rot_local_single_traj_group",
        "question": question,
        "prompt_contract": "system_user_image_history_tokens",
        "prompt_special_tokens": list(PROMPT_SPECIAL_TOKENS),
        "history_quantizer": history_quantizer.metadata(),
        **task_spec.metadata(quantizer),
    }


def _bootstrap_token_contract_from_checkpoint(
    checkpoint: dict,
    processor,
    task_spec: Stage1TaskSpec,
) -> tuple[Stage1TokenRegistry, HistoryTokenRegistry, HistoryTrajectoryQuantizer, Any]:
    if "history_registry" not in checkpoint or not isinstance(checkpoint["history_registry"], dict):
        raise RuntimeError("Checkpoint is missing canonical `history_registry` metadata.")
    history_cfg = checkpoint["history_registry"]
    token_cfg = checkpoint["token_registry"]
    required_token_cfg_keys = ["n_bins", "token_prefix", "start_index"]
    missing_token_cfg_keys = [key for key in required_token_cfg_keys if key not in token_cfg]
    if missing_token_cfg_keys:
        raise RuntimeError(
            "Checkpoint token_registry is missing canonical fields:\n"
            + "\n".join(missing_token_cfg_keys)
        )
    required_history_cfg_keys = ["n_bins", "token_prefix", "start_index"]
    missing_history_cfg_keys = [key for key in required_history_cfg_keys if key not in history_cfg]
    if missing_history_cfg_keys:
        raise RuntimeError(
            "Checkpoint history_registry is missing canonical fields:\n"
            + "\n".join(missing_history_cfg_keys)
        )
    registry = Stage1TokenRegistry(
        n_bins=int(token_cfg["n_bins"]),
        token_prefix=str(token_cfg["token_prefix"]),
        start_index=int(token_cfg["start_index"]),
    )
    registry.add_to_tokenizer(processor.tokenizer)
    history_registry = HistoryTokenRegistry(
        n_bins=int(history_cfg["n_bins"]),
        token_prefix=str(history_cfg["token_prefix"]),
        start_index=int(history_cfg["start_index"]),
    )
    history_registry.add_to_tokenizer(processor.tokenizer)

    if "history_quantizer" not in checkpoint or not isinstance(checkpoint["history_quantizer"], dict):
        raise RuntimeError("Checkpoint is missing canonical `history_quantizer` metadata.")
    history_quantizer_cfg = checkpoint["history_quantizer"]
    required_history_quantizer_keys = [
        "history_steps",
        "n_bins",
        "x_range",
        "y_range",
        "z_range",
        "yaw_range",
        "quantization_mode",
    ]
    missing_history_quantizer_keys = [
        key for key in required_history_quantizer_keys if key not in history_quantizer_cfg
    ]
    if missing_history_quantizer_keys:
        raise RuntimeError(
            "Checkpoint history_quantizer is missing canonical fields:\n"
            + "\n".join(missing_history_quantizer_keys)
        )
    history_quantizer = HistoryTrajectoryQuantizer(
        history_steps=int(history_quantizer_cfg["history_steps"]),
        n_bins=int(history_quantizer_cfg["n_bins"]),
        x_range=tuple(history_quantizer_cfg["x_range"]),
        y_range=tuple(history_quantizer_cfg["y_range"]),
        z_range=tuple(history_quantizer_cfg["z_range"]),
        yaw_range=(
            tuple(history_quantizer_cfg["yaw_range"])
            if history_quantizer_cfg["yaw_range"] is not None
            else None
        ),
        quantization_mode=str(history_quantizer_cfg["quantization_mode"]),
    )

    if "quantizer" not in checkpoint or not isinstance(checkpoint["quantizer"], dict):
        raise RuntimeError("Checkpoint is missing canonical `quantizer` metadata.")
    quantizer = task_spec.quantizer_from_checkpoint(
        checkpoint["quantizer"],
        stage1_metadata=checkpoint["stage1_metadata"],
    )
    return registry, history_registry, history_quantizer, quantizer


def load_components(
    args,
    task_spec: Stage1TaskSpec | None = None,
) -> tuple[
    dict,
    object,
    object,
    Stage1TokenRegistry,
    HistoryTokenRegistry,
    HistoryTrajectoryQuantizer,
    Any,
    torch.dtype,
]:
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    task_spec = task_spec or resolve_task_spec_from_checkpoint(checkpoint)
    checkpoint_args = resolve_checkpoint_args(checkpoint)
    if "stage1_metadata" not in checkpoint or not isinstance(checkpoint["stage1_metadata"], dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    task_spec.validate_checkpoint(checkpoint["stage1_metadata"])
    model_path = str(checkpoint_args["model_path"])
    processor_path = resolve_processor_path(checkpoint_path)
    model_dtype = resolve_dtype(str(checkpoint_args["dtype"]))
    processor_kwargs = build_processor_kwargs(args.image_min_pixels, args.image_max_pixels)
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        **processor_kwargs,
    )
    add_prompt_special_tokens(processor.tokenizer)
    registry, history_registry, history_quantizer, quantizer = _bootstrap_token_contract_from_checkpoint(
        checkpoint,
        processor,
        task_spec,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **build_model_load_kwargs(model_dtype),
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.config.use_cache = True
    model.eval()
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
