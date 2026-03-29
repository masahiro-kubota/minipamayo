"""Canonical Stage 2 inference with Alpamayo-style handoff to Stage 1B expert."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

from ....config import AlpamayoR1Config
from ....contract.prompt import TRAJ_FUTURE_START_TOKEN
from ....action_space.record_adapter import (
    canonicalize_history_batch_for_action_space,
)
from ....helper import create_message, get_processor, to_device
from ....models.alpamayo_r1 import AlpamayoR1
from ....models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN
from ....models.token_utils import StopAfterEOS
from ....stage1.vlm_ce.eval.runner import load_components, resolve_processor_path
from ....stage1.vlm_ce.train import (
    load_checkpoint,
)
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import require_expected_cuda_toolkit
from ....utils.run_metadata import collect_processor_settings
from ..data import ReasoningSftJsonlDataset

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage1b_checkpoint",
    "sample_jsonl",
    "output_json",
}


class TrajectoryLogitsProcessor(LogitsProcessor):
    """Mask discrete trajectory token logits during reasoning rollout."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = int(traj_token_offset)
        self.traj_vocab_size = int(traj_vocab_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical Stage 2 reasoning rollout and hand off to Stage 1B expert."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--sample-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--max-reasoning-tokens", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0)
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
            "Stage 2 inference accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.checkpoint or not args.stage1b_checkpoint or not args.sample_jsonl:
        raise RuntimeError(
            "`checkpoint`, `stage1b_checkpoint`, and `sample_jsonl` must be defined in config JSON."
        )
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    if args.max_reasoning_tokens <= 0:
        raise RuntimeError("`max_reasoning_tokens` must be > 0.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.temperature <= 0.0:
        raise RuntimeError("`temperature` must be > 0.")
    if not (0.0 < args.top_p <= 1.0):
        raise RuntimeError("`top_p` must be in (0, 1].")
    if args.top_k < 0:
        raise RuntimeError("`top_k` must be >= 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def generate_reasoning_handoff(
    *,
    model,
    tokenizer,
    prompt_inputs: dict,
    traj_registry,
    stop_token_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    generation_config = model.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k if top_k > 0 else None
    generation_config.pad_token_id = tokenizer.pad_token_id

    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=stop_token_id)])
    logits_processor = LogitsProcessorList(
        [
            TrajectoryLogitsProcessor(
                traj_token_offset=traj_registry.start_index,
                traj_vocab_size=traj_registry.n_bins,
            )
        ]
    )
    outputs = model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
    )
    if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
        raise RuntimeError("Stage 2 generation did not return `past_key_values` for expert handoff.")

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    generated_ids = outputs.sequences[:, prompt_len:]
    stop_mask = generated_ids == stop_token_id
    has_stop = stop_mask.any(dim=1)
    if not torch.all(has_stop):
        raise RuntimeError(
            "Stage 2 reasoning rollout did not emit `<|traj_future_start|>` within the token budget.\n"
            f"max_reasoning_tokens={max_new_tokens}"
        )
    stop_positions = stop_mask.int().argmax(dim=1)
    handoff_attention_mask = torch.ones_like(
        outputs.sequences,
        dtype=prompt_inputs["attention_mask"].dtype,
    )
    for row_idx, stop_pos in enumerate(stop_positions.tolist()):
        cutoff = prompt_len + stop_pos + 1
        handoff_attention_mask[row_idx, cutoff:] = 0
    return outputs, generated_ids, handoff_attention_mask, stop_positions


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

    wrapper.vlm.config.vocab_size = len(tokenizer)
    if hasattr(wrapper.vlm.config, "text_config"):
        wrapper.vlm.config.text_config.vocab_size = len(tokenizer)
    if hasattr(wrapper.vlm, "generation_config"):
        wrapper.vlm.generation_config.pad_token_id = tokenizer.pad_token_id


def _build_alpamayo_wrapper(
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


def load_stage2_inference_bundle(
    *,
    stage2_checkpoint_path: str | Path,
    stage1b_checkpoint_path: str | Path,
    image_min_pixels: int,
    image_max_pixels: int,
    flow_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(Path(stage2_checkpoint_path))
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict) or "stage1a_checkpoint" not in checkpoint_args:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical `stage1a_checkpoint` args metadata."
        )

    stage1a_checkpoint_path = Path(str(checkpoint_args["stage1a_checkpoint"]))
    stage1_args = argparse.Namespace(
        checkpoint=str(stage1a_checkpoint_path),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    (
        stage1_checkpoint,
        model,
        loaded_processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        _model_dtype,
    ) = load_components(stage1_args)
    processor = get_processor(
        str(resolve_processor_path(stage1a_checkpoint_path)),
        tokenizer=loaded_processor.tokenizer,
    )
    stage2_embed_rows = int(
        checkpoint["model_state_dict"]["model.language_model.embed_tokens.weight"].shape[0]
    )
    target_embed_rows = model.get_input_embeddings().weight.shape[0]
    if stage2_embed_rows != target_embed_rows:
        model.resize_token_embeddings(stage2_embed_rows)
    model.load_state_dict(checkpoint["model_state_dict"])
    if target_embed_rows != stage2_embed_rows:
        model.resize_token_embeddings(target_embed_rows)
    model.config.use_cache = True
    model.to(device)
    model.eval()

    stage1b_checkpoint = torch.load(
        Path(stage1b_checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    wrapper = _build_alpamayo_wrapper(
        stage1_checkpoint=stage1_checkpoint,
        stage1_model=model,
        processor=processor,
        registry=registry,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        quantizer=quantizer,
        stage1b_checkpoint=stage1b_checkpoint,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        flow_steps=flow_steps,
        device=device,
    )
    return {
        "checkpoint": checkpoint,
        "checkpoint_args": checkpoint_args,
        "stage1_checkpoint": stage1_checkpoint,
        "stage1b_checkpoint": stage1b_checkpoint,
        "stage1a_checkpoint_path": stage1a_checkpoint_path,
        "model": model,
        "processor": processor,
        "registry": registry,
        "history_registry": history_registry,
        "history_quantizer": history_quantizer,
        "quantizer": quantizer,
        "wrapper": wrapper,
    }


def load_reasoning_sample(sample_jsonl: str | Path, sample_index: int) -> dict[str, Any]:
    dataset = ReasoningSftJsonlDataset(sample_jsonl)
    if sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index` {sample_index} is out of range for dataset size {len(dataset)}."
        )
    return dataset[sample_index]


def build_wrapper_inputs_for_sample(
    *,
    processor,
    sample: dict[str, Any],
    history_token_count: int,
    device: torch.device,
) -> dict[str, Any]:
    image_path = Path(sample["image_path"])
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        frame_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
    messages = create_message(frame_tensor, num_traj_token=history_token_count)
    tokenized_data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    wrapper_inputs = {
        "ego_history_xyz": sample["ego_history_xyz"].unsqueeze(0).to(dtype=torch.float32),
        "ego_history_rot": sample["ego_history_rot"].unsqueeze(0).to(dtype=torch.float32),
        "tokenized_data": tokenized_data,
    }
    return to_device(wrapper_inputs, device=device)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("Canonical Stage 2 inference currently expects CUDA.")
    require_expected_cuda_toolkit()

    bundle = load_stage2_inference_bundle(
        stage2_checkpoint_path=args.checkpoint,
        stage1b_checkpoint_path=args.stage1b_checkpoint,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        flow_steps=args.flow_steps,
        device=device,
    )
    checkpoint = bundle["checkpoint"]
    checkpoint_args = bundle["checkpoint_args"]
    processor = bundle["processor"]
    history_quantizer = bundle["history_quantizer"]
    wrapper = bundle["wrapper"]
    sample = load_reasoning_sample(args.sample_jsonl, args.sample_index)
    wrapper_inputs = build_wrapper_inputs_for_sample(
        processor=processor,
        sample=sample,
        history_token_count=int(history_quantizer.token_count),
        device=device,
    )
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    with amp_context:
        pred_xyz, pred_rot, extra = wrapper.sample_trajectories_from_data_with_vlm_rollout(
            wrapper_inputs,
            top_p=args.top_p,
            top_k=None if args.top_k <= 0 else int(args.top_k),
            temperature=args.temperature,
            num_traj_samples=1,
            num_traj_sets=1,
            return_extra=True,
            max_generation_length=args.max_reasoning_tokens,
        )
    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN))
    if stop_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|traj_future_start|>`.")
    reasoning_text = str(extra["cot"][0, 0, 0])

    history_xyz, history_rot = canonicalize_history_batch_for_action_space(
        sample["ego_history_xyz"].unsqueeze(0).to(device=device, dtype=torch.float32),
        sample["ego_history_rot"].unsqueeze(0).to(device=device, dtype=torch.float32),
    )
    pred_action = wrapper.action_space.traj_to_action(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        traj_future_xyz=pred_xyz[:, 0, 0],
        traj_future_rot=pred_rot[:, 0, 0],
    )
    pred_waypoints = pred_xyz[0, 0, 0, :, :2].detach().cpu()
    gt_waypoints = sample["gt_waypoints"].to(dtype=torch.float32)
    errors = torch.norm(pred_waypoints - gt_waypoints, dim=1)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1a_checkpoint": str(Path(checkpoint_args["stage1a_checkpoint"]).resolve()),
        "stage1b_checkpoint": str(Path(args.stage1b_checkpoint).resolve()),
        "sample_jsonl": str(Path(args.sample_jsonl).resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": sample["sample_id"],
        "prompt_style": "alpamayo_r1_wrapper",
        "reasoning": {
            "text": reasoning_text,
            "token_ids": None,
            "stop_token_id": stop_token_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "prediction": {
            "action": pred_action[0].reshape(-1).detach().cpu().tolist(),
            "waypoints": pred_waypoints.tolist(),
            "traj_xyz": pred_xyz[0, 0, 0].detach().cpu().tolist(),
            "traj_rot": pred_rot[0, 0, 0].detach().cpu().tolist(),
        },
        "ground_truth": {
            "waypoints": gt_waypoints.tolist(),
            "reasoning_text": sample["reasoning_text"],
        },
        "metrics": {
            "ade_m": float(errors.mean().item()),
            "fde_m": float(errors[-1].item()),
        },
        "processor_settings": collect_processor_settings(
            processor,
            requested_min_pixels=args.image_min_pixels,
            requested_max_pixels=args.image_max_pixels,
        ),
    }

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
