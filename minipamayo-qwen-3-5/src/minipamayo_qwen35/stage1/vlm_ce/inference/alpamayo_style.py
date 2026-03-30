"""Alpamayo-style Stage 1 sample inference.

This keeps the inference path close to `related_repos/alpamayo`:
- fixed `MIN_PIXELS` / `MAX_PIXELS`
- message-based prompt construction
- `processor.apply_chat_template(...)` as the tokenizer entrypoint

Unlike Alpamayo proper, this Stage 1 script currently uses only the single current image
stored in the extracted Stage 1 sample. Ego-motion history is injected through canonical
Stage 1 history tokens inside the Alpamayo-style prompt path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import enforce_runtime_prerequisites
from ....utils.run_metadata import collect_processor_settings
from ....contract.task_spec import CanonicalStage1Spec, KappaOnlyStage1Spec, Stage1TaskSpec
from ....contract.prompt import (
    add_prompt_special_tokens,
    build_multimodal_messages,
    build_stage1_question_user_text,
)
from ....action_space.record_adapter import rollout_waypoints_from_action_tensor
from ...dataset import Stage1JsonlDataset
from ..eval.runner import (
    greedy_generate_action_tokens,
    require_record_field,
    resolve_checkpoint_args,
    resolve_dtype,
    resolve_processor_path,
)
from ....contract.history_tokens import HistoryTokenRegistry, HistoryTrajectoryQuantizer
from ....contract.trajectory_tokens import Stage1TokenRegistry
from ..train import (
    CHECKPOINT_KIND_FULL,
    CHECKPOINT_KIND_MODEL_ONLY,
    build_model_load_kwargs,
    inject_history_inputs_embeds,
    load_checkpoint,
)
from ....helper import MAX_PIXELS, MIN_PIXELS, to_device

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "checkpoint",
    "test_jsonl",
    "output_json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-style Stage 1 inference on one sample."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--test-jsonl", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default="")
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
        base_dirs={
            "project_root": PROJECT_ROOT,
            "config_dir": config_path.parent,
        },
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
            "Stage 1 inference accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.test_jsonl:
        raise RuntimeError("`test_jsonl` must be defined in the config JSON.")
    if not args.output_json:
        raise RuntimeError("`output_json` must be defined in the config JSON.")
    if args.sample_index < 0:
        raise RuntimeError("`sample_index` must be >= 0.")
    return args


def resolve_task_spec(checkpoint: dict) -> Stage1TaskSpec:
    if "stage1_metadata" not in checkpoint or not isinstance(checkpoint["stage1_metadata"], dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    action_representation = checkpoint["stage1_metadata"].get("action_representation")
    if action_representation == "accel_kappa":
        return CanonicalStage1Spec()
    if action_representation == "kappa_only":
        return KappaOnlyStage1Spec()
    raise RuntimeError(f"Unsupported Stage 1 action representation: {action_representation!r}")


def resolve_checkpoint_kind(checkpoint: dict) -> str:
    if "checkpoint_kind" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `checkpoint_kind`.")
    checkpoint_kind = checkpoint["checkpoint_kind"]
    if checkpoint_kind not in {CHECKPOINT_KIND_FULL, CHECKPOINT_KIND_MODEL_ONLY}:
        raise RuntimeError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")
    return checkpoint_kind


def infer_vision_tokens(prompt_inputs: dict) -> tuple[list[int] | None, int | None]:
    if "image_grid_thw" not in prompt_inputs:
        return None, None
    image_grid = prompt_inputs["image_grid_thw"][0].detach().cpu().tolist()
    if len(image_grid) != 3:
        return image_grid, None
    return image_grid, int((image_grid[1] * image_grid[2]) / 4)


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("Stage 1 Alpamayo-style inference currently expects CUDA.")
    enforce_runtime_prerequisites(git_cwd=Path(__file__).resolve().parent)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_kind = resolve_checkpoint_kind(checkpoint)
    checkpoint_args = resolve_checkpoint_args(checkpoint)
    task_spec = resolve_task_spec(checkpoint)
    stage1_metadata = checkpoint["stage1_metadata"]
    task_spec.validate_checkpoint(stage1_metadata)

    processor_path = resolve_processor_path(checkpoint_path)
    model_path = str(checkpoint_args["model_path"])
    model_dtype = resolve_dtype(str(checkpoint_args["dtype"]))
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    tokenizer = processor.tokenizer
    add_prompt_special_tokens(tokenizer)

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
    registry.add_to_tokenizer(tokenizer)
    history_registry = HistoryTokenRegistry(
        n_bins=int(history_cfg["n_bins"]),
        token_prefix=str(history_cfg["token_prefix"]),
        start_index=int(history_cfg["start_index"]),
    )
    history_registry.add_to_tokenizer(tokenizer)
    processor_settings = collect_processor_settings(
        processor,
        requested_min_pixels=MIN_PIXELS,
        requested_max_pixels=MAX_PIXELS,
    )

    if "history_quantizer" not in checkpoint or not isinstance(
        checkpoint["history_quantizer"], dict
    ):
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

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **build_model_load_kwargs(model_dtype),
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.config.use_cache = True
    model.to(device)
    model.eval()

    dataset = Stage1JsonlDataset(args.test_jsonl)
    if args.sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index`={args.sample_index} is out of range for test dataset size {len(dataset)}."
        )
    sample = dataset[args.sample_index]
    record = dataset.records[args.sample_index]

    image_path = Path(sample["image_path"])
    user_text = build_stage1_question_user_text(
        str(stage1_metadata["question"]),
        int(history_quantizer.token_count),
    )
    with Image.open(image_path).convert("RGB") as image:
        frame_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        messages = build_multimodal_messages(frames=frame_tensor, user_text=user_text)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    model_inputs = to_device(inputs, device=device)
    model_inputs = inject_history_inputs_embeds(
        model=model,
        prompt_inputs=model_inputs,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        history_xyz=sample["ego_history_xyz"].unsqueeze(0).to(device=device, dtype=torch.float32),
        history_rot=sample["ego_history_rot"].unsqueeze(0).to(device=device, dtype=torch.float32),
    )
    image_grid_thw, vision_tokens = infer_vision_tokens(model_inputs)

    generated_token_ids = greedy_generate_action_tokens(
        model,
        model_inputs,
        registry,
        int(stage1_metadata["target_dim"]),
        model_dtype,
    )[0]
    pred_token_ids = [int(token_id) for token_id in generated_token_ids.detach().cpu().tolist()]
    pred_bin_ids = registry.decode_token_ids_to_bin_ids(pred_token_ids)
    pred_target = registry.decode_target_token_ids(pred_token_ids, quantizer)
    pred_target_tensor = torch.tensor(pred_target, dtype=torch.float32)
    gt_action_tensor = sample["action"].reshape(-1).to(torch.float32)
    pred_full_action_tensor = task_spec.full_action_from_target_tensor(
        pred_target_tensor,
        gt_action_tensor=gt_action_tensor,
    )

    pred_waypoints = (
        rollout_waypoints_from_action_tensor(
            action=pred_full_action_tensor.view(1, -1, 2),
            history_xyz=sample["ego_history_xyz"].to(dtype=torch.float32),
            history_rot=sample["ego_history_rot"].to(dtype=torch.float32),
            dt=float(stage1_metadata["dt"]),
        )[0].detach().cpu()
    )

    gt_target_tensor = task_spec.target_from_action_tensor(gt_action_tensor)
    gt_waypoints = sample["gt_waypoints"].reshape(-1, 2).to(torch.float32)
    pred_target_tensor_cpu = pred_target_tensor.detach().cpu()
    pred_full_action_tensor_cpu = pred_full_action_tensor.detach().cpu()

    target_mae = float(
        torch.mean(torch.abs(pred_target_tensor_cpu - gt_target_tensor.cpu())).item()
    )
    waypoint_errors = torch.norm(pred_waypoints - gt_waypoints.cpu(), dim=-1)
    ade_m = float(waypoint_errors.mean().item())
    fde_m = float(waypoint_errors[-1].item())

    output_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_kind": checkpoint_kind,
        "test_jsonl": str(Path(args.test_jsonl).resolve()),
        "sample_index": args.sample_index,
        "sample_id": str(require_record_field(record, "sample_id")),
        "source_frame_id": int(require_record_field(record, "source_frame_id")),
        "image_path": str(image_path),
        "message_style": "alpamayo_like_stage1",
        "message": {
            "system_text": messages[0]["content"][0]["text"],
            "user_text": user_text,
            "num_images": 1,
            "history_steps": int(stage1_metadata["history_steps"]),
            "history_token_count": int(stage1_metadata["history_token_count"]),
            "add_generation_prompt": True,
        },
        "prompt_stats": {
            "input_ids_len": int(model_inputs["input_ids"].shape[-1]),
            "image_grid_thw": image_grid_thw,
            "vision_tokens": vision_tokens,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "processor_settings": processor_settings,
        "action_representation": str(stage1_metadata["action_representation"]),
        "question": str(stage1_metadata["question"]),
        "prediction": {
            "token_ids": pred_token_ids,
            "bin_ids": pred_bin_ids,
            "target": pred_target_tensor_cpu.tolist(),
            "full_action": pred_full_action_tensor_cpu.tolist(),
            "waypoints": pred_waypoints.tolist(),
        },
        "ground_truth": {
            "target": gt_target_tensor.cpu().tolist(),
            "full_action": gt_action_tensor.cpu().tolist(),
            "waypoints": gt_waypoints.cpu().tolist(),
        },
        "metrics": {
            "target_mae": target_mae,
            "ade_m": ade_m,
            "fde_m": fde_m,
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "event": "stage1_inference_complete",
                "checkpoint": str(checkpoint_path),
                "sample_id": output_payload["sample_id"],
                "output_json": str(output_path),
                "action_representation": output_payload["action_representation"],
                "vision_tokens": vision_tokens,
                "input_ids_len": output_payload["prompt_stats"]["input_ids_len"],
                "ade_m": ade_m,
                "fde_m": fde_m,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
