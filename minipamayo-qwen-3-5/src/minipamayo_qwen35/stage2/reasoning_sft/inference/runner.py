"""Canonical Stage 2 inference with Alpamayo-style handoff to Stage 1B expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ....sequence.stage3_builder import build_stage2_prompt_text
from ....stage1.expert_cfm.action_space import UnicycleAccelCurvatureActionSpace
from ....stage1.expert_cfm.diffusion import FlowMatchingDiffusion
from ....stage1.expert_cfm.model import load_action_expert_from_checkpoint
from ....stage1.prompt import TRAJ_FUTURE_START_TOKEN
from ....stage1.vlm_ce.eval import load_components
from ....stage1.vlm_ce.train import (
    append_token_to_model_inputs,
    load_checkpoint,
    model_forward_inputs,
    prepare_prompt_inputs_with_history,
)
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.run_metadata import collect_processor_settings
from ..dataset import ReasoningSftJsonlDataset

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "checkpoint",
    "stage1b_checkpoint",
    "sample_jsonl",
    "output_json",
}


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
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def greedy_generate_until_token(
    *,
    model,
    prompt_inputs: dict,
    stop_token_id: int,
    max_new_tokens: int,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    current_inputs = dict(prompt_inputs)
    generated: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**model_forward_inputs(current_inputs))
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token)
        append_token_to_model_inputs(model, current_inputs, next_token)
        if torch.all(next_token == stop_token_id):
            return torch.stack(generated, dim=1), current_inputs
    raise RuntimeError(
        "Stage 2 reasoning rollout did not emit `<|traj_future_start|>` within the token budget.\n"
        f"max_reasoning_tokens={max_new_tokens}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("Canonical Stage 2 inference currently expects CUDA.")

    checkpoint = load_checkpoint(Path(args.checkpoint))
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict) or "stage1a_checkpoint" not in checkpoint_args:
        raise RuntimeError(
            "Stage 2 checkpoint is missing canonical `stage1a_checkpoint` args metadata."
        )

    stage1_args = argparse.Namespace(
        checkpoint=str(checkpoint_args["stage1a_checkpoint"]),
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    (
        stage1_checkpoint,
        model,
        processor,
        _registry,
        history_registry,
        history_quantizer,
        _quantizer,
        model_dtype,
    ) = load_components(stage1_args)
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

    dataset = ReasoningSftJsonlDataset(args.sample_jsonl)
    if args.sample_index >= len(dataset):
        raise RuntimeError(
            f"`sample_index` {args.sample_index} is out of range for dataset size {len(dataset)}."
        )
    sample = dataset[args.sample_index]
    batch = {
        "sample_id": [sample["sample_id"]],
        "image_path": [sample["image_path"]],
        "action": sample["action"].unsqueeze(0),
        "v0": sample["v0"].unsqueeze(0),
        "gt_waypoints": sample["gt_waypoints"].unsqueeze(0),
        "ego_history_xyz": sample["ego_history_xyz"].unsqueeze(0),
        "ego_history_rot": sample["ego_history_rot"].unsqueeze(0),
        "dt": [sample["dt"]],
        "reasoning_text": [sample["reasoning_text"]],
    }

    prompt_text = build_stage2_prompt_text(
        processor,
        float(sample["v0"].item()),
        history_token_count=history_quantizer.token_count,
    )
    prompt_inputs = prepare_prompt_inputs_with_history(
        model=model,
        batch=batch,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        prompt_text=prompt_text,
        device=device,
    )

    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN))
    if stop_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|traj_future_start|>`.")
    reasoning_token_ids, handed_off_inputs = greedy_generate_until_token(
        model=model,
        prompt_inputs=prompt_inputs,
        stop_token_id=stop_token_id,
        max_new_tokens=args.max_reasoning_tokens,
        model_dtype=model_dtype,
    )
    stop_mask = reasoning_token_ids != stop_token_id
    decoded_reasoning_ids = reasoning_token_ids[0][stop_mask[0]].detach().cpu().tolist()
    reasoning_text = processor.tokenizer.decode(
        decoded_reasoning_ids,
        skip_special_tokens=False,
    )

    with torch.no_grad():
        prompt_outputs = model(
            **model_forward_inputs(handed_off_inputs),
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    prompt_cache = prompt_outputs.past_key_values
    prompt_attention_mask = handed_off_inputs["attention_mask"]

    expert, expert_checkpoint = load_action_expert_from_checkpoint(args.stage1b_checkpoint, device)
    if "stage1b_metadata" not in expert_checkpoint or not isinstance(
        expert_checkpoint["stage1b_metadata"], dict
    ):
        raise RuntimeError("Stage 1B checkpoint is missing canonical `stage1b_metadata`.")
    stage1b_metadata = expert_checkpoint["stage1b_metadata"]
    if "dt" not in stage1b_metadata:
        raise RuntimeError("Stage 1B checkpoint metadata is missing canonical `dt`.")
    diffusion = FlowMatchingDiffusion(n_steps=args.flow_steps)
    action_space = UnicycleAccelCurvatureActionSpace(
        k=int(stage1b_metadata["k"]) if "k" in stage1b_metadata else int(sample["action"].shape[0] // 2),
        dt=float(stage1b_metadata["dt"]),
    )
    pred_action = diffusion.sample(
        expert=expert,
        prompt_cache=prompt_cache,
        prompt_attention_mask=prompt_attention_mask,
    ).reshape(1, -1, 2)
    pred_xyz, pred_rot = action_space.action_to_traj(
        traj_history_xyz=batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
        traj_history_rot=batch["ego_history_rot"].to(device=device, dtype=torch.float32),
        action=pred_action,
    )
    pred_waypoints = pred_xyz[0, 0, :, :2].detach().cpu()
    gt_waypoints = sample["gt_waypoints"].to(dtype=torch.float32)
    errors = torch.norm(pred_waypoints - gt_waypoints, dim=1)

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1a_checkpoint": str(Path(checkpoint_args["stage1a_checkpoint"]).resolve()),
        "stage1b_checkpoint": str(Path(args.stage1b_checkpoint).resolve()),
        "sample_jsonl": str(Path(args.sample_jsonl).resolve()),
        "sample_index": int(args.sample_index),
        "sample_id": sample["sample_id"],
        "prompt_style": "alpamayo_like_reasoning_handoff",
        "reasoning": {
            "text": reasoning_text,
            "token_ids": reasoning_token_ids[0].detach().cpu().tolist(),
            "stop_token_id": stop_token_id,
        },
        "prediction": {
            "action": pred_action[0].detach().cpu().tolist(),
            "waypoints": pred_waypoints.tolist(),
            "traj_xyz": pred_xyz[0, 0].detach().cpu().tolist(),
            "traj_rot": pred_rot[0, 0].detach().cpu().tolist(),
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
