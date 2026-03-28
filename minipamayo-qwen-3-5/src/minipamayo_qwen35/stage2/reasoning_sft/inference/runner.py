"""Canonical Stage 2 inference with Alpamayo-style handoff to Stage 1B expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from ....sequence.stage3_builder import build_stage2_prompt_text
from ....stage1.expert_cfm.action_space import UnicycleAccelCurvatureActionSpace
from ....stage1.expert_cfm.diffusion import FlowMatchingDiffusion
from ....stage1.expert_cfm.model import load_action_expert_from_checkpoint
from ....stage1.prompt import COT_END_TOKEN, TRAJ_FUTURE_START_TOKEN
from ....stage1.vlm_ce.eval import load_components
from ....stage1.vlm_ce.train import (
    load_checkpoint,
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


class StopAfterEOS(StoppingCriteria):
    """Stop one token after the first `<|traj_future_start|>` generation."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = int(eos_token_id)
        self.eos_found = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = input_ids.shape[0]
        if self.eos_found is None:
            self.eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        if self.eos_found.all():
            return True
        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id
        self.eos_found = self.eos_found | current_has_eos
        return False


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
    handoff_attention_mask = torch.ones_like(outputs.sequences, dtype=prompt_inputs["attention_mask"].dtype)
    for row_idx, stop_pos in enumerate(stop_positions.tolist()):
        cutoff = prompt_len + stop_pos + 1
        handoff_attention_mask[row_idx, cutoff:] = 0
    return outputs, generated_ids, handoff_attention_mask, stop_positions


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
        registry,
        history_registry,
        history_quantizer,
        _quantizer,
        _model_dtype,
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
    generation_outputs, reasoning_token_ids, prompt_attention_mask, stop_positions = generate_reasoning_handoff(
        model=model,
        tokenizer=processor.tokenizer,
        prompt_inputs=prompt_inputs,
        traj_registry=registry,
        stop_token_id=stop_token_id,
        max_new_tokens=args.max_reasoning_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    decoded_reasoning_ids = reasoning_token_ids[0, : int(stop_positions[0].item())].detach().cpu().tolist()
    cot_end_token_id = int(processor.tokenizer.convert_tokens_to_ids(COT_END_TOKEN))
    if decoded_reasoning_ids and decoded_reasoning_ids[-1] == cot_end_token_id:
        decoded_reasoning_ids = decoded_reasoning_ids[:-1]
    reasoning_text = processor.tokenizer.decode(
        decoded_reasoning_ids,
        skip_special_tokens=False,
    )

    prompt_cache = generation_outputs.past_key_values

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
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
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
