"""Stage 4 trainer for the Qwen3.5 branch.

This is a lightweight GRPO-style post-training loop using only local rewards.
The policy starts from the Stage 3 checkpoint. A frozen Stage 2 decoder is
optional; if absent, rewards fall back to the discrete action tokens generated
by the policy itself.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader

from minipamayo_qwen35.models.trajectory_decoder import cfm_sample, load_decoder_from_checkpoint
from minipamayo_qwen35.sequence.rollout_parser import parse_generated_sequence
from minipamayo_qwen35.sequence.stage3_builder import build_stage3_prompt_text, build_reasoning_text
from minipamayo_qwen35.stage1.vlm_ce.eval import load_components
from minipamayo_qwen35.stage1.vlm_ce.train import (
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    move_inputs_to_device,
    set_seed,
    write_run_config,
)
from minipamayo_qwen35.utils.dynamics import forward_dynamics_batch
from minipamayo_qwen35.utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from minipamayo_qwen35.utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from minipamayo_qwen35.utils.preflight import enforce_training_prerequisites
from minipamayo_qwen35.utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
)
from minipamayo_qwen35.utils.stage34_dataset import Stage34JsonlDataset, stage34_collate

PROJECT_ROOT = Path(__file__).resolve().parents[6]
CONFIG_PATH_KEYS = {
    "stage3_checkpoint",
    "stage2_checkpoint",
    "train_jsonl",
    "save_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 Stage 4 with simplified GRPO.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--stage3-checkpoint", type=str, default="")
    parser.add_argument("--stage2-checkpoint", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="minipamayo-qwen-3-5/checkpoints/stage4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--grpo-beta", type=float, default=0.5)
    parser.add_argument("--lambda-kl", type=float, default=0.1)
    parser.add_argument("--reward-weight-reason", type=float, default=0.25)
    parser.add_argument("--reward-weight-consistency", type=float, default=0.35)
    parser.add_argument("--reward-weight-traj", type=float, default=0.40)
    parser.add_argument("--traj-l2-weight", type=float, default=1.0)
    parser.add_argument("--traj-jerk-weight", type=float, default=0.1)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--wandb-project", type=str, default="minipamayo-qwen35")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
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
        raise RuntimeError("Stage 4 training accepts only --config-json. Put all settings in the JSON file.")

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.stage3_checkpoint:
        raise RuntimeError("`stage3_checkpoint` must be defined in the config JSON.")
    if not args.train_jsonl:
        raise RuntimeError("`train_jsonl` must be defined in the config JSON.")
    if args.num_rollouts < 2:
        raise RuntimeError("`num_rollouts` must be >= 2 for group-relative training.")
    if args.max_gen_tokens <= 0:
        raise RuntimeError("`max_gen_tokens` must be > 0.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def build_dataloader(args: argparse.Namespace) -> tuple[DataLoader, int]:
    dataset = Stage34JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=stage34_collate,
    )
    return loader, len(dataset)


def _load_stage3_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if "args" not in checkpoint:
        raise RuntimeError("Stage 3 checkpoint is missing canonical `args` metadata.")
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Stage 3 checkpoint is missing canonical `model_state_dict`.")
    if "token_registry" not in checkpoint:
        raise RuntimeError("Stage 3 checkpoint is missing canonical `token_registry` metadata.")
    if "quantizer" not in checkpoint:
        raise RuntimeError("Stage 3 checkpoint is missing canonical `quantizer` metadata.")
    return checkpoint


def _load_policy_components(args: argparse.Namespace, stage3_checkpoint: dict):
    stage3_args = stage3_checkpoint["args"]
    if "stage1_checkpoint" not in stage3_args:
        raise RuntimeError("Stage 3 checkpoint args are missing canonical `stage1_checkpoint`.")
    base_args = SimpleNamespace(
        checkpoint=str(stage3_args["stage1_checkpoint"]),
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    _stage1_checkpoint, model, processor, registry, quantizer, model_dtype = load_components(base_args)
    model.load_state_dict(stage3_checkpoint["model_state_dict"])
    return model, processor, registry, quantizer, model_dtype


def _set_stage4_trainable_params(model) -> int:
    trainable_params = 0
    for name, parameter in model.named_parameters():
        lower = name.lower()
        if any(token in lower for token in ("vision", "visual", "patch_embed", "merger")):
            parameter.requires_grad_(False)
        else:
            parameter.requires_grad_(True)
            trainable_params += parameter.numel()
    return trainable_params


def _clone_inputs(batch_inputs: dict) -> dict:
    cloned = {}
    for key, value in batch_inputs.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _append_token_to_inputs(current_inputs: dict, token_id: int) -> None:
    token_tensor = torch.tensor([[token_id]], device=current_inputs["input_ids"].device, dtype=torch.long)
    current_inputs["input_ids"] = torch.cat([current_inputs["input_ids"], token_tensor], dim=1)
    current_inputs["attention_mask"] = torch.cat(
        [
            current_inputs["attention_mask"],
            torch.ones((current_inputs["attention_mask"].shape[0], 1), device=token_tensor.device, dtype=current_inputs["attention_mask"].dtype),
        ],
        dim=1,
    )
    if "mm_token_type_ids" in current_inputs:
        current_inputs["mm_token_type_ids"] = torch.cat(
            [
                current_inputs["mm_token_type_ids"],
                torch.zeros((current_inputs["mm_token_type_ids"].shape[0], 1), device=token_tensor.device, dtype=current_inputs["mm_token_type_ids"].dtype),
            ],
            dim=1,
        )


def prepare_prompt_inputs(batch: dict, processor, device: torch.device) -> dict:
    images = [Image.open(batch["image_path"][0]).convert("RGB")]
    try:
        prompt_text = build_stage3_prompt_text(processor, float(batch["v0"][0].item()))
        prompt_inputs = processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
    finally:
        images[0].close()
    return move_inputs_to_device(prompt_inputs, device)


@torch.no_grad()
def generate_rollout(
    model,
    prompt_inputs: dict,
    model_dtype: torch.dtype,
    max_gen_tokens: int,
    temperature: float,
    eos_token_id: int,
) -> list[int]:
    current_inputs = _clone_inputs(prompt_inputs)
    generated: list[int] = []
    for _ in range(max_gen_tokens):
        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**current_inputs)
        logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        next_token_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_token_id)
        if next_token_id == eos_token_id:
            break
        _append_token_to_inputs(current_inputs, next_token_id)
    return generated


def compute_sequence_log_probs(
    model,
    prompt_inputs: dict,
    token_ids: list[int],
    model_dtype: torch.dtype,
) -> torch.Tensor:
    if not token_ids:
        raise RuntimeError("Stage 4 rollout generation returned an empty sequence.")
    full_inputs = _clone_inputs(prompt_inputs)
    for token_id in token_ids:
        _append_token_to_inputs(full_inputs, token_id)
    with torch.autocast("cuda", dtype=model_dtype):
        outputs = model(**full_inputs)
    prefix_len = prompt_inputs["input_ids"].shape[1]
    logits = outputs.logits[:, prefix_len - 1 : -1, :]
    token_tensor = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs[0, torch.arange(len(token_ids), device=logits.device), token_tensor]


@torch.no_grad()
def extract_condition_hidden_states(
    model,
    prompt_inputs: dict,
    token_ids: list[int],
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_inputs = _clone_inputs(prompt_inputs)
    for token_id in token_ids:
        _append_token_to_inputs(full_inputs, token_id)
    with torch.autocast("cuda", dtype=model_dtype):
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("Stage 4 requires `output_hidden_states=True`, but the model returned none.")
    return outputs.hidden_states[-1].detach(), full_inputs["attention_mask"].detach()


def decision_from_continuous_action(
    action: torch.Tensor,
    v0: float,
    dt: float,
) -> tuple[dict[str, str], torch.Tensor]:
    action_2d = action.view(-1, 2)
    accel = action_2d[:, 0].view(1, -1)
    kappa = action_2d[:, 1].view(1, -1)
    v0_tensor = torch.tensor([v0], dtype=torch.float32, device=action.device)
    waypoints = forward_dynamics_batch(accel, kappa, v0_tensor, dt=dt).squeeze(0).cpu()

    final_speed = float(v0 + dt * accel.sum().item())
    mean_accel = float(accel.mean().item())
    heading_delta = float((kappa.squeeze(0) * max(v0, 0.1) * dt).sum().item())
    lateral_disp = float(waypoints[-1, 1].item())

    if final_speed <= 0.5:
        longitudinal = "stop"
    elif mean_accel < -0.5:
        longitudinal = "yield"
    else:
        longitudinal = "go_straight"

    if heading_delta > 0.4:
        lateral = "turn_left"
    elif heading_delta < -0.4:
        lateral = "turn_right"
    elif lateral_disp > 1.0:
        lateral = "lane_change_left"
    elif lateral_disp < -1.0:
        lateral = "lane_change_right"
    else:
        lateral = "lane_keeping"

    return {"longitudinal": longitudinal, "lateral": lateral}, waypoints


def score_reasoning_text(pred_reasoning: str, target_reasoning: str) -> float:
    pred_text = pred_reasoning.lower()
    target_text = target_reasoning.lower()
    pred_tokens = {token for token in pred_text.replace("\n", " ").split(" ") if token}
    target_tokens = {token for token in target_text.replace("\n", " ").split(" ") if token}
    overlap = pred_tokens & target_tokens
    union = pred_tokens | target_tokens
    overlap_score = len(overlap) / max(len(union), 1)

    field_score = 0.0
    for field in ("longitudinal:", "lateral:"):
        if field in pred_text and field in target_text:
            pred_value = pred_text.split(field, 1)[1].splitlines()[0].strip()
            target_value = target_text.split(field, 1)[1].splitlines()[0].strip()
            if pred_value == target_value:
                field_score += 0.5

    return 0.5 * overlap_score + 0.5 * field_score


def consistency_reward(predicted_decision: dict[str, str] | None, action_decision: dict[str, str]) -> float:
    if predicted_decision is None:
        return 0.0

    predicted_longitudinal = predicted_decision["longitudinal"]
    if predicted_longitudinal == "follow_lead":
        predicted_longitudinal = "go_straight"

    longitudinal_match = predicted_longitudinal == action_decision["longitudinal"]
    lateral_match = predicted_decision["lateral"] == action_decision["lateral"]
    return 1.0 if longitudinal_match and lateral_match else 0.0


def trajectory_reward(
    predicted_action: torch.Tensor,
    gt_waypoints: torch.Tensor,
    v0: float,
    dt: float,
    *,
    lambda_l2: float,
    lambda_jerk: float,
) -> tuple[float, torch.Tensor]:
    _action_decision, pred_waypoints = decision_from_continuous_action(predicted_action, v0, dt)
    l2_penalty = torch.mean((pred_waypoints - gt_waypoints.cpu()) ** 2).item()
    accel = predicted_action.view(-1, 2)[:, 0]
    jerk = torch.mean(torch.abs(accel[1:] - accel[:-1])).item() if accel.numel() > 1 else 0.0
    reward = -(lambda_l2 * l2_penalty + lambda_jerk * jerk)
    return reward, pred_waypoints


def composite_reward(
    r_reason: float,
    r_consistency: float,
    r_traj: float,
    *,
    w_reason: float,
    w_consistency: float,
    w_traj: float,
) -> float:
    return w_reason * r_reason + w_consistency * r_consistency + w_traj * r_traj


def checkpoint_payload(
    model,
    optimizer,
    args: argparse.Namespace,
    stage4_metadata: dict,
    metrics_history: list[dict],
    run_metadata: dict,
    epoch: int,
    global_step: int,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "stage4_metadata": stage4_metadata,
        "metrics_history": metrics_history,
        "run_metadata": run_metadata,
    }


def main() -> None:
    wandb_run = None
    args = parse_args()
    wandb_run = enforce_training_prerequisites(
        project=args.wandb_project,
        config=vars(args),
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        git_cwd=Path(__file__).resolve().parent,
    )
    wall_start = time.perf_counter()

    try:
        device = torch.device(
            args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type != "cuda":
            raise RuntimeError("This Stage 4 trainer is intended to run on CUDA.")
        gpu_preflight = log_gpu_preflight(device)
        gpu_info = collect_gpu_info(device)
        git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
        set_seed(args.seed)

        loader, train_size = build_dataloader(args)
        dataset_fingerprint = collect_dataset_view_fingerprint(loader.dataset)

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stage3_checkpoint = _load_stage3_checkpoint(args.stage3_checkpoint)
        policy_model, processor, registry, quantizer, model_dtype = _load_policy_components(args, stage3_checkpoint)
        policy_model.config.use_cache = False
        if args.gradient_checkpointing:
            policy_model.gradient_checkpointing_enable()
            policy_model.enable_input_require_grads()
        else:
            policy_model.gradient_checkpointing_disable()
        policy_model.to(device)
        policy_model.train()

        reference_model = copy.deepcopy(policy_model)
        reference_model.to(device)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)

        trainable_params = _set_stage4_trainable_params(policy_model)

        decoder = None
        if args.stage2_checkpoint:
            decoder, _decoder_checkpoint = load_decoder_from_checkpoint(args.stage2_checkpoint, device)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        stage2_checkpoint = None
        if args.stage2_checkpoint:
            stage2_checkpoint = args.stage2_checkpoint

        run_metadata = {
            "git": git_metadata,
            "gpu": gpu_info,
            "gpu_preflight": gpu_preflight,
            "datasets": {
                "train": dataset_fingerprint,
            },
            "base_stage3_metadata": stage3_checkpoint.get("stage3_metadata"),
            "stage2_checkpoint": stage2_checkpoint,
            "trainable_params": trainable_params,
        }
        write_run_config(save_dir, args, run_metadata)

        stage4_metadata = {
            "stage3_checkpoint": args.stage3_checkpoint,
            "stage2_checkpoint": stage2_checkpoint,
            "sample_format": "jsonl+images",
            "reward_source": "local_reason_consistency_traj",
            "num_rollouts": args.num_rollouts,
            "flow_steps": args.flow_steps,
        }

        if processor.tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer is missing `eos_token_id`, which Stage 4 requires.")
        eos_token_id = int(processor.tokenizer.eos_token_id)

        print(
            json.dumps(
                {
                    "event": "stage4_setup",
                    "config_json": args.config_json,
                    "run_config_path": str(save_dir / "run_config.json"),
                    "stage4_metadata": stage4_metadata,
                    "train_size": train_size,
                    "num_rollouts": args.num_rollouts,
                    "trainable_params": trainable_params,
                    "decoder_loaded": decoder is not None,
                },
                ensure_ascii=False,
            )
        )
        maybe_wandb_log(
            wandb_run,
            {
                "setup/train_size": train_size,
                "setup/num_rollouts": args.num_rollouts,
                "setup/trainable_params": trainable_params,
                "setup/decoder_loaded": 1 if decoder is not None else 0,
            },
            step=0,
        )

        metrics_history: list[dict] = []
        best_reward = float("-inf")
        best_epoch = 0
        global_step = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            epoch_loss_total = 0.0
            epoch_reward_total = 0.0
            epoch_reason_total = 0.0
            epoch_consistency_total = 0.0
            epoch_traj_total = 0.0
            epoch_samples = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(loader, start=1):
                prompt_inputs = prepare_prompt_inputs(batch, processor, device)
                dt = float(batch["dt"][0])
                v0 = float(batch["v0"][0].item())
                gt_waypoints = batch["gt_waypoints"][0]
                target_reasoning = build_reasoning_text(batch["command"][0], batch["planner_state"][0])

                rollout_rewards: list[float] = []
                rollout_reason_scores: list[float] = []
                rollout_consistency_scores: list[float] = []
                rollout_traj_scores: list[float] = []
                rollout_tokens: list[list[int]] = []
                rollout_parsed: list[dict] = []

                for _ in range(args.num_rollouts):
                    token_ids = generate_rollout(
                        model=policy_model,
                        prompt_inputs=prompt_inputs,
                        model_dtype=model_dtype,
                        max_gen_tokens=args.max_gen_tokens,
                        temperature=args.temperature,
                        eos_token_id=eos_token_id,
                    )
                    parsed = parse_generated_sequence(
                        token_ids=token_ids,
                        tokenizer=processor.tokenizer,
                        registry=registry,
                        quantizer=quantizer,
                        action_len=batch["action"].shape[1],
                    )
                    discrete_action = torch.tensor(parsed["action"], dtype=torch.float32, device=device)
                    if decoder is not None:
                        condition_hidden_states, condition_mask = extract_condition_hidden_states(
                            model=policy_model,
                            prompt_inputs=prompt_inputs,
                            token_ids=token_ids,
                            model_dtype=model_dtype,
                        )
                        continuous_action = cfm_sample(
                            decoder,
                            condition_hidden_states=condition_hidden_states,
                            condition_mask=condition_mask,
                            n_steps=args.flow_steps,
                        )[0].to(device=device, dtype=torch.float32)
                    else:
                        continuous_action = discrete_action

                    action_decision, _ = decision_from_continuous_action(continuous_action, v0, dt)
                    reason_score = score_reasoning_text(parsed["reasoning_text"], target_reasoning)
                    consistency_score = consistency_reward(parsed["decision"], action_decision)
                    traj_score, _pred_waypoints = trajectory_reward(
                        predicted_action=continuous_action,
                        gt_waypoints=gt_waypoints,
                        v0=v0,
                        dt=dt,
                        lambda_l2=args.traj_l2_weight,
                        lambda_jerk=args.traj_jerk_weight,
                    )
                    reward = composite_reward(
                        reason_score,
                        consistency_score,
                        traj_score,
                        w_reason=args.reward_weight_reason,
                        w_consistency=args.reward_weight_consistency,
                        w_traj=args.reward_weight_traj,
                    )
                    rollout_tokens.append(token_ids)
                    rollout_parsed.append(parsed)
                    rollout_rewards.append(reward)
                    rollout_reason_scores.append(reason_score)
                    rollout_consistency_scores.append(consistency_score)
                    rollout_traj_scores.append(traj_score)

                rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
                advantages = rewards_tensor - rewards_tensor.mean()
                rollout_weights = torch.softmax(args.grpo_beta * advantages, dim=0).detach()

                sample_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                for rollout_idx, token_ids in enumerate(rollout_tokens):
                    current_log_probs = compute_sequence_log_probs(
                        model=policy_model,
                        prompt_inputs=prompt_inputs,
                        token_ids=token_ids,
                        model_dtype=model_dtype,
                    )
                    with torch.no_grad():
                        reference_log_probs = compute_sequence_log_probs(
                            model=reference_model,
                            prompt_inputs=prompt_inputs,
                            token_ids=token_ids,
                            model_dtype=model_dtype,
                        )
                    seq_log_prob = current_log_probs.sum()
                    kl_penalty = (current_log_probs - reference_log_probs).mean()
                    sample_loss = sample_loss - rollout_weights[rollout_idx] * (
                        seq_log_prob - args.lambda_kl * kl_penalty
                    )

                (sample_loss / args.grad_accum_steps).backward()
                epoch_loss_total += float(sample_loss.detach().cpu())
                epoch_reward_total += float(rewards_tensor.mean().item())
                epoch_reason_total += float(sum(rollout_reason_scores) / len(rollout_reason_scores))
                epoch_consistency_total += float(sum(rollout_consistency_scores) / len(rollout_consistency_scores))
                epoch_traj_total += float(sum(rollout_traj_scores) / len(rollout_traj_scores))
                epoch_samples += 1

                should_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == len(loader)
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if args.log_every > 0 and global_step % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "loss": round(float(sample_loss.detach().cpu()), 6),
                                    "reward": round(float(rewards_tensor.mean().item()), 6),
                                },
                                ensure_ascii=False,
                            )
                        )
                        maybe_wandb_log(
                            wandb_run,
                            {
                                "train/step_loss": float(sample_loss.detach().cpu()),
                                "train/step_reward": float(rewards_tensor.mean().item()),
                            },
                            step=global_step,
                        )

            avg_loss = epoch_loss_total / max(epoch_samples, 1)
            avg_reward = epoch_reward_total / max(epoch_samples, 1)
            avg_reason = epoch_reason_total / max(epoch_samples, 1)
            avg_consistency = epoch_consistency_total / max(epoch_samples, 1)
            avg_traj = epoch_traj_total / max(epoch_samples, 1)

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(
                        model=policy_model,
                        optimizer=optimizer,
                        args=args,
                        stage4_metadata=stage4_metadata,
                        metrics_history=metrics_history,
                        run_metadata=run_metadata,
                        epoch=epoch,
                        global_step=global_step,
                    ),
                    save_dir / "best.pt",
                )

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "loss": avg_loss,
                "reward": avg_reward,
                "reason_reward": avg_reason,
                "consistency_reward": avg_consistency,
                "traj_reward": avg_traj,
                "best_reward": best_reward,
                "best_epoch": best_epoch,
                "epoch_elapsed_s": round(time.perf_counter() - epoch_start, 3),
            }
            metrics_history.append(epoch_metrics)
            print(json.dumps({"event": "epoch_end", **epoch_metrics}, ensure_ascii=False))
            maybe_wandb_log(
                wandb_run,
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_reward": avg_reward,
                    "train/reason_reward": avg_reason,
                    "train/consistency_reward": avg_consistency,
                    "train/traj_reward": avg_traj,
                    "summary/best_reward": best_reward,
                    "summary/best_epoch": best_epoch,
                },
                step=global_step,
            )

            torch.save(
                checkpoint_payload(
                    model=policy_model,
                    optimizer=optimizer,
                    args=args,
                    stage4_metadata=stage4_metadata,
                    metrics_history=metrics_history,
                    run_metadata=run_metadata,
                    epoch=epoch,
                    global_step=global_step,
                ),
                save_dir / "last.pt",
            )
            with (save_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)

        torch.save(
            checkpoint_payload(
                model=policy_model,
                optimizer=optimizer,
                args=args,
                stage4_metadata=stage4_metadata,
                metrics_history=metrics_history,
                run_metadata=run_metadata,
                epoch=args.max_epochs,
                global_step=global_step,
            ),
            save_dir / "final.pt",
        )

        summary = {
            "config_json": args.config_json,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "run_config_path": str(save_dir / "run_config.json"),
            "train_jsonl": args.train_jsonl,
            "train_size": train_size,
            "completed_epochs": args.max_epochs,
            "best_reward": best_reward,
            "best_epoch": best_epoch,
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
            "total_wall_time_s": round(time.perf_counter() - wall_start, 3),
            "run_metadata": run_metadata,
            "stage4_metadata": stage4_metadata,
            "history": metrics_history,
        }
        with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        maybe_wandb_log(
            wandb_run,
            {
                "summary/best_reward": best_reward,
                "summary/best_epoch": best_epoch,
                "summary/total_wall_time_s": summary["total_wall_time_s"],
            },
            step=global_step,
        )
        print(json.dumps({"event": "stage4_summary", **summary}, ensure_ascii=False))
    finally:
        if wandb_run is not None:
            maybe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
