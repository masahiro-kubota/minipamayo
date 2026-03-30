"""Canonical Stage 3 GRPO-style post-training."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import enforce_training_prerequisites
from ....utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
)
from ....stage1.vlm_ce.runtime import (
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    set_seed,
    write_run_config,
)
from ..common import (
    CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
    STAGE3_REWARD_CONTRACT_V0,
    compute_grpo_loss,
    configure_trainable_policy,
    stage3_checkpoint_payload,
)
from ..dataset import Stage3PostTrainingDataset, stage3_post_training_collate
from ..rewards import RewardWeights, aggregate_rewards, build_reasoning_reward_scorer
from ..rewards.consistency import score_consistency
from ..rewards.trajectory import score_trajectory
from ..rollout import generate_grouped_rollouts, load_stage3_rollout_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "stage2_checkpoint",
    "stage1b_checkpoint",
    "train_jsonl",
    "val_jsonl",
    "manifest_jsonl",
    "save_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train canonical Stage 3 post-training.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--stage2-checkpoint", type=str, default="")
    parser.add_argument("--stage1b-checkpoint", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="minipamayo-qwen-3-5/checkpoints/stage3/post_training",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--lambda-kl", type=float, default=0.1)
    parser.add_argument("--reward-weight-reason", type=float, default=0.0)
    parser.add_argument("--reward-weight-consistency", type=float, default=0.5)
    parser.add_argument("--reward-weight-traj", type=float, default=0.5)
    parser.add_argument("--reasoning-reward-mode", type=str, default="disabled")
    parser.add_argument("--traj-l2-weight", type=float, default=1.0)
    parser.add_argument("--traj-jerk-weight", type=float, default=0.1)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
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
        list_keys={"train_jsonl", "val_jsonl"},
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
            "Stage 3 training accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args

    if not args.stage2_checkpoint:
        raise RuntimeError("`stage2_checkpoint` must be defined in the config JSON.")
    if not args.stage1b_checkpoint:
        raise RuntimeError("`stage1b_checkpoint` must be defined in the config JSON.")
    if not args.train_jsonl:
        raise RuntimeError("`train_jsonl` must be defined in the config JSON.")
    if args.batch_size != 1:
        raise RuntimeError("Canonical Stage 3 currently requires `batch_size == 1`.")
    if args.num_rollouts < 2:
        raise RuntimeError("`num_rollouts` must be >= 2 for group-relative training.")
    if args.max_gen_tokens <= 0:
        raise RuntimeError("`max_gen_tokens` must be > 0.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.grad_accum_steps <= 0:
        raise RuntimeError("`grad_accum_steps` must be > 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = Stage3PostTrainingDataset(
        args.train_jsonl,
        manifest_jsonl=args.manifest_jsonl or None,
        max_samples=args.max_samples,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Stage 3 training dataset is empty.")

    if args.val_jsonl:
        val_dataset = Stage3PostTrainingDataset(args.val_jsonl, max_samples=args.max_samples)
        if len(val_dataset) == 0:
            raise RuntimeError("Stage 3 validation dataset is empty.")
    elif len(train_dataset) >= 2 and args.val_fraction > 0:
        val_size = max(1, int(round(len(train_dataset) * args.val_fraction)))
        val_size = min(val_size, len(train_dataset) - 1)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        val_dataset = None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": stage3_post_training_collate,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
    )


def _sample_view(batch: dict) -> dict:
    sample = {
        "sample_id": batch["sample_id"][0],
        "image_path": batch["image_path"][0],
        "action": batch["action"][0].detach().cpu(),
        "v0": batch["v0"][0].detach().cpu(),
        "gt_waypoints": batch["gt_waypoints"][0].detach().cpu(),
        "dt": float(batch["dt"][0]),
        "ego_history_xyz": batch["ego_history_xyz"][0].detach().cpu(),
        "ego_history_rot": batch["ego_history_rot"][0].detach().cpu(),
        "ego_future_xyz": batch["ego_future_xyz"][0].detach().cpu(),
        "ego_future_rot": batch["ego_future_rot"][0].detach().cpu(),
        "reasoning_text": batch["reasoning_text"][0],
        "sample_weight": float(batch["sample_weight"][0].item()),
    }
    for key in ("command", "planner_state", "decision_longitudinal", "decision_lateral"):
        if key in batch:
            sample[key] = batch[key][0]
    return sample


def _score_rollout(sample: dict, rollout, reasoning_scorer, reward_weights: RewardWeights, args):
    reasoning_result = reasoning_scorer.score(sample, rollout.parsed.reasoning_text)
    consistency_result = score_consistency(
        reasoning_text=rollout.parsed.reasoning_text,
        pred_future_xyz=rollout.pred_future_xyz,
        v0=float(sample["v0"].item()),
        dt=float(sample["dt"]),
    )
    trajectory_result = score_trajectory(
        sample=sample,
        pred_future_xyz=rollout.pred_future_xyz,
        l2_weight=args.traj_l2_weight,
        jerk_weight=args.traj_jerk_weight,
    )
    return aggregate_rewards(
        reasoning=reasoning_result,
        consistency=consistency_result,
        trajectory=trajectory_result,
        weights=reward_weights,
    )


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            raise RuntimeError("This Stage 3 trainer is intended to run on CUDA.")
        gpu_preflight = log_gpu_preflight(device)
        gpu_info = collect_gpu_info(device)
        git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
        set_seed(args.seed)

        train_loader, val_loader, train_size, val_size = build_dataloaders(args)
        train_dataset_fingerprint = collect_dataset_view_fingerprint(train_loader.dataset)
        val_dataset_fingerprint = (
            collect_dataset_view_fingerprint(val_loader.dataset) if val_loader is not None else None
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        bundle = load_stage3_rollout_bundle(
            stage2_checkpoint_path=args.stage2_checkpoint,
            stage1b_checkpoint_path=args.stage1b_checkpoint,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            flow_steps=args.flow_steps,
            device=device,
        )
        policy_model = bundle.policy_model
        reference_model = bundle.reference_model
        policy_model.config.use_cache = True
        if args.gradient_checkpointing:
            policy_model.gradient_checkpointing_enable()
            policy_model.enable_input_require_grads()
        else:
            policy_model.gradient_checkpointing_disable()
        policy_model.eval()

        trainable_params = configure_trainable_policy(policy_model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        stage3_metadata = {
            "stage2_checkpoint": args.stage2_checkpoint,
            "stage1b_checkpoint": args.stage1b_checkpoint,
            "policy_output_contract": CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
            "reward_contract": STAGE3_REWARD_CONTRACT_V0,
            "reasoning_reward_mode": args.reasoning_reward_mode,
            "num_rollouts": args.num_rollouts,
            "lambda_kl": args.lambda_kl,
        }
        run_metadata = {
            "git": git_metadata,
            "gpu": gpu_info,
            "gpu_preflight": gpu_preflight,
            "datasets": {
                "train": train_dataset_fingerprint,
                "val": val_dataset_fingerprint,
            },
            "base_stage2_metadata": bundle.stage2_metadata,
            "trainable_params": trainable_params,
        }
        write_run_config(save_dir, args, run_metadata)

        reasoning_scorer = build_reasoning_reward_scorer(args.reasoning_reward_mode)
        reward_weights = RewardWeights(
            reasoning=args.reward_weight_reason,
            consistency=args.reward_weight_consistency,
            trajectory=args.reward_weight_traj,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        metrics_history: list[dict] = []
        best_reward = float("-inf")
        best_epoch = 0
        global_step = 0

        print(
            json.dumps(
                {
                    "event": "stage3_setup",
                    "config_json": args.config_json,
                    "train_size": train_size,
                    "val_size": val_size,
                    "num_rollouts": args.num_rollouts,
                    "trainable_params": trainable_params,
                    "policy_output_contract": CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
                },
                ensure_ascii=False,
            )
        )

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            epoch_loss_total = 0.0
            epoch_reward_total = 0.0
            epoch_reason_total = 0.0
            epoch_consistency_total = 0.0
            epoch_traj_total = 0.0
            epoch_ade_total = 0.0
            epoch_fde_total = 0.0
            epoch_valid_total = 0.0
            epoch_samples = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                sample = _sample_view(batch)
                rollouts = generate_grouped_rollouts(
                    bundle=bundle,
                    batch=batch,
                    num_rollouts=args.num_rollouts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_gen_tokens,
                    requires_policy_grad=True,
                )
                reward_rows = [
                    _score_rollout(sample, rollout, reasoning_scorer, reward_weights, args)
                    for rollout in rollouts
                ]
                rewards_tensor = torch.tensor(
                    [row.total_reward for row in reward_rows],
                    dtype=torch.float32,
                    device=device,
                )
                policy_logprob_sums = torch.stack(
                    [rollout.policy_logprob_sum for rollout in rollouts],
                    dim=0,
                )
                ref_logprob_sums = torch.stack(
                    [rollout.ref_logprob_sum.detach() for rollout in rollouts],
                    dim=0,
                ).to(device=device, dtype=policy_logprob_sums.dtype)
                loss, loss_stats = compute_grpo_loss(
                    policy_logprob_sums=policy_logprob_sums,
                    ref_logprob_sums=ref_logprob_sums,
                    rewards=rewards_tensor,
                    kl_weight=args.lambda_kl,
                    sample_weight=sample["sample_weight"],
                )
                (loss / args.grad_accum_steps).backward()

                should_step = (
                    batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in policy_model.parameters() if parameter.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                mean_reward = float(rewards_tensor.mean().item())
                mean_reason = sum(row.reasoning_reward for row in reward_rows) / len(reward_rows)
                mean_consistency = (
                    sum(row.consistency_reward for row in reward_rows) / len(reward_rows)
                )
                mean_traj = sum(row.trajectory_reward for row in reward_rows) / len(reward_rows)
                mean_ade = sum(row.ade for row in reward_rows) / len(reward_rows)
                mean_fde = sum(row.fde for row in reward_rows) / len(reward_rows)
                valid_rate = (
                    sum(1 for rollout in rollouts if rollout.parsed.valid) / len(rollouts)
                )

                epoch_loss_total += float(loss.detach().item())
                epoch_reward_total += mean_reward
                epoch_reason_total += mean_reason
                epoch_consistency_total += mean_consistency
                epoch_traj_total += mean_traj
                epoch_ade_total += mean_ade
                epoch_fde_total += mean_fde
                epoch_valid_total += valid_rate
                epoch_samples += 1

                if batch_idx % args.log_every == 0:
                    log_row = {
                        "event": "stage3_train_step",
                        "epoch": epoch,
                        "batch": batch_idx,
                        "sample_id": sample["sample_id"],
                        "loss": round(float(loss.detach().item()), 6),
                        "reward": round(mean_reward, 6),
                        "approx_kl": round(loss_stats["approx_kl"], 6),
                        "valid_rollout_rate": round(valid_rate, 6),
                    }
                    print(json.dumps(log_row, ensure_ascii=False))
                    maybe_wandb_log(
                        wandb_run,
                        {
                            "train/loss": float(loss.detach().item()),
                            "train/reward": mean_reward,
                            "train/reason_reward": mean_reason,
                            "train/consistency_reward": mean_consistency,
                            "train/traj_reward": mean_traj,
                            "train/ade": mean_ade,
                            "train/fde": mean_fde,
                            "train/valid_rollout_rate": valid_rate,
                            "train/approx_kl": loss_stats["approx_kl"],
                        },
                        step=global_step,
                    )

            avg_loss = epoch_loss_total / max(epoch_samples, 1)
            avg_reward = epoch_reward_total / max(epoch_samples, 1)
            avg_reason = epoch_reason_total / max(epoch_samples, 1)
            avg_consistency = epoch_consistency_total / max(epoch_samples, 1)
            avg_traj = epoch_traj_total / max(epoch_samples, 1)
            avg_ade = epoch_ade_total / max(epoch_samples, 1)
            avg_fde = epoch_fde_total / max(epoch_samples, 1)
            avg_valid = epoch_valid_total / max(epoch_samples, 1)

            epoch_metrics = {
                "epoch": epoch,
                "loss": avg_loss,
                "reward": avg_reward,
                "reason_reward": avg_reason,
                "consistency_reward": avg_consistency,
                "traj_reward": avg_traj,
                "ade": avg_ade,
                "fde": avg_fde,
                "valid_rollout_rate": avg_valid,
                "global_step": global_step,
                "epoch_seconds": round(time.perf_counter() - epoch_start, 3),
            }
            metrics_history.append(epoch_metrics)
            _write_json(save_dir / "history.json", metrics_history)

            checkpoint = stage3_checkpoint_payload(
                model=policy_model,
                optimizer=optimizer,
                args=vars(args),
                stage3_metadata=stage3_metadata,
                metrics_history=metrics_history,
                run_metadata=run_metadata,
                epoch=epoch,
                global_step=global_step,
            )
            torch.save(checkpoint, save_dir / "last.pt")
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_epoch = epoch
                torch.save(checkpoint, save_dir / "best.pt")

            summary = {
                "best_reward": best_reward,
                "best_epoch": best_epoch if best_epoch > 0 else None,
                "completed_epochs": epoch,
                "train_size": train_size,
                "val_size": val_size,
                "trainable_params": trainable_params,
                "policy_output_contract": CANONICAL_STAGE3_POLICY_OUTPUT_CONTRACT,
                "peak_reserved_gib": (
                    format_gib(torch.cuda.max_memory_reserved(device))
                    if torch.cuda.is_available()
                    else None
                ),
                "elapsed_seconds": round(time.perf_counter() - wall_start, 3),
            }
            _write_json(save_dir / "summary.json", summary)
            maybe_wandb_log(
                wandb_run,
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_reward": avg_reward,
                    "train/epoch_reason_reward": avg_reason,
                    "train/epoch_consistency_reward": avg_consistency,
                    "train/epoch_traj_reward": avg_traj,
                    "train/epoch_ade": avg_ade,
                    "train/epoch_fde": avg_fde,
                    "train/epoch_valid_rollout_rate": avg_valid,
                    "summary/best_reward": best_reward,
                    "summary/best_epoch": best_epoch if best_epoch > 0 else None,
                },
                step=global_step,
            )

        print(
            json.dumps(
                {
                    "event": "stage3_complete",
                    "save_dir": str(save_dir),
                    "best_reward": best_reward,
                    "best_epoch": best_epoch if best_epoch > 0 else None,
                },
                ensure_ascii=False,
            )
        )
    finally:
        if wandb_run is not None:
            maybe_wandb_finish(wandb_run)
