"""Canonical Stage 2 reasoning SFT trainer.

This stage expects reasoning supervision to be provided by the dataset via
`reasoning_text`. It does not synthesize reasoning targets from planner labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, random_split

from ....contract.sequence_layout import STAGE2_PROMPT_CONTRACT, STAGE2_TARGET_LAYOUT
from ....stage1.vlm_ce.eval import load_components
from ....stage1.vlm_ce.train import (
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    metric_improved,
    release_cuda_memory,
    set_seed,
    write_run_config,
)
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
    collect_processor_settings,
)
from ..common import (
    build_handoff_probe_dataset,
    compute_token_metrics,
    compute_weighted_loss,
    evaluate,
    evaluate_handoff_probe,
    prepare_stage2_batch,
)
from ..dataset import ReasoningSftJsonlDataset, reasoning_sft_collate

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "stage1a_checkpoint",
    "train_jsonl",
    "val_jsonl",
    "handoff_probe_jsonl",
    "save_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train canonical Stage 2 reasoning SFT.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--stage1a-checkpoint", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="minipamayo-qwen-3-5/checkpoints/stage2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--handoff-loss-weight", type=float, default=8.0)
    parser.add_argument("--handoff-probe-jsonl", type=str, default="")
    parser.add_argument("--handoff-probe-samples", type=int, default=0)
    parser.add_argument("--handoff-probe-max-per-jsonl", type=int, default=0)
    parser.add_argument("--handoff-probe-max-reasoning-tokens", type=int, default=256)
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
        list_keys={"train_jsonl", "val_jsonl", "handoff_probe_jsonl"},
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
            "Stage 2 training accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.stage1a_checkpoint:
        raise RuntimeError("`stage1a_checkpoint` must be defined in the config JSON.")
    if not args.train_jsonl:
        raise RuntimeError("`train_jsonl` must be defined in the config JSON.")
    if args.handoff_loss_weight <= 0.0:
        raise RuntimeError("`handoff_loss_weight` must be > 0.")
    if args.handoff_probe_samples < 0:
        raise RuntimeError("`handoff_probe_samples` must be >= 0.")
    if args.handoff_probe_max_per_jsonl < 0:
        raise RuntimeError("`handoff_probe_max_per_jsonl` must be >= 0.")
    if args.handoff_probe_max_reasoning_tokens <= 0:
        raise RuntimeError("`handoff_probe_max_reasoning_tokens` must be > 0.")
    if args.early_stopping_patience < 0:
        raise RuntimeError("`early_stopping_patience` must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise RuntimeError("`early_stopping_min_delta` must be >= 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = ReasoningSftJsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if args.val_jsonl:
        val_dataset = ReasoningSftJsonlDataset(args.val_jsonl)
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty.")
    elif len(train_dataset) >= 2 and args.val_fraction > 0:
        val_size = max(1, int(round(len(train_dataset) * args.val_fraction)))
        val_size = min(val_size, len(train_dataset) - 1)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
    else:
        val_dataset = None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": reasoning_sft_collate,
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


def build_stage2_metadata(dataset, args: argparse.Namespace) -> dict:
    sample = dataset[0]
    gt_waypoints = sample["gt_waypoints"].detach().cpu().reshape(-1, 2)
    action = sample["action"].detach().cpu().reshape(-1)
    return {
        "stage1a_checkpoint": args.stage1a_checkpoint,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl or None,
        "sample_format": "jsonl+images",
        "reasoning_source": "provided_reasoning_text",
        "target_layout": STAGE2_TARGET_LAYOUT,
        "prompt_contract": STAGE2_PROMPT_CONTRACT,
        "k": int(gt_waypoints.shape[0]),
        "action_dim": int(action.shape[0]),
        "dt": float(sample["dt"]),
        "handoff_loss_weight": args.handoff_loss_weight,
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    stage2_metadata: dict,
    token_registry: dict,
    quantizer: dict,
    initial_eval: dict,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
    run_metadata: dict,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "stage2_metadata": stage2_metadata,
        "token_registry": token_registry,
        "quantizer": quantizer,
        "initial_eval": initial_eval,
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
            args.device
            if args.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if device.type != "cuda":
            raise RuntimeError("This Stage 2 trainer is intended to run on CUDA.")
        gpu_preflight = log_gpu_preflight(device)
        gpu_info = collect_gpu_info(device)
        git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
        set_seed(args.seed)

        train_loader, val_loader, train_size, val_size = build_dataloaders(args)
        if len(train_loader) == 0:
            raise RuntimeError("Train DataLoader is empty.")
        explicit_handoff_probe_dataset = build_handoff_probe_dataset(args)
        train_dataset_fingerprint = collect_dataset_view_fingerprint(train_loader.dataset)
        val_dataset_fingerprint = (
            collect_dataset_view_fingerprint(val_loader.dataset) if val_loader is not None else None
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stage1_args = SimpleNamespace(
            checkpoint=args.stage1a_checkpoint,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        (
            checkpoint,
            model,
            processor,
            registry,
            history_registry,
            history_quantizer,
            action_quantizer,
            model_dtype,
        ) = load_components(stage1_args)
        processor_settings = collect_processor_settings(
            processor,
            requested_min_pixels=args.image_min_pixels or None,
            requested_max_pixels=args.image_max_pixels or None,
        )
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()
        model.to(device)
        model.train()

        stage2_metadata = build_stage2_metadata(train_loader.dataset, args)
        stage2_metadata["processor_settings"] = processor_settings
        if args.handoff_probe_jsonl:
            stage2_metadata["handoff_probe_jsonl"] = args.handoff_probe_jsonl
        stage2_metadata["handoff_probe_samples"] = int(args.handoff_probe_samples)
        stage2_metadata["handoff_probe_max_per_jsonl"] = int(args.handoff_probe_max_per_jsonl)
        stage2_metadata["handoff_probe_max_reasoning_tokens"] = int(
            args.handoff_probe_max_reasoning_tokens
        )
        run_metadata = {
            "git": git_metadata,
            "gpu": gpu_info,
            "gpu_preflight": gpu_preflight,
            "datasets": {
                "train": train_dataset_fingerprint,
                "val": val_dataset_fingerprint,
            },
            "processor": processor_settings,
            "base_stage1_metadata": checkpoint.get("stage1_metadata"),
        }
        write_run_config(save_dir, args, run_metadata)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
        total_update_steps = steps_per_epoch * args.max_epochs
        warmup_steps = int(total_update_steps * args.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if total_update_steps <= warmup_steps:
                return 1.0
            progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        initial_eval_loader = val_loader if val_loader is not None else train_loader
        initial_eval_split = "val" if val_loader is not None else "train"
        initial_eval = evaluate(
            model=model,
            dataloader=initial_eval_loader,
            processor=processor,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            device=device,
            model_dtype=model_dtype,
            handoff_loss_weight=args.handoff_loss_weight,
        )
        release_cuda_memory()

        token_registry_payload = {
            "n_bins": registry.n_bins,
            "token_prefix": registry.token_prefix,
            "token_strings": registry.token_strings,
        }
        quantizer_payload = {
            "num_bins": action_quantizer.num_bins,
            "dims_min": list(action_quantizer.dims_min),
            "dims_max": list(action_quantizer.dims_max),
        }

        metrics_history: list[dict] = []
        global_step = 0
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_metric_name = "val_loss" if val_loader is not None else "train_loss"
        best_handoff_success_rate = -1.0
        best_handoff_epoch = 0
        best_handoff_metric = float("inf")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        print(
            json.dumps(
                {
                    "event": "stage2_setup",
                    "config_json": args.config_json,
                    "run_config_path": str(save_dir / "run_config.json"),
                    "stage2_metadata": stage2_metadata,
                    "train_size": train_size,
                    "val_size": val_size,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "max_epochs": args.max_epochs,
                },
                ensure_ascii=False,
            )
        )
        print(
            json.dumps(
                {
                    "event": "initial_eval",
                    "split": initial_eval_split,
                    "loss": initial_eval["loss"],
                    "token_accuracy": initial_eval["token_accuracy"],
                },
                ensure_ascii=False,
            )
        )
        maybe_wandb_log(
            wandb_run,
            {
                "setup/train_size": train_size,
                "setup/val_size": val_size,
                "setup/batch_size": args.batch_size,
                "setup/grad_accum_steps": args.grad_accum_steps,
                f"baseline/{initial_eval_split}_loss": initial_eval["loss"],
                f"baseline/{initial_eval_split}_token_accuracy": initial_eval["token_accuracy"],
            },
            step=0,
        )

        stop_reason = "max_epochs"
        completed_epochs = 0

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss_total = 0.0
            train_batches = 0
            train_correct = 0
            train_tokens = 0
            optimizer_steps_this_epoch = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                full_inputs, labels, loss_weights = prepare_stage2_batch(
                    model=model,
                    batch=batch,
                    processor=processor,
                    history_registry=history_registry,
                    history_quantizer=history_quantizer,
                    device=device,
                    handoff_loss_weight=args.handoff_loss_weight,
                )
                with torch.autocast("cuda", dtype=model_dtype):
                    outputs = model(**model_forward_inputs(full_inputs))
                    loss = compute_weighted_loss(outputs.logits, labels, loss_weights)

                (loss / args.grad_accum_steps).backward()
                metrics = compute_token_metrics(outputs.logits.detach(), labels)
                train_loss_total += float(loss.detach().cpu())
                train_batches += 1
                train_correct += metrics["correct"]
                train_tokens += metrics["total"]

                should_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == len(
                    train_loader
                )
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    if args.log_every > 0 and global_step % args.log_every == 0:
                        payload = {
                            "event": "train_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": round(float(loss.detach().cpu()), 6),
                            "lr": scheduler.get_last_lr()[0],
                        }
                        print(json.dumps(payload, ensure_ascii=False))
                        maybe_wandb_log(
                            wandb_run,
                            {
                                "train/step_loss": float(loss.detach().cpu()),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )

            train_loss = train_loss_total / max(train_batches, 1)
            train_token_accuracy = train_correct / max(train_tokens, 1)

            if val_loader is not None:
                val_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    processor=processor,
                    history_registry=history_registry,
                    history_quantizer=history_quantizer,
                    device=device,
                    model_dtype=model_dtype,
                    handoff_loss_weight=args.handoff_loss_weight,
                )
                release_cuda_memory()
                val_loss = val_metrics["loss"]
                val_token_accuracy = val_metrics["token_accuracy"]
                metric_to_track = val_loss
            else:
                val_loss = None
                val_token_accuracy = None
                metric_to_track = train_loss

            probe_dataset = (
                explicit_handoff_probe_dataset
                if explicit_handoff_probe_dataset is not None
                else (val_loader.dataset if val_loader is not None else train_loader.dataset)
            )
            probe_max_samples = int(args.handoff_probe_samples)
            if probe_max_samples <= 0 and explicit_handoff_probe_dataset is not None:
                probe_max_samples = len(probe_dataset)
            handoff_probe = evaluate_handoff_probe(
                model=model,
                dataset=probe_dataset,
                processor=processor,
                registry=registry,
                history_registry=history_registry,
                history_quantizer=history_quantizer,
                device=device,
                max_samples=probe_max_samples,
                max_reasoning_tokens=args.handoff_probe_max_reasoning_tokens,
                seed=args.seed,
            )
            release_cuda_memory()

            improved = metric_improved(metric_to_track, best_metric, args.early_stopping_min_delta)
            if improved:
                best_metric = metric_to_track
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            handoff_improved = False
            if handoff_probe is not None:
                probe_success_rate = float(handoff_probe["success_rate"])
                if probe_success_rate > best_handoff_success_rate:
                    handoff_improved = True
                elif (
                    probe_success_rate == best_handoff_success_rate
                    and metric_to_track < best_handoff_metric
                ):
                    handoff_improved = True
                if handoff_improved:
                    best_handoff_success_rate = probe_success_rate
                    best_handoff_epoch = epoch
                    best_handoff_metric = metric_to_track

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_token_accuracy": train_token_accuracy,
                "val_loss": val_loss,
                "val_token_accuracy": val_token_accuracy,
                "metric_name": best_metric_name,
                "metric_to_track": metric_to_track,
                "improved": improved,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "lr": scheduler.get_last_lr()[0],
                "epoch_elapsed_s": round(time.perf_counter() - epoch_start, 3),
                "optimizer_steps_this_epoch": optimizer_steps_this_epoch,
            }
            if handoff_probe is not None:
                epoch_metrics["handoff_probe_num_samples"] = int(handoff_probe["num_samples"])
                epoch_metrics["handoff_probe_num_success"] = int(handoff_probe["num_success"])
                epoch_metrics["handoff_probe_success_rate"] = float(handoff_probe["success_rate"])
                epoch_metrics["handoff_probe_mean_stop_position"] = handoff_probe[
                    "mean_stop_position"
                ]
                epoch_metrics["handoff_probe_failure_sample_ids"] = handoff_probe[
                    "failure_sample_ids"
                ]
                epoch_metrics["handoff_probe_first_failure_reason"] = handoff_probe[
                    "first_failure_reason"
                ]
                epoch_metrics["best_handoff_success_rate"] = best_handoff_success_rate
                epoch_metrics["best_handoff_epoch"] = best_handoff_epoch
            metrics_history.append(epoch_metrics)
            print(json.dumps({"event": "epoch_end", **epoch_metrics}, ensure_ascii=False))
            wandb_payload = {
                "train/epoch_loss": train_loss,
                "train/token_accuracy": train_token_accuracy,
                "train/epoch_elapsed_s": epoch_metrics["epoch_elapsed_s"],
                "summary/best_metric_so_far": best_metric,
                "summary/epochs_without_improvement": epochs_without_improvement,
            }
            if val_loss is not None:
                wandb_payload["val/loss"] = val_loss
                wandb_payload["val/token_accuracy"] = val_token_accuracy
            if handoff_probe is not None:
                wandb_payload["handoff_probe/success_rate"] = float(handoff_probe["success_rate"])
                wandb_payload["handoff_probe/num_success"] = int(handoff_probe["num_success"])
                wandb_payload["handoff_probe/num_samples"] = int(handoff_probe["num_samples"])
                if handoff_probe["mean_stop_position"] is not None:
                    wandb_payload["handoff_probe/mean_stop_position"] = float(
                        handoff_probe["mean_stop_position"]
                    )
                wandb_payload["summary/best_handoff_success_rate"] = best_handoff_success_rate
            maybe_wandb_log(wandb_run, wandb_payload, step=global_step)

            if improved:
                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        stage2_metadata=stage2_metadata,
                        token_registry=token_registry_payload,
                        quantizer=quantizer_payload,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                        run_metadata=run_metadata,
                    ),
                    save_dir / "best.pt",
                )

            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    stage2_metadata=stage2_metadata,
                    token_registry=token_registry_payload,
                    quantizer=quantizer_payload,
                    initial_eval=initial_eval,
                    epoch=epoch,
                    global_step=global_step,
                    metrics_history=metrics_history,
                    run_metadata=run_metadata,
                ),
                save_dir / "last.pt",
            )
            if handoff_improved:
                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        stage2_metadata=stage2_metadata,
                        token_registry=token_registry_payload,
                        quantizer=quantizer_payload,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                        run_metadata=run_metadata,
                    ),
                    save_dir / "best_handoff.pt",
                )
            with (save_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)

            completed_epochs = epoch
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                stop_reason = "early_stopping"
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_metric_name": best_metric_name,
                            "best_metric": best_metric,
                            "best_epoch": best_epoch,
                            "patience": args.early_stopping_patience,
                            "min_delta": args.early_stopping_min_delta,
                        },
                        ensure_ascii=False,
                    )
                )
                break

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                stage2_metadata=stage2_metadata,
                token_registry=token_registry_payload,
                quantizer=quantizer_payload,
                initial_eval=initial_eval,
                epoch=completed_epochs,
                global_step=global_step,
                metrics_history=metrics_history,
                run_metadata=run_metadata,
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
            "val_jsonl": args.val_jsonl or None,
            "train_size": train_size,
            "val_size": val_size,
            "completed_epochs": completed_epochs,
            "stop_reason": stop_reason,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_handoff_success_rate": (
                best_handoff_success_rate if best_handoff_epoch > 0 else None
            ),
            "best_handoff_epoch": best_handoff_epoch if best_handoff_epoch > 0 else None,
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
            "total_wall_time_s": round(time.perf_counter() - wall_start, 3),
            "run_metadata": run_metadata,
            "stage2_metadata": stage2_metadata,
            "initial_eval": {
                "split": initial_eval_split,
                "loss": initial_eval["loss"],
                "token_accuracy": initial_eval["token_accuracy"],
            },
            "history": metrics_history,
        }
        with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        maybe_wandb_log(
            wandb_run,
            {
                "summary/best_metric": best_metric,
                "summary/best_epoch": best_epoch,
                "summary/best_handoff_success_rate": (
                    best_handoff_success_rate if best_handoff_epoch > 0 else None
                ),
                "summary/best_handoff_epoch": best_handoff_epoch if best_handoff_epoch > 0 else None,
                "summary/total_wall_time_s": summary["total_wall_time_s"],
            },
            step=global_step,
        )
        print(json.dumps({"event": "stage2_summary", **summary}, ensure_ascii=False))
    finally:
        if wandb_run is not None:
            maybe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
