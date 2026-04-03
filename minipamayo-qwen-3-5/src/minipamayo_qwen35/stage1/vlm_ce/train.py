"""Canonical Stage 1A trainer for the Qwen3.5 branch."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...contract.prompt import DEFAULT_QUESTION
from ...contract.task_spec import CanonicalStage1Spec, Stage1TaskSpec
from ...utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ...utils.checkpoint_paths import checkpoint_scope_from_config_path, resolve_checkpoint_run_dir
from ...utils.json_config import normalize_optional_string_list, normalize_required_string_list
from ...utils.preflight import enforce_training_prerequisites
from ...utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)
from ..stage1_train_data import build_stage1_train_val_dataloaders
from ..stage1_train_runtime import (
    best_metric_from_history,
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    metric_improved,
    move_optimizer_state_to_device,
    release_cuda_memory,
    set_seed,
    write_run_config,
)
from ..stage1a_components import (
    CHECKPOINT_KIND_FULL,
    CHECKPOINT_KIND_MODEL_ONLY,
    build_model_load_kwargs,
    build_processor_kwargs,
    build_stage1_metadata,
    build_training_token_contract,
    load_checkpoint,
    resolve_dtype,
)
from ..stage1a_prompting import model_forward_inputs
from ..stage1a_runtime import (
    Stage1ARuntime,
    compute_token_accuracy,
    prepare_stage1a_training_batch,
    run_stage1a_teacher_forced_batch,
)
from .cli import parse_config_json_only_args

CONFIG_PATH_KEYS = {
    "train_jsonl",
    "val_jsonl",
    "model_path",
    "resume_from_checkpoint",
    "save_dir",
}
MULTI_VALUE_CONFIG_KEYS = {"train_jsonl", "val_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train canonical Stage 1A VLM CE with train/val data and checkpoints."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument("--wandb-project", type=str, default="minipamayo-qwen35")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    gc_group = parser.add_mutually_exclusive_group()
    gc_group.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    gc_group.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_config_json_only_args(
        parser,
        path_keys=CONFIG_PATH_KEYS,
        list_keys=MULTI_VALUE_CONFIG_KEYS,
        error_message="Stage 1 training accepts only --config-json. Put all settings in the JSON file.",
    )
    args.train_jsonl = normalize_required_string_list(args.train_jsonl, key_name="train_jsonl")
    args.val_jsonl = normalize_optional_string_list(args.val_jsonl, key_name="val_jsonl")
    if args.early_stopping_patience < 0:
        raise RuntimeError("`early_stopping_patience` must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise RuntimeError("`early_stopping_min_delta` must be >= 0.")
    args.save_dir = str(
        resolve_checkpoint_run_dir(
            args.save_dir,
            scope=checkpoint_scope_from_config_path(
                args.config_json,
                stage="stage1",
                component="vlm_ce",
                default_track="canonical",
            ),
            run_name=Path(args.config_json).resolve().stem,
        )
    )
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def evaluate(
    runtime: Stage1ARuntime,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict:
    runtime.model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        result = run_stage1a_teacher_forced_batch(
            runtime,
            batch,
            device=device,
            prompt_mode="canonical_teacher_forced",
        )
        outputs = result["outputs"]
        total_loss += float(outputs.loss.detach().cpu())
        total_correct += result["correct"]
        total_tokens += result["total"]
        total_batches += 1

    runtime.model.train()
    return {
        "loss": total_loss / max(total_batches, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def validate_resume_args(args: argparse.Namespace, checkpoint: dict) -> None:
    checkpoint_kind = checkpoint.get("checkpoint_kind")
    if checkpoint_kind != CHECKPOINT_KIND_FULL:
        raise RuntimeError(
            "Resume checkpoint must be a full checkpoint with optimizer and scheduler state. "
            f"Received checkpoint_kind={checkpoint_kind!r}."
        )
    if "optimizer_state_dict" not in checkpoint or "scheduler_state_dict" not in checkpoint:
        raise RuntimeError(
            "Resume checkpoint is missing optimizer or scheduler state required for continuation."
        )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise RuntimeError("Resume checkpoint is missing canonical `args` metadata.")
    keys_to_match = [
        "train_jsonl",
        "val_jsonl",
        "model_path",
        "dtype",
        "gradient_checkpointing",
        "image_min_pixels",
        "image_max_pixels",
    ]
    missing_keys = [key for key in keys_to_match if key not in checkpoint_args]
    if missing_keys:
        raise RuntimeError(
            "Resume checkpoint is missing canonical settings:\n" + "\n".join(missing_keys)
        )
    mismatches = []
    for key in keys_to_match:
        checkpoint_value = checkpoint_args[key]
        current_value = getattr(args, key)
        if checkpoint_value != current_value:
            mismatches.append(f"{key}: checkpoint={checkpoint_value!r}, config={current_value!r}")
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint settings do not match the current config:\n" + "\n".join(mismatches)
        )


def load_resume_state(checkpoint: dict) -> tuple[dict, list[dict], int, int]:
    required_keys = ["initial_eval", "metrics_history", "global_step", "epoch"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise RuntimeError(
            "Resume checkpoint is missing canonical state:\n" + "\n".join(missing_keys)
        )
    initial_eval = checkpoint["initial_eval"]
    metrics_history = checkpoint["metrics_history"]
    if not isinstance(initial_eval, dict):
        raise RuntimeError("Resume checkpoint `initial_eval` must be an object.")
    if not isinstance(metrics_history, list):
        raise RuntimeError("Resume checkpoint `metrics_history` must be a list.")
    return initial_eval, list(metrics_history), int(checkpoint["global_step"]), int(checkpoint["epoch"]) + 1


def checkpoint_payload(
    checkpoint_kind: str,
    runtime: Stage1ARuntime,
    *,
    args: argparse.Namespace,
    initial_eval: dict | None,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
    run_metadata: dict,
) -> dict:
    return {
        "checkpoint_kind": checkpoint_kind,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in runtime.model.state_dict().items()
        },
        "args": vars(args),
        "metrics_history": metrics_history,
        "token_registry": {
            "n_bins": runtime.registry.n_bins,
            "token_prefix": runtime.registry.token_prefix,
            "start_index": runtime.registry.start_index,
            "token_strings": runtime.registry.token_strings,
        },
        "history_registry": runtime.history_registry.metadata(),
        "history_quantizer": runtime.history_quantizer.metadata(),
        "quantizer": runtime.task_spec.quantizer_metadata(runtime.quantizer),
        "stage1_metadata": runtime.stage1_metadata,
        "initial_eval": initial_eval,
        "run_metadata": run_metadata,
    }


def full_checkpoint_payload(
    runtime: Stage1ARuntime,
    optimizer,
    scheduler,
    *,
    args: argparse.Namespace,
    initial_eval: dict | None,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
    run_metadata: dict,
) -> dict:
    payload = checkpoint_payload(
        CHECKPOINT_KIND_FULL,
        runtime,
        args=args,
        initial_eval=initial_eval,
        epoch=epoch,
        global_step=global_step,
        metrics_history=metrics_history,
        run_metadata=run_metadata,
    )
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def model_only_checkpoint_payload(
    runtime: Stage1ARuntime,
    *,
    args: argparse.Namespace,
    initial_eval: dict | None,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
    run_metadata: dict,
) -> dict:
    return checkpoint_payload(
        CHECKPOINT_KIND_MODEL_ONLY,
        runtime,
        args=args,
        initial_eval=initial_eval,
        epoch=epoch,
        global_step=global_step,
        metrics_history=metrics_history,
        run_metadata=run_metadata,
    )


def main(task_spec: Stage1TaskSpec | None = None) -> None:
    wandb_run = None
    task_spec = task_spec or CanonicalStage1Spec()
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
            raise RuntimeError("This Stage 1 trainer is intended to run on CUDA.")
        gpu_preflight = log_gpu_preflight(device)
        gpu_info = collect_gpu_info(device)
        git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
        set_seed(args.seed)

        train_loader, val_loader, train_size, val_size = build_stage1_train_val_dataloaders(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            max_samples=args.max_samples,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            require_validation_split=True,
        )
        if len(train_loader) == 0:
            raise RuntimeError("Train DataLoader is empty.")
        train_dataset_fingerprint = collect_dataset_view_fingerprint(train_loader.dataset)
        val_dataset_fingerprint = (
            collect_dataset_view_fingerprint(val_loader.dataset) if val_loader is not None else None
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        resume_checkpoint = None
        resume_checkpoint_path = None
        processor_source = args.model_path
        if args.resume_from_checkpoint:
            resume_checkpoint_path = Path(args.resume_from_checkpoint).resolve()
            if not resume_checkpoint_path.exists():
                raise RuntimeError(f"Resume checkpoint does not exist: {resume_checkpoint_path}")
            if resume_checkpoint_path.parent != save_dir.resolve():
                raise RuntimeError(
                    "`resume_from_checkpoint` must point inside the same `save_dir` so checkpoints, history, and processor stay consistent."
                )
            resume_checkpoint = load_checkpoint(resume_checkpoint_path)
            validate_resume_args(args, resume_checkpoint)
            stage1_metadata = resume_checkpoint.get("stage1_metadata")
            if not isinstance(stage1_metadata, dict):
                raise RuntimeError("Resume checkpoint is missing canonical `stage1_metadata`.")
            task_spec.validate_checkpoint(stage1_metadata)
            saved_processor_dir = resume_checkpoint_path.parent / "processor"
            if not saved_processor_dir.exists():
                raise RuntimeError(
                    "Resume checkpoint is missing the canonical saved processor directory: "
                    f"{saved_processor_dir}"
                )
            processor_source = str(saved_processor_dir)

        load_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            **build_processor_kwargs(args.image_min_pixels, args.image_max_pixels),
        )
        model_dtype = resolve_dtype(args.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            **build_model_load_kwargs(model_dtype),
        )
        (
            registry,
            history_registry,
            history_quantizer,
            quantizer,
            added_action_tokens,
            added_history_tokens,
        ) = build_training_token_contract(train_loader.dataset, processor, task_spec)
        stage1_metadata = build_stage1_metadata(
            train_loader.dataset,
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            registry=registry,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            quantizer=quantizer,
            task_spec=task_spec,
            question=args.question,
        )
        processor_settings = collect_processor_settings(
            processor,
            requested_min_pixels=args.image_min_pixels or None,
            requested_max_pixels=args.image_max_pixels or None,
        )
        stage1_metadata["processor_settings"] = processor_settings
        model.resize_token_embeddings(len(processor.tokenizer))
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        model.enable_input_require_grads()
        model.to(device)
        model.train()
        processor.save_pretrained(save_dir / "processor")
        load_elapsed = time.perf_counter() - load_start

        runtime = Stage1ARuntime(
            checkpoint={},
            model=model,
            processor=processor,
            registry=registry,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            quantizer=quantizer,
            model_dtype=model_dtype,
            task_spec=task_spec,
            stage1_metadata=stage1_metadata,
            question=str(stage1_metadata["question"]),
            history_token_count=int(stage1_metadata["history_token_count"]),
            target_dim=int(stage1_metadata["target_dim"]),
            full_action_dim=int(stage1_metadata["full_action_dim"]),
            k_steps=int(stage1_metadata["k"]),
            dt=float(stage1_metadata["dt"]),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        best_metric_name = "val_loss" if val_loader is not None else "train_loss"
        initial_eval_split = "val" if val_loader is not None else "train"
        run_metadata = {
            "git": git_metadata,
            "gpu": gpu_info,
            "gpu_preflight": gpu_preflight,
            "datasets": {
                "train": train_dataset_fingerprint,
                "val": val_dataset_fingerprint,
            },
            "processor": processor_settings,
            "resume": {
                "requested": bool(args.resume_from_checkpoint),
                "resume_from_checkpoint": str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None,
                "resumed": resume_checkpoint is not None,
                "checkpoint_epoch": int(resume_checkpoint["epoch"]) if resume_checkpoint is not None else 0,
                "checkpoint_global_step": int(resume_checkpoint["global_step"])
                if resume_checkpoint is not None
                else 0,
            },
            "history_quantizer": history_quantizer.metadata(),
        }
        write_run_config(save_dir, args, run_metadata)

        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            initial_eval, metrics_history, global_step, start_epoch = load_resume_state(resume_checkpoint)
            best_metric, best_epoch = best_metric_from_history(metrics_history, best_metric_name)
            last_epoch = int(metrics_history[-1]["epoch"]) if metrics_history else 0
            epochs_without_improvement = max(0, last_epoch - best_epoch)
        else:
            initial_eval_loader = val_loader if val_loader is not None else train_loader
            initial_eval = evaluate(runtime, initial_eval_loader, device=device)
            release_cuda_memory()
            metrics_history = []
            global_step = 0
            start_epoch = 1
            best_metric = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0

        if start_epoch > args.max_epochs:
            raise RuntimeError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, which is >= configured max_epochs={args.max_epochs}."
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        print(
            json.dumps(
                {
                    "event": "stage1_setup",
                    "config_json": args.config_json or None,
                    "run_config_path": str(save_dir / "run_config.json"),
                    "run_metadata": run_metadata,
                    "train_size": train_size,
                    "val_size": val_size,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "max_epochs": args.max_epochs,
                    "start_epoch": start_epoch,
                    "dtype": args.dtype,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "image_min_pixels": args.image_min_pixels or None,
                    "image_max_pixels": args.image_max_pixels or None,
                    "early_stopping_patience": args.early_stopping_patience,
                    "early_stopping_min_delta": args.early_stopping_min_delta,
                    "total_vocab_size": len(processor.tokenizer),
                    "added_action_tokens": added_action_tokens,
                    "added_history_tokens": added_history_tokens,
                    "model_load_elapsed_s": round(load_elapsed, 3),
                    "stage1_metadata": stage1_metadata,
                    "task_spec": task_spec.name,
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
        setup_wandb_payload = {
            "setup/gpu_query_ok": gpu_preflight["query_ok"],
            "setup/gpu_warning_count": len(gpu_preflight["warning_reasons"]),
            "setup/train_size": train_size,
            "setup/val_size": val_size,
            "setup/batch_size": args.batch_size,
            "setup/grad_accum_steps": args.grad_accum_steps,
            "setup/start_epoch": start_epoch,
            "setup/total_vocab_size": len(processor.tokenizer),
            "setup/added_action_tokens": added_action_tokens,
            "setup/added_history_tokens": added_history_tokens,
            "setup/model_load_elapsed_s": round(load_elapsed, 3),
            "setup/k": stage1_metadata["k"],
            "setup/dt": stage1_metadata["dt"],
            "setup/num_bins": stage1_metadata["num_bins"],
            "setup/target_dim": stage1_metadata["target_dim"],
            "setup/full_action_dim": stage1_metadata["full_action_dim"],
            "setup/history_steps": stage1_metadata["history_steps"],
            "setup/history_token_count": stage1_metadata["history_token_count"],
            "setup/image_min_pixels": args.image_min_pixels,
            "setup/image_max_pixels": args.image_max_pixels,
            "setup/resumed": resume_checkpoint is not None,
            f"baseline/{initial_eval_split}_loss": initial_eval["loss"],
            f"baseline/{initial_eval_split}_token_accuracy": initial_eval["token_accuracy"],
        }
        if gpu_preflight["query_ok"]:
            setup_wandb_payload.update(
                {
                    "setup/gpu_free_gib": round(gpu_preflight["free_mib"] / 1024, 3),
                    "setup/gpu_non_self_compute_gib": round(
                        gpu_preflight["non_self_compute_used_mib"] / 1024, 3
                    ),
                    "setup/gpu_other_used_gib": round(gpu_preflight["other_used_mib"] / 1024, 3),
                }
            )
        maybe_wandb_log(wandb_run, setup_wandb_payload, step=0)

        stop_reason = "max_epochs"
        completed_epochs = start_epoch - 1

        for epoch in range(start_epoch, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss_total = 0.0
            train_batches = 0
            train_correct = 0
            train_tokens = 0
            optimizer_steps_this_epoch = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                full_inputs, labels = prepare_stage1a_training_batch(runtime, batch, device=device)
                with torch.autocast("cuda", dtype=model_dtype):
                    outputs = model(**model_forward_inputs(full_inputs), labels=labels)
                    loss = outputs.loss

                (loss / args.grad_accum_steps).backward()
                correct, token_total = compute_token_accuracy(outputs.logits.detach(), labels)
                train_loss_total += float(loss.detach().cpu())
                train_correct += correct
                train_tokens += token_total
                train_batches += 1

                should_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader)
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    if args.log_every > 0 and global_step % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "loss": round(float(loss.detach().cpu()), 6),
                                    "lr": scheduler.get_last_lr()[0],
                                },
                                ensure_ascii=False,
                            )
                        )
                        maybe_wandb_log(
                            wandb_run,
                            {
                                "train/step_loss": float(loss.detach().cpu()),
                                "train/lr": scheduler.get_last_lr()[0],
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )

            train_loss = train_loss_total / max(train_batches, 1)
            train_accuracy = train_correct / max(train_tokens, 1)

            if val_loader is not None:
                val_metrics = evaluate(runtime, val_loader, device=device)
                release_cuda_memory()
                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["token_accuracy"]
                metric_to_track = val_loss
            else:
                val_loss = None
                val_accuracy = None
                metric_to_track = train_loss

            improved = metric_improved(metric_to_track, best_metric, args.early_stopping_min_delta)
            if improved:
                best_metric = metric_to_track
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_token_accuracy": val_accuracy,
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
            metrics_history.append(epoch_metrics)
            print(json.dumps({"event": "epoch_end", **epoch_metrics}, ensure_ascii=False))

            epoch_log = {
                "train/epoch_loss": train_loss,
                "train/token_accuracy": train_accuracy,
                "train/epoch_elapsed_s": epoch_metrics["epoch_elapsed_s"],
                "train/optimizer_steps_this_epoch": optimizer_steps_this_epoch,
                "train/lr_epoch_end": scheduler.get_last_lr()[0],
                "summary/best_metric_so_far": best_metric,
                "summary/epochs_without_improvement": epochs_without_improvement,
            }
            if val_loss is not None:
                epoch_log["val/loss"] = val_loss
            if val_accuracy is not None:
                epoch_log["val/token_accuracy"] = val_accuracy
            maybe_wandb_log(wandb_run, epoch_log, step=global_step)

            if improved:
                torch.save(
                    model_only_checkpoint_payload(
                        runtime,
                        args=args,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                        run_metadata=run_metadata,
                    ),
                    save_dir / "best.pt",
                )

            if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
                torch.save(
                    full_checkpoint_payload(
                        runtime,
                        optimizer,
                        scheduler,
                        args=args,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                        run_metadata=run_metadata,
                    ),
                    save_dir / f"epoch_{epoch:03d}.pt",
                )

            torch.save(
                full_checkpoint_payload(
                    runtime,
                    optimizer,
                    scheduler,
                    args=args,
                    initial_eval=initial_eval,
                    epoch=epoch,
                    global_step=global_step,
                    metrics_history=metrics_history,
                    run_metadata=run_metadata,
                ),
                save_dir / "last.pt",
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
                            "epochs_without_improvement": epochs_without_improvement,
                        },
                        ensure_ascii=False,
                    )
                )
                maybe_wandb_log(
                    wandb_run,
                    {
                        "summary/early_stopped": 1,
                        "summary/early_stop_epoch": epoch,
                        "summary/best_epoch": best_epoch,
                    },
                    step=global_step,
                )
                break

        torch.save(
            model_only_checkpoint_payload(
                runtime,
                args=args,
                initial_eval=initial_eval,
                epoch=completed_epochs,
                global_step=global_step,
                metrics_history=metrics_history,
                run_metadata=run_metadata,
            ),
            save_dir / "final.pt",
        )

        final_summary = {
            "config_json": args.config_json or None,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "run_config_path": str(save_dir / "run_config.json"),
            "train_jsonl": list(args.train_jsonl),
            "val_jsonl": list(args.val_jsonl) if args.val_jsonl is not None else None,
            "train_size": train_size,
            "val_size": val_size,
            "model_path": args.model_path,
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_epochs": args.max_epochs,
            "completed_epochs": completed_epochs,
            "gradient_checkpointing": args.gradient_checkpointing,
            "image_min_pixels": args.image_min_pixels or None,
            "image_max_pixels": args.image_max_pixels or None,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "stopped_early": stop_reason == "early_stopping",
            "stop_reason": stop_reason,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "resume_from_checkpoint": str(resume_checkpoint_path)
            if resume_checkpoint_path is not None
            else None,
            "resumed": resume_checkpoint is not None,
            "added_action_tokens": added_action_tokens,
            "added_history_tokens": added_history_tokens,
            "total_vocab_size": len(processor.tokenizer),
            "model_load_elapsed_s": round(load_elapsed, 3),
            "total_wall_time_s": round(time.perf_counter() - wall_start, 3),
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
            "run_metadata": run_metadata,
            "stage1_metadata": stage1_metadata,
            "initial_eval": {
                "split": initial_eval_split,
                "loss": initial_eval["loss"],
                "token_accuracy": initial_eval["token_accuracy"],
            },
            "history": metrics_history,
        }
        with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        maybe_wandb_log(
            wandb_run,
            {
                "summary/best_metric": best_metric,
                "summary/best_epoch": best_epoch,
                "summary/peak_allocated_gib": final_summary["peak_allocated_gib"],
                "summary/peak_reserved_gib": final_summary["peak_reserved_gib"],
                "summary/total_wall_time_s": final_summary["total_wall_time_s"],
                "summary/stopped_early": 1 if final_summary["stopped_early"] else 0,
            },
            step=global_step,
        )
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_metric_name": best_metric_name,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "peak_allocated_gib": final_summary["peak_allocated_gib"],
                    "peak_reserved_gib": final_summary["peak_reserved_gib"],
                    "total_wall_time_s": final_summary["total_wall_time_s"],
                    "stopped_early": final_summary["stopped_early"],
                    "stop_reason": stop_reason,
                }
            )

        print(json.dumps({"event": "training_complete", **final_summary}, ensure_ascii=False))
    finally:
        if wandb_run is not None:
            maybe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
