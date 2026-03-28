"""Stage 2 trainer for the Qwen3.5 branch.

This Qwen3.5-specific port avoids the old Qwen2.5 KV-cache expert and instead
conditions the trajectory decoder on the final hidden states of the frozen
Stage 1 VLM.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split

from ..eval.stage1 import load_components
from ..models.trajectory_decoder import TrajectoryDecoder, cfm_loss
from ..sequence.stage3_builder import build_stage2_prompt_text
from ..train.stage1 import (
    first_record_from_dataset,
    format_gib,
    log_gpu_preflight,
    metric_improved,
    maybe_wandb_finish,
    maybe_wandb_log,
    move_inputs_to_device,
    release_cuda_memory,
    set_seed,
    write_run_config,
)
from ..utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ..utils.preflight import enforce_training_prerequisites
from ..utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)
from ..utils.stage34_dataset import Stage34JsonlDataset, stage34_collate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH_KEYS = {
    "stage1_checkpoint",
    "train_jsonl",
    "val_jsonl",
    "save_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Qwen3.5 hidden-state-conditioned Stage 2 flow decoder."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="minipamayo-qwen-3-5/checkpoints/stage2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--decoder-hidden-size", type=int, default=512)
    parser.add_argument("--decoder-num-layers", type=int, default=6)
    parser.add_argument("--decoder-num-attention-heads", type=int, default=8)
    parser.add_argument("--decoder-intermediate-size", type=int, default=2048)
    parser.add_argument("--decoder-attention-dropout", type=float, default=0.0)
    parser.add_argument("--num-fourier-feats", type=int, default=20)
    parser.add_argument("--fourier-max-freq", type=float, default=100.0)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp-num-layers", type=int, default=4)
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
        raise RuntimeError("Stage 2 training accepts only --config-json. Put all settings in the JSON file.")

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.stage1_checkpoint:
        raise RuntimeError("`stage1_checkpoint` must be defined in the config JSON.")
    if not args.train_jsonl:
        raise RuntimeError("`train_jsonl` must be defined in the config JSON.")
    if args.early_stopping_patience < 0:
        raise RuntimeError("`early_stopping_patience` must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise RuntimeError("`early_stopping_min_delta` must be >= 0.")
    if args.image_min_pixels < 0 or args.image_max_pixels < 0:
        raise RuntimeError("`image_min_pixels` and `image_max_pixels` must be >= 0.")
    if args.image_min_pixels > 0 and args.image_max_pixels > 0 and args.image_min_pixels > args.image_max_pixels:
        raise RuntimeError("`image_min_pixels` must be <= `image_max_pixels` when both are set.")
    if args.decoder_hidden_size % args.decoder_num_attention_heads != 0:
        raise RuntimeError("`decoder_hidden_size` must be divisible by `decoder_num_attention_heads`.")
    return args


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = Stage34JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if args.val_jsonl:
        val_dataset = Stage34JsonlDataset(args.val_jsonl)
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty.")
    elif len(train_dataset) >= 2 and args.val_fraction > 0:
        val_size = max(1, int(round(len(train_dataset) * args.val_fraction)))
        val_size = min(val_size, len(train_dataset) - 1)
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
    else:
        val_dataset = None

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": stage34_collate,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, len(train_dataset), len(val_dataset) if val_dataset is not None else 0


def build_stage2_metadata(dataset, args: argparse.Namespace, condition_dim: int) -> dict:
    record = first_record_from_dataset(dataset)
    required_keys = ["gt_waypoints", "action", "dt", "command", "planner_state"]
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise RuntimeError(
            "Training record is missing canonical Stage 2 fields:\n" + "\n".join(missing_keys)
        )
    gt_waypoints = record["gt_waypoints"]
    action = record["action"]
    return {
        "stage1_checkpoint": args.stage1_checkpoint,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl or None,
        "sample_format": "jsonl+images",
        "condition_source": "final_hidden_states",
        "reasoning_source": "synthetic_command_planner_state",
        "k": len(gt_waypoints) if gt_waypoints else len(action) // 2,
        "action_dim": len(action),
        "dt": float(record["dt"]),
        "condition_dim": int(condition_dim),
    }


def _compute_action_stats(dataset) -> dict[str, float]:
    all_accel = []
    all_kappa = []
    for index in range(len(dataset)):
        sample = dataset[index]
        action = sample["action"].cpu().numpy()
        all_accel.append(action[0::2])
        all_kappa.append(action[1::2])
    accel = np.concatenate(all_accel)
    kappa = np.concatenate(all_kappa)
    accel_std = float(np.std(accel))
    kappa_std = float(np.std(kappa))
    if accel_std <= 0.0 or kappa_std <= 0.0:
        raise RuntimeError("Stage 2 action normalization requires non-zero accel and kappa std.")
    return {
        "accel_mean": float(np.mean(accel)),
        "accel_std": accel_std,
        "kappa_mean": float(np.mean(kappa)),
        "kappa_std": kappa_std,
    }


def _build_continuation_rows(tokenizer, texts: list[str]) -> list[list[int]]:
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer is missing `eos_token_id`, which Stage 2 requires.")
    rows: list[list[int]] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if not isinstance(input_ids, list):
            raise RuntimeError("Tokenizer returned a non-list `input_ids` payload for Stage 2.")
        rows.append(list(input_ids) + [int(tokenizer.eos_token_id)])
    return rows


def _append_continuations(prompt_inputs: dict, tokenizer, rows: list[list[int]], device: torch.device) -> dict:
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer is missing `pad_token_id`, which Stage 2 requires.")
    batch_size = len(rows)
    max_len = max(len(row) for row in rows)
    continuation_ids = torch.full(
        (batch_size, max_len),
        fill_value=int(tokenizer.pad_token_id),
        dtype=torch.long,
    )
    continuation_mask = torch.zeros((batch_size, max_len), dtype=prompt_inputs["attention_mask"].dtype)
    for row_idx, row in enumerate(rows):
        row_tensor = torch.tensor(row, dtype=torch.long)
        continuation_ids[row_idx, : len(row)] = row_tensor
        continuation_mask[row_idx, : len(row)] = 1

    input_ids = torch.cat([prompt_inputs["input_ids"], continuation_ids], dim=1)
    attention_mask = torch.cat([prompt_inputs["attention_mask"], continuation_mask], dim=1)

    full_inputs = {
        key: value
        for key, value in prompt_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    full_inputs["input_ids"] = input_ids
    full_inputs["attention_mask"] = attention_mask
    if "mm_token_type_ids" in full_inputs:
        full_inputs["mm_token_type_ids"] = torch.cat(
            [
                full_inputs["mm_token_type_ids"],
                torch.zeros(
                    (batch_size, max_len),
                    dtype=full_inputs["mm_token_type_ids"].dtype,
                ),
            ],
            dim=1,
        )
    return move_inputs_to_device(full_inputs, device)


def prepare_condition_inputs(
    batch: dict,
    processor,
    device: torch.device,
) -> dict:
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    try:
        prompt_texts = [
            build_stage2_prompt_text(processor, float(v0.item()))
            for v0 in batch["v0"]
        ]
        prompt_inputs = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        continuation_rows = _build_continuation_rows(processor.tokenizer, batch["reasoning_text"])
        return _append_continuations(prompt_inputs, processor.tokenizer, continuation_rows, device)
    finally:
        for image in images:
            image.close()


@torch.no_grad()
def extract_condition_hidden_states(
    model,
    full_inputs: dict,
    model_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast("cuda", dtype=model_dtype):
        outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
    if outputs.hidden_states is None:
        raise RuntimeError("Stage 2 requires `output_hidden_states=True`, but the model returned none.")
    return outputs.hidden_states[-1].detach(), full_inputs["attention_mask"].detach()


@torch.no_grad()
def evaluate(
    model,
    decoder: TrajectoryDecoder,
    dataloader: DataLoader,
    processor,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict:
    decoder.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        full_inputs = prepare_condition_inputs(batch, processor, device)
        condition_hidden_states, condition_mask = extract_condition_hidden_states(
            model=model,
            full_inputs=full_inputs,
            model_dtype=model_dtype,
        )
        with torch.autocast("cuda", dtype=model_dtype):
            loss = cfm_loss(
                decoder=decoder,
                gt_action=batch["action"].to(device),
                condition_hidden_states=condition_hidden_states,
                condition_mask=condition_mask,
            )
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    decoder.train()
    return {"loss": total_loss / max(total_batches, 1)}


def checkpoint_payload(
    decoder: TrajectoryDecoder,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    stage2_metadata: dict,
    action_stats: dict,
    initial_eval: dict,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
    run_metadata: dict,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "decoder_state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "stage2_metadata": stage2_metadata,
        "decoder_config": asdict(decoder.export_config()),
        "action_stats": action_stats,
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
            args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
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
        train_dataset_fingerprint = collect_dataset_view_fingerprint(train_loader.dataset)
        val_dataset_fingerprint = (
            collect_dataset_view_fingerprint(val_loader.dataset) if val_loader is not None else None
        )

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        stage1_args = SimpleNamespace(
            checkpoint=args.stage1_checkpoint,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        checkpoint, model, processor, _registry, _quantizer, model_dtype = load_components(stage1_args)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.config.use_cache = False
        model.to(device)
        model.eval()

        processor_settings = collect_processor_settings(
            processor,
            requested_min_pixels=args.image_min_pixels or None,
            requested_max_pixels=args.image_max_pixels or None,
        )
        if not hasattr(model.config, "text_config"):
            raise RuntimeError("Stage 2 expects the Qwen3.5 model config to expose `text_config`.")
        text_hidden_size = int(model.config.text_config.hidden_size)
        stage2_metadata = build_stage2_metadata(train_loader.dataset, args, text_hidden_size)
        stage2_metadata["processor_settings"] = processor_settings

        action_stats = _compute_action_stats(train_loader.dataset)
        decoder = TrajectoryDecoder(
            k=int(stage2_metadata["k"]),
            condition_dim=text_hidden_size,
            hidden_size=args.decoder_hidden_size,
            num_layers=args.decoder_num_layers,
            num_attention_heads=args.decoder_num_attention_heads,
            intermediate_size=args.decoder_intermediate_size,
            attention_dropout=args.decoder_attention_dropout,
            num_fourier_feats=args.num_fourier_feats,
            fourier_max_freq=args.fourier_max_freq,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_num_layers=args.mlp_num_layers,
            accel_mean=action_stats["accel_mean"],
            accel_std=action_stats["accel_std"],
            kappa_mean=action_stats["kappa_mean"],
            kappa_std=action_stats["kappa_std"],
        )
        decoder.to(device)
        decoder.train()

        optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

        initial_eval_loader = val_loader if val_loader is not None else train_loader
        initial_eval_split = "val" if val_loader is not None else "train"
        initial_eval = evaluate(
            model=model,
            decoder=decoder,
            dataloader=initial_eval_loader,
            processor=processor,
            device=device,
            model_dtype=model_dtype,
        )
        release_cuda_memory()
        metrics_history: list[dict] = []
        global_step = 0
        best_metric = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        best_metric_name = "val_cfm_loss" if val_loader is not None else "train_cfm_loss"

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        print(
            json.dumps(
                {
                    "event": "stage2_setup",
                    "config_json": args.config_json,
                    "run_config_path": str(save_dir / "run_config.json"),
                    "stage2_metadata": stage2_metadata,
                    "action_stats": action_stats,
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
                    "cfm_loss": initial_eval["loss"],
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
                f"baseline/{initial_eval_split}_cfm_loss": initial_eval["loss"],
            },
            step=0,
        )

        stop_reason = "max_epochs"
        completed_epochs = 0

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss_total = 0.0
            train_batches = 0
            optimizer_steps_this_epoch = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                full_inputs = prepare_condition_inputs(batch, processor, device)
                condition_hidden_states, condition_mask = extract_condition_hidden_states(
                    model=model,
                    full_inputs=full_inputs,
                    model_dtype=model_dtype,
                )
                with torch.autocast("cuda", dtype=model_dtype):
                    loss = cfm_loss(
                        decoder=decoder,
                        gt_action=batch["action"].to(device),
                        condition_hidden_states=condition_hidden_states,
                        condition_mask=condition_mask,
                    )

                (loss / args.grad_accum_steps).backward()
                train_loss_total += float(loss.detach().cpu())
                train_batches += 1

                should_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader)
                if should_step:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
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
                            "cfm_loss": round(float(loss.detach().cpu()), 6),
                            "lr": scheduler.get_last_lr()[0],
                        }
                        print(json.dumps(payload, ensure_ascii=False))
                        maybe_wandb_log(
                            wandb_run,
                            {
                                "train/step_cfm_loss": float(loss.detach().cpu()),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )

            train_loss = train_loss_total / max(train_batches, 1)
            if val_loader is not None:
                val_metrics = evaluate(
                    model=model,
                    decoder=decoder,
                    dataloader=val_loader,
                    processor=processor,
                    device=device,
                    model_dtype=model_dtype,
                )
                release_cuda_memory()
                val_loss = val_metrics["loss"]
                metric_to_track = val_loss
            else:
                val_loss = None
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
                "train_cfm_loss": train_loss,
                "val_cfm_loss": val_loss,
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
            wandb_payload = {
                "train/epoch_cfm_loss": train_loss,
                "train/epoch_elapsed_s": epoch_metrics["epoch_elapsed_s"],
                "train/optimizer_steps_this_epoch": optimizer_steps_this_epoch,
                "summary/best_metric_so_far": best_metric,
                "summary/epochs_without_improvement": epochs_without_improvement,
            }
            if val_loss is not None:
                wandb_payload["val/cfm_loss"] = val_loss
            maybe_wandb_log(wandb_run, wandb_payload, step=global_step)

            if improved:
                torch.save(
                    checkpoint_payload(
                        decoder=decoder,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        stage2_metadata=stage2_metadata,
                        action_stats=action_stats,
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
                    decoder=decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    stage2_metadata=stage2_metadata,
                    action_stats=action_stats,
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

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
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
                decoder=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                stage2_metadata=stage2_metadata,
                action_stats=action_stats,
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
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
            "total_wall_time_s": round(time.perf_counter() - wall_start, 3),
            "run_metadata": run_metadata,
            "stage2_metadata": stage2_metadata,
            "action_stats": action_stats,
            "initial_eval": {
                "split": initial_eval_split,
                "cfm_loss": initial_eval["loss"],
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
