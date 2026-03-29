"""Canonical Stage 1B expert-CFM trainer."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ....stage1.data.dataset import Stage1JsonlDataset
from ....stage1.vlm_ce.train import (
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    metric_improved,
    release_cuda_memory,
    set_seed,
    stage1_collate,
    write_run_config,
)
from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import (
    load_json_payload,
    normalize_arg_config,
    normalize_optional_string_list,
    normalize_required_string_list,
    resolve_path_base,
)
from ....utils.preflight import enforce_training_prerequisites
from ....utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)
from ....diffusion.action_expert import FlowMatchingDiffusion
from ....models.action_expert import Stage1ActionExpert
from ..core.common import (
    build_stage1b_metadata,
    compute_action_stats,
    extract_prompt_cache,
    freeze_module,
    infer_prompt_text,
    load_stage1_condition_components,
    prepare_condition_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "stage1_checkpoint",
    "train_jsonl",
    "val_jsonl",
    "save_dir",
}
MULTI_VALUE_CONFIG_KEYS = {"train_jsonl", "val_jsonl"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the canonical Stage 1B expert CFM path.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="minipamayo-qwen-3-5/checkpoints/stage1/expert_cfm/canonical",
    )
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
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--decoder-hidden-size", type=int, default=0)
    parser.add_argument("--decoder-num-layers", type=int, default=6)
    parser.add_argument("--decoder-num-attention-heads", type=int, default=0)
    parser.add_argument("--decoder-intermediate-size", type=int, default=2048)
    parser.add_argument("--decoder-attention-dropout", type=float, default=0.0)
    parser.add_argument("--expert-non-causal-attention", type=bool, default=True)
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
        list_keys=MULTI_VALUE_CONFIG_KEYS,
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
            "Stage 1B training accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    args.train_jsonl = normalize_required_string_list(args.train_jsonl, key_name="train_jsonl")
    args.val_jsonl = normalize_optional_string_list(args.val_jsonl, key_name="val_jsonl")
    if not args.stage1_checkpoint:
        raise RuntimeError("`stage1_checkpoint` must be defined in the config JSON.")
    if args.early_stopping_patience < 0:
        raise RuntimeError("`early_stopping_patience` must be >= 0.")
    if args.early_stopping_min_delta < 0:
        raise RuntimeError("`early_stopping_min_delta` must be >= 0.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    if args.decoder_hidden_size < 0:
        raise RuntimeError("`decoder_hidden_size` must be >= 0.")
    if args.decoder_num_layers <= 0:
        raise RuntimeError("`decoder_num_layers` must be > 0.")
    if args.decoder_num_attention_heads < 0:
        raise RuntimeError("`decoder_num_attention_heads` must be >= 0.")
    if args.decoder_intermediate_size <= 0:
        raise RuntimeError("`decoder_intermediate_size` must be > 0.")
    return args


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = Stage1JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    if args.val_jsonl:
        val_dataset = Stage1JsonlDataset(args.val_jsonl)
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
        "collate_fn": stage1_collate,
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


def resolve_expert_text_config(model, args: argparse.Namespace) -> dict:
    if not hasattr(model.config, "text_config"):
        raise RuntimeError("Frozen Stage 1A model is missing `text_config` for Stage 1B.")
    base_text_config = copy.deepcopy(model.config.text_config.to_dict())
    base_hidden_size = int(base_text_config["hidden_size"])
    base_num_heads = int(base_text_config["num_attention_heads"])
    base_num_layers = int(base_text_config["num_hidden_layers"])

    requested_hidden_size = int(args.decoder_hidden_size)
    requested_num_heads = int(args.decoder_num_attention_heads)
    requested_num_layers = int(args.decoder_num_layers)
    requested_intermediate_size = int(args.decoder_intermediate_size)

    if requested_hidden_size not in {0, base_hidden_size}:
        raise RuntimeError(
            "Canonical Stage 1B expert must preserve the Stage 1A text hidden size "
            "to remain prompt-cache compatible.\n"
            f"stage1_hidden_size={base_hidden_size}\n"
            f"requested_decoder_hidden_size={requested_hidden_size}"
        )
    if requested_num_heads not in {0, base_num_heads}:
        raise RuntimeError(
            "Canonical Stage 1B expert must preserve the Stage 1A attention head count "
            "to remain prompt-cache compatible.\n"
            f"stage1_num_attention_heads={base_num_heads}\n"
            f"requested_decoder_num_attention_heads={requested_num_heads}"
        )
    if requested_num_layers > base_num_layers:
        raise RuntimeError(
            "Canonical Stage 1B expert cannot use more layers than the frozen Stage 1A prompt cache.\n"
            f"stage1_num_hidden_layers={base_num_layers}\n"
            f"requested_decoder_num_layers={requested_num_layers}"
        )

    base_text_config["hidden_size"] = base_hidden_size
    base_text_config["num_attention_heads"] = base_num_heads
    base_text_config["num_hidden_layers"] = requested_num_layers
    base_text_config["intermediate_size"] = requested_intermediate_size
    if "layer_types" in base_text_config and isinstance(base_text_config["layer_types"], list):
        base_text_config["layer_types"] = list(
            base_text_config["layer_types"][:requested_num_layers]
        )
    base_text_config["use_cache"] = True
    return base_text_config


@torch.no_grad()
def evaluate(
    expert: Stage1ActionExpert,
    diffusion: FlowMatchingDiffusion,
    model,
    dataloader: DataLoader,
    processor,
    history_registry,
    history_quantizer,
    prompt_text: str,
    device: torch.device,
) -> dict:
    expert.eval()
    total_loss = 0.0
    total_batches = 0
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    for batch in dataloader:
        prompt_inputs = prepare_condition_inputs(
            model=model,
            batch=batch,
            processor=processor,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            prompt_text=prompt_text,
            device=device,
        )
        prompt_cache, prompt_attention_mask = extract_prompt_cache(model, prompt_inputs)
        gt_action = batch["action"].to(device=device, dtype=torch.float32)
        with amp_context:
            loss = diffusion.loss(
                expert=expert,
                gt_action=gt_action,
                prompt_cache=prompt_cache,
                prompt_attention_mask=prompt_attention_mask,
            )
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return {"cfm_loss": total_loss / max(1, total_batches)}


def save_checkpoint(
    save_path: Path,
    *,
    expert: Stage1ActionExpert,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    action_stats: dict,
    stage1b_metadata: dict,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "expert_state_dict": {
                key: value.detach().cpu() for key, value in expert.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "expert_config": vars(expert.export_config()),
            "action_stats": action_stats,
            "stage1b_metadata": stage1b_metadata,
        },
        save_path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)
    release_cuda_memory()

    wandb_run = None
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.json"
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"
    summary_path = save_dir / "summary.json"

    history: list[dict] = []
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    global_step = 0
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0

    try:
        wandb_run = enforce_training_prerequisites(
            project=args.wandb_project,
            config=vars(args),
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            git_cwd=Path(__file__).resolve().parent,
        )
        gpu_preflight = log_gpu_preflight(device)
        train_loader, val_loader, train_size, val_size = build_dataloaders(args)

        (
            stage1_checkpoint,
            model,
            processor,
            _registry,
            history_registry,
            history_quantizer,
            _quantizer,
            _model_dtype,
        ) = load_stage1_condition_components(args)
        freeze_module(model)

        prompt_text = infer_prompt_text(stage1_checkpoint, processor)
        first_batch = next(iter(train_loader))
        action_stats = compute_action_stats(train_loader.dataset)
        expert_text_config = resolve_expert_text_config(model, args)
        expert = Stage1ActionExpert(
            k=int(first_batch["action"].shape[-1] // 2),
            action_dims=(int(first_batch["action"].shape[-1] // 2), 2),
            expert_text_config=expert_text_config,
            keep_same_dtype=True,
            expert_non_causal_attention=bool(args.expert_non_causal_attention),
            num_fourier_feats=args.num_fourier_feats,
            fourier_max_freq=args.fourier_max_freq,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_num_layers=args.mlp_num_layers,
            accel_mean=action_stats["accel_mean"],
            accel_std=action_stats["accel_std"],
            kappa_mean=action_stats["kappa_mean"],
            kappa_std=action_stats["kappa_std"],
        ).to(device=device)
        diffusion = FlowMatchingDiffusion()
        stage1b_metadata = build_stage1b_metadata(
            train_loader.dataset,
            args,
            vars(expert.export_config()),
            action_stats,
        )

        total_optimizer_steps = max(
            1,
            math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * max(1, args.max_epochs),
        )
        warmup_steps = int(round(total_optimizer_steps * args.warmup_ratio))
        optimizer = torch.optim.AdamW(
            expert.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            remaining = max(1, total_optimizer_steps - warmup_steps)
            progress = float(current_step - warmup_steps) / float(remaining)
            return max(0.0, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        run_metadata = {
            "git": collect_git_metadata(),
            "gpu": collect_gpu_info(device),
            "processor": collect_processor_settings(
                processor,
                requested_min_pixels=args.image_min_pixels,
                requested_max_pixels=args.image_max_pixels,
            ),
            "train_dataset": collect_dataset_view_fingerprint(train_loader.dataset),
            "val_dataset": collect_dataset_view_fingerprint(val_loader.dataset)
            if val_loader
            else None,
            "gpu_preflight": gpu_preflight,
        }
        write_run_config(save_dir, args, run_metadata)

        maybe_wandb_log(
            wandb_run,
            {
                "setup/train_size": train_size,
                "setup/val_size": val_size,
                "setup/expert_layers": int(expert.expert_num_layers),
                "setup/expert_hidden_size": int(expert.expert.config.hidden_size),
                "setup/history_steps": int(stage1_checkpoint["stage1_metadata"]["history_steps"]),
            },
            step=0,
        )
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.time()
            expert.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_total = 0.0
            batch_count = 0
            optimizer_steps_this_epoch = 0

            for batch_idx, batch in enumerate(train_loader, start=1):
                prompt_inputs = prepare_condition_inputs(
                    model=model,
                    batch=batch,
                    processor=processor,
                    history_registry=history_registry,
                    history_quantizer=history_quantizer,
                    prompt_text=prompt_text,
                    device=device,
                )
                with torch.no_grad():
                    prompt_cache, prompt_attention_mask = extract_prompt_cache(model, prompt_inputs)
                gt_action = batch["action"].to(device=device, dtype=torch.float32)
                with amp_context:
                    loss = diffusion.loss(
                        expert=expert,
                        gt_action=gt_action,
                        prompt_cache=prompt_cache,
                        prompt_attention_mask=prompt_attention_mask,
                    )
                (loss / float(args.grad_accum_steps)).backward()
                train_loss_total += float(loss.detach().cpu())
                batch_count += 1

                if batch_idx % args.grad_accum_steps == 0 or batch_idx == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(expert.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    optimizer_steps_this_epoch += 1

                    if device.type == "cuda":
                        peak_allocated_bytes = max(
                            peak_allocated_bytes, torch.cuda.max_memory_allocated(device)
                        )
                        peak_reserved_bytes = max(
                            peak_reserved_bytes, torch.cuda.max_memory_reserved(device)
                        )

                    if args.log_every > 0 and global_step % args.log_every == 0:
                        maybe_wandb_log(
                            wandb_run,
                            {
                                "train/step_cfm_loss": float(loss.detach().cpu()),
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )

            epoch_log = {
                "epoch": epoch,
                "train_cfm_loss": train_loss_total / max(1, batch_count),
                "lr": scheduler.get_last_lr()[0],
                "epoch_elapsed_s": time.time() - epoch_start,
                "optimizer_steps_this_epoch": optimizer_steps_this_epoch,
            }

            if val_loader is not None:
                val_metrics = evaluate(
                    expert=expert,
                    diffusion=diffusion,
                    model=model,
                    dataloader=val_loader,
                    processor=processor,
                    history_registry=history_registry,
                    history_quantizer=history_quantizer,
                    prompt_text=prompt_text,
                    device=device,
                )
                epoch_log["val_cfm_loss"] = val_metrics["cfm_loss"]
                improved = metric_improved(
                    val_metrics["cfm_loss"], best_val_loss, args.early_stopping_min_delta
                )
                if improved:
                    best_val_loss = val_metrics["cfm_loss"]
                    best_epoch = epoch
                    stale_epochs = 0
                    save_checkpoint(
                        best_path,
                        expert=expert,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        action_stats=action_stats,
                        stage1b_metadata=stage1b_metadata,
                    )
                else:
                    stale_epochs += 1
            else:
                save_checkpoint(
                    best_path,
                    expert=expert,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                    action_stats=action_stats,
                    stage1b_metadata=stage1b_metadata,
                )

            save_checkpoint(
                last_path,
                expert=expert,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                action_stats=action_stats,
                stage1b_metadata=stage1b_metadata,
            )

            history.append(epoch_log)
            history_path.write_text(
                json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            maybe_wandb_log(
                wandb_run,
                {
                    f"train/{key}"
                    if key.startswith("train_")
                    else f"val/{key[4:]}"
                    if key.startswith("val_")
                    else key: value
                    for key, value in epoch_log.items()
                    if key != "epoch"
                },
                step=global_step,
            )

            if (
                val_loader is not None
                and args.early_stopping_patience > 0
                and stale_epochs >= args.early_stopping_patience
            ):
                break

        summary = {
            "best_epoch": best_epoch if best_epoch > 0 else len(history),
            "best_val_cfm_loss": best_val_loss if math.isfinite(best_val_loss) else None,
            "history_length": len(history),
            "peak_allocated_gib": format_gib(peak_allocated_bytes),
            "peak_reserved_gib": format_gib(peak_reserved_bytes),
            "stage1_checkpoint": args.stage1_checkpoint,
            "condition_source": "prompt_past_key_values",
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        if wandb_run is not None:
            wandb_run.summary.update(summary)
    finally:
        if wandb_run is not None:
            maybe_wandb_finish(wandb_run)
        release_cuda_memory()
