"""Stage 1 trainer for the Qwen3.5 branch.

This is the long-running trainer:
- train/validation split support
- best/final checkpoint saving
- epoch-based loop for real Stage 1 runs

For quick VRAM and throughput probes, use `minipamayo_qwen35.profile_stage1`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, random_split
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..data.stage1_dataset import Stage1JsonlDataset
from ..tokens.action_quantizer import ActionQuantizer
from ..tokens.token_registry import Stage1TokenRegistry
from ..utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ..utils.preflight import enforce_training_prerequisites

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH_KEYS = {
    "dataset_jsonl",
    "val_jsonl",
    "model_path",
    "save_dir",
}

DEFAULT_QUESTION = (
    "Predict the future ego trajectory as action tokens. "
    "Output only the action tokens in order."
)

# Practical presets measured with Qwen3.5-0.8B on this repo's CARLA-derived data:
# - 12 GB class GPU: `--batch-size 1 --gradient-checkpointing`
# - 24 GB class GPU: `--batch-size 2 --no-gradient-checkpointing`
# Keep the defaults conservative, then opt into the 24 GB setting explicitly.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 Stage 1 with validation and checkpoints.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--dataset-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B",
    )
    parser.add_argument("--save-dir", type=str, default="minipamayo-qwen-3-5/checkpoints/stage1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
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
        raise RuntimeError("Stage 1 training accepts only --config-json. Put all settings in the JSON file.")

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.dataset_jsonl:
        raise RuntimeError("`dataset_jsonl` must be defined in the config JSON.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_text(processor, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_inputs_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def stage1_collate(samples: list[dict]) -> dict:
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "image_path": [sample["image_path"] for sample in samples],
        "action": torch.stack([sample["action"] for sample in samples], dim=0),
        "v0": torch.stack([sample["v0"] for sample in samples], dim=0),
        "gt_waypoints": torch.stack([sample["gt_waypoints"] for sample in samples], dim=0),
        "command": [sample["command"] for sample in samples],
    }


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, int, int]:
    train_dataset = Stage1JsonlDataset(args.dataset_jsonl, max_samples=args.max_samples)
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
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
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

    return train_loader, val_loader, len(train_dataset), len(val_dataset) if val_dataset is not None else 0


def first_record_from_dataset(dataset) -> dict:
    if isinstance(dataset, Subset):
        if len(dataset.indices) == 0:
            raise RuntimeError("Subset is empty.")
        base_dataset = dataset.dataset
        base_index = dataset.indices[0]
        return base_dataset.records[base_index]
    if len(dataset.records) == 0:
        raise RuntimeError("Dataset is empty.")
    return dataset.records[0]


def build_stage1_metadata(
    dataset,
    args: argparse.Namespace,
    registry: Stage1TokenRegistry,
    quantizer: ActionQuantizer,
) -> dict:
    record = first_record_from_dataset(dataset)
    gt_waypoints = record.get("gt_waypoints", [])
    action = record.get("action", [])
    return {
        "dataset_jsonl": args.dataset_jsonl,
        "val_jsonl": args.val_jsonl or None,
        "sample_format": "jsonl+images",
        "k": len(gt_waypoints) if gt_waypoints else len(action) // 2,
        "action_dim": len(action),
        "dt": record.get("dt"),
        "n_bins": quantizer.n_bins,
        "a_range": list(quantizer.a_range),
        "kappa_range": list(quantizer.kappa_range),
        "action_token_scheme": "add_tokens",
        "token_prefix": registry.token_prefix,
        "question": args.question,
    }


def prepare_batch(
    batch: dict,
    processor,
    registry: Stage1TokenRegistry,
    quantizer: ActionQuantizer,
    prompt_text: str,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    try:
        action_token_id_rows = [
            registry.encode_action_token_ids(action.cpu().numpy(), quantizer)
            for action in batch["action"]
        ]

        prompt_inputs = processor(
            text=[prompt_text] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        action_token_ids = torch.tensor(action_token_id_rows, dtype=torch.long)

        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_attention_mask = prompt_inputs["attention_mask"]
        batch_size = prompt_input_ids.shape[0]
        action_len = action_token_ids.shape[1]

        input_ids = torch.cat([prompt_input_ids, action_token_ids], dim=1)
        attention_mask = torch.cat(
            [prompt_attention_mask, torch.ones((batch_size, action_len), dtype=prompt_attention_mask.dtype)],
            dim=1,
        )
        labels = torch.full_like(input_ids, -100)
        labels[:, -action_len:] = action_token_ids

        full_inputs = {key: value for key, value in prompt_inputs.items() if key not in {"input_ids", "attention_mask"}}
        full_inputs["input_ids"] = input_ids
        full_inputs["attention_mask"] = attention_mask
        if "mm_token_type_ids" in full_inputs:
            full_inputs["mm_token_type_ids"] = torch.cat(
                [
                    full_inputs["mm_token_type_ids"],
                    torch.zeros((batch_size, action_len), dtype=full_inputs["mm_token_type_ids"].dtype),
                ],
                dim=1,
            )
        full_inputs = move_inputs_to_device(full_inputs, device)
        labels = labels.to(device)
        return full_inputs, labels
    finally:
        for image in images:
            image.close()


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    correct = ((shifted_logits == shifted_labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    processor,
    registry: Stage1TokenRegistry,
    quantizer: ActionQuantizer,
    prompt_text: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        full_inputs, labels = prepare_batch(batch, processor, registry, quantizer, prompt_text, device)
        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**full_inputs, labels=labels)
        correct, token_total = compute_token_accuracy(outputs.logits, labels)
        total_loss += float(outputs.loss.detach().cpu())
        total_correct += correct
        total_tokens += token_total
        total_batches += 1

    model.train()
    return {
        "loss": total_loss / max(total_batches, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    registry: Stage1TokenRegistry,
    quantizer: ActionQuantizer,
    stage1_metadata: dict,
    initial_eval: dict | None,
    epoch: int,
    global_step: int,
    metrics_history: list[dict],
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "metrics_history": metrics_history,
        "token_registry": {
            "n_bins": registry.n_bins,
            "token_prefix": registry.token_prefix,
            "token_strings": registry.token_strings,
        },
        "quantizer": {
            "n_bins": quantizer.n_bins,
            "a_range": list(quantizer.a_range),
            "kappa_range": list(quantizer.kappa_range),
        },
        "stage1_metadata": stage1_metadata,
        "initial_eval": initial_eval,
    }


def maybe_wandb_log(run, data: dict, step: int | None = None) -> None:
    if run is None:
        return
    run.log(data, step=step)


def maybe_wandb_finish(run) -> None:
    if run is None:
        return
    run.finish()


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
    set_seed(args.seed)
    wall_start = time.perf_counter()

    try:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("This Stage 1 trainer is intended to run on CUDA.")

        train_loader, val_loader, train_size, val_size = build_dataloaders(args)
        if len(train_loader) == 0:
            raise RuntimeError("Train DataLoader is empty.")

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_json": args.config_json,
                    "config_payload": args.config_payload,
                    "resolved_args": vars(args),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        load_start = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=model_dtype,
            trust_remote_code=True,
        )

        quantizer = ActionQuantizer()
        registry = Stage1TokenRegistry(n_bins=quantizer.n_bins)
        added = registry.add_to_tokenizer(processor.tokenizer)
        stage1_metadata = build_stage1_metadata(train_loader.dataset, args, registry, quantizer)
        model.resize_token_embeddings(len(processor.tokenizer))
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
        prompt_text = build_prompt_text(processor, args.question)

        initial_eval_split = "val" if val_loader is not None else "train"
        initial_eval_loader = val_loader if val_loader is not None else train_loader
        initial_eval = evaluate(
            model=model,
            dataloader=initial_eval_loader,
            processor=processor,
            registry=registry,
            quantizer=quantizer,
            prompt_text=prompt_text,
            device=device,
            model_dtype=model_dtype,
        )

        metrics_history: list[dict] = []
        best_metric = float("inf")
        best_metric_name = "val_loss" if val_loader is not None else "train_loss"
        global_step = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        setup_payload = {
            "event": "stage1_setup",
            "config_json": args.config_json or None,
            "run_config_path": str(save_dir / "run_config.json"),
            "train_size": train_size,
            "val_size": val_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_epochs": args.max_epochs,
            "dtype": args.dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "total_vocab_size": len(processor.tokenizer),
            "added_action_tokens": added,
            "model_load_elapsed_s": round(load_elapsed, 3),
            "stage1_metadata": stage1_metadata,
        }
        print(json.dumps(setup_payload, ensure_ascii=False))
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
                "setup/total_vocab_size": len(processor.tokenizer),
                "setup/added_action_tokens": added,
                "setup/model_load_elapsed_s": round(load_elapsed, 3),
                "setup/k": stage1_metadata["k"],
                "setup/dt": stage1_metadata["dt"],
                "setup/n_bins": stage1_metadata["n_bins"],
                f"baseline/{initial_eval_split}_loss": initial_eval["loss"],
                f"baseline/{initial_eval_split}_token_accuracy": initial_eval["token_accuracy"],
            },
            step=0,
        )

        for epoch in range(1, args.max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss_total = 0.0
            train_batches = 0
            train_correct = 0
            train_tokens = 0
            optimizer_steps_this_epoch = 0

            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader, start=1):
                full_inputs, labels = prepare_batch(batch, processor, registry, quantizer, prompt_text, device)

                with torch.autocast("cuda", dtype=model_dtype):
                    outputs = model(**full_inputs, labels=labels)
                    loss = outputs.loss

                (loss / args.grad_accum_steps).backward()
                correct, token_total = compute_token_accuracy(outputs.logits.detach(), labels)
                train_loss_total += float(loss.detach().cpu())
                train_correct += correct
                train_tokens += token_total
                train_batches += 1

                should_step = (
                    batch_idx % args.grad_accum_steps == 0
                    or batch_idx == len(train_loader)
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
                        step_payload = {
                            "event": "train_step",
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": round(float(loss.detach().cpu()), 6),
                            "lr": scheduler.get_last_lr()[0],
                        }
                        print(json.dumps(step_payload, ensure_ascii=False))
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
                val_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    processor=processor,
                    registry=registry,
                    quantizer=quantizer,
                    prompt_text=prompt_text,
                    device=device,
                    model_dtype=model_dtype,
                )
                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["token_accuracy"]
                metric_to_track = val_loss
            else:
                val_loss = None
                val_accuracy = None
                metric_to_track = train_loss

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_token_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_token_accuracy": val_accuracy,
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
            }
            if val_loss is not None:
                epoch_log["val/loss"] = val_loss
            if val_accuracy is not None:
                epoch_log["val/token_accuracy"] = val_accuracy
            maybe_wandb_log(wandb_run, epoch_log, step=global_step)

            if metric_to_track < best_metric:
                best_metric = metric_to_track
                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        registry=registry,
                        quantizer=quantizer,
                        stage1_metadata=stage1_metadata,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                    ),
                    save_dir / "best.pt",
                )

            if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
                torch.save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        registry=registry,
                        quantizer=quantizer,
                        stage1_metadata=stage1_metadata,
                        initial_eval=initial_eval,
                        epoch=epoch,
                        global_step=global_step,
                        metrics_history=metrics_history,
                    ),
                    save_dir / f"epoch_{epoch:03d}.pt",
                )

            with (save_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                registry=registry,
                quantizer=quantizer,
                stage1_metadata=stage1_metadata,
                initial_eval=initial_eval,
                epoch=args.max_epochs,
                global_step=global_step,
                metrics_history=metrics_history,
            ),
            save_dir / "final.pt",
        )

        final_summary = {
            "config_json": args.config_json or None,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "run_config_path": str(save_dir / "run_config.json"),
            "dataset_jsonl": args.dataset_jsonl,
            "val_jsonl": args.val_jsonl or None,
            "train_size": train_size,
            "val_size": val_size,
            "model_path": args.model_path,
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "max_epochs": args.max_epochs,
            "gradient_checkpointing": args.gradient_checkpointing,
            "best_metric_name": best_metric_name,
            "best_metric": best_metric,
            "added_action_tokens": added,
            "total_vocab_size": len(processor.tokenizer),
            "model_load_elapsed_s": round(load_elapsed, 3),
            "total_wall_time_s": round(time.perf_counter() - wall_start, 3),
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
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
                "summary/peak_allocated_gib": final_summary["peak_allocated_gib"],
                "summary/peak_reserved_gib": final_summary["peak_reserved_gib"],
                "summary/total_wall_time_s": final_summary["total_wall_time_s"],
            },
            step=global_step,
        )
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_metric_name": best_metric_name,
                    "best_metric": best_metric,
                    "peak_allocated_gib": final_summary["peak_allocated_gib"],
                    "peak_reserved_gib": final_summary["peak_reserved_gib"],
                    "total_wall_time_s": final_summary["total_wall_time_s"],
                }
            )

        print(json.dumps({"event": "training_complete", **final_summary}, ensure_ascii=False))
    finally:
        maybe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
