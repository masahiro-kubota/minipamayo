"""Stage 1 trainer for the Qwen3.5 branch.

This is the long-running trainer:
- train/val split support
- best/final checkpoint saving
- epoch-based loop for real Stage 1 runs

For quick VRAM and throughput probes, use `minipamayo_qwen35.train.profile_stage1`.
"""

from __future__ import annotations

import argparse
import gc
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
from ..utils.preflight import collect_gpu_preflight_snapshot, enforce_training_prerequisites
from ..utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH_KEYS = {
    "train_jsonl",
    "val_jsonl",
    "model_path",
    "resume_from_checkpoint",
    "save_dir",
}

DEFAULT_QUESTION = (
    "Predict the future ego trajectory as action tokens. "
    "Output only the action tokens in order."
)

# Practical Stage 1 VRAM notes for Qwen3.5-0.8B on this repo's CARLA-derived data:
# - Current dataset images are 1280x720. With full image tokens, 12 GB class GPUs still OOM
#   on the first backward even with `batch-size 1` and gradient checkpointing enabled.
# - For 12 GB class GPUs, treat `batch-size 1` + gradient checkpointing as the floor and
#   expect that image-token reduction or smaller input images may still be required.
# - For 24 GB class GPUs, the preferred target is to keep full image tokens and start from
#   `batch-size 2` + `--no-gradient-checkpointing`.
# Update this block whenever VRAM measurements change so the 24 GB path stays documented here.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen3.5 Stage 1 with train/val data and checkpoints.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
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
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
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


def build_processor_kwargs(image_min_pixels: int, image_max_pixels: int) -> dict:
    kwargs = {}
    if image_min_pixels > 0:
        kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels > 0:
        kwargs["max_pixels"] = image_max_pixels
    return kwargs


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


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value_to_device(item, device) for item in value)
    return value


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = move_value_to_device(value, device)


def log_gpu_preflight(device: torch.device) -> dict:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    snapshot = collect_gpu_preflight_snapshot(gpu_index=device_index)
    print(json.dumps({"event": "gpu_preflight", **snapshot}, ensure_ascii=False))
    if snapshot["warning_reasons"]:
        print(
            json.dumps(
                {
                    "event": "gpu_preflight_warning",
                    "gpu_index": device_index,
                    "warning_reasons": snapshot["warning_reasons"],
                    "non_self_compute_processes": snapshot.get("non_self_compute_processes", []),
                },
                ensure_ascii=False,
            )
        )
    return snapshot


def write_run_config(save_dir: Path, args: argparse.Namespace, run_metadata: dict) -> None:
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_json": args.config_json,
                "config_payload": args.config_payload,
                "resolved_args": vars(args),
                "run_metadata": run_metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


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
        "train_jsonl": args.train_jsonl,
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


def validate_resume_args(args: argparse.Namespace, checkpoint: dict) -> None:
    checkpoint_args = checkpoint.get("args", {})
    defaults = {
        "val_jsonl": "",
        "image_min_pixels": 0,
        "image_max_pixels": 0,
    }
    keys_to_match = [
        "train_jsonl",
        "val_jsonl",
        "model_path",
        "dtype",
        "gradient_checkpointing",
        "image_min_pixels",
        "image_max_pixels",
    ]
    mismatches = []
    for key in keys_to_match:
        checkpoint_value = checkpoint_args.get(key, defaults.get(key))
        current_value = getattr(args, key)
        if checkpoint_value != current_value:
            mismatches.append(f"{key}: checkpoint={checkpoint_value!r}, config={current_value!r}")
    if mismatches:
        raise RuntimeError(
            "Resume checkpoint settings do not match the current config:\n" + "\n".join(mismatches)
        )


def metric_improved(current: float, best: float, min_delta: float) -> bool:
    if math.isinf(best):
        return True
    return current < (best - min_delta)


def best_metric_from_history(metrics_history: list[dict], metric_name: str) -> tuple[float, int]:
    best_metric = float("inf")
    best_epoch = 0
    for metrics in metrics_history:
        value = metrics.get(metric_name)
        if value is None:
            continue
        value = float(value)
        if value < best_metric:
            best_metric = value
            best_epoch = int(metrics.get("epoch", 0))
    return best_metric, best_epoch


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
    run_metadata: dict,
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
        "run_metadata": run_metadata,
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
    wall_start = time.perf_counter()

    try:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise RuntimeError("This Stage 1 trainer is intended to run on CUDA.")
        gpu_preflight = log_gpu_preflight(device)
        gpu_info = collect_gpu_info(device)
        git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
        set_seed(args.seed)

        train_loader, val_loader, train_size, val_size = build_dataloaders(args)
        if len(train_loader) == 0:
            raise RuntimeError("Train DataLoader is empty.")
        train_dataset_fingerprint = collect_dataset_view_fingerprint(train_loader.dataset)
        val_dataset_fingerprint = collect_dataset_view_fingerprint(val_loader.dataset) if val_loader is not None else None

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
            resume_checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
            validate_resume_args(args, resume_checkpoint)
            saved_processor_dir = resume_checkpoint_path.parent / "processor"
            if saved_processor_dir.exists():
                processor_source = str(saved_processor_dir)

        load_start = time.perf_counter()
        processor_kwargs = build_processor_kwargs(args.image_min_pixels, args.image_max_pixels)
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, **processor_kwargs)
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
                "resume_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
                "resumed": resume_checkpoint is not None,
                "checkpoint_epoch": int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint is not None else 0,
                "checkpoint_global_step": int(resume_checkpoint.get("global_step", 0)) if resume_checkpoint is not None else 0,
            },
        }
        write_run_config(save_dir, args, run_metadata)

        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            initial_eval = resume_checkpoint.get("initial_eval")
            metrics_history = list(resume_checkpoint.get("metrics_history", []))
            global_step = int(resume_checkpoint.get("global_step", 0))
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
            best_metric, best_epoch = best_metric_from_history(metrics_history, best_metric_name)
            last_epoch = int(metrics_history[-1]["epoch"]) if metrics_history else 0
            epochs_without_improvement = max(0, last_epoch - best_epoch)
        else:
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
            release_cuda_memory()
            metrics_history = []
            global_step = 0
            start_epoch = 1
            best_metric = float("inf")
            best_epoch = 0
            epochs_without_improvement = 0

        if initial_eval is None:
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
            release_cuda_memory()

        if start_epoch > args.max_epochs:
            raise RuntimeError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, which is >= configured max_epochs={args.max_epochs}."
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        setup_payload = {
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
        setup_wandb_payload = {
            "setup/gpu_query_ok": gpu_preflight["query_ok"],
            "setup/gpu_warning_count": len(gpu_preflight["warning_reasons"]),
            "setup/train_size": train_size,
            "setup/val_size": val_size,
            "setup/batch_size": args.batch_size,
            "setup/grad_accum_steps": args.grad_accum_steps,
            "setup/start_epoch": start_epoch,
            "setup/total_vocab_size": len(processor.tokenizer),
            "setup/added_action_tokens": added,
            "setup/model_load_elapsed_s": round(load_elapsed, 3),
            "setup/k": stage1_metadata["k"],
            "setup/dt": stage1_metadata["dt"],
            "setup/n_bins": stage1_metadata["n_bins"],
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
                    "setup/gpu_non_self_compute_gib": round(gpu_preflight["non_self_compute_used_mib"] / 1024, 3),
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
                        run_metadata=run_metadata,
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
                        run_metadata=run_metadata,
                    ),
                    save_dir / f"epoch_{epoch:03d}.pt",
                )

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
                    run_metadata=run_metadata,
                ),
                save_dir / "last.pt",
            )

            with (save_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(metrics_history, f, indent=2, ensure_ascii=False)
            completed_epochs = epoch

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                stop_reason = "early_stopping"
                early_stop_payload = {
                    "event": "early_stopping",
                    "epoch": epoch,
                    "best_metric_name": best_metric_name,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "patience": args.early_stopping_patience,
                    "min_delta": args.early_stopping_min_delta,
                    "epochs_without_improvement": epochs_without_improvement,
                }
                print(json.dumps(early_stop_payload, ensure_ascii=False))
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
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                registry=registry,
                quantizer=quantizer,
                stage1_metadata=stage1_metadata,
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
            "train_jsonl": args.train_jsonl,
            "val_jsonl": args.val_jsonl or None,
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
            "resume_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
            "resumed": resume_checkpoint is not None,
            "added_action_tokens": added,
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
        maybe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
