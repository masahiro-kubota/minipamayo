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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, random_split

from ....stage1.common.prompt import COT_END_TOKEN, TRAJ_FUTURE_START_TOKEN
from ....stage1.vlm_ce.eval import load_components
from ....stage1.vlm_ce.train import (
    format_gib,
    inject_history_inputs_embeds,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    metric_improved,
    model_forward_inputs,
    move_inputs_to_device,
    prepare_prompt_inputs_with_history,
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
from ..data import ReasoningSftJsonlDataset, reasoning_sft_collate
from ..inference.runner import generate_reasoning_handoff
from ..prompt import build_stage2_prompt_text

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
        "target_layout": "reasoning_then_cot_end_then_traj_future_start_then_eos",
        "prompt_contract": "alpamayo_like_reasoning_with_cot_prefill",
        "k": int(gt_waypoints.shape[0]),
        "action_dim": int(action.shape[0]),
        "dt": float(sample["dt"]),
        "handoff_loss_weight": args.handoff_loss_weight,
    }


def build_handoff_probe_dataset(args: argparse.Namespace):
    if args.handoff_probe_jsonl:
        max_per_jsonl = args.handoff_probe_max_per_jsonl
        if max_per_jsonl <= 0:
            raise RuntimeError(
                "`handoff_probe_max_per_jsonl` must be > 0 when `handoff_probe_jsonl` is set."
            )
        probe_samples = []
        for jsonl_path in args.handoff_probe_jsonl:
            source_dataset = ReasoningSftJsonlDataset(jsonl_path)
            source_count = min(max_per_jsonl, len(source_dataset))
            for idx in range(source_count):
                probe_samples.append(source_dataset[idx])
        if not probe_samples:
            raise RuntimeError("Explicit handoff probe dataset resolved to zero samples.")
        return probe_samples
    return None


def _build_target_rows(
    tokenizer, batch: dict
) -> list[list[int]]:
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer is missing `eos_token_id`, which Stage 2 requires.")
    cot_end_token_id = int(tokenizer.convert_tokens_to_ids(COT_END_TOKEN))
    if cot_end_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|cot_end|>`.")
    traj_future_start_token_id = int(tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN))
    if traj_future_start_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|traj_future_start|>`.")
    target_rows: list[list[int]] = []
    for reasoning_text in batch["reasoning_text"]:
        reasoning_prefix = tokenizer(reasoning_text, add_special_tokens=False)
        reasoning_ids = reasoning_prefix["input_ids"]
        if not isinstance(reasoning_ids, list):
            raise RuntimeError("Tokenizer returned a non-list `input_ids` payload for Stage 2.")
        row = list(reasoning_ids) + [
            cot_end_token_id,
            traj_future_start_token_id,
            int(tokenizer.eos_token_id),
        ]
        target_rows.append(row)
    return target_rows


def prepare_stage2_batch(
    model,
    batch: dict,
    processor,
    history_registry,
    history_quantizer,
    device: torch.device,
    handoff_loss_weight: float,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    try:
        prompt_texts = [
            build_stage2_prompt_text(
                processor,
                float(v0.item()),
                history_token_count=history_quantizer.token_count,
            )
            for v0 in batch["v0"]
        ]
        prompt_inputs = processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = move_inputs_to_device(prompt_inputs, device)
        prompt_inputs = inject_history_inputs_embeds(
            model=model,
            prompt_inputs=prompt_inputs,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            history_xyz=batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
            history_rot=batch["ego_history_rot"].to(device=device, dtype=torch.float32),
        )
        target_rows = _build_target_rows(
            processor.tokenizer,
            batch,
        )
        if processor.tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer is missing `pad_token_id`, which Stage 2 requires.")
        batch_size = len(target_rows)
        max_len = max(len(row) for row in target_rows)
        target_ids = torch.full(
            (batch_size, max_len),
            fill_value=int(processor.tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        target_mask = torch.zeros(
            (batch_size, max_len),
            dtype=prompt_inputs["attention_mask"].dtype,
            device=device,
        )
        weight_tensor = torch.ones((batch_size, max_len), dtype=torch.float32, device=device)
        cot_end_token_id = int(processor.tokenizer.convert_tokens_to_ids(COT_END_TOKEN))
        traj_future_start_token_id = int(
            processor.tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN)
        )
        if processor.tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer is missing `eos_token_id`, which Stage 2 requires.")
        handoff_token_ids = {
            int(processor.tokenizer.eos_token_id),
            cot_end_token_id,
            traj_future_start_token_id,
        }

        for row_idx, row in enumerate(target_rows):
            row_len = len(row)
            target_ids[row_idx, :row_len] = torch.tensor(row, dtype=torch.long, device=device)
            target_mask[row_idx, :row_len] = 1
            for token_idx, token_id in enumerate(row):
                token_id = int(row[token_idx])
                if token_id in handoff_token_ids:
                    weight_tensor[row_idx, token_idx] = handoff_loss_weight
                else:
                    weight_tensor[row_idx, token_idx] = 1.0

        input_ids = torch.cat([prompt_inputs["input_ids"], target_ids], dim=1)
        attention_mask = torch.cat([prompt_inputs["attention_mask"], target_mask], dim=1)
        full_inputs = {
            key: value
            for key, value in prompt_inputs.items()
            if key not in {"input_ids", "attention_mask", "inputs_embeds"}
        }
        full_inputs["input_ids"] = input_ids
        full_inputs["attention_mask"] = attention_mask
        if "inputs_embeds" in prompt_inputs:
            target_embeds = model.get_input_embeddings()(target_ids)
            full_inputs["inputs_embeds"] = torch.cat(
                [prompt_inputs["inputs_embeds"], target_embeds], dim=1
            )
        if "mm_token_type_ids" in full_inputs:
            full_inputs["mm_token_type_ids"] = torch.cat(
                [
                    full_inputs["mm_token_type_ids"],
                    torch.zeros(
                        (batch_size, max_len),
                        dtype=full_inputs["mm_token_type_ids"].dtype,
                        device=full_inputs["mm_token_type_ids"].device,
                    ),
                ],
                dim=1,
            )
        labels = torch.full_like(input_ids, -100)
        offset = prompt_inputs["input_ids"].shape[1]
        labels[:, offset:] = target_ids

        full_loss_weights = torch.ones_like(labels, dtype=torch.float32)
        full_loss_weights[:, offset:] = weight_tensor

        return (
            full_inputs,
            labels,
            full_loss_weights,
        )
    finally:
        for image in images:
            image.close()


def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = loss_weights[:, 1:].contiguous()
    vocab_size = shifted_logits.shape[-1]

    token_loss = F.cross_entropy(
        shifted_logits.view(-1, vocab_size),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    valid_mask = shifted_labels != -100
    weighted_loss = token_loss[valid_mask] * shifted_weights[valid_mask]
    denominator = shifted_weights[valid_mask].sum().clamp_min(1.0)
    return weighted_loss.sum() / denominator


def compute_token_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, int]:
    shifted_preds = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels != -100

    total = int(valid_mask.sum().item())
    correct = int(((shifted_preds == shifted_labels) & valid_mask).sum().item())
    return {
        "correct": correct,
        "total": total,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    processor,
    history_registry,
    history_quantizer,
    device: torch.device,
    model_dtype: torch.dtype,
    handoff_loss_weight: float,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        full_inputs, labels, loss_weights = prepare_stage2_batch(
            model=model,
            batch=batch,
            processor=processor,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            device=device,
            handoff_loss_weight=handoff_loss_weight,
        )
        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**model_forward_inputs(full_inputs))
            loss = compute_weighted_loss(outputs.logits, labels, loss_weights)

        metrics = compute_token_metrics(outputs.logits, labels)
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        total_correct += metrics["correct"]
        total_tokens += metrics["total"]

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


@torch.inference_mode()
def evaluate_handoff_probe(
    *,
    model,
    dataset,
    processor,
    registry,
    history_registry,
    history_quantizer,
    device: torch.device,
    max_samples: int,
    max_reasoning_tokens: int,
    seed: int,
) -> dict | None:
    if max_samples <= 0 or len(dataset) == 0:
        return None

    num_samples = min(max_samples, len(dataset))
    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids(TRAJ_FUTURE_START_TOKEN))
    if stop_token_id < 0:
        raise RuntimeError("Tokenizer is missing canonical `<|traj_future_start|>`.")

    prev_use_cache = bool(getattr(model.config, "use_cache", False))
    was_training = model.training
    model.config.use_cache = True
    model.eval()

    success_count = 0
    failure_sample_ids: list[str] = []
    success_positions: list[int] = []
    failure_reasons: list[str] = []
    try:
        for sample_idx in range(num_samples):
            sample = dataset[sample_idx]
            torch.manual_seed(seed + sample_idx)
            torch.cuda.manual_seed_all(seed + sample_idx)

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
            try:
                _, _, _, stop_positions = generate_reasoning_handoff(
                    model=model,
                    tokenizer=processor.tokenizer,
                    prompt_inputs=prompt_inputs,
                    traj_registry=registry,
                    stop_token_id=stop_token_id,
                    max_new_tokens=max_reasoning_tokens,
                    temperature=0.6,
                    top_p=0.98,
                    top_k=0,
                )
                success_count += 1
                success_positions.append(int(stop_positions[0].item()))
            except RuntimeError as exc:
                failure_sample_ids.append(str(sample["sample_id"]))
                failure_reasons.append(str(exc))
    finally:
        model.config.use_cache = prev_use_cache
        if was_training:
            model.train()

    success_rate = success_count / max(num_samples, 1)
    mean_stop_position = (
        sum(success_positions) / len(success_positions) if success_positions else None
    )
    first_failure_reason = failure_reasons[0] if failure_reasons else ""
    return {
        "num_samples": num_samples,
        "num_success": success_count,
        "success_rate": success_rate,
        "mean_stop_position": mean_stop_position,
        "failure_sample_ids": failure_sample_ids,
        "first_failure_reason": first_failure_reason,
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
            "n_bins": action_quantizer.n_bins,
            "a_range": list(action_quantizer.a_range),
            "kappa_range": list(action_quantizer.kappa_range),
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
