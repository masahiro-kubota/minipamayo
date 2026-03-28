"""Stage 1 smoke-test and VRAM profiler for the Qwen3.5 branch.

This script is intentionally small and is used first as:
1. a smoke test that full fine-tuning works end-to-end
2. a VRAM / time profiler for planning later experiments
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..data.stage1_dataset import Stage1JsonlDataset
from ..tokens.action_quantizer import ActionQuantizer
from ..tokens.token_registry import Stage1TokenRegistry


DEFAULT_QUESTION = (
    "Predict the future ego trajectory as action tokens. "
    "Output only the action tokens in order."
)

# Measured Stage 1 presets on the CARLA-derived smoke dataset in this repo:
# - 12 GB class GPU: keep `--batch-size 1 --gradient-checkpointing`
#   (about 11.36 GiB peak reserved with Qwen3.5-0.8B in bf16).
# - 24 GB class GPU: prefer `--batch-size 2 --no-gradient-checkpointing`
#   (fastest measured setting so far, about 20.08 GiB peak reserved).
# - Conservative 24 GB fallback: `--batch-size 2 --gradient-checkpointing`
#   (about 14.31 GiB peak reserved if you want more headroom).
# These numbers are a planning guide for Stage 1 only; longer Stage 3/4
# sequences will need more VRAM.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short Qwen3.5 Stage 1 training trial.")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/masa/minipamayo/shared_checkpoints/hf_models/Qwen3.5-0.8B",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    parser.add_argument(
        "--output-json",
        type=str,
        default="/home/masa/minipamayo/minipamayo-qwen-3-5/artifacts/stage1_trial_summary.json",
    )
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
    return parser.parse_args()


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


def get_batch(dataset: Stage1JsonlDataset, start_index: int, batch_size: int) -> list[dict]:
    return [dataset[(start_index + i) % len(dataset)] for i in range(batch_size)]


def build_stage1_metadata(
    dataset: Stage1JsonlDataset,
    registry: Stage1TokenRegistry,
    quantizer: ActionQuantizer,
    question: str,
) -> dict:
    record = dataset.records[0]
    gt_waypoints = record.get("gt_waypoints", [])
    action = record.get("action", [])
    return {
        "train_jsonl": str(dataset.jsonl_path),
        "sample_format": "jsonl+images",
        "k": len(gt_waypoints) if gt_waypoints else len(action) // 2,
        "action_dim": len(action),
        "dt": record.get("dt"),
        "n_bins": quantizer.n_bins,
        "a_range": list(quantizer.a_range),
        "kappa_range": list(quantizer.kappa_range),
        "action_token_scheme": "add_tokens",
        "token_prefix": registry.token_prefix,
        "question": question,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    wall_start = time.perf_counter()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("This Stage 1 probe is intended to be run on CUDA to measure VRAM.")

    dataset = Stage1JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

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
    stage1_metadata = build_stage1_metadata(dataset, registry, quantizer, args.question)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.enable_input_require_grads()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    load_elapsed = time.perf_counter() - load_start

    prompt_text = build_prompt_text(processor, args.question)
    total_steps = args.warmup_steps + args.measure_steps
    losses: list[float] = []
    warmup_times: list[float] = []
    measured_times: list[float] = []
    measured_allocated: list[int] = []
    measured_reserved: list[int] = []
    measured_samples_per_second: list[float] = []

    start_total = time.perf_counter()
    for step in range(total_steps):
        batch_samples = get_batch(dataset, step * args.batch_size, args.batch_size)
        images = [Image.open(sample["image_path"]).convert("RGB") for sample in batch_samples]
        action_token_id_rows = [
            registry.encode_action_token_ids(sample["action"].numpy(), quantizer)
            for sample in batch_samples
        ]

        prompt_inputs = processor(
            text=[prompt_text] * args.batch_size,
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

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**full_inputs, labels=labels)
            loss = outputs.loss

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        step_elapsed = time.perf_counter() - step_start
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        losses.append(float(loss.detach().cpu()))
        if step < args.warmup_steps:
            warmup_times.append(step_elapsed)
        else:
            measured_times.append(step_elapsed)
            measured_allocated.append(peak_allocated)
            measured_reserved.append(peak_reserved)
            measured_samples_per_second.append(args.batch_size / step_elapsed)

        print(
            json.dumps(
                {
                    "step": step,
                    "loss": round(losses[-1], 6),
                    "batch_size": args.batch_size,
                    "elapsed_s": round(step_elapsed, 3),
                    "samples_per_s": round(args.batch_size / step_elapsed, 3),
                    "peak_allocated_gib": format_gib(peak_allocated),
                    "peak_reserved_gib": format_gib(peak_reserved),
                },
                ensure_ascii=False,
            )
        )
        for image in images:
            image.close()

    train_elapsed = time.perf_counter() - start_total
    total_elapsed = time.perf_counter() - wall_start
    summary = {
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "num_optimizer_steps": total_steps,
        "num_samples_seen": total_steps * args.batch_size,
        "dataset_size": len(dataset),
        "device": str(device),
        "dtype": args.dtype,
        "full_fine_tune": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "added_action_tokens": added,
        "total_vocab_size": len(processor.tokenizer),
        "stage1_metadata": stage1_metadata,
        "warmup_step_times_s": warmup_times,
        "measured_step_times_s": measured_times,
        "mean_measured_step_time_s": sum(measured_times) / max(len(measured_times), 1),
        "measured_samples_per_second": measured_samples_per_second,
        "mean_measured_samples_per_second": (
            sum(measured_samples_per_second) / max(len(measured_samples_per_second), 1)
        ),
        "peak_allocated_gib_max": format_gib(max(measured_allocated) if measured_allocated else 0),
        "peak_reserved_gib_max": format_gib(max(measured_reserved) if measured_reserved else 0),
        "losses": losses,
        "model_load_elapsed_s": round(load_elapsed, 3),
        "train_loop_elapsed_s": round(train_elapsed, 3),
        "total_wall_time_s": round(total_elapsed, 3),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
