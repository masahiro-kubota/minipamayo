"""Stage 1 smoke-test and VRAM profiler for the Qwen3.5 branch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...contract.prompt import DEFAULT_QUESTION
from ...contract.task_spec import CanonicalStage1Spec
from ...utils.json_config import normalize_required_string_list
from ...utils.preflight import enforce_runtime_prerequisites
from ..dataset import Stage1JsonlDataset, stage1_collate
from .cli import parse_config_json_only_args
from .components import build_model_load_kwargs, build_stage1_metadata, build_training_token_contract, resolve_dtype
from .runtime import Stage1ARuntime, format_gib, model_forward_inputs, prepare_stage1a_training_batch, set_seed

CONFIG_PATH_KEYS = {"train_jsonl", "model_path", "output_json"}
MULTI_VALUE_CONFIG_KEYS = {"train_jsonl"}
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short canonical Stage 1A VLM CE training trial."
    )
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
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
    parser.add_argument("--forward-only", action="store_true", default=False)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/stage1/vlm_ce/profile/trial_summary.json"),
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
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_config_json_only_args(
        parser,
        path_keys=CONFIG_PATH_KEYS,
        list_keys=MULTI_VALUE_CONFIG_KEYS,
        error_message="Stage 1 profile accepts only --config-json. Put all settings in the JSON file.",
    )
    args.train_jsonl = normalize_required_string_list(args.train_jsonl, key_name="train_jsonl")
    return args


def get_batch(dataset: Stage1JsonlDataset, start_index: int, batch_size: int) -> list[dict]:
    return [dataset[(start_index + i) % len(dataset)] for i in range(batch_size)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    wall_start = time.perf_counter()

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("This Stage 1 probe is intended to be run on CUDA to measure VRAM.")
    enforce_runtime_prerequisites(git_cwd=Path(__file__).resolve().parent)

    dataset = Stage1JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model_dtype = resolve_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **build_model_load_kwargs(model_dtype),
    )

    task_spec = CanonicalStage1Spec()
    (
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        added_action_tokens,
        added_history_tokens,
    ) = build_training_token_contract(dataset, processor, task_spec)
    stage1_metadata = build_stage1_metadata(
        dataset,
        train_jsonl=args.train_jsonl,
        val_jsonl=None,
        registry=registry,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        quantizer=quantizer,
        task_spec=task_spec,
        question=args.question,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    model.enable_input_require_grads()
    model.to(device)
    model.train()
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

    total_steps = args.warmup_steps + args.measure_steps
    losses: list[float] = []
    warmup_times: list[float] = []
    measured_times: list[float] = []
    measured_allocated: list[int] = []
    measured_reserved: list[int] = []
    measured_samples_per_second: list[float] = []

    start_total = time.perf_counter()
    for step in range(total_steps):
        batch = stage1_collate(get_batch(dataset, step * args.batch_size, args.batch_size))
        full_inputs, labels = prepare_stage1a_training_batch(runtime, batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        with torch.autocast("cuda", dtype=model_dtype):
            outputs = model(**model_forward_inputs(full_inputs), labels=labels)
            loss = outputs.loss

        if not args.forward_only:
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

    train_elapsed = time.perf_counter() - start_total
    total_elapsed = time.perf_counter() - wall_start
    summary = {
        "model_path": args.model_path,
        "train_jsonl": list(args.train_jsonl),
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
        "forward_only": args.forward_only,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "added_action_tokens": added_action_tokens,
        "added_history_tokens": added_history_tokens,
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
