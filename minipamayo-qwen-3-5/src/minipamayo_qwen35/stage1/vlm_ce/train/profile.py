"""Stage 1 smoke-test and VRAM profiler for the Qwen3.5 branch.

This script is intentionally small and is used first as:
1. a smoke test that full fine-tuning works end-to-end
2. a VRAM / time profiler for planning later experiments
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....utils.json_config import (
    load_json_payload,
    normalize_arg_config,
    normalize_required_string_list,
    resolve_path_base,
)
from ....utils.preflight import require_expected_cuda_toolkit
from ....contract.task_spec import CanonicalStage1Spec
from ...dataset import Stage1JsonlDataset
from ....contract.prompt import DEFAULT_QUESTION, add_prompt_special_tokens, build_prompt_text
from ....contract.history_tokens import HistoryTokenRegistry, HistoryTrajectoryQuantizer
from ....contract.trajectory_tokens import Stage1TokenRegistry
from .runner import format_gib, model_forward_inputs, prepare_batch, stage1_collate
from .runner import build_model_load_kwargs

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {"train_jsonl", "model_path", "output_json"}
MULTI_VALUE_CONFIG_KEYS = {"train_jsonl"}

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

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        args = parser.parse_args()
        args.train_jsonl = normalize_required_string_list(args.train_jsonl, key_name="train_jsonl")
        return args

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Stage 1 profile accepts only --config-json. Put all settings in the JSON file."
        )

    config_path, payload = load_json_payload(pre_args.config_json)
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
    parser.set_defaults(**config_args, config_json=str(config_path))
    args = parser.parse_args()
    args.config_json = str(config_path)
    args.config_payload = payload
    args.config_args = config_args
    args.train_jsonl = normalize_required_string_list(args.train_jsonl, key_name="train_jsonl")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_batch(dataset: Stage1JsonlDataset, start_index: int, batch_size: int) -> list[dict]:
    return [dataset[(start_index + i) % len(dataset)] for i in range(batch_size)]


def build_stage1_metadata(
    dataset: Stage1JsonlDataset,
    registry: Stage1TokenRegistry,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    task_spec: CanonicalStage1Spec,
    quantizer,
    question: str,
) -> dict:
    sample = dataset[0]
    gt_waypoints = sample["gt_waypoints"].detach().cpu().reshape(-1, 2)
    action = sample["action"].detach().cpu().reshape(-1).numpy()
    ego_history_xyz = sample["ego_history_xyz"].detach().cpu().numpy()
    dt_value = float(sample["dt"].item())
    return {
        "train_jsonl": [str(path) for path in dataset.jsonl_paths],
        "sample_format": "jsonl+images",
        "k": int(gt_waypoints.shape[0]),
        "target_dim": int(task_spec.target_from_action_array(action).shape[0]),
        "full_action_dim": int(action.shape[0]),
        "dt": dt_value,
        "action_token_scheme": "alpamayo_like_discrete_tokens",
        "token_prefix": registry.token_prefix,
        "token_start_index": registry.start_index,
        "history_token_scheme": "placeholder_input_ids_discrete_tokens",
        "history_token_prefix": history_registry.token_prefix,
        "history_token_start_index": history_registry.start_index,
        "history_steps": int(ego_history_xyz.shape[-2]),
        "history_token_count": history_quantizer.token_count,
        "question": question,
        **task_spec.metadata(quantizer),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    wall_start = time.perf_counter()

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("This Stage 1 probe is intended to be run on CUDA to measure VRAM.")
    require_expected_cuda_toolkit()

    dataset = Stage1JsonlDataset(args.train_jsonl, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        **build_model_load_kwargs(model_dtype),
    )

    task_spec = CanonicalStage1Spec()
    history_quantizer = HistoryTrajectoryQuantizer()
    add_prompt_special_tokens(processor.tokenizer)
    quantizer = task_spec.build_quantizer(dataset)
    quantizer_n_bins = int(
        quantizer.num_bins if hasattr(quantizer, "num_bins") else quantizer.n_bins
    )
    registry = Stage1TokenRegistry(n_bins=quantizer_n_bins, start_index=0)
    added = registry.add_to_tokenizer(processor.tokenizer)
    history_registry = HistoryTokenRegistry(
        n_bins=history_quantizer.n_bins,
        start_index=registry.start_index + registry.n_bins,
    )
    history_added = history_registry.add_to_tokenizer(processor.tokenizer)
    stage1_metadata = build_stage1_metadata(
        dataset,
        registry,
        history_registry,
        history_quantizer,
        task_spec,
        quantizer,
        args.question,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    load_elapsed = time.perf_counter() - load_start

    prompt_text = build_prompt_text(
        processor,
        args.question,
        history_token_count=history_quantizer.token_count,
    )
    total_steps = args.warmup_steps + args.measure_steps
    losses: list[float] = []
    warmup_times: list[float] = []
    measured_times: list[float] = []
    measured_allocated: list[int] = []
    measured_reserved: list[int] = []
    measured_samples_per_second: list[float] = []

    start_total = time.perf_counter()
    for step in range(total_steps):
        batch_samples = stage1_collate(get_batch(dataset, step * args.batch_size, args.batch_size))
        full_inputs, labels = prepare_batch(
            model,
            batch_samples,
            processor,
            registry,
            history_registry,
            history_quantizer,
            quantizer,
            task_spec,
            prompt_text,
            device,
        )

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
        "added_action_tokens": added,
        "added_history_tokens": history_added,
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
