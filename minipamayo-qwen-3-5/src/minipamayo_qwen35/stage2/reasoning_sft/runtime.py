"""Shared runtime helpers for canonical Stage 2 reasoning SFT."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList

from ...contract.prompt import (
    COT_END_TOKEN,
    TRAJ_FUTURE_START_TOKEN,
    build_reasoning_prompt_text,
)
from ...models.token_utils import StopAfterEOS
from ...stage1.stage1a_prompting import (
    inject_history_inputs_embeds,
    model_forward_inputs,
    move_inputs_to_device,
    prepare_prompt_inputs_with_history,
)


class TrajectoryLogitsProcessor(LogitsProcessor):
    """Mask discrete trajectory token logits during reasoning rollout."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = int(traj_token_offset)
        self.traj_vocab_size = int(traj_vocab_size)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


def _stage2_amp_context(device: torch.device, model_dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=model_dtype)
    return nullcontext()


def _build_target_rows(tokenizer, batch: dict) -> list[list[int]]:
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
        prompt_text = build_reasoning_prompt_text(
            processor,
            history_token_count=history_quantizer.token_count,
        )
        prompt_inputs = processor(
            text=[prompt_text] * len(images),
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
        target_rows = _build_target_rows(processor.tokenizer, batch)
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
                if int(token_id) in handoff_token_ids:
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
        return full_inputs, labels, full_loss_weights
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


def compute_token_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    shifted_preds = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels != -100
    total = int(valid_mask.sum().item())
    correct = int(((shifted_preds == shifted_labels) & valid_mask).sum().item())
    return {"correct": correct, "total": total}


def run_stage2_teacher_forced_batch(
    *,
    model,
    batch: dict,
    processor,
    history_registry,
    history_quantizer,
    device: torch.device,
    model_dtype: torch.dtype,
    handoff_loss_weight: float,
) -> dict[str, Any]:
    full_inputs, labels, loss_weights = prepare_stage2_batch(
        model=model,
        batch=batch,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        device=device,
        handoff_loss_weight=handoff_loss_weight,
    )
    with _stage2_amp_context(device, model_dtype):
        outputs = model(**model_forward_inputs(full_inputs))
        loss = compute_weighted_loss(outputs.logits, labels, loss_weights)
    metrics = compute_token_metrics(outputs.logits.detach(), labels)
    return {
        "outputs": outputs,
        "loss": loss,
        "labels": labels,
        "loss_weights": loss_weights,
        "correct": metrics["correct"],
        "total": metrics["total"],
    }


@torch.no_grad()
def evaluate_stage2(
    *,
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
        result = run_stage2_teacher_forced_batch(
            model=model,
            batch=batch,
            processor=processor,
            history_registry=history_registry,
            history_quantizer=history_quantizer,
            device=device,
            model_dtype=model_dtype,
            handoff_loss_weight=handoff_loss_weight,
        )
        total_loss += float(result["loss"].detach().cpu())
        total_batches += 1
        total_correct += result["correct"]
        total_tokens += result["total"]

    model.train()
    return {
        "loss": total_loss / max(total_batches, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def generate_reasoning_handoff(
    *,
    model,
    tokenizer,
    prompt_inputs: dict,
    traj_registry,
    stop_token_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    generation_config = model.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = top_k if top_k > 0 else None
    generation_config.pad_token_id = tokenizer.pad_token_id

    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=stop_token_id)])
    logits_processor = LogitsProcessorList(
        [
            TrajectoryLogitsProcessor(
                traj_token_offset=traj_registry.start_index,
                traj_vocab_size=traj_registry.n_bins,
            )
        ]
    )
    outputs = model.generate(
        **prompt_inputs,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
    )
    if not hasattr(outputs, "past_key_values") or outputs.past_key_values is None:
        raise RuntimeError("Stage 2 generation did not return `past_key_values` for expert handoff.")

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    generated_ids = outputs.sequences[:, prompt_len:]
    stop_mask = generated_ids == stop_token_id
    has_stop = stop_mask.any(dim=1)
    if not torch.all(has_stop):
        raise RuntimeError(
            "Stage 2 reasoning rollout did not emit `<|traj_future_start|>` within the token budget.\n"
            f"max_reasoning_tokens={max_new_tokens}"
        )
    stop_positions = stop_mask.int().argmax(dim=1)
    handoff_attention_mask = torch.ones_like(
        outputs.sequences,
        dtype=prompt_inputs["attention_mask"].dtype,
    )
    for row_idx, stop_pos in enumerate(stop_positions.tolist()):
        cutoff = prompt_len + stop_pos + 1
        handoff_attention_mask[row_idx, cutoff:] = 0
    return outputs, generated_ids, handoff_attention_mask, stop_positions


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
            prompt_text = build_reasoning_prompt_text(
                processor,
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
