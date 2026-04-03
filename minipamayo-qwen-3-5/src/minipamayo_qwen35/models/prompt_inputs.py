"""Shared prompt-input helpers for multimodal history-conditioned models."""

from __future__ import annotations

from PIL import Image
import torch

from ..contract.history_tokens import (
    HistoryTokenRegistry,
    HistoryTrajectoryQuantizer,
    encode_history_token_id_rows,
)


def move_inputs_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def model_forward_inputs(model_inputs: dict) -> dict:
    if "inputs_embeds" not in model_inputs:
        return model_inputs
    return {key: value for key, value in model_inputs.items() if key != "input_ids"}


def inject_history_token_ids(
    *,
    prompt_inputs: dict,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> dict:
    if history_registry.placeholder_token_id is None:
        raise RuntimeError("History token registry is not attached to a tokenizer yet.")
    input_ids = prompt_inputs["input_ids"]
    placeholder_mask = input_ids == history_registry.placeholder_token_id
    placeholder_count = int(placeholder_mask[0].sum().item()) if input_ids.shape[0] > 0 else 0
    if placeholder_count != history_quantizer.token_count:
        raise RuntimeError(
            "Prompt history placeholder count does not match the canonical history token count.\n"
            f"prompt_placeholder_count={placeholder_count}\n"
            f"history_token_count={history_quantizer.token_count}"
        )
    history_token_id_rows = encode_history_token_id_rows(
        history_xyz=history_xyz,
        history_rot=history_rot,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
    )
    fused_inputs = dict(prompt_inputs)
    fused_inputs["input_ids"] = history_registry.replace_placeholder_ids(
        input_ids,
        history_token_id_rows,
    )
    return fused_inputs


def inject_history_inputs_embeds(
    *,
    model,
    prompt_inputs: dict,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
) -> dict:
    del model
    return inject_history_token_ids(
        prompt_inputs=prompt_inputs,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        history_xyz=history_xyz,
        history_rot=history_rot,
    )


def append_token_to_model_inputs(model, model_inputs: dict, next_token: torch.Tensor) -> None:
    model_inputs["input_ids"] = torch.cat(
        [model_inputs["input_ids"], next_token.unsqueeze(1)], dim=1
    )
    if "inputs_embeds" in model_inputs:
        next_embed = model.get_input_embeddings()(next_token.unsqueeze(1))
        model_inputs["inputs_embeds"] = torch.cat(
            [model_inputs["inputs_embeds"], next_embed], dim=1
        )
    model_inputs["attention_mask"] = torch.cat(
        [
            model_inputs["attention_mask"],
            torch.ones(
                (next_token.shape[0], 1),
                device=next_token.device,
                dtype=model_inputs["attention_mask"].dtype,
            ),
        ],
        dim=1,
    )
    if "mm_token_type_ids" in model_inputs:
        model_inputs["mm_token_type_ids"] = torch.cat(
            [
                model_inputs["mm_token_type_ids"],
                torch.zeros(
                    (next_token.shape[0], 1),
                    device=next_token.device,
                    dtype=model_inputs["mm_token_type_ids"].dtype,
                ),
            ],
            dim=1,
        )


def prepare_prompt_inputs_with_history(
    model,
    batch: dict,
    processor,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    prompt_text: str,
    device: torch.device,
) -> dict:
    images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
    try:
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
        return prompt_inputs
    finally:
        for image in images:
            image.close()

