from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ..contract.history_tokens import (
    HistoryTokenRegistry,
    HistoryTrajectoryQuantizer,
)
from ..contract.prompt import build_multimodal_messages, build_stage1_question_user_text
from ..contract.task_spec import Stage1TaskSpec
from ..contract.trajectory_tokens import Stage1TokenRegistry
from ..helper import to_device
from ..models.prompt_inputs import (
    append_token_to_model_inputs,
    inject_history_inputs_embeds,
    inject_history_token_ids,
    model_forward_inputs,
    move_inputs_to_device,
    prepare_prompt_inputs_with_history,
)


def prepare_alpamayo_prompt_inputs_with_history(
    *,
    model,
    batch: dict,
    processor,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    question: str,
    history_token_count: int,
    device: torch.device,
) -> dict:
    user_text = build_stage1_question_user_text(question, history_token_count)
    message_batch: list[list[dict]] = []
    for image_path in batch["image_path"]:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            frame_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        message_batch.append(build_multimodal_messages(frames=frame_tensor, user_text=user_text))
    prompt_inputs = processor.apply_chat_template(
        message_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = to_device(prompt_inputs, device=device)
    return inject_history_inputs_embeds(
        model=model,
        prompt_inputs=prompt_inputs,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        history_xyz=batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
        history_rot=batch["ego_history_rot"].to(device=device, dtype=torch.float32),
    )


def build_full_inputs_from_prompt_inputs(
    *,
    model,
    prompt_inputs: dict,
    batch: dict,
    registry: Stage1TokenRegistry,
    quantizer,
    task_spec: Stage1TaskSpec,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    target_token_id_rows = task_spec.encode_target_token_rows_from_batch(
        batch,
        registry,
        quantizer,
    )
    target_token_ids = torch.tensor(target_token_id_rows, dtype=torch.long, device=device)

    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attention_mask = prompt_inputs["attention_mask"]
    batch_size = prompt_input_ids.shape[0]
    target_len = target_token_ids.shape[1]

    input_ids = torch.cat([prompt_input_ids, target_token_ids], dim=1)
    attention_mask = torch.cat(
        [
            prompt_attention_mask,
            torch.ones(
                (batch_size, target_len),
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            ),
        ],
        dim=1,
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, -target_len:] = target_token_ids

    full_inputs = {
        key: value
        for key, value in prompt_inputs.items()
        if key not in {"input_ids", "attention_mask", "inputs_embeds"}
    }
    full_inputs["input_ids"] = input_ids
    full_inputs["attention_mask"] = attention_mask
    if "inputs_embeds" in prompt_inputs:
        target_embeds = model.get_input_embeddings()(target_token_ids)
        full_inputs["inputs_embeds"] = torch.cat([prompt_inputs["inputs_embeds"], target_embeds], dim=1)
    if "mm_token_type_ids" in full_inputs:
        full_inputs["mm_token_type_ids"] = torch.cat(
            [
                full_inputs["mm_token_type_ids"],
                torch.zeros(
                    (batch_size, target_len),
                    dtype=full_inputs["mm_token_type_ids"].dtype,
                    device=full_inputs["mm_token_type_ids"].device,
                ),
            ],
            dim=1,
        )
    return full_inputs, labels


def prepare_batch(
    model,
    batch: dict,
    processor,
    registry: Stage1TokenRegistry,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    quantizer,
    task_spec: Stage1TaskSpec,
    prompt_text: str,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    prompt_inputs = prepare_prompt_inputs_with_history(
        model,
        batch,
        processor,
        history_registry,
        history_quantizer,
        prompt_text,
        device,
    )
    return build_full_inputs_from_prompt_inputs(
        model=model,
        prompt_inputs=prompt_inputs,
        batch=batch,
        registry=registry,
        quantizer=quantizer,
        task_spec=task_spec,
        device=device,
    )
