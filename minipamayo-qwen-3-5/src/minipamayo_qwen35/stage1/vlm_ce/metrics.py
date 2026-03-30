from __future__ import annotations

import torch


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    correct = ((shifted_logits == shifted_labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def require_record_field(record: dict, key: str):
    if key not in record:
        raise RuntimeError(f"Evaluation record is missing canonical field `{key}`: {record!r}")
    return record[key]


def infer_vision_tokens(prompt_inputs: dict) -> tuple[list[int] | None, int | None]:
    if "image_grid_thw" not in prompt_inputs:
        return None, None
    image_grid = prompt_inputs["image_grid_thw"][0].detach().cpu().tolist()
    if len(image_grid) != 3:
        return image_grid, None
    return image_grid, int((image_grid[1] * image_grid[2]) / 4)
