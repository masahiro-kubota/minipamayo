from __future__ import annotations


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
