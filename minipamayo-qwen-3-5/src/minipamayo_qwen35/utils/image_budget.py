"""Canonical Qwen image-token budget settings."""

from __future__ import annotations

CANONICAL_IMAGE_MIN_PIXELS = 163840
CANONICAL_IMAGE_MAX_PIXELS = 196608


def validate_canonical_image_budget(image_min_pixels: int, image_max_pixels: int) -> None:
    if image_min_pixels != CANONICAL_IMAGE_MIN_PIXELS:
        raise RuntimeError(
            "`image_min_pixels` must match the canonical fixed value "
            f"{CANONICAL_IMAGE_MIN_PIXELS}."
        )
    if image_max_pixels != CANONICAL_IMAGE_MAX_PIXELS:
        raise RuntimeError(
            "`image_max_pixels` must match the canonical fixed value "
            f"{CANONICAL_IMAGE_MAX_PIXELS}."
        )

