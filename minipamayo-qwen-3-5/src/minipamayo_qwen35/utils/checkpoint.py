"""Shared checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path) -> dict:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint must deserialize to a dict: {checkpoint_path}")
    return checkpoint

