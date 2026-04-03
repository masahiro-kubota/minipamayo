"""Adapters that normalize stage-specific artifacts for the inspector."""

from .stage1a import load_stage1a_run
from .stage1b import load_stage1b_run
from .stage2_eval import load_stage2_eval_run
from .stage2_inference import load_stage2_inference_run
from .stage3_eval import load_stage3_eval_run

__all__ = [
    "load_stage1a_run",
    "load_stage1b_run",
    "load_stage2_eval_run",
    "load_stage2_inference_run",
    "load_stage3_eval_run",
]
