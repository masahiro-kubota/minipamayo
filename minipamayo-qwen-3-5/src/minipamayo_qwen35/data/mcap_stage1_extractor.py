"""Backward-compatible wrapper for `minipamayo_qwen35.data.extract_stage1`."""

from .extract_stage1 import *  # noqa: F401,F403
from .extract_stage1 import main


if __name__ == "__main__":
    main()
