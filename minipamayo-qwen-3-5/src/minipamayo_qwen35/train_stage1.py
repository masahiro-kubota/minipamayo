"""Backward-compatible wrapper for `minipamayo_qwen35.train.stage1`."""

from .train.stage1 import *  # noqa: F401,F403
from .train.stage1 import main


if __name__ == "__main__":
    main()
