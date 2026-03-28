"""Backward-compatible wrapper for `minipamayo_qwen35.eval.stage1`."""

from .eval.stage1 import *  # noqa: F401,F403
from .eval.stage1 import main


if __name__ == "__main__":
    main()
