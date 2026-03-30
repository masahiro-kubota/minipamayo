"""Steer-only Stage 1A evaluation thin wrapper."""

from __future__ import annotations

from ...contract.task_spec import KappaOnlyStage1Spec
from .eval import main as run_stage1_eval


def main() -> None:
    run_stage1_eval(task_spec=KappaOnlyStage1Spec())


if __name__ == "__main__":
    main()
