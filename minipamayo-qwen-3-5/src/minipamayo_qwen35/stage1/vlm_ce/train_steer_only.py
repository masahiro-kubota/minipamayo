"""Steer-only Stage 1A training thin wrapper."""

from __future__ import annotations

from ...contract.task_spec import KappaOnlyStage1Spec
from .train import main as run_stage1_training


def main() -> None:
    run_stage1_training(task_spec=KappaOnlyStage1Spec())


if __name__ == "__main__":
    main()
