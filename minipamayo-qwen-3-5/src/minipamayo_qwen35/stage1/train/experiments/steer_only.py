"""Steer-only Stage 1 train entrypoint."""

from ... import KappaOnlyStage1Spec
from ..runner import main as run_stage1_training


def main() -> None:
    run_stage1_training(KappaOnlyStage1Spec())


if __name__ == "__main__":
    main()
