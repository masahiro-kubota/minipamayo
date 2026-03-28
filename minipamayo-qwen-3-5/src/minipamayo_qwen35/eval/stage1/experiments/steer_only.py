"""Steer-only Stage 1 eval entrypoint."""

from ....stage1 import KappaOnlyStage1Spec
from ..runner import main as run_stage1_eval


def main() -> None:
    run_stage1_eval(KappaOnlyStage1Spec())


if __name__ == "__main__":
    main()
