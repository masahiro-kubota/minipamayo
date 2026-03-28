"""Canonical Stage 1 eval entrypoint."""

from ... import CanonicalStage1Spec
from .runner import main as run_stage1_eval


def main() -> None:
    run_stage1_eval(CanonicalStage1Spec())


if __name__ == "__main__":
    main()
