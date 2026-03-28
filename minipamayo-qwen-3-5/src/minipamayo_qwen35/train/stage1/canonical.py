"""Canonical Stage 1 train entrypoint."""

from ...stage1 import CanonicalStage1Spec
from .runner import main as run_stage1_training


def main() -> None:
    run_stage1_training(CanonicalStage1Spec())


if __name__ == "__main__":
    main()
