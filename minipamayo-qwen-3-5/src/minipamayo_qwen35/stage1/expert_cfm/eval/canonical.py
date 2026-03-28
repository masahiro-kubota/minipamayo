"""Canonical Stage 1B evaluation entrypoint.

This is a naming-aligned wrapper over the current Stage 2 evaluation placeholder.
The implementation remains pending until the expert path is migrated fully.
"""

from __future__ import annotations

import argparse
import sys

from ....eval.stage2.canonical import main as _legacy_stage2_eval_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the canonical Stage 1B expert CFM path.")
    parser.add_argument("--config-json", type=str, default="")
    return parser


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        build_parser().print_help()
        return
    _legacy_stage2_eval_main()
