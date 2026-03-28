"""Canonical Stage 3 post-training evaluation placeholder."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical Stage 3 post-training.")
    parser.add_argument("--config-json", type=str, default="")
    return parser


def main() -> None:
    parser = build_parser()
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        parser.parse_args()
        return
    raise RuntimeError("Canonical Stage 3 post-training evaluation is not implemented yet.")
