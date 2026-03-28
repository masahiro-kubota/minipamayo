#!/usr/bin/env python3
"""Validate simple conventional commit messages without scopes."""

from __future__ import annotations

import re
import sys
from pathlib import Path


CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|refactor|docs|chore|test|perf|ci|build|revert): [a-z0-9].+$"
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Expected exactly one commit message file path.", file=sys.stderr)
        return 1

    commit_msg_path = Path(argv[1])
    message = commit_msg_path.read_text(encoding="utf-8").splitlines()
    subject = message[0].strip() if message else ""

    if CONVENTIONAL_COMMIT_RE.fullmatch(subject):
        return 0

    print(
        "Commit subject must use a simple conventional commit prefix like "
        "`feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`, or `chore: ...`.",
        file=sys.stderr,
    )
    print(f"Got: {subject!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
