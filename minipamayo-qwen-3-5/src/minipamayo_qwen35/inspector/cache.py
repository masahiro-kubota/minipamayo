"""Small cached file readers used by the local eval inspector."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=512)
def read_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str).resolve()
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=128)
def read_jsonl(path_str: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_str).resolve()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return tuple(rows)


@lru_cache(maxsize=512)
def read_bytes(path_str: str) -> bytes:
    return Path(path_str).resolve().read_bytes()
