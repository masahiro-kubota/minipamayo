"""Canonical Stage 1B action-expert training entrypoint.

This currently reuses the existing detached-hidden-state decoder training path.
It is exposed under `stage1.expert_cfm` so the public stage naming matches the
Alpamayo paper more closely while the internal conditioning path is upgraded.
"""

from __future__ import annotations

from ....train.stage2.canonical import main as _legacy_stage2_main


def main() -> None:
    _legacy_stage2_main()

