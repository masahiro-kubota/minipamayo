# Stage-First Refactor Checklist

This document is the working checklist for refactoring `minipamayo_qwen35` into a stage-first layout.

The goal is to make the codebase easier to navigate by aligning:

- source layout
- config layout
- entrypoint names
- stage-specific shared code

around the same top-level axis: `stage1`, `stage2`, `stage3`, `stage4`.

## Goal

The intended end state is:

```text
src/minipamayo_qwen35/
  backbones/
  conditioning/
  runtime/
  dynamics/

  stage1/
    data/
    tokenization/
    task_spec.py
    train/
      canonical.py
      runner.py
      profile.py
      experiments/
        steer_only.py
    eval/
      canonical.py
      runner.py
      experiments/
        steer_only.py

  stage2/
    train/
      canonical.py
    eval/
      canonical.py

  stage3/
    train/
      canonical.py
    eval/
      canonical.py

  stage4/
    train/
      canonical.py
    eval/
      canonical.py

  analysis/
    visualize.py
```

Config should mirror the same stage-first structure:

```text
configs/
  stage1/
    data/
    train/
      canonical/
      experiments/
        steer_only/
    eval/
      canonical/
      experiments/
        steer_only/
  stage2/
    train/
    eval/
  stage3/
    train/
    eval/
  stage4/
    train/
    eval/
```

## Refactor Principles

- [ ] Keep `stage` as the first organizational axis.
- [ ] Treat `train`, `eval`, and `experiments` as stage-local concerns, not top-level concerns.
- [ ] Move stage-specific shared code under the owning stage.
- [ ] Keep only truly cross-stage code at the top level.
- [ ] Avoid backward-compatibility wrappers unless explicitly required.
- [ ] Fail loudly with `RuntimeError` when canonical artifacts are missing or malformed.
- [ ] Keep config execution recordable: config-only entrypoints remain config-only after refactor.
- [ ] Update README and example commands only after the new paths are actually live.

## Current Pain Points To Remove

- [x] Eliminate the split ownership between `train/stage1`, `eval/stage1`, and top-level `stage1/`.
- [x] Eliminate top-level `tokens/` usage for Stage 1-only tokenization logic.
- [ ] Eliminate top-level `sequence/` usage for stage-specific prompt builders.
- [x] Eliminate top-level `data/` usage for Stage 1-specific extraction and dataset loading.
- [ ] Eliminate `utils/` as a catch-all for stage-specific code such as `stage34_dataset.py`.
- [x] Eliminate config layout drift between `src/` and `configs/` for Stage 1.

## Proposed Ownership Rules

### Stage-local code

- [ ] `stage1` owns:
  - extraction and Stage 1 dataset readers
  - action tokenization and quantization
  - Stage 1 task specs
  - Stage 1 prompt builders
  - Stage 1 train/eval/profile entrypoints
- [ ] `stage2` owns:
  - trajectory decoder definition
  - Stage 2 prompt builder
  - Stage 2 train/eval entrypoints
- [ ] `stage3` owns:
  - reasoning prompt builder
  - Stage 3 train/eval entrypoints
  - Stage 3 dataset shaping if it is not reused elsewhere
- [ ] `stage4` owns:
  - RL or post-training loop code
  - Stage 4 train/eval entrypoints

### Cross-stage shared code

- [ ] Keep `backbones/` only for reusable backbone abstractions and implementations.
- [ ] Keep `conditioning/` only for reusable conditioning modules.
- [ ] Replace broad `utils/` with more specific shared buckets:
  - `runtime/` for config loading, preflight checks, metadata helpers
  - `dynamics/` for unicycle / rollout math
  - `analysis/` for visualization and offline inspection tools

## Phase 1: Freeze The Target Layout

- [x] Confirm the final naming convention:
  - `src/minipamayo_qwen35/stageN/train/canonical.py`
  - `src/minipamayo_qwen35/stageN/eval/canonical.py`
  - `configs/stageN/train/canonical/*.json`
  - `configs/stageN/eval/canonical/*.json`
- [x] Confirm experiment naming convention:
  - `src/minipamayo_qwen35/stage1/train/experiments/steer_only.py`
  - `src/minipamayo_qwen35/stage1/eval/experiments/steer_only.py`
  - `configs/stage1/train/experiments/steer_only/*.json`
  - `configs/stage1/eval/experiments/steer_only/*.json`
- [ ] Confirm that old top-level `train/` and `eval/` packages will be removed once migration is complete.
- [x] Confirm that old `configs/train/...`, `configs/eval/...`, and `configs/data/...` roots will be removed once migration is complete.

## Phase 2: Move Stage 1 Fully Under `stage1/`

### Entry points

- [x] Move `src/minipamayo_qwen35/train/stage1/*` into `src/minipamayo_qwen35/stage1/train/*`.
- [x] Move `src/minipamayo_qwen35/eval/stage1/*` into `src/minipamayo_qwen35/stage1/eval/*`.
- [x] Move Stage 1 profiling under `src/minipamayo_qwen35/stage1/train/profile.py`.
- [x] Ensure canonical entrypoints remain:
  - `python -m minipamayo_qwen35.stage1.train`
  - `python -m minipamayo_qwen35.stage1.eval`
- [x] Ensure experiment entrypoints remain:
  - `python -m minipamayo_qwen35.stage1.train.experiments.steer_only`
  - `python -m minipamayo_qwen35.stage1.eval.experiments.steer_only`

### Data and tokenization

- [x] Move `src/minipamayo_qwen35/data/stage1.py` to `src/minipamayo_qwen35/stage1/data/extract.py`.
- [x] Move `src/minipamayo_qwen35/data/stage1_dataset.py` to `src/minipamayo_qwen35/stage1/data/dataset.py`.
- [x] Move `src/minipamayo_qwen35/tokens/action_quantizer.py` to `src/minipamayo_qwen35/stage1/tokenization/quantizer.py`.
- [x] Move `src/minipamayo_qwen35/tokens/token_registry.py` to `src/minipamayo_qwen35/stage1/tokenization/registry.py`.
- [x] Update all imports so Stage 1 does not depend on top-level `data/` or `tokens/`.

### Prompt / sequence helpers

- [ ] Move `src/minipamayo_qwen35/sequence/stage1_builder.py` to `src/minipamayo_qwen35/stage1/prompt.py`.
- [ ] Update Stage 1 callers to import the new prompt builder location.

### Verify Stage 1 after move

- [x] `py_compile` Stage 1 modules.
- [x] Run Stage 1 pre-commit hooks on changed files.
- [x] Confirm `--help` works for Stage 1 canonical train/eval entrypoints.
- [x] Confirm `--help` works for Stage 1 steer-only train/eval entrypoints.
- [x] Confirm canonical train config parse still works.
- [x] Confirm steer-only train config parse still works.
- [x] Confirm steer-only `kappa_range` derivation still records deterministic metadata.

## Phase 3: Mirror Configs To Stage-First Layout

### Stage 1 data configs

- [x] Move `configs/data/stage1/*.json` to `configs/stage1/data/*.json`.
- [x] Update README and commands to use `configs/stage1/data/...`.

### Stage 1 train configs

- [x] Move `configs/train/stage1/canonical/*.json` to `configs/stage1/train/canonical/*.json`.
- [x] Move `configs/train/stage1/experiments/steer_only/*.json` to `configs/stage1/train/experiments/steer_only/*.json`.
- [x] Update `save_dir` values to match the desired checkpoint hierarchy.
- [x] Decide on the checkpoint hierarchy:
  - `checkpoints/stage1/canonical/<run_name>/`
  - `checkpoints/stage1/experiments/steer_only/<run_name>/`

### Stage 1 eval configs

- [x] Move `configs/eval/stage1/canonical/*.json` to `configs/stage1/eval/canonical/*.json`.
- [x] Move `configs/eval/stage1/experiments/steer_only/*.json` to `configs/stage1/eval/experiments/steer_only/*.json`.
- [x] Update `checkpoint` paths to the new checkpoint hierarchy.
- [x] Update `output_json` paths to a consistent artifact hierarchy.

### Verify Stage 1 configs after move

- [x] Confirm all Stage 1 config paths still resolve correctly with `path_base`.
- [x] Confirm train config parse still works after path move.
- [x] Confirm eval config parse still works after path move.
- [x] Confirm extractor config parse still works after path move.

## Phase 4: Move Stage 2 Fully Under `stage2/`

- [ ] Move Stage 2 train code into `src/minipamayo_qwen35/stage2/train/canonical.py`.
- [ ] Move Stage 2 eval code into `src/minipamayo_qwen35/stage2/eval/canonical.py`.
- [ ] Move Stage 2-specific model code into `src/minipamayo_qwen35/stage2/model.py`.
- [ ] Move Stage 2 prompt helpers into `src/minipamayo_qwen35/stage2/prompt.py` if they are not shared more broadly.
- [ ] Update imports from Stage 2 into Stage 1 using the new stage-first paths.
- [ ] Move Stage 2 configs into:
  - `configs/stage2/train/...`
  - `configs/stage2/eval/...`
- [ ] Verify `python -m minipamayo_qwen35.stage2.train --help`.

## Phase 5: Move Stage 3 Fully Under `stage3/`

- [ ] Move Stage 3 train code into `src/minipamayo_qwen35/stage3/train/canonical.py`.
- [ ] Move Stage 3 eval code into `src/minipamayo_qwen35/stage3/eval/canonical.py`.
- [ ] Move `sequence/stage3_builder.py` into `src/minipamayo_qwen35/stage3/prompt.py`.
- [ ] Move Stage 3-specific dataset shaping into `src/minipamayo_qwen35/stage3/dataset.py`.
- [ ] Decide whether any current `stage34` shared pieces should live in `stage3/` or a more explicit shared reasoning module.
- [ ] Move Stage 3 configs into:
  - `configs/stage3/train/...`
  - `configs/stage3/eval/...`
- [ ] Verify `python -m minipamayo_qwen35.stage3.train --help`.

## Phase 6: Move Stage 4 Fully Under `stage4/`

- [ ] Move Stage 4 train code into `src/minipamayo_qwen35/stage4/train/canonical.py`.
- [ ] Move Stage 4 eval code into `src/minipamayo_qwen35/stage4/eval/canonical.py`.
- [ ] Keep Stage 4-specific reward / rollout helpers close to Stage 4 unless truly shared.
- [ ] Move Stage 4 configs into:
  - `configs/stage4/train/...`
  - `configs/stage4/eval/...`
- [ ] Verify `python -m minipamayo_qwen35.stage4.train --help`.

## Phase 7: Split Top-Level Shared Modules By Meaning

### Runtime helpers

- [ ] Replace top-level `utils/` as the primary import bucket.
- [ ] Move `json_config.py` into `src/minipamayo_qwen35/runtime/config.py`.
- [ ] Move `preflight.py` into `src/minipamayo_qwen35/runtime/preflight.py`.
- [ ] Move `run_metadata.py` into `src/minipamayo_qwen35/runtime/run_metadata.py`.

### Dynamics helpers

- [ ] Move `utils/dynamics.py` into `src/minipamayo_qwen35/dynamics/unicycle.py`.
- [ ] Update rollout and evaluation imports accordingly.

### Analysis helpers

- [ ] Move `visualize.py` into `src/minipamayo_qwen35/analysis/visualize.py`.
- [ ] Update any documented usage paths.

### Verify shared-module split

- [ ] Run `rg` to confirm no imports still point at removed `utils/*` paths.
- [ ] Run `py_compile` after runtime/dynamics split.

## Phase 8: Remove Obsolete Top-Level Packages

- [ ] Remove top-level `train/` package once all stage entrypoints live under `stageN/train`.
- [ ] Remove top-level `eval/` package once all stage entrypoints live under `stageN/eval`.
- [ ] Remove top-level `data/` package once Stage 1 data code has been moved.
- [ ] Remove top-level `tokens/` package once Stage 1 tokenization code has been moved.
- [ ] Remove top-level `sequence/` package once stage-specific prompt builders have been moved.
- [ ] Remove top-level `utils/` package once runtime and dynamics split is complete.

## Phase 9: Update Docs, Examples, And Commands

- [ ] Update `README.md` to show stage-first entrypoints and config paths.
- [ ] Update dataset README if data config paths change.
- [ ] Update any plan docs that reference old `train/stage1.py` or `eval/stage1.py`.
- [ ] Update example commands in docs to use:
  - `minipamayo_qwen35.stage1.train`
  - `minipamayo_qwen35.stage1.eval`
  - `minipamayo_qwen35.stage1.train.experiments.steer_only`
- [ ] Confirm README does not mention removed paths.

## Phase 10: Final Verification

- [ ] `git status --short` is clean after the final refactor commit series.
- [ ] `pre-commit run --all-files` passes for the repository scope that is meant to be enforced.
- [ ] `py_compile` passes on all Python files under `src/minipamayo_qwen35/`.
- [ ] Canonical Stage 1 train config parses and runs preflight.
- [ ] Canonical Stage 1 eval config parses.
- [ ] Steer-only Stage 1 train config parses and records a derived `kappa_range`.
- [ ] Stage 2 train entrypoint parses.
- [ ] Stage 3 train entrypoint parses.
- [ ] Stage 4 train entrypoint parses.
- [ ] Placeholder stage eval entrypoints clearly fail with explicit `RuntimeError` until implemented.

## Suggested Commit Plan

- [ ] `refactor: move qwen35 stage1 code to stage-first layout`
- [ ] `refactor: move qwen35 stage1 configs to stage-first layout`
- [ ] `refactor: move qwen35 stage2 through stage4 entrypoints to stage-first layout`
- [ ] `refactor: split qwen35 runtime and dynamics modules`
- [ ] `docs: update qwen35 stage-first commands and config paths`

## Notes For Execution

- [ ] Do not mix code moves, config moves, and documentation moves into one giant commit unless absolutely necessary.
- [ ] Keep each phase runnable before starting the next phase.
- [ ] After each major move, run a compile pass immediately rather than waiting until the end.
- [ ] Prefer explicit `RuntimeError` over fallback imports or compatibility shims.
- [ ] Keep config-only execution invariant intact through the whole migration.
