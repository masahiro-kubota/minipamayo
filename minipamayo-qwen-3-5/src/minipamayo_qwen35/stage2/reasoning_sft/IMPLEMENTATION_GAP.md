# Stage 2 Reasoning SFT Implementation Gap

This document compares the paper-based design in [DESIGN.md](./DESIGN.md) against the current repo implementation.

## Design Baseline

- Baseline document: [DESIGN.md](./DESIGN.md)
- Paper focus: CoC-labeled reasoning data and joint reasoning-plus-action SFT from §4 and §5.2.
- Expected role: train one policy that autoregressively models both the reasoning trace and the future action sequence.

## Current Implementation Snapshot

- The main trainer is [train.py](./train.py), which fine-tunes a Stage 1A-derived policy on `reasoning_text` supervision and a handoff-oriented contract.
- Target construction and weighted loss live in [runtime.py](./runtime.py).
- Dataset loading uses [dataset.py](./dataset.py) on top of [../../reasoning/dataset.py](../../reasoning/dataset.py).
- Inference is handled by [inference.py](./inference.py), which generates reasoning and then hands execution to a separate Stage 1B expert.
- Data preparation is handled by [preprocess.py](./preprocess.py), which builds reasoning JSONL files from Stage 1 records.

## Aligned Areas

- `[aligned]` [train.py](./train.py) and [../../reasoning/dataset.py](../../reasoning/dataset.py) explicitly require `reasoning_text`, which matches the paper's emphasis on reasoning-supervised training.
- `[aligned]` [eval.py](./eval.py) emits reasoning-focused evaluation artifacts and per-sample reasoning payloads, which is consistent with the paper's claim that the model should produce explicit reasoning outputs.
- `[aligned]` [train.py](./train.py) keeps Stage 2 as an SFT stage rather than an RL stage, which matches the order described in §5.2.

## Divergences from Paper-Based Design

- `[diverged]` [runtime.py](./runtime.py) builds the teacher-forced target as `reasoning_text + cot_end + traj_future_start + eos`, without discrete action tokens. In §5.2, the paper's objective is the joint likelihood of reasoning and action tokens.
- `[diverged]` [inference.py](./inference.py) requires a separate [../../stage1/expert_cfm/inference.py](../../stage1/expert_cfm/inference.py)-style Stage 1B decoder path and performs a reasoning handoff, while the paper describes one policy that generates the reasoning-action sequence directly.
- `[diverged]` [preprocess.py](./preprocess.py) synthesizes `reasoning_text` deterministically from `command` and `planner_state` through [../../reasoning/synthetic.py](../../reasoning/synthetic.py), whereas §4 describes a CoC dataset built from explicit driving decisions, critical components, and composed reasoning traces.
- `[partial]` [train.py](./train.py) introduces handoff-specific loss weighting and handoff probe metrics. Those are useful repo-level checks, but they are not part of the paper's stated Stage 2 objective in §5.2.

## Likely Intentional Repo Adaptations

- The handoff path in [inference.py](./inference.py) appears to preserve compatibility with the repo's separate Stage 1B continuous decoder while Stage 2 remains reasoning-centric.
- The synthetic reasoning builder in [../../reasoning/synthetic.py](../../reasoning/synthetic.py) appears to provide a low-cost surrogate for CoC-style supervision before the full paper-style labeling workflow is available in the repo.

## Open Questions

- Should Stage 2 be upgraded to emit discrete action tokens directly so that its contract matches Eq. (9) and can feed Stage 3 without an intermediate handoff?
- Should the repo continue to rely on synthetic planner-derived reasoning targets, or should it move toward a closer implementation of the paper's structured CoC data pipeline?
