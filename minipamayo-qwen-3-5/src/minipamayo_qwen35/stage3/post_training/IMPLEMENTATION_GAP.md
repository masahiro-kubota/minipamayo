# Stage 3 Post-Training Implementation Gap

This document compares the paper-based design in [DESIGN.md](./DESIGN.md) against the current repo implementation.

## Design Baseline

- Baseline document: [DESIGN.md](./DESIGN.md)
- Paper focus: GRPO-style RL with three rewards from §5.3, §5.3.1, and §5.3.2.
- Expected role: post-train a reasoning-and-action policy with rollout feedback, an LRM-based reasoning judge, a consistency reward, and a trajectory-quality reward.

## Current Implementation Snapshot

- The trainer is [train.py](./train.py), which samples grouped rollouts, computes a GRPO-style loss, and writes Stage 3 checkpoints.
- Rollout bundle assembly and Stage 2 contract checks live in [bundle.py](./bundle.py) and [common.py](./common.py).
- Reward logic is implemented in [rewards.py](./rewards.py) and structured-text parsing lives in [parser.py](./parser.py).
- Dataset filtering and curation manifest support live in [dataset.py](./dataset.py) and [preprocess.py](./preprocess.py).
- Evaluation is handled by [eval.py](./eval.py).

## Aligned Areas

- `[aligned]` [train.py](./train.py) samples multiple rollouts per input and optimizes rollout-level rewards, which matches the paper's post-training direction from §5.3.
- `[aligned]` [bundle.py](./bundle.py) keeps both a policy model and a frozen reference model, which matches the KL-regularized two-policy structure described in §5.3.1.
- `[aligned]` [rewards.py](./rewards.py) separates reasoning, consistency, and trajectory terms into distinct reward components, which matches the top-level reward decomposition in §5.3.2.

## Divergences from Paper-Based Design

- `[diverged]` [rewards.py](./rewards.py) supports only `disabled` and `exact_match` reasoning reward modes, while §5.3.2 describes an LRM-based reasoning critic with graded feedback over behavior consistency and causal reasoning quality.
- `[partial]` [rewards.py](./rewards.py) computes consistency from a small structured-text parser in [parser.py](./parser.py) plus heuristic trajectory labels, while §5.3.2 describes consistency against meta-actions derived from predicted motion and parsed CoC intent.
- `[diverged]` [rewards.py](./rewards.py) implements trajectory reward from L2 imitation and jerk only; the collision term described in Eq. (11) is absent.
- `[partial]` [common.py](./common.py) uses centered rewards and a mean log-probability difference for its GRPO-style loss, but it does not implement the paper's explicit `beta`-weighted relative-advantage form from Eq. (10).
- `[diverged]` [bundle.py](./bundle.py) requires a Stage 2 checkpoint with a `reason_and_action_tokens` contract and explicitly rejects the current reasoning-handoff Stage 2 checkpoints. The paper assumes Stage 3 continues from the Stage 2 policy rather than from a parallel contract that Stage 2 does not presently emit.

## Likely Intentional Repo Adaptations

- [preprocess.py](./preprocess.py) adds disagreement-score-based curation manifests. The paper-derived baseline in this directory does not specify this preprocessing step, so this looks like a repo-only data-selection convenience.
- [dataset.py](./dataset.py) propagates per-sample weights and disagreement scores. That appears to be infrastructure for experimentation rather than a direct claim from the paper.

## Open Questions

- Should Stage 3 first be brought into contract alignment with the current Stage 2 output format, or should Stage 2 be changed to match the Stage 3 assumption of direct reasoning-and-action token generation?
- Should the repo prioritize adding the LRM judge and collision-aware trajectory reward before refining the GRPO loss shape, since those are the largest paper-to-implementation gaps?
