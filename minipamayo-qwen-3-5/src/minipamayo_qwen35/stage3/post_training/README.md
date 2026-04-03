# Stage 3 (`post_training`)

## Role in Alpamayo-R1

This directory implements the repo's RL post-training and alignment stage.

The paper presents RL post-training as the stage that refines reasoning quality, enforces reasoning-action consistency, and improves trajectory quality beyond what CoC SFT can achieve. This repository isolates that alignment phase as `stage3`.

As with the earlier stages, the paper does not define a repo-style standalone "Stage 3" module. The repo splits the paper's architectural and training components into `stage1a`, `stage1b`, `stage2`, and `stage3` for implementation and experimentation.

Source text for all paper-backed citations in this README: [`paper/alpamayo/alpamayo-paper.txt`](../../../../paper/alpamayo/alpamayo-paper.txt).

## Paper Mapping

- Main paper sections:
  - `§5.3 "RL-based Post-Training"`
  - `§5.3.1 "Post-Training Algorithm"`
  - `§5.3.2 "Reward Model"`
- Supporting paper sections:
  - `§6.3 "Improvements of Reasoning, Consistency, and Safety via RL Post-Training"`
  - the consistency discussion later in the same experimental section

## Paper-Backed Notes

- Paper citation: `§5.3 "RL-based Post-Training"`
  The paper says RL post-training optimizes three complementary rewards: `"reasoning quality"`, `"reasoning-action consistency"`, and `"trajectory quality"`. Those are the core targets of this stage in the repo as well.

- Paper citation: `§5.3 "RL-based Post-Training"`
  The paper contrasts RL with SFT by saying RL provides `"explicit inference feedback on the model's own rollouts"` and aligns optimization with deployment-time behavior. This is the main reason Stage 3 exists as a separate post-SFT alignment phase.

- Paper citation: `§5.3.1 "Post-Training Algorithm"`
  The paper states `"We adopt GRPO"` and explains that it optimizes `"relative advantages within a group of sampled model rollouts"`. This is the algorithmic center of the repo's post-training implementation.

- Paper citation: `§5.3.2 "Reward Model"`
  The paper says the reward model evaluates `"both what the model reasons and how it acts"`, combining reasoning quality, reasoning-action consistency, and low-level trajectory quality. That is the exact paper-backed decomposition for this stage's reward design.

- Paper citation: `§5.3.2 "Reward Model"`
  The paper describes large reasoning models as critics that evaluate logical soundness, causal alignment, and contextual consistency, using the `"generation-verification gap"` to justify critic-style grading. This maps to the reasoning-quality side of the Stage 3 reward path.

- Paper citation: `§6.3 "Improvements of Reasoning, Consistency, and Safety via RL Post-Training"`
  The paper says SFT alone `"does not guarantee that these traces are causally grounded"` or that actions faithfully reflect the reasoning. This is the paper-side explanation for why Stage 3 must come after Stage 2.

- Paper citation: `§6.3 "Improvements of Reasoning, Consistency, and Safety via RL Post-Training"`
  The paper says the consistency reward is `"crucial for anchoring reasoning to physically realizable behaviors"` and reports a `"45% improvement in the reasoning score"` together with a `"37% increase in reasoning-action consistency"`. That is the clearest paper evidence for the alignment goals of this stage.

## Repo Mapping

- `train.py` is the main Stage 3 post-training entrypoint.
- `rewards.py`, `runtime.py`, and `common.py` implement the reward decomposition, rollout scoring, and GRPO-style optimization logic.
- `dataset.py`, `preprocess.py`, and `disagreement.py` support the repo's curation and weighting flow for post-training samples.
- `eval.py` measures post-training outputs in terms of reward components, rollout quality, and reasoning-action alignment.

## Related Stages

- Stage 2 reasoning SFT: [`../../stage2/reasoning_sft/README.md`](../../stage2/reasoning_sft/README.md)
- Stage 1B action expert: [`../../stage1/expert_cfm/README.md`](../../stage1/expert_cfm/README.md)
