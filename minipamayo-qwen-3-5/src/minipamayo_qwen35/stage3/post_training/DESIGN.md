# Stage 3 Post-Training Design

This document is the paper-first design baseline for this implementation stage. See [README.md](./README.md) for the short orientation and [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md) for repo-specific differences from this design. The authoritative paper source for this document is [alpamayo-paper.txt](../../../../paper/alpamayo/alpamayo-paper.txt).

## Purpose

This stage represents Alpamayo-R1's RL-based post-training phase. Its role is to refine reasoning quality, tighten reasoning-action consistency, and improve trajectory quality beyond what CoC-supervised SFT can provide on its own.

## Paper Mapping

- Paper citation: §5.3 "RL-based Post-Training". The paper introduces an RL stage that optimizes reasoning quality, reasoning-action consistency, and trajectory quality together.
- Paper citation: §5.3.1 "Post-Training Algorithm". The paper adopts GRPO with grouped rollouts and KL regularization to a reference policy.
- Paper citation: §5.3.2 "Reward Model". The paper defines three reward components: reasoning quality, CoC-action consistency, and low-level trajectory quality.
- Paper citation: §6.3 "Improvements of Reasoning, Consistency, and Safety via RL Post-Training". The paper reports that this stage improves reasoning scores, consistency, and safety-related behavior.

## Intended Inputs and Outputs

- Paper citation: §5.3.1 "Post-Training Algorithm". The stage consumes grouped rollouts sampled from the current policy for each observation.
- Paper citation: §5.3.2 "Reward Model". Each rollout is evaluated with a scalar reward composed from reasoning, consistency, and trajectory-quality signals.
- Paper citation: §5.3.2 "Reward Model". The model outputs remain a reasoning trace plus an action or trajectory prediction, but the training signal now comes from rollout-level rewards instead of teacher forcing.

## Intended Architecture and Data Flow

- Paper citation: §5.3.1 "Post-Training Algorithm". A current policy and a reference policy are both needed so KL regularization can constrain the update.
- Paper citation: §5.3.2 "Reward Model". The reasoning reward is produced by a large reasoning model acting as a critic over predicted reasoning versus ground-truth CoC reasoning.
- Paper citation: §5.3.2 "Reward Model". The consistency reward compares generated reasoning against the predicted motion behavior.
- Paper citation: §5.3.2 "Reward Model". The trajectory-quality reward evaluates the predicted motion in continuous space.

## Intended Training Objective

- Paper citation: §5.3.1 "Post-Training Algorithm". The intended objective is GRPO, using relative advantages within a group of sampled rollouts.
- Paper citation: §5.3.1 "Post-Training Algorithm". The update includes a KL penalty to the reference policy to prevent reward over-optimization.
- Paper citation: §5.3.2 "Reward Model". The reward surface should encourage grounded reasoning, coherent reasoning-action coupling, and safe comfortable motion together.

## Intended Inference / Evaluation Behavior

- Paper citation: §5.3 "RL-based Post-Training". The stage is meant to improve test-time rollout behavior rather than only teacher-forced metrics.
- Paper citation: §6.3 "Improvements of Reasoning, Consistency, and Safety via RL Post-Training". Evaluation should consider reasoning quality, reasoning-action consistency, and motion quality together.
- Paper citation: §5.3.2 "Reward Model". The reasoning outputs are expected to remain interpretable while corresponding actions become more faithful to the stated rationale.

## Repo-Level Decomposition Note

The paper describes post-training as the third training stage of a single policy. This repo places that work under `stage3/post_training` and also adds repo-specific preprocessing and artifact flows around it. Those implementation details are not part of the paper-derived baseline in this document.

## Related Stages

- Local overview: [README.md](./README.md)
- Local implementation comparison: [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md)
- Upstream paper-based design: [Stage 2 DESIGN.md](../../stage2/reasoning_sft/DESIGN.md)
- Upstream orientation: [Stage 2 README](../../stage2/reasoning_sft/README.md)
