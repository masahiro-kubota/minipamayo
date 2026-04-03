# Stage 1B Expert CFM Design

This document is the paper-first design baseline for this implementation stage. See [README.md](./README.md) for the short orientation and [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md) for repo-specific differences from this design. The authoritative paper source for this document is [alpamayo-paper.txt](../../../../paper/alpamayo/alpamayo-paper.txt).

## Purpose

This stage represents the continuous action-decoding side of Alpamayo-R1. In the repo's decomposition, it isolates the action-expert and flow-matching decoder that transform VLM context into physically meaningful control outputs and future trajectories.

## Paper Mapping

- Paper citation: §3.2.2 "Trajectory Decoding". The paper replaces raw waypoint prediction with a control-based action representation governed by unicycle dynamics.
- Paper citation: §5.1 "Action Modality Injection". The paper introduces a separate action-expert that decodes continuous actions with flow matching.
- Paper citation: §6.6 "Ablation: Action Modality Injection". The paper reports that flow-matching continuous decoding improves both quality and inference speed over a discrete autoregressive baseline.

## Intended Inputs and Outputs

- Paper citation: §5.1 "Action Modality Injection". The expert receives VLM-side context, a noisy control sample, and the diffusion-time signal.
- Paper citation: §3.2.2 "Trajectory Decoding". The target output is a continuous control sequence over acceleration and curvature rather than raw waypoints.
- Paper citation: §3.2.2 "Trajectory Decoding". The final motion trajectory is produced by mapping the predicted controls back into future waypoints through the unicycle dynamics.

## Intended Architecture and Data Flow

- Paper citation: §5.1 "Action Modality Injection". The expert is a separate decoder attached to the main VLM rather than a second unrelated policy.
- Paper citation: §5.1 "Action Modality Injection". The expert conditions on the VLM sequence context that includes observations and reasoning.
- Paper citation: §5.1 "Action Modality Injection". The decoder predicts a vector field over noisy controls and is sampled through Euler integration during inference.
- Paper citation: §3.2.2 "Trajectory Decoding". The action representation is designed to preserve physical fidelity while remaining fast enough for real-time decoding.

## Intended Training Objective

- Paper citation: §5.1 "Action Modality Injection". The action-expert is trained with a conditional flow-matching objective.
- Paper citation: §5.1 "Action Modality Injection". The paper uses the Gaussian conditional OT path and trains the expert to match the target vector field.
- Paper citation: §5.1 "Action Modality Injection". Gradients from the action-expert are stopped at the VLM KV-cache so that the continuous decoder does not back-propagate into the VLM weights.

## Intended Inference / Evaluation Behavior

- Paper citation: §5.1 "Action Modality Injection". Inference begins from Gaussian noise and iteratively denoises the control sequence through Euler integration.
- Paper citation: §3.2.2 "Trajectory Decoding". The decoded controls are converted into future waypoints through the unicycle-dynamics rollout.
- Paper citation: §6.6 "Ablation: Action Modality Injection". The design goal is to obtain better closed-loop behavior and faster decoding than a purely autoregressive discrete decoder.

## Repo-Level Decomposition Note

The paper does not define a standalone "Stage 1B." It describes the action-expert as part of the overall action-modality injection strategy. This repo pulls that decoder into `stage1/expert_cfm` so that the continuous expert can be trained and evaluated independently from the VLM-side discrete-token path in `stage1/vlm_ce`.

## Related Stages

- Local overview: [README.md](./README.md)
- Local implementation comparison: [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md)
- Upstream repo stage: [Stage 1A README](../vlm_ce/README.md)
- Upstream paper-based design: [Stage 1A DESIGN.md](../vlm_ce/DESIGN.md)
- Downstream reasoning stage: [Stage 2 README](../../stage2/reasoning_sft/README.md)
