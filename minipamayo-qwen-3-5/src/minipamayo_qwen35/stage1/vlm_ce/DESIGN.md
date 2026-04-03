# Stage 1A VLM CE Design

This document is the paper-first design baseline for this implementation stage. See [README.md](./README.md) for the short orientation and [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md) for repo-specific differences from this design. The authoritative paper source for this document is [alpamayo-paper.txt](../../../../paper/alpamayo/alpamayo-paper.txt).

## Purpose

This stage represents the VLM-side portion of Alpamayo-R1's action-modality pipeline. In the repo's decomposition, it isolates the observation encoder, language backbone, and discrete trajectory-token training path that the paper presents as part of a larger reasoning-and-action architecture.

## Paper Mapping

- Paper citation: §3.1 "VLM Backbone: Cosmos-Reason". The paper uses Cosmos-Reason as the base VLM and positions it as the policy backbone for embodied driving reasoning.
- Paper citation: §3.2.1 "Vision Encoding". The paper treats vision encoding as the front-end that converts driving observations into a compact token stream for the backbone.
- Paper citation: §5.1 "Action Modality Injection". The paper injects the action modality through discrete trajectory tokens inside the VLM training sequence.
- Paper citation: §6.7 "Ablation: Efficient Vision Encoding". The paper keeps single-image tokenization as the default configuration while discussing more efficient alternatives.

## Intended Inputs and Outputs

- Paper citation: §3.2.1 "Vision Encoding". The intended observation input is a multimodal driving scene made of image observations and ego-motion context.
- Paper citation: §5.1 "Action Modality Injection". The intended supervision output for this stage is a discrete control-token sequence derived from the future trajectory.
- Paper citation: §3.2.2 "Trajectory Decoding". The control target is not raw waypoint coordinates; it is the control-based representation built from acceleration and curvature over a fixed horizon.

## Intended Architecture and Data Flow

- Paper citation: §3.1 "VLM Backbone: Cosmos-Reason". A driving observation is encoded into multimodal tokens and processed by the VLM backbone.
- Paper citation: §3.2.1 "Vision Encoding". The default design uses a single-image tokenization path, but the architecture is meant to remain compatible with more efficient multi-camera or video tokenizers.
- Paper citation: §5.1 "Action Modality Injection". The backbone is trained autoregressively so that reasoning-related text tokens and discrete action tokens can share one sequence model.
- Paper citation: §3.2.2 "Trajectory Decoding". The discrete action-token path is the training-side counterpart of the continuous trajectory decoder used later for deployment.

## Intended Training Objective

- Paper citation: §5.1 "Action Modality Injection". The VLM is trained with cross-entropy over the training token sequence after action tokens are appended to the sequence.
- Paper citation: §5.1 "Action Modality Injection". Each future trajectory is represented as 64 control steps with two quantized values per step, yielding 128 discrete trajectory tokens.
- Paper citation: §3.2.2 "Trajectory Decoding". The target control sequence is derived from the future trajectory under the unicycle-dynamics representation rather than learned directly in raw waypoint space.

## Intended Inference / Evaluation Behavior

- Paper citation: §5.1 "Action Modality Injection". The paper explicitly separates training-time discrete action tokens from deployment-time continuous decoding.
- Paper citation: §5.1 "Action Modality Injection". This stage is therefore a training-side policy component, not the paper's final trajectory-decoding endpoint.
- Paper citation: §6.7 "Ablation: Efficient Vision Encoding". Evaluation should remain compatible with the default single-image tokenizer and with later efficient-tokenizer variants.

## Repo-Level Decomposition Note

The paper does not define a standalone "Stage 1A." It presents a VLM with action-modality injection and a downstream continuous decoder inside one overall training strategy. This repo splits that design into `stage1/vlm_ce` for the discrete VLM path and `stage1/expert_cfm` for the continuous decoder so that the two components can be trained, evaluated, and iterated on separately.

## Related Stages

- Local overview: [README.md](./README.md)
- Local implementation comparison: [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md)
- Next stage in the repo split: [Stage 1B README](../expert_cfm/README.md)
- Adjacent paper-based design: [Stage 1B DESIGN.md](../expert_cfm/DESIGN.md)
