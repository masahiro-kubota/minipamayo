# Stage 2 Reasoning SFT Design

This document is the paper-first design baseline for this implementation stage. See [README.md](./README.md) for the short orientation and [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md) for repo-specific differences from this design. The authoritative paper source for this document is [alpamayo-paper.txt](../../../../paper/alpamayo/alpamayo-paper.txt).

## Purpose

This stage represents the supervised reasoning phase that teaches Alpamayo-R1 to explain driving behavior with structured, causally grounded chain-of-causation traces while still predicting the corresponding future action sequence.

## Paper Mapping

- Paper citation: §4 "Chain of Causation Dataset: Learning Causally Grounded Reasoning VLAs". The paper defines the CoC dataset and the structured labeling protocol for decision-grounded reasoning.
- Paper citation: §4.1 "Structured Chain of Causation". The paper decomposes each sample into driving decision, critical components, and a composed CoC trace.
- Paper citation: §5.2 "Eliciting Reasoning". The paper uses SFT on CoC data to teach the model to generate reasoning and action jointly.
- Paper citation: §6.2 "Policy Improvements from Reasoning". The paper reports that explicit CoC reasoning improves open-loop and closed-loop behavior over action-only baselines.

## Intended Inputs and Outputs

- Paper citation: §5.2 "Eliciting Reasoning". Each training sample consists of an observation, a structured CoC reasoning trace, and the corresponding control-based trajectory representation.
- Paper citation: §4.1 "Structured Chain of Causation". The reasoning target is explicitly decision-grounded and causally linked to observable critical components.
- Paper citation: §5.2 "Eliciting Reasoning". The intended model output is a joint reasoning-and-action sequence rather than reasoning alone.

## Intended Architecture and Data Flow

- Paper citation: §4.1 "Structured Chain of Causation". Data preparation starts from CoC-labeled samples that include explicit driving decisions and decision-relevant causal factors.
- Paper citation: §5.2 "Eliciting Reasoning". SFT teaches the policy to generate the reasoning trace and the control-token sequence in one autoregressive framework.
- Paper citation: §5.2 "Eliciting Reasoning". The reasoning trace is expected to stay grounded in visible history and aligned with the action sequence it accompanies.

## Intended Training Objective

- Paper citation: §5.2 "Eliciting Reasoning". The objective is the conditional log-likelihood of the joint sequence `pi_theta(Reason, a | o)`.
- Paper citation: §5.2 "Eliciting Reasoning". Cross-entropy is applied over both reasoning tokens and the discrete trajectory tokens from the action-modality stage.
- Paper citation: §5.2 "Eliciting Reasoning". The purpose of this stage is to scaffold reasoning ability before later RL-based refinement.

## Intended Inference / Evaluation Behavior

- Paper citation: §6.2 "Policy Improvements from Reasoning". During inference, the model should emit explicit reasoning together with trajectory predictions.
- Paper citation: §5.2 "Eliciting Reasoning". The design expectation is that the generated reasoning remains causally grounded but still requires later RL to improve visual grounding and reasoning-action consistency.
- Paper citation: §6.2 "Policy Improvements from Reasoning". Evaluation should measure both trajectory quality and the benefits of reasoning in challenging scenarios.

## Repo-Level Decomposition Note

The paper does not define a standalone "Stage 2" as a separate module boundary. It presents CoC supervision as the second stage in a single reasoning-capable VLA pipeline that still predicts actions jointly. This repo isolates that reasoning-SFT work under `stage2/reasoning_sft`, while later repo stages and helpers decide how reasoning is handed off to downstream action generation.

## Related Stages

- Local overview: [README.md](./README.md)
- Local implementation comparison: [IMPLEMENTATION_GAP.md](./IMPLEMENTATION_GAP.md)
- Upstream paper-based designs: [Stage 1A DESIGN.md](../../stage1/vlm_ce/DESIGN.md) and [Stage 1B DESIGN.md](../../stage1/expert_cfm/DESIGN.md)
- Downstream stage: [Stage 3 README](../../stage3/post_training/README.md)
