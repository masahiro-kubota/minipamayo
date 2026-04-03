# Stage 1A (`vlm_ce`)

## Role in Alpamayo-R1

This directory implements the repo's Stage 1A VLM policy stage.

The paper does not name a separate "Stage 1A". Instead, the paper presents architectural and training components at a higher level, while this repository splits them into `stage1a`, `stage1b`, `stage2`, and `stage3` for implementation and experimentation.

In this repo, `vlm_ce` is the stage that teaches the VLM side of the reasoning-action architecture: multimodal observation encoding, the VLM backbone, and discrete trajectory-token training.

Source text for all paper-backed citations in this README: [`paper/alpamayo/alpamayo-paper.txt`](../../../../paper/alpamayo/alpamayo-paper.txt).

## Paper Mapping

- Main paper sections:
  - `§3.1 "VLM Backbone: Cosmos-Reason"`
  - `§3.2.1 "Vision Encoding"`
  - `§5.1 "Action Modality Injection"`
- Supporting paper sections:
  - `§6.2 "Policy Improvements from Reasoning"`
  - `§6.7 "Ablation: Efficient Vision Encoding"`

## Paper-Backed Notes

- Paper citation: `§3.1 "VLM Backbone: Cosmos-Reason"`
  The paper says AR1 adopts Cosmos-Reason as the VLM backbone and describes it as a `"VLM specifically designed for Physical AI applications"`. This is the paper-side foundation for the VLM implemented here.

- Paper citation: `§3.2.1 "Vision Encoding"`
  The paper frames vision encoding as a deployment bottleneck and says the encoder must produce `"as few tokens as possible while preserving relevant semantic information"`. That maps to the observation-side VLM setup handled in this stage.

- Paper citation: `§3.2.1 "Vision Encoding"`
  The default AR1 setup in the paper still uses `"single-image tokenization"` as its default tokenizer for the main experiments, even though it also discusses triplane and Flex alternatives. That matches this repo stage being the baseline VLM policy path rather than the continuous action expert.

- Paper citation: `§5.1 "Action Modality Injection"`
  The paper states that during training they `"inject the action modality to the VLM through discrete tokens"` and train the VLM with cross-entropy over the unified token sequence. This is the clearest paper match for why this stage is trained with discrete trajectory-token supervision.

- Paper citation: `§5.1 "Action Modality Injection"`
  The paper also says discrete tokenization enables `"unified autoregressive training in which reasoning and trajectories share a common token space"`. In repo terms, Stage 1A is the VLM side of that shared token-space training setup.

- Paper citation: `§6.2 "Policy Improvements from Reasoning"`
  The paper evaluates a base model `"pre-trained on D_overall with action modality injection (Sec. 5.1)"` before CoC fine-tuning. This repo stage corresponds to that pre-CoC VLM policy foundation.

## Repo Mapping

- `train.py` is the main Stage 1A training entrypoint for the VLM policy.
- `eval.py` and `inference.py` evaluate and sample the Stage 1A model before the later reasoning and RL stages.
- The surrounding Stage 1A runtime and prompting utilities under `stage1/` provide the shared token-contract and multimodal sequence machinery that this stage trains.
- This stage is intentionally separate from the continuous action expert. The paper's flow-matching decoder belongs primarily to the Stage 1B mapping in this repo.

## Related Stages

- Stage 1B action expert: [`../expert_cfm/README.md`](../expert_cfm/README.md)
- Stage 2 reasoning SFT: [`../../stage2/reasoning_sft/README.md`](../../stage2/reasoning_sft/README.md)
