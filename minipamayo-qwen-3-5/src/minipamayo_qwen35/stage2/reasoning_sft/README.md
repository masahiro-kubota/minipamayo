# Stage 2 (`reasoning_sft`)

## Role in Alpamayo-R1

This directory implements the repo's reasoning-SFT stage.

The paper presents structured CoC reasoning and reasoning-action sequence training as part of the overall Alpamayo-R1 training strategy. This repository isolates that work into `stage2`, where the model learns to generate structured reasoning traces on top of the Stage 1 policy foundation.

In repo terms, Stage 2 is where CoC supervision is injected into the model. It also functions as the reasoning-to-action handoff stage in this codebase, even though the paper discusses the joint reasoning-action sequence more abstractly.

Source text for all paper-backed citations in this README: [`paper/alpamayo/alpamayo-paper.txt`](../../../../paper/alpamayo/alpamayo-paper.txt).

## Paper Mapping

- Main paper sections:
  - `§4 "Chain of Causation Dataset: Learning Causally Grounded Reasoning VLAs"`
  - `§5.2 "Eliciting Reasoning"`
- Supporting paper sections:
  - `§6.2 "Policy Improvements from Reasoning"`

## Paper-Backed Notes

- Paper citation: `§4 "Chain of Causation Dataset"`
  The paper argues that reasoning data must be `"closely correlated with the ego trajectory"` and criticizes existing datasets for vague behavior descriptions, superficial reasoning, and causal confusion. This is the paper motivation for the supervised reasoning stage implemented here.

- Paper citation: `§4 "Chain of Causation Dataset"`
  The paper defines structured CoC in terms of driving decision, critical components, and a natural-language reasoning trace. That structure is the conceptual basis for the reasoning supervision consumed by this stage.

- Paper citation: `§5.2 "Eliciting Reasoning"`
  The paper says it leverages the CoC dataset to teach the model `"to generate reasoning traces through imitation"`, with reasoning anchored to `"explicit driving decisions"` and `"critical scene components"`. This is the direct paper description of what this stage is for.

- Paper citation: `§5.2 "Eliciting Reasoning"`
  The SFT objective in the paper maximizes the likelihood of the `"reasoning-action sequence"` in a `"unified autoregressive framework"`. In repo terms, this stage teaches the model to emit reasoning in the same sequence regime used by later action generation.

- Paper citation: `§5.2 "Eliciting Reasoning"`
  The paper also says `"SFT on CoC data already yields measurable improvements"` but remains limited by data bias, limited generalization, weak visual grounding, and reasoning-action inconsistency. That is why this stage exists as a distinct pre-RL step rather than the final alignment stage.

- Paper citation: `§6.2 "Policy Improvements from Reasoning"`
  The paper reports that models trained with CoC reasoning `"generate explicit reasoning outputs alongside trajectory predictions"` and better handle `"challenging scenarios that require multi-step decision making"`. This is the clearest experimental description of the role this stage plays.

## Repo Mapping

- `train.py` is the main Stage 2 reasoning-SFT training entrypoint.
- `preprocess.py` and the reasoning dataset helpers prepare and consume reasoning-supervised data derived from CoC-style annotations.
- `inference.py` and `batch_inference.py` expose the repo's reasoning-to-action handoff behavior, which is how this implementation operationalizes the paper's joint reasoning-action sequence.
- This stage assumes the Stage 1A token contract and Stage 1B decoding path already exist and adds structured reasoning on top of them.

## Related Stages

- Stage 1A VLM policy: [`../../stage1/vlm_ce/README.md`](../../stage1/vlm_ce/README.md)
- Stage 1B action expert: [`../../stage1/expert_cfm/README.md`](../../stage1/expert_cfm/README.md)
- Stage 3 RL post-training: [`../../stage3/post_training/README.md`](../../stage3/post_training/README.md)
