# Stage 1A VLM CE Implementation Gap

This document compares the paper-based design in [DESIGN.md](./DESIGN.md) against the current repo implementation.

## Design Baseline

- Baseline document: [DESIGN.md](./DESIGN.md)
- Paper focus: VLM backbone, vision encoding, and discrete action-token training from §3.1, §3.2.1, and §5.1.
- Expected role: training-side discrete action modeling that feeds the broader Alpamayo-R1 reasoning-and-action stack.

## Current Implementation Snapshot

- The main trainer is [train.py](./train.py), which fine-tunes a multimodal VLM with canonical Stage 1 datasets and checkpoint outputs.
- Token-contract construction lives in [../stage1a_components.py](../stage1a_components.py), which adds discrete action tokens and history tokens to the tokenizer.
- Standalone evaluation and one-sample inference live in [eval.py](./eval.py) and [inference.py](./inference.py).
- The dataset contract is [../dataset.py](../dataset.py), which currently exposes one `image_path` plus ego-history tensors per sample.

## Aligned Areas

- `[aligned]` [train.py](./train.py) and [../stage1a_components.py](../stage1a_components.py) train the VLM with cross-entropy over a discrete action-token contract, which matches §5.1 at the objective level.
- `[aligned]` [../stage1a_components.py](../stage1a_components.py) stores control quantization and token-registry metadata, which is consistent with the paper's discrete control-token representation from §3.2.2 and §5.1.
- `[aligned]` [train.py](./train.py) keeps the VLM as the central policy backbone, which matches the role described in §3.1.

## Divergences from Paper-Based Design

- `[partial]` [../dataset.py](../dataset.py) models the observation around a single `image_path` plus history tensors, while the paper describes a multi-camera observation stack and discusses multi-camera and video tokenizers in §3.2.1.
- `[diverged]` [inference.py](./inference.py) exposes Stage 1A as a direct discrete-token inference endpoint, while §5.1 states that the paper does not use discrete trajectory tokens for inference.
- `[partial]` [eval.py](./eval.py) turns Stage 1A into a standalone benchmarked trajectory predictor. That is useful for the repo, but the paper treats this component mainly as part of the larger action-modality pipeline rather than as a deployment endpoint on its own.

## Likely Intentional Repo Adaptations

- [inference.py](./inference.py) and [eval.py](./eval.py) make the discrete policy path inspectable before Stage 1B is available. The paper does not require this intermediate visibility, but it is a practical engineering aid.
- [../dataset.py](../dataset.py) narrows the observation interface to the current dataset format. This looks like a staging simplification rather than a claim that the paper's full multi-camera design is unnecessary.

## Open Questions

- Should Stage 1A remain a public standalone inference target, or should it become a training-only artifact once Stage 1B or later stages are the default policy endpoints?
- If the repo wants closer paper alignment, should multi-camera or multi-frame tokenization enter this stage, or should Stage 1A explicitly stay scoped to the paper's default single-image path?
