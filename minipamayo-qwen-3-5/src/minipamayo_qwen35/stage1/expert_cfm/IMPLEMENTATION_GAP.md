# Stage 1B Expert CFM Implementation Gap

This document compares the paper-based design in [DESIGN.md](./DESIGN.md) against the current repo implementation.

## Design Baseline

- Baseline document: [DESIGN.md](./DESIGN.md)
- Paper focus: continuous control decoding, flow matching, and unicycle-based trajectory rollout from §3.2.2 and §5.1.
- Expected role: a continuous decoder that consumes VLM context and turns it into physically plausible controls and trajectories.

## Current Implementation Snapshot

- The trainer is [train.py](./train.py), which freezes a Stage 1A checkpoint and trains a dedicated action-expert.
- Conditioning is assembled through [../stage1a_conditioning.py](../stage1a_conditioning.py), which loads Stage 1A prompt-state components and extracts prompt KV-cache.
- Flow-matching configuration is defined in [../stage1b_diffusion_adapter.py](../stage1b_diffusion_adapter.py).
- Runtime decoding and optional control overrides live in [runtime.py](./runtime.py), [eval.py](./eval.py), and [inference.py](./inference.py).

## Aligned Areas

- `[aligned]` [train.py](./train.py), [../stage1b_diffusion_adapter.py](../stage1b_diffusion_adapter.py), and [runtime.py](./runtime.py) implement a separate flow-matching action decoder with Euler-style sampling, which matches §5.1.
- `[aligned]` [metadata.py](./metadata.py) records an unicycle acceleration-curvature action-space contract, which matches the control-based design described in §3.2.2.
- `[aligned]` [../stage1a_conditioning.py](../stage1a_conditioning.py) freezes the upstream Stage 1A module before expert training, which is consistent with the paper's stop-gradient boundary between the expert and the VLM cache.

## Divergences from Paper-Based Design

- `[partial]` [../stage1a_conditioning.py](../stage1a_conditioning.py) conditions the expert on Stage 1A prompt cache only, while §5.1 describes conditioning on the VLM context that includes observations and reasoning. In the current repo split, Stage 1B does not yet consume Stage 2 reasoning text as part of its conditioning cache.
- `[partial]` [train.py](./train.py) learns the expert from a frozen Stage 1A checkpoint before the repo's reasoning stage, while the paper presents the expert as part of a single reasoning-capable VLA pipeline.
- `[partial]` [eval.py](./eval.py) and [inference.py](./inference.py) evaluate Stage 1B mainly in open-loop ADE/FDE terms. The paper's motivation in §5.1 and §6.6 emphasizes deployment speed and closed-loop benefit more strongly than the current local evaluation contract does.

## Likely Intentional Repo Adaptations

- [runtime.py](./runtime.py), [eval.py](./eval.py), and [inference.py](./inference.py) add an optional PID override path. The paper does not specify such a controller overlay, so this looks like a repo-only diagnostic or control-sanity tool rather than a contradiction.
- [train.py](./train.py) exposes explicit architectural knobs such as decoder depth and MLP size. The paper fixes the general design direction but does not prescribe this exact configuration surface.

## Open Questions

- Should Stage 1B eventually condition on Stage 2 reasoning outputs so that the conditioning contract matches the paper more literally?
- Is the current standalone Stage 1B evaluation contract sufficient, or should the repo add a closed-loop or latency-focused benchmark to reflect the claims made in §6.6?
