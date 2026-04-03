# Stage 1B (`expert_cfm`)

## Role in Alpamayo-R1

This directory implements the repo's Stage 1B action expert and flow-matching trajectory decoder.

The paper does not define a standalone "Stage 1B" by that name. The paper describes an action-expert / flow-matching trajectory decoder as one component inside the larger Alpamayo-R1 architecture, and this repository isolates that component as its own stage for implementation and experimentation.

In this repo, `expert_cfm` is the continuous action-decoding side of the system: it consumes the VLM-side conditioning and produces physically grounded trajectories via flow matching.

Source text for all paper-backed citations in this README: [`paper/alpamayo/alpamayo-paper.txt`](../../../../paper/alpamayo/alpamayo-paper.txt).

## Paper Mapping

- Main paper sections:
  - `§3.2.2 "Trajectory Decoding"`
  - `§5.1 "Action Modality Injection"`
  - `§6.6 "Ablation: Action Modality Injection"`
- Supporting paper sections:
  - `§3.2 "Domain-Specific Adaptations"`

## Paper-Backed Notes

- Paper citation: `§3.2.2 "Trajectory Decoding"`
  The paper says trajectory decoding must preserve `"fidelity and multi-modality"`, be fast enough for `"real-time inference"`, and integrate into the VLA training pipeline. This repo stage is the concrete decoder side of those requirements.

- Paper citation: `§3.2.2 "Trajectory Decoding"`
  The paper says it adopts an action representation `"governed by unicycle dynamics"` instead of raw waypoint prediction. That maps directly to the control-based action representation used by the expert path in this repo.

- Paper citation: `§3.2.2 "Trajectory Decoding"`
  The paper states that AR1 combines `"discrete trajectory tokens learned within the VLM with an action-expert"` that decodes trajectories into continuous representations `"using a flow matching framework"`. This repo stage is that action-expert component.

- Paper citation: `§5.1 "Action Modality Injection"`
  The paper says it adopts `"a separate action-expert to decode actions via flow matching"` and that the expert consumes the VLM KV-cache plus noisy control inputs. That is the closest paper description of the `expert_cfm` stage in this codebase.

- Paper citation: `§5.1 "Action Modality Injection"`
  The paper further says `"we train the action-expert using a vanilla conditional flow matching loss"`. This is the paper-side justification for the Stage 1B training path in this repo.

- Paper citation: `§6.6 "Ablation: Action Modality Injection"`
  The paper reports that `"flow-matching yields substantial improvements"` in open-loop and closed-loop metrics, while also improving comfort and inference speed. This is the main experimental motivation for keeping the action expert separate from the VLM token path.

## Repo Mapping

- `train.py` is the main Stage 1B training entrypoint for the action expert.
- `runtime.py`, `pid.py`, and the Stage 1B diffusion / expert modules implement the continuous control decoding path used after VLM conditioning.
- `eval.py` and `inference.py` measure how well the expert path converts conditioned policy context into trajectories and control-like outputs.
- This stage depends on the VLM-side contract established by Stage 1A and should be read as the repo split of the paper's continuous trajectory decoding component.

## Related Stages

- Stage 1A VLM policy: [`../vlm_ce/README.md`](../vlm_ce/README.md)
- Stage 2 reasoning SFT: [`../../stage2/reasoning_sft/README.md`](../../stage2/reasoning_sft/README.md)
