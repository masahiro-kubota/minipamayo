# Shared Checkpoints

This directory collects the main checkpoints discussed in this repository so they can be shared with other people without asking them to navigate the full project tree.

Included files:

| File | Size | Role | Recommended use |
| --- | --- | --- | --- |
| `qwen-vl-mini_stage1_checkpoint-2325.pt` | 59 MiB | Stage 1 feature-alignment checkpoint for the DINOv2 + Qwen2.5 VLM | Use only if you want the adapter-only intermediate state |
| `qwen-vl-mini_stage2_checkpoint-2464.pt` | 3.8 GiB | First full VLM checkpoint after visual instruction tuning | Use for historical comparison with Stage 2.1 |
| `qwen-vl-mini_stage2.1_checkpoint-5247.pt` | 3.8 GiB | Main generic VLM checkpoint built from DINOv2 + Qwen2.5-0.5B | Use as the default generic VLM checkpoint |
| `cosmos-reason-mini_rl-mini-merged_checkpoint-final.pt` | 1.3 GiB | Domain-adapted VLM after Cosmos Reason Mini RL | Use as the default checkpoint for MiniPamayo initialization |

Also included:

| File | Purpose |
| --- | --- |
| `SHA256SUMS.txt` | Integrity hashes for all copied `.pt` files |
| `*.README.md` | Per-checkpoint explanation files |

Additional local model directory:

| Path | Purpose |
| --- | --- |
| `hf_models/Qwen3.5-0.8B/` | Local Hugging Face snapshot for `Qwen/Qwen3.5-0.8B` |

Source paths inside the original repository:

| Shared file | Original path |
| --- | --- |
| `qwen-vl-mini_stage1_checkpoint-2325.pt` | `qwen-vl-mini/checkpoints/stage1/checkpoint-2325.pt` |
| `qwen-vl-mini_stage2_checkpoint-2464.pt` | `qwen-vl-mini/checkpoints/stage2/checkpoint-2464.pt` |
| `qwen-vl-mini_stage2.1_checkpoint-5247.pt` | `qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt` |
| `cosmos-reason-mini_rl-mini-merged_checkpoint-final.pt` | `cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt` |

Quick guidance:

- `qwen-vl-mini_stage2.1_checkpoint-5247.pt` is the best single file to share when someone asks for the DINOv2 + Qwen2.5 VLM itself.
- `cosmos-reason-mini_rl-mini-merged_checkpoint-final.pt` is the best single file to share when someone asks for the domain-adapted checkpoint actually used to initialize MiniPamayo.

## GUI inspector

A small browser UI is available at [vlm_gui_app.py](/home/masa/minipamayo/shared_checkpoints/vlm_gui_app.py).

```bash
cd /home/masa/minipamayo/shared_checkpoints/vlm_gui
uv sync
uv run streamlit run app.py
```

See [VLM_GUI.md](/home/masa/minipamayo/shared_checkpoints/VLM_GUI.md) and [shared_checkpoints/vlm_gui](/home/masa/minipamayo/shared_checkpoints/vlm_gui) for details.

The GUI can load both:

- repo-native `.pt` checkpoints from this directory
- local Hugging Face multimodal model directories under `shared_checkpoints/hf_models/`
