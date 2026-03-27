# VLM GUI

`shared_checkpoints/vlm_gui_app.py` is a small Streamlit app for manually probing the shared VLM checkpoints with an arbitrary image.

## What it does

- lets you choose one of the `.pt` files in [shared_checkpoints](/home/masa/minipamayo/shared_checkpoints)
- lets you choose a local Hugging Face VLM directory under [hf_models](/home/masa/minipamayo/shared_checkpoints/hf_models)
- uploads a local image
- optionally downsizes the image before inference
- optionally JPEG-recompresses the image to simulate aggressive compression
- shows three previews: original, post-compression, and final model-input preview
- runs the existing `qwen-vl-mini` / `cosmos-reason-mini` inference path
- can also run a local Hugging Face `Image-Text-to-Text` model such as `Qwen3.5-0.8B`
- shows the generated answer and the preprocessing metadata

## Run

```bash
cd /home/masa/minipamayo/shared_checkpoints/vlm_gui
uv sync
uv run streamlit run app.py
```

Project files:

- [pyproject.toml](/home/masa/minipamayo/shared_checkpoints/vlm_gui/pyproject.toml)
- [README.md](/home/masa/minipamayo/shared_checkpoints/vlm_gui/README.md)
- [app.py](/home/masa/minipamayo/shared_checkpoints/vlm_gui/app.py)

## Notes

- The app only changes the image *before* the model transform. The repo checkpoints still end in the standard `224x224` `IMAGE_TRANSFORM`; Hugging Face VLMs use their own processor.
- `qwen-vl-mini_stage1_*` is adapter-only, so answers are mainly useful as a sanity check.
- The full checkpoints to compare in practice are:
  - `qwen-vl-mini_stage2.1_checkpoint-5247.pt`
  - `cosmos-reason-mini_rl-mini-merged_checkpoint-final.pt`
- For `Qwen/Qwen3.5-0.8B`, the model card says the latest `transformers` is required:
  - https://huggingface.co/Qwen/Qwen3.5-0.8B
