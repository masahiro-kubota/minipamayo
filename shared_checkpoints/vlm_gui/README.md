# VLM GUI UV Project

This directory is a dedicated `uv` project for the Streamlit VLM inspector.

## Why this exists

- keeps the GUI dependencies separate from the training repos
- allows newer Hugging Face dependencies for `Qwen3.5-0.8B`
- avoids mixing the app runtime with the repo's ad hoc local Python environment

## Install

```bash
cd /home/masa/minipamayo/shared_checkpoints/vlm_gui
uv sync
```

## Run

```bash
cd /home/masa/minipamayo/shared_checkpoints/vlm_gui
uv run streamlit run app.py
```

## Notes

- `transformers` is pinned to `main` because `Qwen/Qwen3.5-0.8B` requires a very recent multimodal stack.
- The actual GUI implementation lives in [../vlm_gui_app.py](/home/masa/minipamayo/shared_checkpoints/vlm_gui_app.py). `app.py` is only a thin wrapper.
