"""Streamlit app for inspecting local VLM checkpoints on uploaded images.

Usage:
    python -m streamlit run shared_checkpoints/vlm_gui_app.py
"""

from __future__ import annotations

import gc
import io
import logging
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
QWEN_SRC = ROOT / "qwen-vl-mini" / "src"
COSMOS_SRC = ROOT / "cosmos-reason-mini" / "src"
HF_MODEL_DIR = Path(__file__).resolve().parent / "hf_models"


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    loader_kind: str
    label: str


_ACTIVE_MODEL_KEY: str | None = None
_ACTIVE_MODEL = None
_HF_PROCESSOR_CACHE: dict[str, object] = {}


def get_qwen35_acceleration_status() -> dict[str, str | bool]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except Exception:
        chunk_gated_delta_rule = None
        fused_recurrent_gated_delta_rule = None

    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except Exception:
        causal_conv1d_fn = None
        causal_conv1d_update = None

    has_fla = bool(chunk_gated_delta_rule and fused_recurrent_gated_delta_rule)
    has_causal_conv1d = bool(causal_conv1d_fn and causal_conv1d_update)

    if has_fla and has_causal_conv1d:
        mode = "full"
        summary = "flash-linear-attention and causal-conv1d are both active."
    elif has_fla:
        mode = "partial"
        summary = (
            "flash-linear-attention is active. causal-conv1d is unavailable on the current "
            "torch/CUDA stack, so Qwen3.5 still uses torch fallback for the conv step."
        )
    else:
        mode = "none"
        summary = "Qwen3.5 is running without flash-linear-attention acceleration."

    return {
        "mode": mode,
        "has_fla": has_fla,
        "has_causal_conv1d": has_causal_conv1d,
        "summary": summary,
    }


@contextmanager
def suppress_qwen35_partial_fast_path_warning(enabled: bool):
    if not enabled:
        yield
        return

    logger = logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def ensure_repo_src_paths():
    for src_dir in (QWEN_SRC, COSMOS_SRC):
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def discover_checkpoints(directory: Path) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    for path in sorted(directory.glob("*.pt")):
        name = path.name
        if name.startswith("qwen-vl-mini_stage1_"):
            loader_kind = "qwen_stage1"
            label = f"{name}  [Qwen Stage 1]"
        elif name.startswith("qwen-vl-mini_stage2"):
            loader_kind = "qwen_stage2"
            label = f"{name}  [Qwen Stage 2]"
        elif name.startswith("cosmos-reason-mini_"):
            loader_kind = "cosmos"
            label = f"{name}  [Cosmos Reason Mini]"
        else:
            loader_kind = "qwen_stage2"
            label = f"{name}  [Unknown -> Qwen Stage 2]"
        specs.append(CheckpointSpec(path=path, loader_kind=loader_kind, label=label))
    hf_dir = directory / "hf_models"
    if hf_dir.exists():
        for path in sorted(hf_dir.iterdir()):
            if path.is_dir() and (path / "config.json").exists():
                specs.append(
                    CheckpointSpec(
                        path=path,
                        loader_kind="hf_image_text",
                        label=f"{path.name}  [HF Image-Text-to-Text]",
                    )
                )
    return specs


def infer_loader_kind(path: Path) -> str:
    name = path.name
    if path.is_dir() and (path / "config.json").exists():
        return "hf_image_text"
    if name.startswith("qwen-vl-mini_stage1_"):
        return "qwen_stage1"
    if name.startswith("qwen-vl-mini_stage2"):
        return "qwen_stage2"
    if name.startswith("cosmos-reason-mini_"):
        return "cosmos"
    return "qwen_stage2"


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def preprocess_uploaded_image(
    image: Image.Image,
    max_edge: int,
    jpeg_quality: int,
) -> tuple[Image.Image, dict[str, str]]:
    work = image.convert("RGB")
    original_size = work.size
    original_bytes = pil_to_png_bytes(work)

    if max_edge > 0 and max(work.size) > max_edge:
        scale = max_edge / max(work.size)
        resized = (
            max(1, round(work.size[0] * scale)),
            max(1, round(work.size[1] * scale)),
        )
        work = work.resize(resized, Image.Resampling.LANCZOS)

    compressed_bytes = None
    if jpeg_quality < 100:
        buf = io.BytesIO()
        work.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        compressed_bytes = buf.getvalue()
        work = Image.open(io.BytesIO(compressed_bytes)).convert("RGB")

    preview_bytes = compressed_bytes if compressed_bytes is not None else pil_to_png_bytes(work)
    info = {
        "original_resolution": f"{original_size[0]}x{original_size[1]}",
        "processed_resolution": f"{work.size[0]}x{work.size[1]}",
        "original_size": format_bytes(len(original_bytes)),
        "processed_size": format_bytes(len(preview_bytes)),
    }
    return work, info


def unload_active_model():
    global _ACTIVE_MODEL_KEY, _ACTIVE_MODEL
    if _ACTIVE_MODEL is None:
        return
    try:
        model_obj = _ACTIVE_MODEL["model"] if isinstance(_ACTIVE_MODEL, dict) else _ACTIVE_MODEL
        model_obj.to("cpu")
    except Exception:
        pass
    _ACTIVE_MODEL = None
    _ACTIVE_MODEL_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_repo_image_transform():
    ensure_repo_src_paths()
    from qwen_vl_mini.model import IMAGE_TRANSFORM

    return IMAGE_TRANSFORM


def get_hf_processor(model_path: Path):
    cache_key = str(model_path.resolve())
    if cache_key in _HF_PROCESSOR_CACHE:
        return _HF_PROCESSOR_CACHE[cache_key]

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    _HF_PROCESSOR_CACHE[cache_key] = processor
    return processor


def build_model_input_preview(
    image: Image.Image,
    loader_kind: str,
    selected_path: Path,
) -> tuple[Image.Image, str]:
    if loader_kind in {"qwen_stage1", "qwen_stage2", "cosmos"}:
        preview = image.resize((224, 224), Image.Resampling.LANCZOS)
        return preview, "224x224 after repo IMAGE_TRANSFORM resize"

    if loader_kind == "hf_image_text":
        processor = get_hf_processor(selected_path)
        image_processor = processor.image_processor
        processed = image_processor.preprocess([image], return_tensors="pt")
        grid_t, grid_h, grid_w = processed["image_grid_thw"][0].tolist()
        patch_size = int(getattr(image_processor, "patch_size", 16))
        merge_size = int(getattr(image_processor, "merge_size", 1))
        final_h = grid_h * patch_size * merge_size
        final_w = grid_w * patch_size * merge_size
        preview = image.resize((final_w, final_h), Image.Resampling.LANCZOS)
        return preview, (
            f"{final_w}x{final_h} processor raster before patchify "
            f"(grid={grid_t}x{grid_h}x{grid_w}, patch={patch_size}, merge={merge_size})"
        )

    return image.copy(), "Unknown final input format"


def load_repo_model(checkpoint_path: Path, loader_kind: str, device: str):
    ensure_repo_src_paths()

    if loader_kind in {"qwen_stage1", "qwen_stage2"}:
        from qwen_vl_mini.eval_qualitative import load_model as load_qwen_checkpoint

        stage = 1 if loader_kind == "qwen_stage1" else 2
        model = load_qwen_checkpoint(str(checkpoint_path), stage=stage, device=device)
    elif loader_kind == "cosmos":
        from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint

        model = load_vlm_from_checkpoint(str(checkpoint_path), device=device)
    else:
        raise ValueError(f"Unsupported repo loader kind: {loader_kind}")

    model.eval()
    return {"backend": "repo", "model": model}


def load_hf_image_text_model(model_path: Path, device: str):
    try:
        from transformers import AutoModelForImageTextToText
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Hugging Face multimodal classes. "
            "Qwen3.5 requires a recent transformers build. "
            "The official model card recommends installing transformers from main."
        ) from exc

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = get_hf_processor(model_path)
    accel_status = get_qwen35_acceleration_status()
    with suppress_qwen35_partial_fast_path_warning(accel_status["mode"] == "partial"):
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=torch_dtype,
            trust_remote_code=True,
        )
    model = model.to(device).eval()
    return {
        "backend": "hf_image_text",
        "model": model,
        "processor": processor,
        "runtime_status": accel_status,
    }


def load_model(checkpoint_path: Path, loader_kind: str, device: str):
    global _ACTIVE_MODEL_KEY, _ACTIVE_MODEL
    key = f"{loader_kind}:{checkpoint_path}:{device}"
    if _ACTIVE_MODEL is not None and _ACTIVE_MODEL_KEY == key:
        return _ACTIVE_MODEL

    unload_active_model()

    if loader_kind in {"qwen_stage1", "qwen_stage2", "cosmos"}:
        model = load_repo_model(checkpoint_path, loader_kind, device)
    elif loader_kind == "hf_image_text":
        model = load_hf_image_text_model(checkpoint_path, device)
    else:
        raise ValueError(f"Unsupported loader kind: {loader_kind}")

    _ACTIVE_MODEL = model
    _ACTIVE_MODEL_KEY = key
    return model


def generate_answer(
    model,
    image: Image.Image,
    question: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> tuple[str, float]:
    backend = model["backend"]
    model_obj = model["model"]

    if backend == "repo":
        image_transform = get_repo_image_transform()
        pixel_values = image_transform(image).unsqueeze(0).to(device)
        prompt = model_obj.prepare_prompt(question)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        generate_kwargs = {
            "pixel_values": pixel_values,
            "input_ids": prompt["input_ids"].to(device),
            "attention_mask": prompt["attention_mask"].to(device),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            with autocast_ctx:
                output_ids = model_obj.generate(**generate_kwargs)
        elapsed = time.perf_counter() - start
        answer = model_obj.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer, elapsed

    if backend == "hf_image_text":
        processor = model["processor"]
        tokenizer = processor.tokenizer
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        if do_sample:
            generate_kwargs["temperature"] = temperature

        start = time.perf_counter()
        with torch.no_grad():
            with autocast_ctx:
                output_ids = model_obj.generate(**inputs, **generate_kwargs)
        elapsed = time.perf_counter() - start
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_len:]
        answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return answer, elapsed

    raise ValueError(f"Unsupported backend: {backend}")


def main():
    st.set_page_config(page_title="MiniPamayo VLM Inspector", layout="wide")
    st.title("MiniPamayo VLM Inspector")
    st.caption("Upload an image, optionally downscale / recompress it, then run one of the shared VLM checkpoints.")

    checkpoint_dir = Path(
        st.sidebar.text_input("Checkpoint directory", str(Path(__file__).resolve().parent))
    )
    checkpoint_specs = discover_checkpoints(checkpoint_dir) if checkpoint_dir.exists() else []

    device_options = ["auto", "cuda", "cpu"]
    device_choice = st.sidebar.selectbox("Device", device_options, index=0)
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    if device == "cuda" and not torch.cuda.is_available():
        st.sidebar.warning("CUDA was selected, but no GPU is available. Falling back to CPU.")
        device = "cpu"

    selected_path = None
    loader_kind = None
    if checkpoint_specs:
        labels = [spec.label for spec in checkpoint_specs]
        selected_label = st.sidebar.selectbox("Shared checkpoint", labels)
        spec = checkpoint_specs[labels.index(selected_label)]
        selected_path = spec.path
        loader_kind = spec.loader_kind
    else:
        st.sidebar.warning("No .pt checkpoints found in the selected directory.")

    manual_checkpoint = st.sidebar.text_input("Manual checkpoint path (optional)", "")
    if manual_checkpoint.strip():
        selected_path = Path(manual_checkpoint).expanduser().resolve()
        loader_kind = st.sidebar.selectbox(
            "Manual checkpoint type",
            ["qwen_stage1", "qwen_stage2", "cosmos", "hf_image_text"],
            index=["qwen_stage1", "qwen_stage2", "cosmos", "hf_image_text"].index(
                infer_loader_kind(selected_path)
            ),
        )

    st.sidebar.markdown("---")
    max_new_tokens = st.sidebar.slider("Max new tokens", min_value=16, max_value=512, value=160)
    do_sample = st.sidebar.checkbox("Enable sampling", value=False)
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.1,
        disabled=not do_sample,
    )

    st.sidebar.markdown("---")
    max_edge = st.sidebar.number_input(
        "Optional max edge before model",
        min_value=0,
        max_value=4096,
        value=0,
        step=32,
        help="0 disables downscaling. Repo checkpoints still end in 224x224. HF models use their own processor.",
    )
    jpeg_quality = st.sidebar.slider(
        "Optional JPEG recompression quality",
        min_value=10,
        max_value=100,
        value=100,
        step=5,
        help="100 disables JPEG recompression.",
    )

    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    question = st.text_area(
        "Prompt",
        value="Describe this image in detail.",
        height=120,
    )

    if selected_path is None:
        st.info("Select a checkpoint first.")
        return

    if not selected_path.exists():
        st.error(f"Checkpoint not found: {selected_path}")
        return

    if uploaded is None:
        st.info("Upload an image to start.")
        return

    original_image = Image.open(uploaded).convert("RGB")
    processed_image, info = preprocess_uploaded_image(original_image, max_edge, jpeg_quality)

    final_preview, final_preview_info = build_model_input_preview(
        processed_image,
        loader_kind,
        selected_path,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(original_image, use_container_width=True)
        st.caption(f"{info['original_resolution']}  |  {info['original_size']}")
    with col2:
        st.subheader("After Optional Compression")
        st.image(processed_image, use_container_width=True)
        same_note = "  |  unchanged" if processed_image.size == original_image.size else ""
        st.caption(
            f"{info['processed_resolution']}  |  {info['processed_size']}{same_note}"
        )
    with col3:
        st.subheader("Final Model Input Preview")
        st.image(final_preview, use_container_width=True)
        st.caption(final_preview_info)

    run_clicked = st.button("Run inference", type="primary", use_container_width=True)
    if not run_clicked:
        return

    with st.spinner(f"Loading {selected_path.name} on {device}..."):
        model = load_model(selected_path, loader_kind, device)

    runtime_status = model.get("runtime_status")
    if runtime_status is not None and runtime_status["mode"] != "full":
        st.info(f"Qwen3.5 acceleration status: {runtime_status['summary']}")

    with st.spinner("Generating..."):
        answer, elapsed = generate_answer(
            model=model,
            image=processed_image,
            question=question,
            device=device,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    st.subheader("Answer")
    st.write(answer)
    st.caption(
        f"checkpoint={selected_path.name}  |  loader={loader_kind}  |  device={device}  |  latency={elapsed:.2f}s"
    )

    with st.expander("Debug info"):
        st.json(
            {
                "checkpoint": str(selected_path),
                "loader_kind": loader_kind,
                "device": device,
                "original_resolution": info["original_resolution"],
                "processed_resolution": info["processed_resolution"],
                "original_size": info["original_size"],
                "processed_size": info["processed_size"],
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "runtime_status": runtime_status,
            }
        )


if __name__ == "__main__":
    main()
