"""Qualitative evaluation: generate captions/answers from images."""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image

from qwen_vl_mini.model import IMAGE_TRANSFORM, QwenVLMini

EVAL_QUESTIONS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the weather like in this image?",
    "How many people are in this image?",
    "What colors are dominant in this image?",
]


def load_model(checkpoint_path: str | None, stage: int, device: str) -> QwenVLMini:
    """Load model, optionally from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint, or None for random init
        stage: 1 (adapter only) or 2 (full model)
        device: "cuda" or "cpu"
    """
    model = QwenVLMini()

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if stage == 1:
            model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        elif stage == 2:
            model.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
            model.adapter.load_state_dict(ckpt["adapter_state_dict"])
            model.llm.load_state_dict(ckpt["llm_state_dict"])

    model.to(device).eval()
    return model


def generate_answer(
    model: QwenVLMini,
    image_path: str,
    question: str,
    device: str,
    max_new_tokens: int = 200,
) -> str:
    """Generate answer for a single image + question pair."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    prompt = model.prepare_prompt(question)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            pixel_values,
            prompt["input_ids"].to(device),
            prompt["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # When generate() receives inputs_embeds, output contains only generated tokens
    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def select_images(image_dir: str, n: int = 5, seed: int = 42) -> list[str]:
    """Randomly select n images from directory."""
    rng = random.Random(seed)
    images = sorted(Path(image_dir).glob("*.jpg"))
    selected = rng.sample(images, min(n, len(images)))
    return [str(p) for p in selected]


def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (omit for random init baseline)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stage number (1: adapter only, 2: full model)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/coco/val2014",
        help="Directory with evaluation images",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="Number of images to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.checkpoint is not None:
        print(f"Loading Stage {args.stage} checkpoint: {args.checkpoint}")
    else:
        print("Loading model with random init (no checkpoint)")
    model = load_model(args.checkpoint, args.stage, device)
    print(f"Model loaded. Device: {device}")

    images = select_images(args.image_dir, args.n_images, args.seed)
    print(f"Selected {len(images)} images from {args.image_dir}\n")
    print("=" * 80)

    for img_path in images:
        print(f"\nImage: {Path(img_path).name}")
        print("-" * 40)
        for question in EVAL_QUESTIONS:
            answer = generate_answer(model, img_path, question, device, args.max_new_tokens)
            print(f"  Q: {question}")
            print(f"  A: {answer}\n")
        print("=" * 80)


if __name__ == "__main__":
    main()
