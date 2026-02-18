"""Stage 1 Pre-training Dataset: LLaVA-CC3M-Pretrain-595K."""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# 11 question prompt variants (LLaVA Table 11)
PRETRAIN_PROMPTS = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

IGNORE_INDEX = -100


class PretrainDataset(Dataset):
    """LLaVA-CC3M-Pretrain-595K for Stage 1 Feature Alignment.

    Each sample: image + caption → train Adapter to project DINOv2 features
    into Qwen2.5-0.5B's input space.
    """

    def __init__(self, json_path: str, image_dir: str, tokenizer, transform, max_length: int = 512):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        with open(json_path) as f:
            self.data = json.load(f)

        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # --- Image ---
        image_path = self.image_dir / sample["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Exception):
            # Broken image: return a random other sample
            return self[random.randint(0, len(self) - 1)]
        pixel_values = self.transform(image)

        # --- Text ---
        # Extract caption from conversations
        caption = ""
        for turn in sample["conversations"]:
            if turn["from"] == "gpt":
                caption = turn["value"]
                break

        # Random prompt variant
        prompt = random.choice(PRETRAIN_PROMPTS)

        # Build Qwen2.5 chat template
        # Prompt part: system + user + generation prompt
        messages = [
            {"role": "system", "content": "You are a visual assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Full text: prompt + caption + <|im_end|>
        full_text = prompt_text + caption + "<|im_end|>"

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Truncate
        full_ids = full_ids[: self.max_length]

        # Labels: -100 for prompt, actual IDs for caption
        prompt_len = len(prompt_ids)
        labels = [IGNORE_INDEX] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]

        # Ensure same length
        assert len(labels) == len(full_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PretrainCollator:
    """Pad sequences to same length within a batch."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        pixel_values = torch.stack([s["pixel_values"] for s in batch])

        # Find max length in batch
        max_len = min(max(len(s["input_ids"]) for s in batch), self.max_length)

        input_ids = []
        attention_mask = []
        labels = []

        for s in batch:
            seq_len = len(s["input_ids"])
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids.append(torch.cat([
                    s["input_ids"],
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                ]))
                attention_mask.append(torch.cat([
                    s["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ]))
                labels.append(torch.cat([
                    s["labels"],
                    torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                ]))
            else:
                input_ids.append(s["input_ids"][:max_len])
                attention_mask.append(s["attention_mask"][:max_len])
                labels.append(s["labels"][:max_len])

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
