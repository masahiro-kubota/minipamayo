"""Stage 2 Visual Instruction Tuning Dataset: LLaVA-Instruct-150K."""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class InstructDataset(Dataset):
    """LLaVA-Instruct-150K for Stage 2 Visual Instruction Tuning.

    Each sample: image + multi-turn QA → train full model (VE + Adapter + LLM)
    for visual question answering and description.
    """

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        tokenizer,
        transform,
        max_length: int = 2048,
    ):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

        with open(json_path) as f:
            self.data = json.load(f)

        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __len__(self) -> int:
        return len(self.data)

    def _build_messages(self, conversations: list[dict]) -> list[dict]:
        """Convert LLaVA conversations to Qwen2.5 chat messages.

        Removes <image> tag from text (visual tokens injected by model separately).
        """
        messages = [{"role": "system", "content": "You are a visual assistant."}]

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = (
                turn["value"]
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            )
            content = content.strip()
            if content:
                messages.append({"role": role, "content": content})

        return messages

    def _tokenize_with_labels(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """Tokenize multi-turn messages with loss mask.

        Only assistant responses have valid labels. System, user, and
        special tokens (<|im_start|>, role, <|im_end|>) are masked with -100.
        """
        all_ids = []
        all_labels = []

        for msg in messages:
            # Build the formatted text for this message:
            # <|im_start|>{role}\n{content}<|im_end|>\n
            header = f"<|im_start|>{msg['role']}\n"
            body = msg["content"]
            footer = "<|im_end|>\n"

            header_ids = self.tokenizer.encode(header, add_special_tokens=False)
            body_ids = self.tokenizer.encode(body, add_special_tokens=False)
            footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)

            turn_ids = header_ids + body_ids + footer_ids

            if msg["role"] == "assistant":
                # Header is prompt (masked), body + footer have loss
                turn_labels = [IGNORE_INDEX] * len(header_ids) + body_ids + footer_ids
            else:
                # System/user: all masked
                turn_labels = [IGNORE_INDEX] * len(turn_ids)

            all_ids.extend(turn_ids)
            all_labels.extend(turn_labels)

        return all_ids, all_labels

    def _resolve_image_path(self, image_name: str) -> Path:
        """Resolve image filename to full path.

        LLaVA-Instruct uses bare IDs like '000000033471.jpg' but
        COCO train2014 files have prefix 'COCO_train2014_000000033471.jpg'.
        Try the bare name first, then try with COCO prefix.
        """
        path = self.image_dir / image_name
        if path.exists():
            return path
        # Try with COCO prefix
        coco_name = f"COCO_train2014_{image_name}"
        coco_path = self.image_dir / coco_name
        if coco_path.exists():
            return coco_path
        return path  # Return original (will fail in Image.open and trigger fallback)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # --- Image ---
        image_path = self._resolve_image_path(sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Exception):
            return self[random.randint(0, len(self) - 1)]
        pixel_values = self.transform(image)

        # --- Text ---
        messages = self._build_messages(sample["conversations"])
        input_ids, labels = self._tokenize_with_labels(messages)

        # Truncate
        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class InstructCollator:
    """Pad sequences to same length within a batch."""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.pad_token_id = tokenizer.pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        pixel_values = torch.stack([s["pixel_values"] for s in batch])

        max_len = min(max(len(s["input_ids"]) for s in batch), self.max_length)

        input_ids = []
        attention_mask = []
        labels = []

        for s in batch:
            seq_len = len(s["input_ids"])
            pad_len = max_len - seq_len

            if pad_len > 0:
                input_ids.append(
                    torch.cat(
                        [
                            s["input_ids"],
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        ]
                    )
                )
                attention_mask.append(
                    torch.cat(
                        [
                            s["attention_mask"],
                            torch.zeros(pad_len, dtype=torch.long),
                        ]
                    )
                )
                labels.append(
                    torch.cat(
                        [
                            s["labels"],
                            torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                        ]
                    )
                )
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
