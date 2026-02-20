"""運転ドメイン QA データセット。

InstructDataset (qwen-vl-mini) と同じトークン化方式を使用する。
差分は画像パス解決のみ(nuScenes 単一ソース)。
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

IGNORE_INDEX = -100


class DrivingQADataset(Dataset):
    """Phase 1 で生成した LLaVA 形式の運転 QA データを読み込む。

    JSON フォーマット:
    [
        {
            "id": "driving_understanding_000001",
            "image": "samples/CAM_FRONT/xxx.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nQuestion?"},
                {"from": "gpt", "value": "Answer."}
            ]
        },
        ...
    ]
    """

    def __init__(
        self,
        json_path: str,
        image_root: str,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 2048,
        transform=None,
    ):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_root = Path(image_root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.max_length = max_length
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _build_messages(self, conversations: list[dict]) -> list[dict]:
        """LLaVA conversations -> Qwen2.5 chat messages に変換。
        InstructDataset._build_messages() と同一ロジック。
        """
        messages = [{"role": "system", "content": "You are a visual assistant."}]
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            content = (
                turn["value"]
                .replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            ).strip()
            if content:
                messages.append({"role": role, "content": content})
        return messages

    def _tokenize_with_labels(self, messages: list[dict]) -> tuple[list[int], list[int]]:
        """メッセージをトークン化し、ラベルマスクを生成。
        InstructDataset._tokenize_with_labels() と同一ロジック。
        assistant の body + footer のみ loss 対象。
        """
        all_ids = []
        all_labels = []
        for msg in messages:
            header = f"<|im_start|>{msg['role']}\n"
            body = msg["content"]
            footer = "<|im_end|>\n"

            header_ids = self.tokenizer.encode(header, add_special_tokens=False)
            body_ids = self.tokenizer.encode(body, add_special_tokens=False)
            footer_ids = self.tokenizer.encode(footer, add_special_tokens=False)
            turn_ids = header_ids + body_ids + footer_ids

            if msg["role"] == "assistant":
                turn_labels = [IGNORE_INDEX] * len(header_ids) + body_ids + footer_ids
            else:
                turn_labels = [IGNORE_INDEX] * len(turn_ids)

            all_ids.extend(turn_ids)
            all_labels.extend(turn_labels)
        return all_ids, all_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # --- Image ---
        image_path = self.image_root / sample["image"]
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, Exception):
            return self[random.randint(0, len(self) - 1)]
        pixel_values = self.transform(image)

        # --- Text (InstructDataset と同一方式) ---
        messages = self._build_messages(sample["conversations"])
        input_ids, labels = self._tokenize_with_labels(messages)

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
