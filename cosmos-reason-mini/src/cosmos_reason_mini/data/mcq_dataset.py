"""MCQ データセット。RL ロールアウト用。"""

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class MCQDataset(Dataset):
    """MCQ データを読み込み、プロンプト形式に変換する。"""

    def __init__(
        self,
        json_path: str,
        image_root: str,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B",
        max_length: int = 2048,
        shuffle_options: bool = True,
        transform=None,
    ):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_root = Path(image_root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.max_length = max_length
        self.shuffle_options = shuffle_options
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

        # --- MCQ Prompt ---
        options = dict(sample["options"])
        correct = sample["correct"]

        # 選択肢シャッフル
        if self.shuffle_options:
            keys = list(options.keys())
            values = [options[k] for k in keys]
            correct_text = options[correct]
            random.shuffle(values)
            new_options = {k: v for k, v in zip(keys, values, strict=False)}
            new_correct = next(k for k, v in new_options.items() if v == correct_text)
            options = new_options
            correct = new_correct

        options_text = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
        prompt = (
            f"{sample['question']}\n{options_text}\n\n"
            "Please think step by step and answer in the format: "
            "<think> your reasoning </think> <answer> the letter </answer>."
        )

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "correct": correct,
            "id": sample["id"],
        }
