"""Qwen2.5-VL Mini: DINOv2 ViT-B/14 + Qwen2.5-0.5B miniature VLM."""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer, Dinov2Model

# ImageNet normalization (DINOv2 standard)
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VisionEncoder(nn.Module):
    """DINOv2 ViT-B/14 wrapper. Returns patch tokens only (CLS excluded)."""

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.dinov2.config.hidden_size  # 768
        self.freeze()

    def freeze(self):
        self.dinov2.requires_grad_(False)

    def unfreeze(self):
        self.dinov2.requires_grad_(True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) ImageNet-normalized
        Returns:
            patch_tokens: (B, 256, 768)
        """
        output = self.dinov2(pixel_values=pixel_values)
        # last_hidden_state: (B, 1+256, 768) — CLS + patch tokens
        # Exclude CLS token (index 0)
        return output.last_hidden_state[:, 1:, :]


class Adapter(nn.Module):
    """2-layer MLP projector (COMM Table 6: Ratio 4)."""

    def __init__(self, vision_dim: int = 768, llm_dim: int = 896, ratio: int = 4):
        super().__init__()
        hidden_dim = vision_dim * ratio  # 3072
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N, 768)
        Returns:
            projected: (B, N, 896)
        """
        return self.mlp(vision_features)


class QwenVLMini(nn.Module):
    """DINOv2 + Adapter + Qwen2.5-0.5B unified VLM."""

    IGNORE_INDEX = -100

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov2-base",
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        neftune_alpha: float = 0.0,
    ):
        super().__init__()
        self.neftune_alpha = neftune_alpha
        self.vision_encoder = VisionEncoder(vision_model_name)
        self.adapter = Adapter(
            vision_dim=self.vision_encoder.hidden_size,
            llm_dim=896,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    def _embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings via LLM's embedding layer."""
        return self.llm.model.embed_tokens(input_ids)

    def _build_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Build combined visual + text inputs for the LLM.

        Layout: [visual_tokens (256)] [text_tokens (T)]
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Vision: image → DINOv2 → Adapter → (B, 256, 896)
        with torch.no_grad() if not self.vision_encoder.dinov2.training else torch.enable_grad():
            vision_features = self.vision_encoder(pixel_values)
        visual_embeds = self.adapter(vision_features)
        num_visual = visual_embeds.shape[1]  # 256

        # Text: input_ids → embedding → (B, T, 896)
        text_embeds = self._embed_text(input_ids)

        # Concat: (B, 256+T, 896)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        # NEFTune: add uniform noise during training
        if self.training and self.neftune_alpha > 0:
            dims = torch.tensor(
                inputs_embeds.shape[1] * inputs_embeds.shape[2],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            mag = self.neftune_alpha / torch.sqrt(dims)
            inputs_embeds = inputs_embeds + torch.zeros_like(inputs_embeds).uniform_(-mag, mag)

        # Attention mask: [1...1(256), attention_mask(T)]
        visual_mask = torch.ones(B, num_visual, dtype=attention_mask.dtype, device=device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": combined_mask,
        }

        # Labels: [-100...-100(256), labels(T)]
        if labels is not None:
            visual_labels = torch.full(
                (B, num_visual), self.IGNORE_INDEX, dtype=labels.dtype, device=device
            )
            result["labels"] = torch.cat([visual_labels, labels], dim=1)

        return result

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            pixel_values: (B, 3, 224, 224) ImageNet-normalized images
            input_ids: (B, T) text token IDs
            attention_mask: (B, T) text attention mask
            labels: (B, T) labels for loss (-100 for ignored positions)
        Returns:
            CausalLMOutputWithPast with loss (if labels provided) and logits
        """
        inputs = self._build_inputs(pixel_values, input_ids, attention_mask, labels)
        return self.llm(**inputs)

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text given image + prompt.

        Args:
            pixel_values: (B, 3, 224, 224)
            input_ids: (B, T) prompt token IDs
            attention_mask: (B, T) prompt attention mask
            **generate_kwargs: passed to model.generate()
        Returns:
            generated token IDs
        """
        inputs = self._build_inputs(pixel_values, input_ids, attention_mask)
        return self.llm.generate(**inputs, **generate_kwargs)

    def prepare_prompt(self, question: str) -> dict:
        """Tokenize a user question using Qwen2.5 chat template.

        Returns input_ids, attention_mask, and label mask for training.
        """
        # Build chat-formatted text
        messages = [
            {"role": "system", "content": "You are a visual assistant."},
            {"role": "user", "content": question},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def set_stage1(self):
        """Stage 1: Freeze VE + LLM, only Adapter is trainable."""
        self.vision_encoder.freeze()
        self.llm.requires_grad_(False)
        self.adapter.requires_grad_(True)

    def set_stage2(self):
        """Stage 2: Unfreeze all (VE + Adapter + LLM)."""
        self.vision_encoder.unfreeze()
        self.adapter.requires_grad_(True)
        self.llm.requires_grad_(True)
