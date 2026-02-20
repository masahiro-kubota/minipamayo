"""QwenVLMini モデルのロードユーティリティ。"""

import torch

from qwen_vl_mini.model import QwenVLMini


def load_vlm_from_checkpoint(
    checkpoint_path: str,
    neftune_alpha: float = 0.0,
    device: str = "cuda",
) -> QwenVLMini:
    """Stage 2.1 チェックポイントから VLM をロードする。

    チェックポイント形式(train_stage2.py の save_checkpoint):
        {
            "vision_encoder_state_dict": ...,
            "adapter_state_dict": ...,
            "llm_state_dict": ...,
            "optimizer_state_dict": ...,
            "scheduler_state_dict": ...,
            "global_step": int,
            "epoch": int,
        }
    """
    model = QwenVLMini(neftune_alpha=neftune_alpha)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 3 つの state_dict を個別にロード(train_stage2.py と同じ方式)
    model.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
    model.adapter.load_state_dict(ckpt["adapter_state_dict"])
    model.llm.load_state_dict(ckpt["llm_state_dict"])

    model = model.to(device)
    return model
