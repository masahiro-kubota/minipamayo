"""GRPO (Group Relative Policy Optimization) 実装。

単一 GPU 向けの同期的実装。
Cosmos-Reason1 の完全非同期フレームワークとは異なるが、
アルゴリズムは同一。
"""

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """GRPO ハイパーパラメータ。"""

    num_rollouts: int = 4  # K: 質問あたりのロールアウト数
    max_new_tokens: int = 256  # 生成最大トークン数
    temperature: float = 0.7  # サンプリング温度
    clip_epsilon: float = 0.2  # PPO clipping
    kl_coeff: float = 0.005  # KL 正則化係数
    lr: float = 4e-6  # 学習率
    mu: int = 4  # μ: rollout データに対する最適化ステップ数


def extract_answer(text: str) -> str:
    """<answer>X</answer> から選択肢を抽出。"""
    # Primary: <answer>X</answer> タグ
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback 1: "(X)" 形式 (例: "(A) The vehicle...")
    match = re.search(r"\(([A-D])\)", text)
    if match:
        return match.group(1).upper()
    # Fallback 2: 行頭の単独文字 (例: "A\nThe vehicle...")
    match = re.match(r"\s*([A-D])\s*[\n.)\s]", text)
    if match:
        return match.group(1).upper()
    return ""


def compute_reward(generated_text: str, correct_answer: str) -> float:
    """MCQ の正解判定。正解=1, 不正解=0。"""
    extracted = extract_answer(generated_text)
    return 1.0 if extracted == correct_answer else 0.0


def compute_format_reward(generated_text: str) -> float:
    """<think>...</think><answer>...</answer> 形式ボーナス。"""
    has_think = bool(re.search(r"<think>.*</think>", generated_text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*</answer>", generated_text, re.DOTALL))
    if has_think and has_answer:
        return 0.1
    return 0.0


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """グループ内で advantage を計算。A_i = (R_i - mean) / std"""
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r < 1e-8:
        return torch.zeros_like(rewards)
    return (rewards - mean_r) / (std_r + 1e-8)


def compute_sequence_log_probs(model, pixel_values, prompt_ids, generated_ids):
    """生成トークン列の log probability を計算。

    Args:
        model: QwenVLMini
        pixel_values: (1, 3, 224, 224)
        prompt_ids: (1, T_prompt) — プロンプトの token IDs
        generated_ids: (1, T_gen) — 生成された token IDs

    Returns:
        log_prob: scalar — 生成シーケンス全体の平均 log probability
    """
    # 完全なシーケンス: prompt + generated
    full_ids = torch.cat([prompt_ids, generated_ids], dim=1)
    full_mask = torch.ones_like(full_ids)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            pixel_values=pixel_values,
            input_ids=full_ids,
            attention_mask=full_mask,
        )

    # logits の shape: (1, 256 + T_prompt + T_gen, vocab)
    # visual tokens (256) が先頭に付加されている
    logits = outputs.logits
    num_visual = 256
    prompt_len = prompt_ids.shape[1]
    gen_len = generated_ids.shape[1]

    # 生成部分に対応する logits:
    # position (num_visual + prompt_len - 1) の logits が generated[0] を予測
    # position (num_visual + prompt_len + gen_len - 2) の logits が generated[-1] を予測
    start = num_visual + prompt_len - 1
    end = num_visual + prompt_len + gen_len - 1
    gen_logits = logits[:, start:end, :]  # (1, gen_len, vocab)

    log_probs = F.log_softmax(gen_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (1, gen_len)

    return token_log_probs.mean()


class GRPOTrainer:
    """GRPO の学習ループを管理するクラス。"""

    def __init__(self, policy_model, ref_model, tokenizer, config: GRPOConfig):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.config = config

        for param in self.ref.parameters():
            param.requires_grad = False
        self.ref.eval()

    @torch.no_grad()
    def rollout(self, pixel_values, prompt_ids, attention_mask):
        """K 個の応答を生成し、old policy の log_prob を記録する。

        Returns:
            generated_texts: list[str]
            generated_ids_list: list[Tensor] — 各 (1, T_gen)
            old_log_probs: list[Tensor] — 各 scalar (detached)
        """
        self.policy.eval()
        generated_texts = []
        generated_ids_list = []
        old_log_probs = []

        for _ in range(self.config.num_rollouts):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.policy.generate(
                    pixel_values=pixel_values,
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(text)
            generated_ids_list.append(output_ids)

            # old policy の log_prob を記録 (multi-step 更新用)
            if output_ids.shape[1] > 0:
                old_lp = compute_sequence_log_probs(
                    self.policy, pixel_values, prompt_ids, output_ids
                )
                old_log_probs.append(old_lp.detach())
            else:
                old_log_probs.append(torch.tensor(0.0, device=pixel_values.device))

        self.policy.train()
        return generated_texts, generated_ids_list, old_log_probs

    def compute_loss(self, pixel_values, prompt_ids, generated_ids_list, advantages, old_log_probs):
        """GRPO の policy loss + KL loss を計算。

        PPO clipping ratio は old policy (= ロールアウト時の policy) に対して計算。
        KL regularization は reference policy (= SFT model, frozen) に対して計算。

        multi-step 更新 (μ > 1) では old_log_probs はロールアウト時に記録した値を
        使い回すため、2 ステップ目以降で ratio ≠ 1.0 となり clipping が有効に機能する。

        Args:
            pixel_values: (1, 3, 224, 224)
            prompt_ids: (1, T_prompt)
            generated_ids_list: list[Tensor] — K 個の生成トークン列
            advantages: Tensor — K 個の advantage
            old_log_probs: list[Tensor] — ロールアウト時の log_prob (detached)
        """
        total_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)
        total_kl = 0.0
        total_pg = 0.0
        total_ratio = 0.0
        count = 0

        for i, gen_ids in enumerate(generated_ids_list):
            if gen_ids.shape[1] == 0:
                continue

            # Policy log prob (with grad)
            policy_log_prob = compute_sequence_log_probs(
                self.policy, pixel_values, prompt_ids, gen_ids
            )

            # Old policy log prob (ロールアウト時に記録済み)
            old_log_prob = old_log_probs[i]

            # Reference log prob (KL 正則化用, frozen SFT model)
            with torch.no_grad():
                ref_log_prob = compute_sequence_log_probs(
                    self.ref, pixel_values, prompt_ids, gen_ids
                )

            # KL divergence (against reference)
            kl = policy_log_prob - ref_log_prob
            total_kl += kl.item()

            # PPO-style clipped ratio (against old policy)
            ratio = torch.exp(policy_log_prob - old_log_prob)
            total_ratio += ratio.item()
            adv = advantages[i]
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            )
            pg_loss = -torch.min(ratio * adv, clipped_ratio * adv)
            kl_loss = self.config.kl_coeff * kl

            total_loss = total_loss + pg_loss + kl_loss
            total_pg += pg_loss.item()
            count += 1

        if count == 0:
            return total_loss, {"pg_loss": 0.0, "kl": 0.0, "ratio": 1.0}

        avg_loss = total_loss / count
        stats = {
            "pg_loss": total_pg / count,
            "kl": total_kl / count,
            "ratio": total_ratio / count,
        }
        return avg_loss, stats
