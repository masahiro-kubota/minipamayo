# Phase 3: Physical AI RL (GRPO) — 具体的実装プラン

## 目的

Cosmos-Reason1 の Physical AI RL（§4 + §5.2 + §7.2）を小規模再現する。
Phase 2 で SFT 済みのモデルに対し、**GRPO（Group Relative Policy Optimization）** で MCQ ベースの強化学習を行い、運転シーンの推論能力をさらに向上させる。

### Cosmos-Reason1 論文との対応

| Cosmos-Reason1 §4+§7.2 | Cosmos Reason Mini Phase 3 | 差分 |
|---|---|---|
| アルゴリズム: GRPO | GRPO | **一致** |
| 報酬: MCQ 正解率（ルールベース） | MCQ 正解率（ルールベース） | **一致** |
| 回答形式: `<think>...<answer>` タグ | `<think>...<answer>` タグ | **一致** |
| RL データ: ~30,304 MCQ（3 カテゴリ） | ~1,000〜2,000 MCQ（運転のみ） | ~15 倍差 |
| ロールアウト: K=9 / 質問 | K=4〜8 / 質問 | VRAM 制約 |
| バッチ: 128 質問 | 4〜8 質問 | VRAM 制約 |
| 完全非同期 RL フレームワーク | 同期的シンプルループ | RTX 4090 単体 |
| 結果: 物理的常識 +5.0%, 直観的物理学 +7.0% | MCQ 正解率の改善を確認 | 規模差 |

### GRPO を選択する理由

1. **Critic モデル不要**: PPO と異なり、別の価値関数ネットワークを訓練する必要がない → VRAM 節約
2. **ルールベース報酬と相性が良い**: MCQ の正解/不正解で明確な報酬を計算可能
3. **Cosmos-Reason1 と同じアルゴリズム**: 設計思想の再現に最適

---

## 前提条件

- Phase 2 完了: SFT チェックポイントが `checkpoints/sft/checkpoint-XXX.pt` に存在
- Phase 1 の eval データ: `data/sft/qa_eval.json` が存在
- GPT-4o API キー（MCQ 変換用）

---

## GRPO アルゴリズム概要

### 処理フロー

```
1. バッチ分の MCQ 質問を選択
2. Phase 1 (Rollout): 各質問に対して K 個の応答をサンプリング
   - old policy の log_prob を記録（multi-step 更新用）
3. 各応答の報酬を計算（正解=1, 不正解=0）
4. グループ内で advantage を計算:
   A_i = (R(o_i) - mean(R)) / std(R)
5. Phase 2 (Multi-step 最適化): μ 回のポリシー更新:
   r(θ) = π_θ(o_i|q) / π_old(o_i|q)  ← old policy (ロールアウト時) に対する ratio
   L = -E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ] + β * KL(π_θ || π_ref)
   ※ μ=1 では ratio ≈ 1.0 で clipping が無効。μ > 1 で ratio ≠ 1.0 となり clipping が有効に機能
6. 1〜5 を繰り返す
```

### 数式

```
# Advantage（グループ相対）
A_i = (R(o_i) - μ_G) / σ_G
  where μ_G = mean(R(o_1), ..., R(o_K))
        σ_G = std(R(o_1), ..., R(o_K))

# Policy loss（PPO-style clipping, multi-step）
# 注意: ratio は old policy（ロールアウト時）に対して計算。ref ではない。
# old_log_prob はロールアウト時に記録し、μ 回の最適化ステップで使い回す。
# μ=1 では ratio ≈ 1.0 で clipping が実質無効。
# μ > 1 の 2 ステップ目以降で ratio ≠ 1.0 となり clipping が有効に機能。
r(θ) = π_θ(o_i|q) / π_old(o_i|q)
L_clip = min(r(θ) * A_i, clip(r(θ), 1-ε, 1+ε) * A_i)

# KL 正則化（reference policy = SFT model に対して計算）
L_KL = β * KL(π_θ || π_ref)

# 総合 loss
L = -E[L_clip] + L_KL
```

---

## プロジェクト構成（Phase 3 で追加するファイル）

```
cosmos-reason-mini/src/cosmos_reason_mini/
├── data/
│   ├── generate_mcq.py          # 新規: SFT QA → MCQ 変換
│   ├── convert_mcq_to_sft.py    # 新規: MCQ → SFT 用 LLaVA 形式変換 (CoT 付き)
│   └── mcq_dataset.py           # 新規: MCQ 用 Dataset
├── grpo.py                      # 新規: GRPO アルゴリズム実装 (multi-step 対応)
├── train_rl.py                  # 新規: RL 学習スクリプト (multi-step GRPO)
└── eval_mcq.py                  # 新規: MCQ 評価スクリプト
```

---

## Step 0: MCQ データ作成

### 目的

Phase 1 の推論 QA（行動予測、因果推論、安全判断）を MCQ 形式に変換する。
Cosmos-Reason1 §5.2 のアプローチに対応。

### MCQ 形式

Cosmos-Reason1 に倣い、4 択 MCQ + `<think>...<answer>` 形式:

```
Question: この状況で ego vehicle が次にとるべき行動は？
(A) 加速して前方の車両を追い越す
(B) 減速して車間距離を確保する
(C) 右車線に車線変更する
(D) 停車して歩行者を待つ

Please think step by step and answer in the format: <think> your reasoning </think> <answer> the letter </answer>.

期待する出力:
<think>
The image shows a pedestrian crossing ahead. The traffic light is yellow.
The ego vehicle should slow down to maintain a safe distance.
</think>
<answer>B</answer>
```

### MCQ 生成プロンプト

#### prompts/mcq_generation.txt

```
Given the following driving scene question and its correct answer, create a multiple choice question with 4 options (A, B, C, D).

Rules:
1. One option must be the correct answer (mark it)
2. The other 3 options must be plausible but incorrect for this specific scene
3. Options should be specific enough that visual reasoning is required to answer
4. Options should cover different action types (accelerate, decelerate, lane change, stop, etc.)
5. Randomize the position of the correct answer

Question: {question}
Correct Answer: {answer}

Output as JSON:
{
  "question": "reformulated question (concise, suitable for MCQ)",
  "options": {
    "A": "option text",
    "B": "option text",
    "C": "option text",
    "D": "option text"
  },
  "correct": "A"
}
```

### 新規ファイル: data/generate_mcq.py

```python
"""SFT の推論 QA を MCQ 形式に変換する。"""
import argparse, json, os, time
from openai import OpenAI
from tqdm import tqdm


def generate_mcq(client, question: str, answer: str, prompt_template: str) -> dict:
    prompt = prompt_template.replace("{question}", question).replace("{answer}", answer)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="qa_train.json (LLaVA format)")
    parser.add_argument("--prompt", default="prompts/mcq_generation.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter_category", default="reasoning",
                        help="推論 QA のみ MCQ に変換")
    args = parser.parse_args()

    client = OpenAI()
    with open(args.prompt) as f:
        prompt_template = f.read().strip()
    with open(args.input) as f:
        data = json.load(f)

    # 推論 QA のみフィルタ
    reasoning_qa = [
        item for item in data
        if args.filter_category in item.get("id", "")
    ]
    print(f"Reasoning QA: {len(reasoning_qa)} / {len(data)} total")

    results = []
    for item in tqdm(reasoning_qa, desc="Generating MCQ"):
        question = item["conversations"][0]["value"].replace("<image>\n", "")
        answer = item["conversations"][1]["value"]
        try:
            mcq = generate_mcq(client, question, answer, prompt_template)
            results.append({
                "id": item["id"].replace("driving_", "mcq_"),
                "image": item["image"],
                "question": mcq["question"],
                "options": mcq["options"],
                "correct": mcq["correct"],
            })
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(0.2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(results)} MCQ")


if __name__ == "__main__":
    main()
```

### 実行

```bash
# Train MCQ: 推論 QA (~3,000) → MCQ (~3,000)
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_mcq \
    --input data/sft/qa_train.json \
    --output data/rl/mcq_train.json

# Eval MCQ: eval の推論 QA → MCQ
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_mcq \
    --input data/sft/qa_eval.json \
    --output data/rl/mcq_eval.json
```

### MCQ データ形式

```json
[
    {
        "id": "mcq_reasoning_000123",
        "image": "samples/CAM_FRONT/n008_xxx.jpg",
        "question": "What should the ego vehicle do next?",
        "options": {
            "A": "Accelerate to overtake the vehicle ahead",
            "B": "Slow down and maintain safe following distance",
            "C": "Change to the right lane",
            "D": "Come to a complete stop"
        },
        "correct": "B"
    }
]
```

### API コスト見積もり

| 項目 | Train | Eval |
|---|---|---|
| MCQ 生成数 | ~3,000 | ~500 |
| コスト (gpt-4o-mini) | ~$2 | ~$0.30 |

---

## Step 1: MCQ Dataset 実装

### 新規ファイル: data/mcq_dataset.py

```python
"""MCQ データセット。RL ロールアウト用。"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MCQDataset(Dataset):
    """MCQ データを読み込み、プロンプト形式に変換する。

    各サンプルは画像 + MCQ 質問を返す。
    選択肢の順序はエポックごとにシャッフル可能（汎化促進）。
    """

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

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
        options = sample["options"]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
        correct = sample["correct"]  # "B"

        # 選択肢シャッフル（Cosmos-Reason1 と同様、汎化促進）
        if self.shuffle_options:
            keys = list(options.keys())
            values = [options[k] for k in keys]
            random.shuffle(values)
            new_options = {k: v for k, v in zip(keys, values)}
            # 正解のキーを更新
            correct_text = options[correct]
            new_correct = [k for k, v in new_options.items() if v == correct_text][0]
            options = new_options
            correct = new_correct

        # プロンプト構築
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
```

---

## Step 2: GRPO 実装

### 新規ファイル: grpo.py

Cosmos-Reason1 §4.1 の GRPO を RTX 4090 向けに簡略化実装。

```python
"""GRPO (Group Relative Policy Optimization) 実装。

単一 GPU 向けの同期的実装。
Cosmos-Reason1 の完全非同期フレームワークとは異なるが、
アルゴリズムは同一。
"""
import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """GRPO ハイパーパラメータ。"""
    num_rollouts: int = 4       # K: 質問あたりのロールアウト数
    max_new_tokens: int = 256   # 生成最大トークン数
    temperature: float = 0.7    # サンプリング温度
    clip_epsilon: float = 0.2   # PPO clipping
    kl_coeff: float = 0.005     # KL 正則化係数
    lr: float = 4e-6            # 学習率


def extract_answer(text: str) -> str:
    """<answer>X</answer> から選択肢を抽出。"""
    match = re.search(r"<answer>\s*([A-D])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: 最後の単独の A-D 文字
    match = re.search(r"\b([A-D])\b", text[::-1])
    if match:
        return match.group(1).upper()
    return ""


def compute_reward(generated_text: str, correct_answer: str) -> float:
    """MCQ の正解判定。正解=1, 不正解=0。

    Cosmos-Reason1 と同じルールベース報酬。
    """
    extracted = extract_answer(generated_text)
    return 1.0 if extracted == correct_answer else 0.0


def compute_format_reward(generated_text: str) -> float:
    """出力フォーマットの正確さを報酬化。

    <think>...</think><answer>...</answer> 形式かどうか。
    """
    has_think = bool(re.search(r"<think>.*</think>", generated_text, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*</answer>", generated_text, re.DOTALL))
    if has_think and has_answer:
        return 0.1  # ボーナス
    return 0.0


def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """グループ内で advantage を計算。

    A_i = (R_i - mean(R)) / (std(R) + eps)
    """
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r < 1e-8:
        # 全て同じ報酬（全正解 or 全不正解）→ advantage = 0
        return torch.zeros_like(rewards)
    return (rewards - mean_r) / (std_r + 1e-8)


def compute_sequence_log_probs(model, pixel_values, prompt_ids, generated_ids):
    """生成トークン列の log probability を計算。

    Visual tokens (256) が先頭に付加されるため、offset を考慮する。
    """
    full_ids = torch.cat([prompt_ids, generated_ids], dim=1)
    full_mask = torch.ones_like(full_ids)

    # autocast 必須: DINOv2 (fp32) + LLM (bf16) の dtype 不一致を解決
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            pixel_values=pixel_values,
            input_ids=full_ids,
            attention_mask=full_mask,
        )

    logits = outputs.logits
    num_visual = 256  # visual tokens が先頭に付加される
    prompt_len = prompt_ids.shape[1]
    gen_len = generated_ids.shape[1]

    # 生成部分に対応する logits (visual token offset を考慮)
    start = num_visual + prompt_len - 1
    end = num_visual + prompt_len + gen_len - 1
    gen_logits = logits[:, start:end, :]

    log_probs = F.log_softmax(gen_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=generated_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_log_probs.mean()


class GRPOTrainer:
    """GRPO の学習ループを管理するクラス。"""

    def __init__(self, policy_model, ref_model, tokenizer, config: GRPOConfig):
        self.policy = policy_model
        self.ref = ref_model       # frozen
        self.tokenizer = tokenizer
        self.config = config

        # ref model を完全に freeze
        for param in self.ref.parameters():
            param.requires_grad = False
        self.ref.eval()

    @torch.no_grad()
    def rollout(self, pixel_values, prompt_ids, attention_mask):
        """K 個の応答を生成する。

        Returns:
            generated_texts: list[str] (K 個)
            generated_ids: list[Tensor] (K 個)
        """
        self.policy.eval()
        generated_texts = []
        generated_ids_list = []

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

        self.policy.train()
        return generated_texts, generated_ids_list

    def compute_loss(
        self, pixel_values, prompt_ids, generated_ids_list, advantages
    ):
        """GRPO の policy loss + KL loss を計算。"""
        total_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)
        total_kl = 0.0
        count = 0

        for i, gen_ids in enumerate(generated_ids_list):
            if gen_ids.shape[1] == 0:
                continue

            # Policy log prob (with grad)
            policy_log_prob = compute_sequence_log_probs(
                self.policy, pixel_values, prompt_ids, gen_ids
            )

            # Reference log prob (no grad)
            with torch.no_grad():
                ref_log_prob = compute_sequence_log_probs(
                    self.ref, pixel_values, prompt_ids, gen_ids
                )

            # KL divergence
            kl = (policy_log_prob - ref_log_prob)
            total_kl += kl.item()

            # Policy gradient loss with clipping
            ratio = torch.exp(policy_log_prob - ref_log_prob.detach())
            adv = advantages[i]
            clipped_ratio = torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon,
            )
            pg_loss = -torch.min(ratio * adv, clipped_ratio * adv)
            kl_loss = self.config.kl_coeff * kl

            total_loss = total_loss + pg_loss + kl_loss
            count += 1

        if count == 0:
            return total_loss, {"pg_loss": 0.0, "kl": 0.0}

        avg_loss = total_loss / count
        stats = {
            "pg_loss": avg_loss.item(),
            "kl": total_kl / count,
        }
        return avg_loss, stats
```

### Cosmos-Reason1 との GRPO 実装差分

| 項目 | Cosmos-Reason1 | Cosmos Reason Mini |
|---|---|---|
| フレームワーク | cosmos-rl（完全非同期） | PyTorch 同期ループ |
| ロールアウト | 分散アクターノード | 単一 GPU で逐次生成 |
| 並列度 | 5D 並列（DP, PP, CP, FSDP, TP） | なし |
| Reference policy | 別ノードで推論 | 同一 GPU に frozen で保持 |
| 報酬関数 | single_choice + format | **同一**（single_choice + format） |
| Advantage | グループ相対 | **同一** |
| Clipping | ε=0.2 | **同一** |
| KL 係数 | 0.005 | **同一** |
| Multi-step (μ) | multi-step 最適化 | **同一** (μ=4) |
| ratio 計算 | π_θ / π_old（old policy ベース） | **同一** |

---

## Step 3: MCQ 評価スクリプト

### 新規ファイル: eval_mcq.py

Phase 2 の SFT ベースライン記録と、Phase 3 の RL 後評価の両方で使用。

```python
"""MCQ 評価スクリプト。

SFT モデルまたは RL モデルの MCQ 正解率を計測する。
Cosmos-Reason1 §6 のベンチマーク評価に対応。
"""
import argparse, json, random
import torch
from tqdm import tqdm
from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint
from cosmos_reason_mini.data.mcq_dataset import MCQDataset
from cosmos_reason_mini.grpo import extract_answer

# MCQ プロンプト付きで生成し、正解判定
def evaluate_mcq(model, dataset, device, max_new_tokens=256,
                 temperature=0.1, num_seeds=1) -> dict:
    """MCQ 正解率を計測。"""
    all_results = []

    for seed in range(num_seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        correct = 0
        total = 0

        for idx in tqdm(range(len(dataset)), desc=f"Seed {seed}"):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            prompt = sample["prompt"]
            correct_answer = sample["correct"]

            # model.prepare_prompt() で統一的にトークン化
            prompt_dict = model.prepare_prompt(prompt)
            input_ids = prompt_dict["input_ids"].to(device)
            attention_mask = prompt_dict["attention_mask"].to(device)

            # Generate (autocast 必須)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            generated = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            extracted = extract_answer(generated)

            if extracted == correct_answer:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        all_results.append({"seed": seed, "accuracy": accuracy, "correct": correct, "total": total})

    # 5 シード平均
    avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
    return {
        "avg_accuracy": avg_accuracy,
        "per_seed": all_results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mcq_json", default="data/rl/mcq_eval.json")
    parser.add_argument("--image_root", default="data/nuscenes")
    parser.add_argument("--output", default="results/mcq_eval.json")
    parser.add_argument("--num_seeds", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = load_vlm_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    dataset = MCQDataset(
        args.mcq_json, args.image_root,
        shuffle_options=args.num_seeds > 1,
    )

    results = evaluate_mcq(model, dataset, device, num_seeds=args.num_seeds)
    print(f"MCQ Accuracy ({args.num_seeds}-seed avg): {results['avg_accuracy']*100:.1f}%")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

---

## Step 4: SFT ベースライン記録

RL の効果を測るため、SFT モデルの MCQ 正解率をベースラインとして記録。

```bash
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_mcq \
    --checkpoint checkpoints/sft/checkpoint-XXX.pt \
    --mcq_json data/rl/mcq_eval.json \
    --output results/mcq_sft_baseline.json
```

### 期待値

| 指標 | 値 | 根拠 |
|---|---|---|
| ランダム正解率 | 25.0% | 4 択 MCQ |
| SFT ベースライン | 30〜50% | SFT で推論能力を獲得済み |

**注意**: plan.md §小規模 VLM 論文からの注意事項 §8 に記載の通り、0.5B モデルが SFT 後に 25% を大幅に超えないと RL の学習信号が不足する。SFT ベースラインが 30% 未満の場合、MCQ の難易度を下げることを検討。

---

## Step 5: RL 学習スクリプト

### 新規ファイル: train_rl.py

```python
"""Physical AI RL (GRPO) 学習スクリプト。

Usage:
    cd cosmos-reason-mini && PYTHONUNBUFFERED=1 uv run python -m cosmos_reason_mini.train_rl \
        --sft_checkpoint checkpoints/sft/checkpoint-XXX.pt \
        --mcq_train data/rl/mcq_train.json \
        --mcq_eval data/rl/mcq_eval.json \
        --output_dir checkpoints/rl
"""
import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb

from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint
from cosmos_reason_mini.data.mcq_dataset import MCQDataset
from cosmos_reason_mini.grpo import (
    GRPOConfig, GRPOTrainer,
    compute_reward, compute_format_reward, compute_advantages,
)


DEFAULTS = {
    "sft_checkpoint": "checkpoints/sft/checkpoint-XXX.pt",
    "mcq_train": "data/rl/mcq_train.json",
    "mcq_eval": "data/rl/mcq_eval.json",
    "image_root": "data/nuscenes",
    "output_dir": "checkpoints/rl",
    "num_rollouts": 4,
    "lr": 4e-6,
    "kl_coeff": 0.005,
    "clip_epsilon": 0.2,
    "temperature": 0.7,
    "max_new_tokens": 256,
    "iterations": 200,
    "eval_every": 50,
    "save_every": 50,
    "batch_size": 4,  # 質問数 / iteration
    "logging_steps": 5,
    "no_wandb": False,
}


def main():
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(default), default=default)
    args = parser.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    # --- Policy Model (trainable) ---
    policy = load_vlm_from_checkpoint(args.sft_checkpoint, device=device)

    # RL では LLM のみ trainable（VE + Adapter は frozen）
    for name, param in policy.named_parameters():
        if not name.startswith("llm."):
            param.requires_grad = False
    policy.train()

    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Trainable: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M")

    # --- Reference Model (frozen) ---
    ref = load_vlm_from_checkpoint(args.sft_checkpoint, device=device)
    ref.eval()

    # --- GRPO ---
    config = GRPOConfig(
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        clip_epsilon=args.clip_epsilon,
        kl_coeff=args.kl_coeff,
        lr=args.lr,
    )
    trainer = GRPOTrainer(policy, ref, tokenizer, config)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.lr, betas=(0.9, 0.95),
    )

    # --- Dataset ---
    train_dataset = MCQDataset(args.mcq_train, args.image_root, shuffle_options=True)

    # --- wandb ---
    wandb.init(project="cosmos-reason-mini", name="phase3-rl-grpo", config=vars(args))

    # --- Training Loop ---
    for iteration in range(args.iterations):
        # ランダムに batch_size 個の質問を選択
        indices = random.sample(range(len(train_dataset)), min(args.batch_size, len(train_dataset)))

        iter_reward = 0.0
        iter_correct = 0
        iter_total = 0

        for idx in indices:
            sample = train_dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            correct_answer = sample["correct"]
            prompt = sample["prompt"]

            # model.prepare_prompt() で統一的にトークン化
            prompt_dict = policy.prepare_prompt(prompt)
            input_ids = prompt_dict["input_ids"].to(device)
            attention_mask = prompt_dict["attention_mask"].to(device)

            # 1. Rollout: K 個の応答を生成
            generated_texts, generated_ids_list = trainer.rollout(
                pixel_values, input_ids, attention_mask,
            )

            # 2. 報酬計算
            rewards = []
            for text in generated_texts:
                r = compute_reward(text, correct_answer)
                r += compute_format_reward(text)
                rewards.append(r)
            rewards_tensor = torch.tensor(rewards, device=device)

            # 統計
            iter_reward += sum(rewards)
            iter_correct += sum(1 for r in rewards if r >= 1.0)
            iter_total += len(rewards)

            # 3. Advantage 計算
            advantages = compute_advantages(rewards_tensor)

            # 4. Policy update
            if advantages.abs().sum() > 0:  # 全て同じ報酬ならスキップ
                loss, stats = trainer.compute_loss(
                    pixel_values, input_ids,
                    generated_ids_list, advantages,
                )
                loss.backward()

        # Gradient step（batch_size 個の質問分を蓄積後）
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        avg_reward = iter_reward / max(iter_total, 1)
        accuracy = iter_correct / max(iter_total, 1)
        wandb.log({
            "iteration": iteration,
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "kl": stats.get("kl", 0),
        })

        if iteration % args.logging_steps == 0:
            print(f"Iter {iteration}: reward={avg_reward:.3f}, "
                  f"acc={accuracy*100:.1f}%, kl={stats.get('kl', 0):.4f}")

        # Save
        if (iteration + 1) % args.save_every == 0:
            save_path = Path(args.output_dir) / f"checkpoint-{iteration+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "vision_encoder_state_dict": policy.vision_encoder.state_dict(),
                "adapter_state_dict": policy.adapter.state_dict(),
                "llm_state_dict": policy.llm.state_dict(),
                "iteration": iteration,
            }, save_path)

        # Eval
        if (iteration + 1) % args.eval_every == 0:
            from cosmos_reason_mini.eval_mcq import evaluate_mcq
            eval_dataset = MCQDataset(args.mcq_eval, args.image_root, shuffle_options=False)
            eval_results = evaluate_mcq(policy, tokenizer, eval_dataset, device, num_seeds=1)
            wandb.log({"eval_mcq_accuracy": eval_results["avg_accuracy"]})
            print(f"  Eval MCQ Accuracy: {eval_results['avg_accuracy']*100:.1f}%")
            policy.train()

    # Final save
    save_path = Path(args.output_dir) / f"checkpoint-final.pt"
    torch.save({
        "vision_encoder_state_dict": policy.vision_encoder.state_dict(),
        "adapter_state_dict": policy.adapter.state_dict(),
        "llm_state_dict": policy.llm.state_dict(),
    }, save_path)
    print(f"RL training complete. Final: {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
```

---

## Step 6: RL 学習実行

```bash
cd cosmos-reason-mini && PYTHONUNBUFFERED=1 uv run python -m cosmos_reason_mini.train_rl \
    --sft_checkpoint checkpoints/sft/checkpoint-XXX.pt \
    --mcq_train data/rl/mcq_train.json \
    --mcq_eval data/rl/mcq_eval.json \
    --output_dir checkpoints/rl \
    --num_rollouts 4 \
    --lr 4e-6 \
    --kl_coeff 0.005 \
    --iterations 200 \
    --batch_size 4 \
    --eval_every 50 \
    --save_every 50
```

### 推定

| 項目 | 値 |
|---|---|
| MCQ データ | ~3,000 (train), ~500 (eval) |
| イテレーション | 200 |
| 質問/iter | 4 |
| ロールアウト/質問 | 4 (K=4) |
| μ（最適化ステップ/iter） | 4 |
| 生成/iter | 16 (4×4) |
| 推定時間 | ~3〜5 時間 |
| VRAM | ~11-12 GB（policy + ref model） |

### VRAM 見積もり

| コンポーネント | メモリ |
|---|---|
| Policy model (582M, bf16) | ~1.2 GB |
| Reference model (582M, bf16, frozen) | ~1.2 GB |
| Optimizer states (LLM 494M only, ×12 bytes) | ~5.9 GB |
| Activation + ロールアウトバッファ | ~3 GB |
| **合計** | **~11.3 GB** |

RTX 4090 (24 GB) に十分収まる。~13 GB の余裕。

### 監視項目

1. **MCQ 正解率**: イテレーションとともに上昇すること
2. **KL divergence**: 0.01〜0.1 の範囲に収まること。急増したら `kl_coeff` を上げる
3. **平均報酬**: 上昇トレンドがあること
4. **Eval 正解率**: 50 iter ごとに eval で確認

---

## Step 7: 最終評価

### 7a: MCQ 評価（5 シード平均）

```bash
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_mcq \
    --checkpoint checkpoints/rl/checkpoint-final.pt \
    --mcq_json data/rl/mcq_eval.json \
    --output results/mcq_rl_final.json \
    --num_seeds 5
```

### 7b: 定性的評価

```bash
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.eval_qualitative \
    --checkpoint checkpoints/rl/checkpoint-final.pt \
    --eval_json data/sft/qa_eval.json \
    --num_samples 20 \
    --seed 42
```

### 7c: SFT vs RL 比較

| 指標 | SFT (Phase 2) | RL (Phase 3) | Cosmos-Reason1 改善幅 |
|---|---|---|---|
| MCQ 正解率 (5-seed avg) | XX% | XX% | +5.0% (物理的常識+具現化推論) |
| 定性: 推論の質 | ベースライン | 改善期待 | — |
| 定性: フォーマット遵守 | 不安定 | `<think>...<answer>` 遵守 | — |
| KL divergence | — | <0.1 | — |

### 期待する結果

| 指標 | 目標 |
|---|---|
| MCQ 正解率 | SFT ベースラインより **+3〜10%** 改善 |
| `<answer>` タグの遵守率 | **>80%** |
| KL | 0.01〜0.1 |
| 定性的な推論品質 | SFT と同等以上 |

---

## Step 8: MiniPamayo への引き継ぎ

### 8a: 最終チェックポイントの保存

```bash
# RL 後の最終重みを MiniPamayo 用にコピー
cp cosmos-reason-mini/checkpoints/rl/checkpoint-final.pt \
   minipamayo/checkpoints/cosmos-reason-mini.pt
```

### 8b: 引き継ぎ確認

```python
# MiniPamayo の Stage 0 で重みがロードできることを確認
from qwen_vl_mini.model import QwenVLMini
import torch

model = QwenVLMini()
ckpt = torch.load("checkpoints/cosmos-reason-mini.pt", weights_only=True)
model.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
model.adapter.load_state_dict(ckpt["adapter_state_dict"])
model.llm.load_state_dict(ckpt["llm_state_dict"])
print("Cosmos Reason Mini → MiniPamayo 引き継ぎ: OK")
```

### 8c: 引き継ぎる知見

| 知見 | 内容 |
|---|---|
| Adapter 方式 | MLP（Cross-Attention は未検証） |
| 最適 LR (SFT) | 2e-5（LLM+Adapter）/ 1e-5（VE） |
| 最適 LR (RL) | 4e-6 |
| NEFTune | alpha=5 が有効 |
| GRPO 設定 | K=4, ε=0.2, KL=0.005 |
| データ品質 | GPT-4o キャプション + gpt-4o-mini QA が実用的 |
| 過学習リスク | 小データ (< 10K) では save_steps=25 が重要 |

---

## RL が期待通りに進まない場合

| 問題 | 対策 |
|---|---|
| SFT ベースラインが 25% 前後（ランダムレベル） | MCQ の難易度を下げる（選択肢を2択にする等） |
| 正解率が全く上がらない | K を増やす（K=8）、iterations を増やす |
| KL が急増（>1.0） | `kl_coeff` を 0.01 → 0.05 に上げる |
| 全て同じ回答を生成（mode collapse） | `temperature` を上げる（0.9）、KL を強める |
| VRAM 不足 | K=2 に減らす、batch_size=2 に |
| 定性的な品質が劣化 | RL を早期終了し、中間チェックポイントを使用 |

---

## 完了状況

| Step | 状態 | 備考 |
|------|------|------|
| Step 0: MCQ データ作成 | ✅ 完了 | 101 MCQ (Mini) → train/eval split (61/40) |
| Step 1: MCQ Dataset 実装 | ✅ 完了 | mcq_dataset.py |
| Step 2: GRPO 実装 | ✅ 完了 (v3) | grpo.py — ratio バグ修正 + multi-step (μ) 対応 |
| Step 3: MCQ 評価スクリプト | ✅ 完了 | eval_mcq.py |
| Step 4: SFT ベースライン記録 | ✅ 完了 (v3) | **38.3%** (no MCQ) / **92.5%** (MCQ+CoT) ※eval split |
| Step 5: RL 学習スクリプト | ✅ 完了 (v2) | train_rl.py — multi-step GRPO + --mu 引数 |
| Step 6: RL 学習実行 | ✅ 完了 (Mini v3) | 20 iter, batch=2, K=4, train MCQ のみ |
| Step 7: 最終評価 | ✅ 完了 (Mini v3) | **86.7%** (no MCQ) / **93.3%** (MCQ+CoT) ※eval split リークなし、GRPO も train MCQ のみで学習 |
| Step 8: MiniPamayo 引き継ぎ | 未着手 | 本番データ完了後に実施 |

---

## Mini 検証 v1 で発見・修正したバグ

### Bug 1: MCQ の正解が全問 "A" (データ生成バグ)

- **原因**: `prompts/mcq_generation.txt` の JSON 例が `"correct": "A"` → GPT-4o-mini が全 101 問で正解を A に配置
- **影響**: `shuffle_options=False` の評価で全問 correct_answer="A" → モデルが A を出すだけで正解 → 評価結果が信用できない
- **修正**: `generate_mcq.py` にコード側シャッフル (`shuffle_mcq()`) + テキストクリーニング (`clean_option_text()`) を追加。`reshuffle_mcq.py` で既存データを API なしで再シャッフル可能

### Bug 2: GRPO ratio が ref ベース (アルゴリズムバグ)

- **原因**: `ratio = exp(policy_log_prob - ref_log_prob)` だが、PPO の ratio は `exp(policy_log_prob - old_log_prob)` であるべき
- **影響**: iteration が進むと policy が ref から乖離し、ratio が 1 から大きくずれるため、clipping が本来の目的（更新幅の制限）ではなく ref からの乖離制限として動作。KL ペナルティとの二重効果で学習が過度に保守的になる
- **修正**: `old_log_prob = policy_log_prob.detach()` を使用。ratio は old policy に対して計算し、KL は ref に対して計算するよう分離

### Bug 3: extract_answer フォールバックがノイジー

- **原因**: `\b([A-D])\b` が "A bus is located..." の "A" にマッチし得る
- **修正**: Fallback 1: `\(([A-D])\)` (括弧付き), Fallback 2: `^\s*([A-D])\s*[\n.)\s]` (行頭のみ)

### Bug 4: 選択肢テキストに "(correct)" がリーク

- **影響**: 1/101 のみ（軽微）
- **修正**: `clean_option_text()` で "(correct)", "[correct]", "(correct answer)" を除去

---

## Mini パイプライン検証結果 (v2: バグ修正後)

### MCQ データ

| 項目 | 値 |
|------|------|
| 入力 (reasoning QA) | 101 |
| 生成 MCQ | 101 (100% 成功) |
| 正解分布 (シャッフル後) | A:20, B:21, C:29, D:31 |
| API | gpt-4o-mini |

### GRPO 学習設定 (Mini)

| パラメータ | 値 |
|------|------|
| iterations | 20 |
| batch_size | 2 |
| num_rollouts (K) | 4 |
| mu (最適化ステップ) | 4 |
| lr | 4e-6 |
| kl_coeff | 0.005 |
| clip_epsilon | 0.2 |
| temperature | 0.7 |
| trainable params | 494.0M (LLM のみ) |

### 全モデル比較 (v3: リーク修正後、eval split 40 MCQ)

| モデル | MCQ 正解率 | vs ランダム | 備考 |
|------|------|------|------|
| ランダム | 25.0% | — | 4 択 MCQ |
| SFT (no MCQ) | 38.3% | +13.3% | catastrophic forgetting |
| **Qwen2.5-VL Mini (Cosmos 学習前)** | **54.2%** | **+29.2%** | 汎用 VLM のベースライン |
| GRPO (no MCQ in SFT) | 86.7% | +61.7% | SFT 劣化を大幅回復 |
| **SFT (MCQ+CoT)** | **92.5%** | **+67.5%** | MCQ+CoT 混合で forgetting 解消 |
| **GRPO (MCQ+CoT in SFT)** | **93.3%** | **+68.3%** | 最高精度 |

### v1→v2→v3 の比較

| 指標 | v1 (バグあり) | v2 (修正後) | v3 (リーク修正) | 備考 |
|------|------|------|------|------|
| SFT ベースライン (no MCQ) | 64.4% | 32.7% | 38.3% | v3: eval split (40 MCQ) |
| SFT + MCQ+CoT | — | 86.8% ⚠️ | 92.5% | v2 はリーク (train=eval) |
| GRPO (no MCQ) | 86.1% | 69.3% ⚠️ | 86.7% | v2 はリーク (train=eval)、v3 は train MCQ のみで再学習 |
| GRPO (MCQ+CoT) | — | 95.7% ⚠️ | 93.3% | v2 はリーク (train=eval) |

⚠️ v2 の評価データ (101 MCQ) は学習データと同一だったためリークあり。v3 で画像ベースで train (61 MCQ, 30 images) / eval (40 MCQ, 20 images) に分割し修正。

### 分析 (v3: リーク修正後)

1. **SFT (no MCQ) で catastrophic forgetting が発生** (54.2% → 38.3%)
   - 201 QA の少量 SFT で汎用的な回答能力が損なわれた
   - 自由形式の回答を学んだ結果、MCQ フォーマット（選択肢文字の出力）が下手に

2. **MCQ+CoT の SFT 混合で forgetting 完全解消** (54.2% → 92.5%, +38.3pt)
   - SFT データに MCQ+CoT (61問) を混合するだけで、MCQ 正解率が大幅向上
   - リークなしでも 92.5% を達成 → MCQ+CoT 混合は真に効果的

3. **GRPO は両パイプラインで効果あり**
   - no MCQ: 38.3% → 86.7% (+48.4pt) — forgetting を大幅回復
   - MCQ+CoT: 92.5% → 93.3% (+0.8pt) — SFT で既に高精度のため微改善

4. **RL 後の open-ended 生成品質は劣化**
   - 自問自答ループの発生、中国語の混入（Qwen2.5 の事前学習由来）
   - MCQ 特化の副作用。少データでは想定内

### 本番への教訓

- **SFT データに MCQ + CoT を必ず含める**: リークなしでも 92.5% を達成。Cosmos-Reason1 と同じアプローチが 0.5B モデルでも有効
- SFT データは 3,000+ QA にスケールしないと forgetting リスクが高い（MCQ+CoT なしの場合）
- **評価データのリーク防止**: train/eval は画像単位で分割すべき（質問単位だと同一画像で複数 MCQ がある場合にリーク）
- MCQ データ生成時は GPT の JSON 例バイアスに注意（コード側でシャッフルすべき）
- 評価時は正解分布を必ず確認（全部 A のようなバイアスを検出）
- GRPO の ratio は old policy に対して計算（ref ではない）
- **GRPO は multi-step (μ > 1) が必要**: μ=1 では ratio が常に 1.0 で PPO clipping が無効
