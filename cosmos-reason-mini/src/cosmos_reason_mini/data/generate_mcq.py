"""SFT の推論 QA を MCQ 形式に変換する。"""

import argparse
import json
import os
import random
import re
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

LABELS = ["A", "B", "C", "D"]


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


def clean_option_text(text: str) -> str:
    """選択肢テキストからリーク表現を除去する。"""
    text = re.sub(r"\s*\(correct\)\s*", " ", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*\[correct\]\s*", " ", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*\(correct answer\)\s*", " ", text, flags=re.IGNORECASE).strip()
    return text


def shuffle_mcq(options: dict, correct: str) -> tuple[dict, str]:
    """正解位置をランダムにシャッフルする。

    GPT が常に正解を A に配置する問題を回避するため、
    コード側で選択肢の順序をランダム化する。
    """
    correct_text = options[correct]
    values = list(options.values())
    random.shuffle(values)
    new_options = {label: val for label, val in zip(LABELS, values, strict=True)}
    new_correct = next(k for k, v in new_options.items() if v == correct_text)
    return new_options, new_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="qa.json (LLaVA format)")
    parser.add_argument("--prompt", default="prompts/mcq_generation.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--filter_category",
        default="reasoning",
        help="MCQ に変換するカテゴリ (reasoning or all)",
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    load_dotenv()
    client = OpenAI()
    with open(args.prompt) as f:
        prompt_template = f.read().strip()
    with open(args.input) as f:
        data = json.load(f)

    # カテゴリフィルタ
    if args.filter_category == "all":
        filtered = data
    else:
        filtered = [item for item in data if args.filter_category in item.get("id", "")]
    print(f"Filtered QA: {len(filtered)} / {len(data)} total")

    results = []
    for item in tqdm(filtered, desc="Generating MCQ"):
        question = item["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
        answer = item["conversations"][1]["value"]
        try:
            mcq = generate_mcq(client, question, answer, prompt_template)
            # テキストクリーニング
            options = {k: clean_option_text(v) for k, v in mcq["options"].items()}
            correct = mcq["correct"]
            # 正解位置をランダムにシャッフル
            options, correct = shuffle_mcq(options, correct)
            results.append(
                {
                    "id": item["id"].replace("driving_", "mcq_"),
                    "image": item["image"],
                    "question": mcq["question"],
                    "options": options,
                    "correct": correct,
                }
            )
        except Exception as e:
            print(f"Error for {item['id']}: {e}")
        time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 正解分布の確認
    from collections import Counter

    dist = Counter(r["correct"] for r in results)
    print(f"Generated {len(results)} MCQ, correct distribution: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
