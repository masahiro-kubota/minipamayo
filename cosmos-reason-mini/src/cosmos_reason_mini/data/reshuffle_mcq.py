"""既存 MCQ JSON の正解位置をシャッフルし、リーク表現を除去する。

API 呼び出し不要。generate_mcq.py で生成済みデータの修正用。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.reshuffle_mcq \
        --input data/rl/mcq_mini.json \
        --output data/rl/mcq_mini.json
"""

import argparse
import json
import random
from collections import Counter

from cosmos_reason_mini.data.generate_mcq import clean_option_text, shuffle_mcq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    print(f"Input: {len(data)} MCQ")

    # 修正前の分布
    before_dist = Counter(d["correct"] for d in data)
    print(f"Before: {dict(sorted(before_dist.items()))}")

    for item in data:
        # テキストクリーニング
        item["options"] = {k: clean_option_text(v) for k, v in item["options"].items()}
        # シャッフル
        item["options"], item["correct"] = shuffle_mcq(item["options"], item["correct"])

    # 修正後の分布
    after_dist = Counter(d["correct"] for d in data)
    print(f"After:  {dict(sorted(after_dist.items()))}")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
