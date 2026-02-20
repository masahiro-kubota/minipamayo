"""QA データのクリーニングと LLaVA 形式への変換。"""

import argparse
import json
import os
import re

# キャプション参照パターン
CAPTION_REFS = [
    r"(?i)as described in the (caption|description)",
    r"(?i)according to the (caption|description)",
    r"(?i)the (caption|description) (mentions|states|indicates|describes)",
    r"(?i)based on the (caption|description)",
    r"(?i)from the (caption|description)",
]


def clean_text(text: str) -> str:
    for pattern in CAPTION_REFS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_llava_format(qa_items: list) -> list:
    """QA リストを LLaVA JSON 形式に変換。"""
    llava_data = []
    for i, item in enumerate(qa_items):
        question = clean_text(item["question"]).strip()
        answer = clean_text(item["answer"]).strip()

        # 品質フィルタ
        if len(answer) < 10:
            continue
        if not question.endswith("?"):
            question += "?"

        llava_data.append(
            {
                "id": f"driving_{item.get('category', 'unknown')}_{i:06d}",
                "image": item["image"],  # samples/CAM_FRONT/xxx.jpg
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )
    return llava_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)
    cleaned = to_llava_format(raw)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    # 統計
    categories = {}
    for item in raw:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print(f"Input: {len(raw)} → Output: {len(cleaned)} QA pairs")
    print(f"Categories: {categories}")


if __name__ == "__main__":
    main()
