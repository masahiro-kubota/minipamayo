"""キャプションから QA ペアを生成する。"""

import argparse
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def generate_qa(client, caption: str, prompt_template: str, model="gpt-4o-mini") -> list:
    """キャプションから QA を生成。テキストのみなので gpt-4o-mini で十分。"""
    prompt = prompt_template.replace("{caption}", caption)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    # JSON パース(配列 or オブジェクト内の配列)
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--prompt_understanding", default="prompts/qa_understanding.txt")
    parser.add_argument("--prompt_reasoning", default="prompts/qa_reasoning.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()
    with open(args.prompt_understanding) as f:
        prompt_u = f.read().strip()
    with open(args.prompt_reasoning) as f:
        prompt_r = f.read().strip()
    with open(args.captions) as f:
        captions = json.load(f)

    results = []
    for item in tqdm(captions, desc="Generating QA"):
        if not item.get("caption"):
            continue
        caption = item["caption"]

        # 理解 QA(2 問)
        try:
            qa_u = generate_qa(client, caption, prompt_u, args.model)
            for qa in qa_u:
                results.append(
                    {
                        "sample_token": item["sample_token"],
                        "image": item["image_path"],
                        "category": "understanding",
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )
        except Exception as e:
            print(f"Understanding QA error for {item['sample_token']}: {e}")

        # 推論 QA(1-2 問)
        try:
            qa_r = generate_qa(client, caption, prompt_r, args.model)
            for qa in qa_r:
                results.append(
                    {
                        "sample_token": item["sample_token"],
                        "image": item["image_path"],
                        "category": "reasoning",
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )
        except Exception as e:
            print(f"Reasoning QA error for {item['sample_token']}: {e}")

        time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(results)} QA pairs")


if __name__ == "__main__":
    main()
