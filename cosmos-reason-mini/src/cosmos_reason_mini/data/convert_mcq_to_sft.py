"""MCQ データを SFT 用 LLaVA 形式に変換する。

論文 Cosmos-Reason1 では SFT データに MCQ が含まれており、
モデルは SFT の段階で <think>...</think><answer>X</answer> 形式を学習する。
本スクリプトはその再現のため、MCQ JSON + 元の QA JSON から
LLaVA 形式の SFT データを生成する。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.convert_mcq_to_sft \
        --mcq_json data/rl/mcq_mini.json \
        --qa_json data/sft/qa_mini.json \
        --output data/sft/mcq_sft_mini.json

    # 自由形式 QA と統合:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.convert_mcq_to_sft \
        --mcq_json data/rl/mcq_mini.json \
        --qa_json data/sft/qa_mini.json \
        --merge_with data/sft/qa_mini.json \
        --output data/sft/qa_merged_mini.json
"""

import argparse
import json
import os


def build_mcq_prompt(question: str, options: dict) -> str:
    """MCQ プロンプトを構築する。MCQDataset と同じ形式。"""
    options_text = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return (
        f"{question}\n{options_text}\n\n"
        "Please think step by step and answer in the format: "
        "<think> your reasoning </think> <answer> the letter </answer>."
    )


def build_cot_answer(original_answer: str, correct: str, max_sentences: int = 3) -> str:
    """元の QA 回答から短い CoT + answer タグを構築する。

    SmolVLM 知見: 0.5B モデルでは長い CoT が有害。2-3 文に制限。
    """
    # 元の回答を文で分割し、最初の max_sentences 文を使用
    sentences = [s.strip() for s in original_answer.replace("\n", " ").split(".") if s.strip()]
    short_reasoning = ". ".join(sentences[:max_sentences])
    if short_reasoning and not short_reasoning.endswith("."):
        short_reasoning += "."
    return f"<think>{short_reasoning}</think><answer>{correct}</answer>"


def convert_mcq_to_llava(mcq_data: list, qa_data: list, max_sentences: int = 3) -> list:
    """MCQ JSON を LLaVA 形式に変換する。

    Args:
        mcq_data: MCQ JSON リスト
        qa_data: 元の QA JSON リスト (CoT 生成用)
        max_sentences: CoT の最大文数

    Returns:
        LLaVA 形式の SFT データリスト
    """
    # QA データを ID でインデックス化
    qa_by_id = {item["id"]: item for item in qa_data}

    results = []
    for mcq in mcq_data:
        # 元の QA を検索 (mcq_reasoning_XXX → driving_reasoning_XXX)
        original_id = mcq["id"].replace("mcq_", "driving_")
        original_qa = qa_by_id.get(original_id)

        if original_qa is None:
            print(f"Warning: no matching QA for {mcq['id']}")
            continue

        # 元の回答を取得
        original_answer = original_qa["conversations"][1]["value"]

        # MCQ プロンプトと CoT 回答を構築
        prompt = build_mcq_prompt(mcq["question"], mcq["options"])
        answer = build_cot_answer(original_answer, mcq["correct"], max_sentences)

        results.append(
            {
                "id": mcq["id"].replace("mcq_", "mcq_sft_"),
                "image": mcq["image"],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq_json", required=True, help="MCQ JSON (generate_mcq.py の出力)")
    parser.add_argument("--qa_json", required=True, help="元の QA JSON (LLaVA 形式)")
    parser.add_argument("--merge_with", default=None, help="統合する自由形式 QA JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_sentences", type=int, default=3, help="CoT の最大文数")
    args = parser.parse_args()

    with open(args.mcq_json) as f:
        mcq_data = json.load(f)
    with open(args.qa_json) as f:
        qa_data = json.load(f)

    # MCQ → LLaVA 変換
    mcq_sft = convert_mcq_to_llava(mcq_data, qa_data, args.max_sentences)
    print(f"Converted {len(mcq_sft)} MCQ to SFT format")

    # 統合
    if args.merge_with:
        with open(args.merge_with) as f:
            freeform_data = json.load(f)
        merged = freeform_data + mcq_sft
        print(f"Merged: {len(freeform_data)} free-form + {len(mcq_sft)} MCQ = {len(merged)} total")
    else:
        merged = mcq_sft

    # サンプル表示
    if mcq_sft:
        sample = mcq_sft[0]
        print(f"\nSample (id={sample['id']}):")
        print(f"  Q: {sample['conversations'][0]['value'][:120]}...")
        print(f"  A: {sample['conversations'][1]['value'][:120]}...")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
