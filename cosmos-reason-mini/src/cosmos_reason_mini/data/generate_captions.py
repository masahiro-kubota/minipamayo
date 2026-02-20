"""GPT-4o を使って nuScenes フレームのキャプションを生成する。"""

import argparse
import base64
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(client, image_path: str, prompt: str) -> str:
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="frames.json path")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--prompt", default="prompts/caption.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="Skip already-generated")
    parser.add_argument("--delay", type=float, default=0.5, help="API call delay (sec)")
    parser.add_argument("--save_interval", type=int, default=100, help="Intermediate save interval")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()  # OPENAI_API_KEY from env
    with open(args.prompt) as f:
        prompt = f.read().strip()
    with open(args.frames) as f:
        frames = json.load(f)

    # Resume support
    existing = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing = {item["sample_token"]: item for item in json.load(f)}

    results = list(existing.values())
    for frame in tqdm(frames, desc="Generating captions"):
        if frame["sample_token"] in existing:
            continue
        image_path = os.path.join(args.dataroot, frame["image_path"])
        try:
            caption = generate_caption(client, image_path, prompt)
            results.append({**frame, "caption": caption})
        except Exception as e:
            print(f"Error for {frame['sample_token']}: {e}")
            results.append({**frame, "caption": "", "error": str(e)})
        # Rate limiting
        time.sleep(args.delay)
        if len(results) % args.save_interval == 0:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ok_count = len([r for r in results if r.get("caption")])
    print(f"Generated {ok_count}/{len(results)} captions")


if __name__ == "__main__":
    main()
