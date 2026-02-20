# Phase 1: SFT データ作成 — 具体的実装プラン

## 目的

Cosmos-Reason1 の Physical AI SFT パイプライン（§5.1）を小規模再現する。
nuScenes の運転画像から、教師 VLM（GPT-4o）を使って **運転シーン理解 QA** と **運転行動推論 QA** を生成し、
Qwen2.5-VL Mini を運転ドメインに特化させるための SFT データを作成する。

### Cosmos-Reason1 論文との対応

| Cosmos-Reason1 §5.1 | Cosmos Reason Mini Phase 1 | 差分 |
|---|---|---|
| 人間キュレーション動画（5 ドメイン） | nuScenes 画像（運転のみ） | 動画→画像、1ドメインに特化 |
| 詳細キャプション生成 | GPT-4o キャプション生成 | 同思想 |
| QA ペア構築 | GPT-4o QA 生成 | 同思想 |
| DeepSeek-R1 推論トレース | GPT-4o/Claude 推論トレース（短く簡潔） | 教師モデルが異なる |
| クリーニング & リライト | ルールベースクリーニング | 簡略化 |
| ~4M サンプル | **~5,000〜10,000 サンプル** | ~400 倍差（意図的） |

---

## 前提条件

- Qwen2.5-VL Mini Stage 2.1 が完了していること（[checkpoint-5247.pt](../../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt)）
- GPT-4o API キーが利用可能であること（OpenAI）
- nuScenes アカウント登録済み（https://www.nuscenes.org/）

---

## データ設計

### データソース: nuScenes

[datasets.md](../../docs/datasets.md) の推奨に基づき、nuScenes を使用する。

| 項目 | nuScenes Mini | nuScenes Full |
|---|---|---|
| 用途 | パイプライン検証 | 本番データ生成 |
| シーン数 | 10 | 850 (train) + 150 (val) |
| キーフレーム数 | ~400 | ~34,000 (train) + ~6,000 (val) |
| サイズ | ~4 GB | ~300 GB |
| 画像解像度 | 1600×900 | 1600×900 |
| カメラ | 6台（CAM_FRONT のみ使用） | 同左 |

**段階的アプローチ**:
1. nuScenes Mini でパイプライン全体を検証（~400 フレーム → ~100 QA）
2. nuScenes Full (train split) で本番データ生成（~3,000 フレーム → ~7,500 QA）
3. nuScenes Full (val split) で評価データ生成（~500 フレーム → ~1,250 QA）

### QA カテゴリと目標数

Cosmos-Reason1 の 2 カテゴリ（物理的常識 + 具現化推論）を運転ドメインに特化:

| カテゴリ | サブタイプ | 例 | 目標数 | 比率 |
|---|---|---|---|---|
| **運転シーン理解** | シーン記述 | 「この運転シーンを説明してください」 | ~1,500 | 20% |
| | 空間関係 | 「先行車との相対位置は？」 | ~2,000 | 27% |
| | 物体属性 | 「信号の状態は？」 | ~1,000 | 13% |
| **運転行動推論** | 因果推論 | 「なぜ先行車は減速しているか？」 | ~1,000 | 13% |
| | 行動予測 | 「ego vehicle は次にどう行動すべきか？」 | ~1,500 | 20% |
| | 安全判断 | 「この車線変更は安全か？」 | ~500 | 7% |
| **合計** | | | **~7,500** | 100% |

**空間関係 QA を重視する理由**: DINOv2 は細粒度の空間情報に優れる（COMM 論文: CLIP より grounding で +2.8pt）。この強みを活かすデータ構成にする。

### データ分割

| 分割 | ソース | フレーム数 | QA 数 | 用途 |
|---|---|---|---|---|
| train | nuScenes train split | ~3,000 | ~7,500 | SFT 学習 |
| eval | nuScenes val split | ~500 | ~1,250 | SFT 評価 + RL MCQ 評価 |

**train/eval の分離**: nuScenes の公式 train/val split を使用するため、画像レベルでの data leak は発生しない。

---

## プロジェクト構成（Phase 1 で追加するファイル）

```
cosmos-reason-mini/
├── pyproject.toml                          # 新規: プロジェクト設定
├── src/cosmos_reason_mini/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── select_frames.py                # 新規: nuScenes フレーム選定
│       ├── generate_captions.py            # 新規: GPT-4o キャプション生成
│       ├── generate_qa.py                  # 新規: QA ペア生成
│       └── clean_qa.py                     # 新規: クリーニング・統合
├── data/                                   # データディレクトリ（.gitignore）
│   ├── nuscenes/                           # nuScenes データ
│   └── sft/                                # 生成された SFT データ
│       ├── captions_train.json
│       ├── captions_eval.json
│       ├── qa_train_raw.json
│       ├── qa_eval_raw.json
│       ├── qa_train.json                   # クリーニング済み最終版
│       └── qa_eval.json                    # クリーニング済み最終版
└── prompts/                                # プロンプトテンプレート
    ├── caption.txt
    ├── qa_understanding.txt
    └── qa_reasoning.txt
```

---

## Step 0: nuScenes セットアップ

### 0a: nuScenes Mini ダウンロード（パイプライン検証用）

```bash
# nuScenes サイト（https://www.nuscenes.org/nuscenes）から手動ダウンロード
# ※要アカウント登録・利用規約同意
# v1.0-mini (~4 GB) を data/nuscenes/ に展開

# 期待されるディレクトリ構造:
# cosmos-reason-mini/data/nuscenes/
# ├── v1.0-mini/           # メタデータ JSON
# │   ├── scene.json
# │   ├── sample.json
# │   ├── sample_data.json
# │   ├── ego_pose.json
# │   └── ...
# └── samples/
#     ├── CAM_FRONT/       # キーフレーム画像 (1600×900)
#     ├── CAM_FRONT_LEFT/
#     └── ...
```

### 0b: pyproject.toml 作成

```toml
[project]
name = "cosmos-reason-mini"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.37.0",
    "pillow>=10.0.0",
    "nuscenes-devkit>=1.1.11",
    "openai>=1.0.0",
    "tqdm",
]

[project.optional-dependencies]
dev = ["ruff"]

[tool.uv.sources]
qwen-vl-mini = { path = "../qwen-vl-mini", editable = true }

[build-system]
requires = ["uv_build>=0.10.2,<0.11.0"]
build-backend = "uv_build"
```

### 0c: 動作確認

```bash
cd cosmos-reason-mini && uv sync
cd cosmos-reason-mini && uv run python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes')
print(f'Scenes: {len(nusc.scene)}')
print(f'Samples: {len(nusc.sample)}')
"
# 期待出力: Scenes: 10, Samples: 404
```

---

## Step 1: フレーム選定

### 選定基準

1. **CAM_FRONT のみ**: MiniPamayo はフロントカメラ 1 台の想定
2. **キーフレーム（2Hz）のみ**: アノテーション付きフレーム
3. **多様性の確保**: 同一シーンから連続フレームを取りすぎない（最大 5 フレーム/シーンを間引き選定）
4. **興味深い状況を優先**: 交差点、歩行者あり、車線変更、信号変化等

### 新規ファイル: data/select_frames.py

```python
"""nuScenes からフレームを選定し、メタデータ付き JSON を出力する。"""
import argparse, json, os, random
from nuscenes.nuscenes import NuScenes

def select_frames(nusc, max_per_scene=5, seed=42):
    random.seed(seed)
    frames = []
    for scene in nusc.scene:
        # シーン内の全キーフレームを収集
        sample_token = scene["first_sample_token"]
        scene_frames = []
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
            scene_frames.append({
                "sample_token": sample_token,
                "image_path": cam_data["filename"],  # samples/CAM_FRONT/xxx.jpg
                "scene_name": scene["name"],
                "scene_description": scene["description"],
                "ego_translation": ego_pose["translation"],
                "ego_rotation": ego_pose["rotation"],
                "timestamp": sample["timestamp"],
            })
            sample_token = sample["next"] if sample["next"] else None

        # 等間隔でサンプリング（多様性確保）
        if len(scene_frames) > max_per_scene:
            indices = [int(i * len(scene_frames) / max_per_scene) for i in range(max_per_scene)]
            scene_frames = [scene_frames[i] for i in indices]
        frames.extend(scene_frames)
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--max_per_scene", type=int, default=5)
    parser.add_argument("--output", default="data/sft/frames.json")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    frames = select_frames(nusc, max_per_scene=args.max_per_scene)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"Selected {len(frames)} frames from {len(nusc.scene)} scenes")

if __name__ == "__main__":
    main()
```

### 実行

```bash
# Mini（検証用）: ~50 フレーム
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.select_frames \
    --version v1.0-mini --max_per_scene 5 \
    --output data/sft/frames_mini.json

# Full（本番用）: ~3,000 フレーム (train)
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.select_frames \
    --version v1.0-trainval --max_per_scene 4 \
    --output data/sft/frames_train.json

# Full（評価用）: ~500 フレーム (val)
# ※ val split のシーンのみ使用するフィルタを追加
```

---

## Step 2: キャプション生成（GPT-4o）

### 目的

各フレームに対して GPT-4o で詳細なシーンキャプションを生成する。
Cosmos-Reason1 §5.1.1 の「詳細キャプション生成」に対応。

### プロンプト設計

Cosmos-Reason1 は動画に対するキャプションだが、Cosmos Reason Mini は静止画なのでプロンプトを調整する。

#### prompts/caption.txt

```
You are an expert driving scene analyst. Describe the following front-camera driving image in detail.

Include ALL of the following aspects:
1. **Road**: Road type (highway, urban, intersection, etc.), number of lanes, road surface condition
2. **Traffic**: Other vehicles (type, color, position relative to ego), pedestrians, cyclists
3. **Signals/Signs**: Traffic lights (state: red/yellow/green), stop signs, speed limits, lane markings
4. **Environment**: Weather (clear, rain, overcast), time of day (day, dusk, night), visibility
5. **Ego vehicle context**: Current lane, approximate speed impression, direction of travel

Be factual and precise about spatial relationships (e.g., "a white sedan approximately 20 meters ahead in the same lane").
Do NOT speculate about things not visible in the image.
Respond in English. Keep the description between 100-200 words.
```

### 新規ファイル: data/generate_captions.py

```python
"""GPT-4o を使って nuScenes フレームのキャプションを生成する。"""
import argparse, json, os, base64, time
from openai import OpenAI
from tqdm import tqdm

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_caption(client, image_path: str, prompt: str) -> str:
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
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
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated {len([r for r in results if r.get('caption')])} captions")

if __name__ == "__main__":
    main()
```

### 実行

```bash
# Mini（検証用）: ~50 フレーム → ~$2-3
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_captions \
    --frames data/sft/frames_mini.json \
    --output data/sft/captions_mini.json \
    --resume

# Full（本番用）: ~3,000 フレーム → ~$30-50
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_captions \
    --frames data/sft/frames_train.json \
    --output data/sft/captions_train.json \
    --resume
```

### API コスト見積もり

| 項目 | Mini (検証) | Full (本番) |
|---|---|---|
| フレーム数 | ~50 | ~3,500 |
| 入力（画像+プロンプト）/件 | ~1,500 tokens | ~1,500 tokens |
| 出力/件 | ~300 tokens | ~300 tokens |
| 合計入力 | ~75K tokens | ~5.25M tokens |
| 合計出力 | ~15K tokens | ~1.05M tokens |
| コスト（GPT-4o） | ~$0.35 | ~$24 |

---

## Step 3: QA 生成

### 目的

キャプションを元に、理解 QA と推論 QA を生成する。
画像は見せず、キャプション（テキスト）のみで QA を生成するため、**テキスト API（安価）で処理可能**。

### プロンプト設計

#### prompts/qa_understanding.txt

```
Given the following description of a driving scene from a front-facing camera, generate 2 question-answer pairs that test UNDERSTANDING of the scene.

Types of questions to generate:
- Spatial relationship: position of objects relative to ego vehicle
- Object attributes: vehicle type, color, signal state
- Scene description: road type, weather, traffic density

Rules:
- Questions should be answerable from the image alone (not the caption)
- Answers should be concise (1-3 sentences)
- Include specific spatial terms (ahead, left, right, behind, approximately X meters)
- Do NOT reference "the caption" or "the description" in questions or answers

Caption:
{caption}

Output as JSON array:
[
  {"question": "...", "answer": "..."},
  {"question": "...", "answer": "..."}
]
```

#### prompts/qa_reasoning.txt

```
Given the following description of a driving scene from a front-facing camera, generate 1-2 question-answer pairs that test REASONING about driving actions and causality.

Types of questions to generate:
- Action prediction: "What should the ego vehicle do next?"
- Causal reasoning: "Why is the vehicle ahead slowing down?"
- Safety judgment: "Is it safe to change lanes in this situation?"

Rules:
- Questions should require reasoning, not just observation
- Answers should include brief reasoning (1-3 sentences explaining WHY)
- Be grounded in what's visible in the scene
- Do NOT reference "the caption" or "the description"

Caption:
{caption}

Output as JSON array:
[
  {"question": "...", "answer": "..."}
]
```

### 新規ファイル: data/generate_qa.py

```python
"""キャプションから QA ペアを生成する。"""
import argparse, json, os, time
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
    # JSON パース（配列 or オブジェクト内の配列）
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

        # 理解 QA（2 問）
        try:
            qa_u = generate_qa(client, caption, prompt_u, args.model)
            for qa in qa_u:
                results.append({
                    "sample_token": item["sample_token"],
                    "image": item["image_path"],
                    "category": "understanding",
                    "question": qa["question"],
                    "answer": qa["answer"],
                })
        except Exception as e:
            print(f"Understanding QA error: {e}")

        # 推論 QA（1-2 問）
        try:
            qa_r = generate_qa(client, caption, prompt_r, args.model)
            for qa in qa_r:
                results.append({
                    "sample_token": item["sample_token"],
                    "image": item["image_path"],
                    "category": "reasoning",
                    "question": qa["question"],
                    "answer": qa["answer"],
                })
        except Exception as e:
            print(f"Reasoning QA error: {e}")

        time.sleep(args.delay)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(results)} QA pairs")

if __name__ == "__main__":
    main()
```

### 実行

```bash
# Mini（検証用）: ~50 キャプション → ~150 QA, ~$0.10
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_qa \
    --captions data/sft/captions_mini.json \
    --output data/sft/qa_mini_raw.json

# Full（本番用）: ~3,000 キャプション → ~7,500 QA, ~$5-10
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.generate_qa \
    --captions data/sft/captions_train.json \
    --output data/sft/qa_train_raw.json
```

### API コスト見積もり（QA 生成）

QA 生成はテキストのみなので **gpt-4o-mini**（入力 $0.15/1M, 出力 $0.60/1M）を使用。

| 項目 | Mini (検証) | Full (本番) |
|---|---|---|
| API コール数 | ~100 (50×2 prompts) | ~7,000 |
| 合計入力 | ~150K tokens | ~10.5M tokens |
| 合計出力 | ~50K tokens | ~3.5M tokens |
| コスト（gpt-4o-mini） | ~$0.05 | ~$3.7 |

**Phase 1 合計コスト: ~$28（Mini $0.40 + Full $27.7）**

---

## Step 4: クリーニング・検証

### クリーニングルール

Cosmos-Reason1 §5.1.1 のクリーニングに対応:

1. **キャプション参照の除去**: 「As described in the caption」「The description mentions」等
2. **画像参照の修正**: 「In the image」→ そのまま許可（画像を見る前提の QA なので自然）
3. **空の回答の除外**: answer が空または極端に短い（<10 文字）ものを除外
4. **重複 QA の除外**: 同一画像に対する類似 QA をフィルタ
5. **フォーマット統一**: question は `?` で終わる、answer は `.` で終わる

### 新規ファイル: data/clean_qa.py

```python
"""QA データのクリーニングと LLaVA 形式への変換。"""
import argparse, json, re, os

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

        llava_data.append({
            "id": f"driving_{item.get('category', 'unknown')}_{i:06d}",
            "image": item["image"],  # samples/CAM_FRONT/xxx.jpg
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        })
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
```

### 実行

```bash
# Mini
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.clean_qa \
    --input data/sft/qa_mini_raw.json \
    --output data/sft/qa_mini.json

# Full (train)
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.clean_qa \
    --input data/sft/qa_train_raw.json \
    --output data/sft/qa_train.json

# Full (eval)
cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.data.clean_qa \
    --input data/sft/qa_eval_raw.json \
    --output data/sft/qa_eval.json
```

---

## Step 5: 品質チェック

### 5a: 統計の確認

```bash
cd cosmos-reason-mini && uv run python -c "
import json
with open('data/sft/qa_train.json') as f:
    data = json.load(f)
print(f'Total QA: {len(data)}')

# カテゴリ分布
cats = {}
for item in data:
    cat = item['id'].split('_')[1]
    cats[cat] = cats.get(cat, 0) + 1
for cat, count in sorted(cats.items()):
    print(f'  {cat}: {count} ({count/len(data)*100:.1f}%)')

# 回答長の分布
lens = [len(item['conversations'][1]['value']) for item in data]
print(f'Answer length: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}')

# 画像のユニーク数
images = set(item['image'] for item in data)
print(f'Unique images: {len(images)}')
print(f'QA per image: {len(data)/len(images):.1f}')
"
```

### 5b: ランダムサンプル目視確認

```bash
# 10 サンプルをランダムに表示して品質を確認
cd cosmos-reason-mini && uv run python -c "
import json, random
random.seed(42)
with open('data/sft/qa_train.json') as f:
    data = json.load(f)
for item in random.sample(data, min(10, len(data))):
    print(f'--- {item[\"id\"]} ---')
    print(f'Image: {item[\"image\"]}')
    print(f'Q: {item[\"conversations\"][0][\"value\"].replace(chr(10), \" \")}')
    print(f'A: {item[\"conversations\"][1][\"value\"]}')
    print()
"
```

### 5c: 期待する結果

| 指標 | 目標 |
|---|---|
| 合計 QA 数（train） | 5,000〜10,000 |
| 合計 QA 数（eval） | 1,000〜1,500 |
| ユニーク画像数（train） | ~3,000 |
| QA / 画像 | 2〜3 |
| 理解 QA 比率 | ~60% |
| 推論 QA 比率 | ~40% |
| 回答平均文字数 | 50〜150 |
| キャプション参照の残存 | 0 件 |

---

## Step 6: DataLoader テスト

Phase 2 で使用する Dataset クラスの基本動作を事前に確認する。

```bash
cd cosmos-reason-mini && uv run python -c "
import json
from PIL import Image
import os

with open('data/sft/qa_mini.json') as f:
    data = json.load(f)

# 最初のサンプルの画像が読めるか確認
item = data[0]
image_path = os.path.join('data/nuscenes', item['image'])
img = Image.open(image_path)
print(f'Image size: {img.size}')  # 期待: (1600, 900)
print(f'Q: {item[\"conversations\"][0][\"value\"]}')
print(f'A: {item[\"conversations\"][1][\"value\"]}')
print('DataLoader test: OK')
"
```

---

## 教師モデルの選択肢

| モデル | 用途 | コスト | 品質 |
|---|---|---|---|
| **GPT-4o** | キャプション生成（画像入力） | $2.50/1M in + $10/1M out | 最高 |
| **gpt-4o-mini** | QA 生成（テキストのみ） | $0.15/1M in + $0.60/1M out | 十分 |
| **Claude 3.5 Sonnet** | キャプション or QA（代替） | $3/1M in + $15/1M out | GPT-4o 同等 |
| **Qwen2.5-VL-72B** | キャプション生成（代替） | API 依存 | 良好 |

**推奨**: キャプション生成に GPT-4o、QA 生成に gpt-4o-mini。コスト効率が最も良い。

---

## 完了状況

| Step | 状態 | 備考 |
|------|------|------|
| Step 0: nuScenes セットアップ | **完了** ✅ | nuScenes Mini 展開済み、pyproject.toml + uv sync 完了 |
| Step 1: フレーム選定 | **完了** ✅ | 50 フレーム（10 scenes × 5）→ `data/sft/frames_mini.json` |
| Step 2: キャプション生成 | **完了** ✅ | 50/50 成功 → `data/sft/captions_mini.json` |
| Step 3: QA 生成 | **完了** ✅ | 201 QA ペア → `data/sft/qa_mini_raw.json` |
| Step 4: クリーニング | **完了** ✅ | 201 → 201（フィルタ脱落 0）→ `data/sft/qa_mini.json` |
| Step 5: 品質チェック | **完了** ✅ | 統計・目視確認 OK（下記参照） |
| Step 6: DataLoader テスト | **完了** ✅ | 全 201 QA の画像アクセス確認 OK |

### Mini パイプライン検証結果（2026-02-20）

| 指標 | 結果 | 目標 |
|------|------|------|
| 合計 QA 数 | 201 | ~150（超過 OK） |
| ユニーク画像数 | 50 | 50 |
| QA / 画像 | 4.0 | 2〜3（やや多い） |
| 理解 QA 比率 | 49.8% | ~60% |
| 推論 QA 比率 | 50.2% | ~40% |
| 回答平均文字数 | 160 | 50〜150（やや長い） |
| キャプション参照の残存 | 0 件 | 0 件 |

### 判断メモ

1. **理解/推論の比率が 50:50 になった件**: 計画では 60:40 を目標としていたが、推論プロンプトが 1-2 問のところ概ね 2 問生成したため 50:50 になった。Full データ生成時にプロンプトを調整するか、理解プロンプトを 3 問に増やすことで調整可能。Mini 検証としては問題なし。
2. **回答長がやや長い件**: 平均 160 文字で目標の 50-150 をやや超過。0.5B モデルの学習には長すぎる可能性がある。Full データ生成時にプロンプトに「Keep answers under 100 words」を追加することを検討。
3. **nuScenes Full への拡張**: Mini パイプラインが問題なく動作することを確認済み。Full データ（~300 GB）のダウンロードとフレーム選定に進む準備ができている。コスト見積もり: キャプション ~$24 + QA ~$3.7 = **~$28**。
