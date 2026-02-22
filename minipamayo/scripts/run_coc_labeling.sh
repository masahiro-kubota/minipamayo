#!/bin/bash
# CoC labeling for full nuScenes dataset (CPU only, OpenAI API)
# Run with: nohup bash scripts/run_coc_labeling.sh > logs/coc_labeling.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

export PATH="/home/masahirokubota/.local/bin:$PATH"

# Load .env for OPENAI_API_KEY
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "=== CoC Labeling Start: $(date) ==="
echo "Dataset: /mnt/ssd/nuscenes (v1.0-trainval)"

PYTHONUNBUFFERED=1 uv run python -m minipamayo.data.coc_labeling \
    --nuscenes_root /mnt/ssd/nuscenes \
    --nuscenes_version v1.0-trainval \
    --output data/coc_annotations_trainval.jsonl \
    --K 6 \
    --concurrency 20

echo "=== CoC Labeling Done: $(date) ==="
