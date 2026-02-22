#!/bin/bash
# Full MiniPamayo training pipeline with full nuScenes dataset
# Run with: nohup bash scripts/run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
#
# Pipeline: [wait CoC] → Phase 4 → Stage 1 → Stage 2 → Stage 3 → Stage 4

set -e
cd "$(dirname "$0")/.."

export PATH="/home/masahirokubota/.local/bin:$PATH"
NUSCENES_ROOT="/mnt/ssd/nuscenes"
NUSCENES_VERSION="v1.0-trainval"
COC_DATA="data/coc_annotations_trainval.jsonl"
K=6
TIMING_LOG="logs/pipeline_timing.txt"

# Load .env for OPENAI_API_KEY (needed for Stage 4 r_reason)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

mkdir -p logs

echo "=== Full Pipeline Start: $(date) ==="
echo "nuScenes: $NUSCENES_ROOT ($NUSCENES_VERSION)"
echo "K=$K"

# Timing log header
echo "=== Pipeline Timing Report ===" > "$TIMING_LOG"
echo "Start: $(date)" >> "$TIMING_LOG"
echo "" >> "$TIMING_LOG"
PIPELINE_START=$SECONDS

# ============================================================
# Wait for CoC labeling to complete
# ============================================================
echo ""
echo "Waiting for CoC annotations: $COC_DATA"
COC_WAIT_START=$SECONDS
while [ ! -f "$COC_DATA" ] || pgrep -f "coc_labeling" > /dev/null 2>&1; do
    LINES=$(wc -l < "$COC_DATA" 2>/dev/null || echo 0)
    echo "  CoC progress: $LINES / ~28199 lines ($(date))"
    sleep 60
done
LINES=$(wc -l < "$COC_DATA")
COC_WAIT_ELAPSED=$(( SECONDS - COC_WAIT_START ))
echo "CoC annotations ready: $LINES lines (waited ${COC_WAIT_ELAPSED}s = $(( COC_WAIT_ELAPSED / 60 ))min)"
echo "CoC Wait:  ${COC_WAIT_ELAPSED}s ($(( COC_WAIT_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

# ============================================================
# Phase 4: Control-based regression (3 epochs for full dataset)
# ============================================================
echo ""
echo "=========================================="
echo "Phase 4: Starting at $(date)"
echo "=========================================="
STAGE_START=$SECONDS
PYTHONUNBUFFERED=1 uv run python -m minipamayo.train_phase4 \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version "$NUSCENES_VERSION" \
    --K $K \
    --max_epochs 3 \
    --use_wandb
STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
echo "Phase 4: Done at $(date) (${STAGE_ELAPSED}s = $(( STAGE_ELAPSED / 60 ))min)"
echo "Phase 4:   ${STAGE_ELAPSED}s ($(( STAGE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

# ============================================================
# Stage 1: Discrete action tokens (3 epochs)
# ============================================================
echo ""
echo "=========================================="
echo "Stage 1: Starting at $(date)"
echo "=========================================="
STAGE_START=$SECONDS
PYTHONUNBUFFERED=1 uv run python -m minipamayo.train_stage1 \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version "$NUSCENES_VERSION" \
    --K $K \
    --max_epochs 3 \
    --use_wandb
STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
echo "Stage 1: Done at $(date) (${STAGE_ELAPSED}s = $(( STAGE_ELAPSED / 60 ))min)"
echo "Stage 1:   ${STAGE_ELAPSED}s ($(( STAGE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

# ============================================================
# Stage 2: Flow Matching decoder (10 epochs)
# ============================================================
echo ""
echo "=========================================="
echo "Stage 2: Starting at $(date)"
echo "=========================================="
STAGE_START=$SECONDS
PYTHONUNBUFFERED=1 uv run python -m minipamayo.train_stage2 \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version "$NUSCENES_VERSION" \
    --K $K \
    --max_epochs 10 \
    --use_wandb
STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
echo "Stage 2: Done at $(date) (${STAGE_ELAPSED}s = $(( STAGE_ELAPSED / 60 ))min)"
echo "Stage 2:   ${STAGE_ELAPSED}s ($(( STAGE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

# ============================================================
# Stage 3: CoC SFT (5 epochs)
# ============================================================
echo ""
echo "=========================================="
echo "Stage 3: Starting at $(date)"
echo "=========================================="
STAGE_START=$SECONDS
PYTHONUNBUFFERED=1 uv run python -m minipamayo.train_stage3 \
    --coc_data "$COC_DATA" \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version "$NUSCENES_VERSION" \
    --K $K \
    --max_epochs 5 \
    --use_wandb
STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
echo "Stage 3: Done at $(date) (${STAGE_ELAPSED}s = $(( STAGE_ELAPSED / 60 ))min)"
echo "Stage 3:   ${STAGE_ELAPSED}s ($(( STAGE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

# ============================================================
# Stage 4: GRPO RL (3 epochs, no_reason_reward for speed)
# ============================================================
echo ""
echo "=========================================="
echo "Stage 4: Starting at $(date)"
echo "=========================================="
STAGE_START=$SECONDS
PYTHONUNBUFFERED=1 uv run python -m minipamayo.train_stage4 \
    --coc_data "$COC_DATA" \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version "$NUSCENES_VERSION" \
    --K_traj $K \
    --max_epochs 3 \
    --no_reason_reward \
    --use_wandb
STAGE_ELAPSED=$(( SECONDS - STAGE_START ))
echo "Stage 4: Done at $(date) (${STAGE_ELAPSED}s = $(( STAGE_ELAPSED / 60 ))min)"
echo "Stage 4:   ${STAGE_ELAPSED}s ($(( STAGE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"

PIPELINE_ELAPSED=$(( SECONDS - PIPELINE_START ))
echo "" >> "$TIMING_LOG"
echo "Total:     ${PIPELINE_ELAPSED}s ($(( PIPELINE_ELAPSED / 60 ))min)" >> "$TIMING_LOG"
echo "End: $(date)" >> "$TIMING_LOG"

echo ""
echo "=== Full Pipeline Complete: $(date) ==="
echo "=== Timing Report ==="
cat "$TIMING_LOG"
