#!/bin/bash
# 520 Cross-seed: baseline vs T=1 with seeds 2027, 2048
# Seed 2025 already done in 519. Runs seeds 2027 and 2048.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=0

PROJECT_DIR="/home/zetyun/nanowhale"
cd "$PROJECT_DIR"
LOG_DIR="/data/zetyun/eval/520/logs"
mkdir -p "$LOG_DIR"

TRAIN_PARQUET="/data/zetyun/datasets/nanowhale_515_small_pretrain_train.parquet"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "520 Cross-seed: baseline vs T=1"
echo "Seeds: 2027, 2048  |  Steps: 2000"
echo "Started: $(date)"
echo "============================================"

for SEED in 2027 2048; do
    for CONFIG in 519_pilot_base 519_pilot_t1; do
        LABEL="${CONFIG}_s${SEED}"
        OUTPUT_DIR="/data/zetyun/phase2_520_20ngfull_${CONFIG}_s${SEED}"
        VAL_TEXT="/data/zetyun/eval/520/val_${LABEL}.txt"
        SPLIT_META="/data/zetyun/eval/520/split_${LABEL}.json"
        LOG_FILE="$LOG_DIR/520_cross_seed_${LABEL}_${TIMESTAMP}.log"

        echo ""
        echo "=== $LABEL (2000 steps) ==="

        python3 scripts/train_pretrain.py \
            --config "configs/519/${CONFIG}.yaml" \
            --train_parquet "$TRAIN_PARQUET" \
            --val_ratio 0.05 \
            --split_seed "$SEED" \
            --val_text_out "$VAL_TEXT" \
            --split_meta_out "$SPLIT_META" \
            --no_compile --no_bf16 \
            --seed "$SEED" \
            --max_steps 2000 \
            --output_dir "$OUTPUT_DIR" \
            2>&1 | tee "$LOG_FILE"

        echo "  $LABEL done: $(date)"
    done
done

echo ""
echo "============================================"
echo "Cross-seed complete: $(date)"
echo "============================================"
