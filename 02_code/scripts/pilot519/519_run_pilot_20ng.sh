#!/bin/bash
# 519 Pilot: Recurrent-depth training on 20 Newsgroups data
# accelerator0 only, fp32, 2000 steps each
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/home/zetyun/nanowhale"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PILOT_PARQUET="/data/zetyun/datasets/nanowhale_515_small_pretrain_train.parquet"
EVAL_DIR="/data/zetyun/eval"

echo "============================================"
echo "519 Pilot: T-depth on 20 Newsgroups"
echo "Started at: $(date)"
echo "Data: $PILOT_PARQUET"
echo "============================================"

run_training() {
    local config=$1
    local label=$2
    local max_steps=$3
    local output_dir=$4
    local val_text="$EVAL_DIR/519_pilot_${label}_val.txt"
    local split_meta="$EVAL_DIR/519_pilot_${label}_split.json"

    echo ""
    echo "============================================"
    echo "  Pilot: $label ($max_steps steps)"
    echo "  Started: $(date)"
    echo "============================================"

    python3 scripts/train_pretrain.py \
        --config "configs/519/${config}.yaml" \
        --train_parquet "$PILOT_PARQUET" \
        --val_ratio 0.05 \
        --split_seed 2025 \
        --val_text_out "$val_text" \
        --split_meta_out "$split_meta" \
        --no_compile \
        --no_bf16 \
        --seed 2025 \
        --max_steps "$max_steps" \
        --output_dir "$output_dir" \
        2>&1 | tee "$LOG_DIR/519_pilot_${label}_${TIMESTAMP}.log"

    echo "  Completed: $label at $(date)"
}

# Run all 4 configs sequentially
run_training "519_pilot_base" "base" 2000 "/data/zetyun/phase2_519_20ng_pilot_base_s2025"
run_training "519_pilot_t1" "t1" 2000 "/data/zetyun/phase2_519_20ng_pilot_t1_s2025"
run_training "519_pilot_t2" "t2" 2000 "/data/zetyun/phase2_519_20ng_pilot_t2_s2025"
run_training "519_pilot_t4" "t4" 2000 "/data/zetyun/phase2_519_20ng_pilot_t4_s2025"

echo ""
echo "============================================"
echo "519 Pilot Complete at: $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  /data/zetyun/phase2_519_20ng_pilot_base_s2025/"
echo "  /data/zetyun/phase2_519_20ng_pilot_t1_s2025/"
echo "  /data/zetyun/phase2_519_20ng_pilot_t2_s2025/"
echo "  /data/zetyun/phase2_519_20ng_pilot_t4_s2025/"
echo ""
echo "Logs: $LOG_DIR/"
