#!/bin/bash
# 519 Pilot: Recurrent-depth PPL across training stages on FineWeb-Edu
# Run AFTER 519_setup_and_smoke.sh completes successfully.
# Usage: bash /home/zetyun/nanowhale/reports/519reports/519_run_pilot.sh
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

PILOT_PARQUET="/data/zetyun/datasets/nanowhale_519_fineweb_pilot.parquet"
EVAL_DIR="/data/zetyun/eval"
VAL_TEXT="$EVAL_DIR/519_pilot_val_split.txt"
SPLIT_META="$EVAL_DIR/519_pilot_split_meta.json"

echo "============================================"
echo "519 Pilot: T-depth PPL on FineWeb-Edu Pilot"
echo "Started at: $(date)"
echo "============================================"

# Verify pilot data exists
if [ ! -f "$PILOT_PARQUET" ]; then
    echo "ERROR: Pilot parquet not found: $PILOT_PARQUET"
    echo "Run 519_setup_and_smoke.sh first."
    exit 1
fi

echo ""
echo "Pilot dataset: $PILOT_PARQUET"
python3 -c "
import pandas as pd
df = pd.read_parquet('$PILOT_PARQUET')
lens = df['text'].str.len()
total_chars = int(lens.sum())
print(f'  rows: {len(df)}, chars: {total_chars:,}')
print(f'  mean_len: {lens.mean():.0f}, median: {lens.median():.0f}, p95: {lens.quantile(0.95):.0f}')
"

# ============================================================
# Run training runs sequentially
# ============================================================

run_training() {
    local config=$1
    local label=$2
    local max_steps=$3

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
        --val_text_out "$VAL_TEXT" \
        --split_meta_out "$SPLIT_META" \
        --no_compile \
        --no_bf16 \
        --seed 2025 \
        --max_steps "$max_steps" \
        2>&1 | tee "$LOG_DIR/519_pilot_${label}_${TIMESTAMP}.log"

    echo "  Completed: $label at $(date)"
}

# Run baseline (2000 steps)
run_training "519_pilot_base" "base" 2000

# Run T=1 (2000 steps)
run_training "519_pilot_t1" "t1" 2000

# Run T=2 (2000 steps)
run_training "519_pilot_t2" "t2" 2000

# Run T=4 (2000 steps)
run_training "519_pilot_t4" "t4" 2000

echo ""
echo "============================================"
echo "519 Pilot Complete at: $(date)"
echo "============================================"
echo ""
echo "Results in:"
echo "  /data/zetyun/phase2_519_fineweb_pilot_base_s2025/"
echo "  /data/zetyun/phase2_519_fineweb_pilot_t1_s2025/"
echo "  /data/zetyun/phase2_519_fineweb_pilot_t2_s2025/"
echo "  /data/zetyun/phase2_519_fineweb_pilot_t4_s2025/"
echo ""
echo "Logs: $LOG_DIR/"
