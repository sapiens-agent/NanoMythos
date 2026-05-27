#!/usr/bin/env bash
# Phase 520: wall-matched baseline + L5 + t1 on real LM parquet (518 split).
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR="${NANOWHALE_DEVICE_SELECTOR:-0}"
PY="${PY:-/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python}"

# 519 summary_1k: base_s2025 steps/s 6.075, t4_s2025 3.252 -> match t4@1000 wall: 1000*6.075/3.252
WALL_STEPS="${WALL_STEPS:-1868}"

COMMON=(
  --train_parquet /data/zetyun/datasets/nanowhale_phase2_lm_real_train.parquet
  --parquet_text_field text
  --val_ratio 0.05
  --split_seed 2025
  --split_method hash
  --val_text_out /data/zetyun/eval/518_lm_real_val_split.txt
  --split_meta_out /data/zetyun/eval/518_lm_real_split_meta.json
  --no_bf16
  --no_compile
  --collect_diagnostics
  --seed 2025
)

echo "=== 520-1 baseline wall-match (${WALL_STEPS} steps) ==="
"$PY" scripts/train_pretrain.py --config configs/phase2_smoke_baseline.yaml \
  --output_dir /data/zetyun/phase2_520_lmreal_base_wall_s2025 \
  --max_steps "$WALL_STEPS" \
  "${COMMON[@]}" 2>&1 | tee /data/zetyun/phase2_520_lmreal_base_wall_s2025.log

echo "=== 520-2 L5 depth control (1000 steps) ==="
"$PY" scripts/train_pretrain.py --config configs/phase2_baseline_L5.yaml \
  --output_dir /data/zetyun/phase2_520_lmreal_L5_1000_s2025 \
  --max_steps 1000 \
  "${COMMON[@]}" 2>&1 | tee /data/zetyun/phase2_520_lmreal_L5_1000_s2025.log

echo "=== 520-3 t1_on backup (1000 steps) ==="
"$PY" scripts/train_pretrain.py --config configs/phase2_rec_t1_on.yaml \
  --output_dir /data/zetyun/phase2_520_lmreal_t1on1000_s2025 \
  --max_steps 1000 \
  "${COMMON[@]}" 2>&1 | tee /data/zetyun/phase2_520_lmreal_t1on1000_s2025.log

echo "520 train chain done"
