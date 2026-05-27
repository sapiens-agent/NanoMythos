#!/usr/bin/env bash
# Stage 516: seed 2027 confirmation (50/100/200/full) + T-depth on 50-char (T2,T4 s2025).
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python

common_args() {
  local parquet="$1" val_text="$2" val_meta="$3"
  echo --train_parquet "$parquet" --parquet_text_field text     --val_ratio 0.05 --split_seed 2025 --split_method hash     --val_text_out "$val_text" --split_meta_out "$val_meta"     --no_bf16 --no_compile --collect_diagnostics --max_steps 3000
}

run_one() {
  local cfg="$1" out="$2" seed="$3"; shift 3
  echo "=== $out seed=$seed ==="
  $PY scripts/train_pretrain.py --config "$cfg" --output_dir "$out" --seed "$seed" "$@" 2>&1 | tee "${out}.log"
}

# ===== Part A: Seed 2027 confirmation (50, 100, 200, full) =====
echo '========== PART A: Seed 2027 =========='

# 50 chars
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc50.parquet /data/zetyun/eval/516_trunc50_s2027_val_split.txt /data/zetyun/eval/516_trunc50_s2027_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc50_base3000_s2027 2027 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc50_t1on3000_s2027 2027 $ARGS

# 100 chars
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc100.parquet /data/zetyun/eval/516_trunc100_s2027_val_split.txt /data/zetyun/eval/516_trunc100_s2027_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc100_base3000_s2027 2027 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc100_t1on3000_s2027 2027 $ARGS

# 200 chars
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc200.parquet /data/zetyun/eval/516_trunc200_s2027_val_split.txt /data/zetyun/eval/516_trunc200_s2027_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc200_base3000_s2027 2027 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc200_t1on3000_s2027 2027 $ARGS

# full
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_full.parquet /data/zetyun/eval/516_full_s2027_val_split.txt /data/zetyun/eval/516_full_s2027_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_full_base3000_s2027 2027 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_full_t1on3000_s2027 2027 $ARGS

# ===== Part B: T-depth on 50-char (T2, T4 at seed 2025) =====
echo '========== PART B: T-depth on 50-char =========='

ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc50.parquet /data/zetyun/eval/516_trunc50_val_split.txt /data/zetyun/eval/516_trunc50_split_meta.json)
run_one configs/phase2_rec_t2_on.yaml /data/zetyun/phase2_516_trunc50_t2on3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t4_on.yaml /data/zetyun/phase2_516_trunc50_t4on3000_s2025 2025 $ARGS

echo '516 seed2027 + tdepth done'
