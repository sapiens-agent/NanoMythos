#!/usr/bin/env bash
# Stage 516: length ablation — baseline vs T=1 at 50, 200, 500 chars.
# (100-char and full reused from 515/514.)
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python
SEED=2025

common_args() {
  local parquet="$1" val_text="$2" val_meta="$3"
  echo --train_parquet "$parquet" --parquet_text_field text     --val_ratio 0.05 --split_seed 2025 --split_method hash     --val_text_out "$val_text" --split_meta_out "$val_meta"     --no_bf16 --no_compile --collect_diagnostics --max_steps 3000
}

run_one() {
  local cfg="$1" out="$2" seed="$3"; shift 3
  echo "=== $out seed=$seed ==="
  $PY scripts/train_pretrain.py --config "$cfg" --output_dir "$out" --seed "$seed" "$@" 2>&1 | tee "${out}.log"
}

# --- 50 chars ---
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc50.parquet /data/zetyun/eval/516_trunc50_val_split.txt /data/zetyun/eval/516_trunc50_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc50_base3000_s$SEED $SEED $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc50_t1on3000_s$SEED $SEED $ARGS

# --- 200 chars ---
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc200.parquet /data/zetyun/eval/516_trunc200_val_split.txt /data/zetyun/eval/516_trunc200_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc200_base3000_s$SEED $SEED $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc200_t1on3000_s$SEED $SEED $ARGS

# --- 500 chars ---
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc500.parquet /data/zetyun/eval/516_trunc500_val_split.txt /data/zetyun/eval/516_trunc500_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_516_trunc500_base3000_s$SEED $SEED $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_516_trunc500_t1on3000_s$SEED $SEED $ARGS

echo '516 length ablation done seed='$SEED
