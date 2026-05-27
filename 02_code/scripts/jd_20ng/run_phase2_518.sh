#!/usr/bin/env bash
# Stage 518: cross-corpus MSRVTT + compute-matched on 50-char 20ng.
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

# ===== Part A: MSRVTT cross-corpus (50 + 100 chars) =====
echo '========== PART A: MSRVTT 50-char =========='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_518_msrvtt_50.parquet /data/zetyun/eval/518_msrvtt50_val_split.txt /data/zetyun/eval/518_msrvtt50_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_518_msrvtt50_base3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_518_msrvtt50_t1on3000_s2025 2025 $ARGS

echo '========== PART A: MSRVTT 100-char =========='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_518_msrvtt_100.parquet /data/zetyun/eval/518_msrvtt100_val_split.txt /data/zetyun/eval/518_msrvtt100_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_518_msrvtt100_base3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_518_msrvtt100_t1on3000_s2025 2025 $ARGS

# ===== Part B: Compute-matched baseline on 50-char 20ng =====
# T1 wall time for 50-char 20ng seed2025 = 367.5s, baseline steps/s ≈ 8.16
# cm_steps = round(367.5 * 8.16) ≈ 2999 → essentially same as 3000
# T1 and baseline have nearly identical speed on 50-char, so compute-match is already done.
# Instead, run a 2x steps baseline (6000 steps) to see if more steps closes the gap.
echo '========== PART B: 50-char 2x steps baseline =========='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_516_20ng_trunc50.parquet /data/zetyun/eval/518_trunc50_2x_val_split.txt /data/zetyun/eval/518_trunc50_2x_split_meta.json)
# Note: override max_steps to 6000
$PY scripts/train_pretrain.py --config configs/phase2_smoke_baseline.yaml   --output_dir /data/zetyun/phase2_518_trunc50_base6000_s2025 --seed 2025   --train_parquet /data/zetyun/datasets/nanowhale_516_20ng_trunc50.parquet   --parquet_text_field text --val_ratio 0.05 --split_seed 2025 --split_method hash   --val_text_out /data/zetyun/eval/518_trunc50_2x_val_split.txt   --split_meta_out /data/zetyun/eval/518_trunc50_2x_split_meta.json   --no_bf16 --no_compile --collect_diagnostics --max_steps 6000   2>&1 | tee /data/zetyun/phase2_518_trunc50_base6000_s2025.log

echo '518 all done'
