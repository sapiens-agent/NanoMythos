#!/usr/bin/env bash
# 522: Wiki50 2000-step cross-seed + compute-matched baseline.
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python

D=/data/zetyun/datasets/nanowhale_518_wiki_text_short50.parquet
V=/data/zetyun/eval/520_wiki50_val_split.txt

run_one() {
  local cfg="$1" out="$2" seed="$3" steps="$4"
  echo "=== $out steps=$steps seed=$seed ==="
  $PY scripts/train_pretrain.py --config "$cfg" --output_dir "$out" --seed "$seed"     --train_parquet "$D" --parquet_text_field text     --val_ratio 0.05 --split_seed 2025 --split_method hash     --val_text_out "$V" --split_meta_out /dev/null     --no_bf16 --no_compile --max_steps "$steps"     2>&1 | tee "${out}.log"
}

# ===== Part A: 2000-step cross-seed =====
echo '=== 2000-step seed=2027 ==='
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_522_wiki50_base2000_s2027 2027 2000
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_522_wiki50_t1on2000_s2027 2027 2000

echo '=== 2000-step seed=2048 ==='
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_522_wiki50_base2000_s2048 2048 2000
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_522_wiki50_t1on2000_s2048 2048 2000

# ===== Part B: Compute-matched baseline =====
# T1 2000-step wall-clock for seed2025 = 248.7s (from 521 data)
# Baseline speed on Wiki50 @ 2000 steps = 2000/248.7 ≈ 8.04 steps/s (from 521 base: 248.5s)
# T1 and baseline have nearly identical speed on Wiki50 short text
# cm_steps = round(248.7 * (2000/248.5)) ≈ 2002 → effectively the same as 2000
# T1 at 2000 steps IS already compute-matched to baseline at 2000 steps
echo '=== Compute-matched: T1 and baseline same speed on Wiki50 ==='

echo '522 all done'
