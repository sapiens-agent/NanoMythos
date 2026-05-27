#!/usr/bin/env bash
# 520: Wiki 50 deep validation — cross-seed + T-depth + 10000-step.
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python

DATASET=/data/zetyun/datasets/nanowhale_518_wiki_text_short50.parquet

common_args() {
  local steps="${1:-3000}"
  echo --train_parquet "$DATASET" --parquet_text_field text     --val_ratio 0.05 --split_seed 2025 --split_method hash     --val_text_out /data/zetyun/eval/520_wiki50_val_split.txt     --split_meta_out /data/zetyun/eval/520_wiki50_split_meta.json     --no_bf16 --no_compile --collect_diagnostics --max_steps "$steps"
}

run_one() {
  local cfg="$1" out="$2" seed="$3"; shift 3
  echo "=== $out seed=$seed ==="
  $PY scripts/train_pretrain.py --config "$cfg" --output_dir "$out" --seed "$seed" "$@" 2>&1 | tee "${out}.log"
}

# ===== Part A: Wiki 50 seed 2027 =====
echo '=== Wiki50 seed=2027 ==='
ARGS=$(common_args 3000)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_520_wiki50_base3000_s2027 2027 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_520_wiki50_t1on3000_s2027 2027 $ARGS

# ===== Part B: Wiki 50 seed 2048 =====
echo '=== Wiki50 seed=2048 ==='
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_520_wiki50_base3000_s2048 2048 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_520_wiki50_t1on3000_s2048 2048 $ARGS

# ===== Part C: Wiki 50 T-depth (T2, T4) =====
echo '=== Wiki50 T-depth seed=2025 ==='
run_one configs/phase2_rec_t2_on.yaml /data/zetyun/phase2_520_wiki50_t2on3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t4_on.yaml /data/zetyun/phase2_520_wiki50_t4on3000_s2025 2025 $ARGS

# ===== Part D: Wiki 50 10000-step =====
echo '=== Wiki50 10000-step seed=2025 ==='
ARGS10K=$(common_args 10000)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_520_wiki50_base10000_s2025 2025 $ARGS10K
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_520_wiki50_t1on10000_s2025 2025 $ARGS10K

echo '520 all done'
