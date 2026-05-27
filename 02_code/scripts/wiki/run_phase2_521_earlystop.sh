#!/usr/bin/env bash
# 521: Wiki 50 early-stopping sweep — 1000/2000/5000 steps.
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

for STEPS in 1000 2000 5000; do
  run_one configs/phase2_smoke_baseline.yaml "/data/zetyun/phase2_521_wiki50_base${STEPS}_s2025" 2025 $STEPS
  run_one configs/phase2_rec_t1_on.yaml "/data/zetyun/phase2_521_wiki50_t1on${STEPS}_s2025" 2025 $STEPS
done
echo '521 all done'
