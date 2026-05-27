#!/usr/bin/env bash
# Stage 518: 3000-step baseline / T=1 / T=2 on Chinese JD parquet (accelerator 0).
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python

COMMON=(
  --train_parquet /data/zetyun/datasets/nanowhale_phase1_train.parquet
  --parquet_text_field text
  --val_ratio 0.05
  --split_seed 2025
  --split_method hash
  --val_text_out /data/zetyun/eval/518_jd3000_val_split.txt
  --split_meta_out /data/zetyun/eval/518_jd3000_split_meta.json
  --no_bf16
  --no_compile
  --collect_diagnostics
  --max_steps 3000
)

run_one() {
  local cfg="$1" out="$2" seed="$3"
  echo "=== $out seed=$seed ==="
  $PY scripts/train_pretrain.py --config "$cfg" --output_dir "$out" --seed "$seed"     "${COMMON[@]}" 2>&1 | tee "${out}.log"
}

SEED="${1:-2025}"
run_one configs/phase2_smoke_baseline.yaml "/data/zetyun/phase2_518_base3000_s${SEED}" "$SEED"
run_one configs/phase2_rec_t1_on.yaml "/data/zetyun/phase2_518_t1on3000_s${SEED}" "$SEED"
run_one configs/phase2_rec_t2_on.yaml "/data/zetyun/phase2_518_t2on3000_s${SEED}" "$SEED"
echo "518 JD 3000step train done seed=$SEED"
