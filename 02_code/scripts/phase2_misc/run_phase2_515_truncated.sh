#!/usr/bin/env bash
# Stage 515: truncated 20ng (100 chars) — baseline vs T=1, 3000 steps.
set -euo pipefail
cd /home/zetyun/nanowhale
export NANOWHALE_DEVICE_SELECTOR=0
export HF_HOME=/data/zetyun/.cache/huggingface
export TMPDIR=/data/zetyun/tmp
PY=/home/zetyun/miniconda3/envs/nanowhale_runtime/bin/python

COMMON=(
  --train_parquet /data/zetyun/datasets/nanowhale_515_truncated_20ng.parquet
  --parquet_text_field text
  --val_ratio 0.05
  --split_seed 2025
  --split_method hash
  --val_text_out /data/zetyun/eval/515_truncated_val_split.txt
  --split_meta_out /data/zetyun/eval/515_truncated_split_meta.json
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
run_one configs/phase2_smoke_baseline.yaml "/data/zetyun/phase2_515_trunc_base3000_s${SEED}" "$SEED"
run_one configs/phase2_rec_t1_on.yaml "/data/zetyun/phase2_515_trunc_t1on3000_s${SEED}" "$SEED"
echo '515 truncated train done seed='$SEED
