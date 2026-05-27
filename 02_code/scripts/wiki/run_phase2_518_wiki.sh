#!/usr/bin/env bash
# 518: Wiki text PPL experiment — full + short50 + short100.
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

# ===== Full Wiki =====
echo '=== Full Wiki ==='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_518_wiki_text_train.parquet /data/zetyun/eval/518_wiki_full_val_split.txt /data/zetyun/eval/518_wiki_full_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_518_wikitext_base3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_518_wikitext_t1on3000_s2025 2025 $ARGS

# ===== Short Wiki 50 =====
echo '=== Short Wiki 50 ==='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_518_wiki_text_short50.parquet /data/zetyun/eval/518_wiki_short50_val_split.txt /data/zetyun/eval/518_wiki_short50_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_518_wiki50_base3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_518_wiki50_t1on3000_s2025 2025 $ARGS

# ===== Short Wiki 100 =====
echo '=== Short Wiki 100 ==='
ARGS=$(common_args /data/zetyun/datasets/nanowhale_518_wiki_text_short100.parquet /data/zetyun/eval/518_wiki_short100_val_split.txt /data/zetyun/eval/518_wiki_short100_split_meta.json)
run_one configs/phase2_smoke_baseline.yaml /data/zetyun/phase2_518_wiki100_base3000_s2025 2025 $ARGS
run_one configs/phase2_rec_t1_on.yaml /data/zetyun/phase2_518_wiki100_t1on3000_s2025 2025 $ARGS

echo '518 wiki done'
