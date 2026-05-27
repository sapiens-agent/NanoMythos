#!/bin/bash
# 519 FineWeb-Edu Recurrent Depth Experiment Setup & Smoke
# Run on internal accelerator server:
#   bash /home/zetyun/nanowhale/reports/519reports/519_setup_and_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/home/zetyun/nanowhale"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime

export NANOWHALE_DEVICE_SELECTOR=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "519 FineWeb-Edu Experiment Setup"
echo "Started at: $(date)"
echo "============================================"

# ============================================================
# Step 1: Download FineWeb-Edu from ModelScope (if needed)
# ============================================================
echo ""
echo "[Step 1] Checking FineWeb-Edu data..."

FEWEB_CACHE="$HOME/.cache/modelscope/hub/datasets/AI-ModelScope___fineweb-edu"
FEWEB_PARQUETS=$(find "$FEWEB_CACHE" -name '*.parquet' 2>/dev/null | wc -l)

if [ "$FEWEB_PARQUETS" -gt 100 ]; then
    echo "  FineWeb-Edu already cached ($FEWEB_PARQUETS parquet files)"
else
    echo "  Downloading FineWeb-Edu from ModelScope (~2.4GB, ~5 min)..."
    python3 -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('AI-ModelScope/fineweb-edu', split='train')
print(f'FineWeb-Edu rows: {len(ds)}')
" 2>&1 | tee "$LOG_DIR/519_fineweb_download_${TIMESTAMP}.log"
    echo "  Download complete."
fi

# ============================================================
# Step 2: Build smoke and pilot parquet datasets
# ============================================================
echo ""
echo "[Step 2] Building parquet datasets..."

SMOKE_PARQUET="/data/zetyun/datasets/nanowhale_519_fineweb_smoke.parquet"
PILOT_PARQUET="/data/zetyun/datasets/nanowhale_519_fineweb_pilot.parquet"

if [ -f "$SMOKE_PARQUET" ]; then
    echo "  Smoke parquet exists: $SMOKE_PARQUET"
else
    echo "  Building smoke parquet (1000 rows)..."
    python3 scripts/build_phase2_lm_parquet.py \
        --source fineweb \
        --max_rows 1000 \
        --min_chars 100 \
        --seed 2025 \
        --fineweb_take 50000 \
        --output "$SMOKE_PARQUET" \
        2>&1 | tee "$LOG_DIR/519_build_smoke_${TIMESTAMP}.log"
    echo "  Smoke parquet built."
fi

if [ -f "$PILOT_PARQUET" ]; then
    echo "  Pilot parquet exists: $PILOT_PARQUET"
else
    echo "  Building pilot parquet (20000 rows)..."
    python3 scripts/build_phase2_lm_parquet.py \
        --source fineweb \
        --max_rows 20000 \
        --min_chars 100 \
        --seed 2025 \
        --fineweb_take 100000 \
        --output "$PILOT_PARQUET" \
        2>&1 | tee "$LOG_DIR/519_build_pilot_${TIMESTAMP}.log"
    echo "  Pilot parquet built."
fi

# Verify datasets
echo ""
echo "  Verifying datasets..."
python3 -c "
import pandas as pd
for label, path in [('Smoke', '$SMOKE_PARQUET'), ('Pilot', '$PILOT_PARQUET')]:
    df = pd.read_parquet(path)
    lens = df['text'].str.len()
    total_chars = int(lens.sum())
    print(f'  {label}: {len(df)} rows, {total_chars:,} chars')
    print(f'    text len: mean={lens.mean():.0f}, median={lens.median():.0f}, p95={lens.quantile(0.95):.0f}')
    print(f'    sample: {df[\"text\"].iloc[0][:200]}')
"

# ============================================================
# Step 3: Create 519 experiment configs
# ============================================================
echo ""
echo "[Step 3] Creating 519 experiment configs..."

mkdir -p configs/519

# Base config template (non-recurrent baseline)
cat > configs/519/519_smoke_base.yaml << 'EOF'
model:
  vocab_size: 129280
  hidden_size: 320
  num_hidden_layers: 8
  num_attention_heads: 8
  num_key_value_heads: 1
  moe_intermediate_size: 640
  n_routed_experts: 4
  n_shared_experts: 1
  num_experts_per_tok: 2
  q_lora_rank: 160
  head_dim: 96
  qk_rope_head_dim: 32
  o_groups: 2
  o_lora_rank: 80
  hc_mult: 4
  hc_sinkhorn_iters: 2
  hc_eps: 1.0e-6
  num_hash_layers: 0
  swiglu_limit: 0.0
  scoring_func: sqrtsoftplus
  routed_scaling_factor: 1.5
  max_position_embeddings: 2048
  rms_norm_eps: 1.0e-6
  rope_theta: 10000.0
  initializer_range: 0.02
  tie_word_embeddings: false
  attention_bias: false
  attention_dropout: 0.0
  compress_ratios: [0, 0, 0, 0, 0, 0, 0, 0, 0]
  hidden_smoothness_lambda: 0.0
  hidden_norm_lambda: 0.0
  recurrent_enabled: false
  recurrent_prelude_layers: 2
  recurrent_core_layers: 4
  recurrent_steps: 1
  recurrent_coda_layers: 2
  recurrent_use_loop_embedding: true
  recurrent_max_steps: 8
training:
  max_seq_length: 512
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 6.0e-4
  weight_decay: 0.1
  adam_beta1: 0.9
  adam_beta2: 0.95
  max_grad_norm: 1.0
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  bf16: false
  max_steps: 300
  save_steps: 300
  logging_steps: 10
  gradient_checkpointing: true
EOF

# Recurrent configs differ only in recurrent_enabled and recurrent_steps
for t in 1 2 4; do
    sed "s/recurrent_enabled: false/recurrent_enabled: true/; s/recurrent_steps: 1/recurrent_steps: $t/" \
        configs/519/519_smoke_base.yaml > "configs/519/519_smoke_t${t}.yaml"
    echo "  Created configs/519/519_smoke_t${t}.yaml"
done
echo "  Created configs/519/519_smoke_base.yaml"

# Pilot configs (2000 steps, same model)
for t in base t1 t2 t4; do
    case $t in
        base) steps_val=1; rec_enabled=false ;;
        t1) steps_val=1; rec_enabled=true ;;
        t2) steps_val=2; rec_enabled=true ;;
        t4) steps_val=4; rec_enabled=true ;;
    esac
    sed "s/max_steps: 300/max_steps: 2000/; s/save_steps: 300/save_steps: 500/; s/recurrent_enabled: false/recurrent_enabled: $rec_enabled/; s/recurrent_steps: 1/recurrent_steps: $steps_val/" \
        configs/519/519_smoke_base.yaml > "configs/519/519_pilot_${t}.yaml"
    echo "  Created configs/519/519_pilot_${t}.yaml"
done

echo "  All configs created."

# ============================================================
# Step 4: Run smoke tests
# ============================================================
echo ""
echo "[Step 4] Running smoke tests (300 steps each)..."
echo "  Runtime device: $(runtime-device-info --query-runtime=name --format=csv,noheader | head -1)"

VAL_TEXT_OUT="/data/zetyun/eval/519_smoke_val_split.txt"
SPLIT_META_OUT="/data/zetyun/eval/519_smoke_split_meta.json"

# Baseline smoke
echo ""
echo "  === Smoke: Baseline ==="
python3 scripts/train_pretrain.py \
    --config configs/519/519_smoke_base.yaml \
    --train_parquet "$SMOKE_PARQUET" \
    --val_ratio 0.05 \
    --split_seed 2025 \
    --val_text_out "$VAL_TEXT_OUT" \
    --split_meta_out "$SPLIT_META_OUT" \
    --no_compile \
    --no_bf16 \
    --seed 2025 \
    2>&1 | tee "$LOG_DIR/519_smoke_base_${TIMESTAMP}.log"

echo "  Baseline smoke complete."

# T=1 smoke
echo ""
echo "  === Smoke: T=1 Recurrent ==="
python3 scripts/train_pretrain.py \
    --config configs/519/519_smoke_t1.yaml \
    --train_parquet "$SMOKE_PARQUET" \
    --val_ratio 0.05 \
    --split_seed 2025 \
    --val_text_out "$VAL_TEXT_OUT" \
    --split_meta_out "$SPLIT_META_OUT" \
    --no_compile \
    --no_bf16 \
    --seed 2025 \
    2>&1 | tee "$LOG_DIR/519_smoke_t1_${TIMESTAMP}.log"

echo "  T=1 smoke complete."

echo ""
echo "============================================"
echo "519 Setup & Smoke Complete at: $(date)"
echo "============================================"
echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "Next: Run pilot experiments with:"
echo "  cd /home/zetyun/nanowhale"
echo "  bash reports/519reports/519_run_pilot.sh"
