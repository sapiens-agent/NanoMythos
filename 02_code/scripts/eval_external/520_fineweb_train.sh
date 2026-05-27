#!/bin/bash
# Step C: FineWeb-Edu baseline vs T=1 (2000 steps, seed 2025)
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=0
cd /home/zetyun/nanowhale

FW10K=/data/zetyun/datasets/fineweb_edu_10k.parquet
[ -f "$FW10K" ] || { echo "ERROR: $FW10K not found"; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=/data/zetyun/eval/520/logs
mkdir -p $LOG_DIR

echo "============================================"
echo "Step C: FineWeb-Edu baseline vs T=1 (2000 steps)"
echo "Data: $FW10K"
echo "Started: $(date)"
echo "============================================"

# Baseline
echo ""
echo "=== FineWeb Baseline 2000 steps ==="
python3 scripts/train_pretrain.py \
    --config configs/519/519_pilot_base.yaml \
    --train_parquet "$FW10K" \
    --val_ratio 0.05 --split_seed 2025 \
    --val_text_out /data/zetyun/eval/520/val_fw_base_s2025.txt \
    --split_meta_out /data/zetyun/eval/520/split_fw_base_s2025.json \
    --no_compile --no_bf16 --seed 2025 --max_steps 2000 \
    --output_dir /data/zetyun/phase2_520_fineweb_base_s2025 \
    2>&1 | tee "$LOG_DIR/520_fineweb_base_2000_${TIMESTAMP}.log"

echo "  Baseline done: $(date)"

# T=1
echo ""
echo "=== FineWeb T=1 2000 steps ==="
python3 scripts/train_pretrain.py \
    --config configs/519/519_pilot_t1.yaml \
    --train_parquet "$FW10K" \
    --val_ratio 0.05 --split_seed 2025 \
    --val_text_out /data/zetyun/eval/520/val_fw_t1_s2025.txt \
    --split_meta_out /data/zetyun/eval/520/split_fw_t1_s2025.json \
    --no_compile --no_bf16 --seed 2025 --max_steps 2000 \
    --output_dir /data/zetyun/phase2_520_fineweb_t1_s2025 \
    2>&1 | tee "$LOG_DIR/520_fineweb_t1_2000_${TIMESTAMP}.log"

echo "  T=1 done: $(date)"

# Held-out PPL on both
echo ""
echo "=== Held-out PPL eval ==="
VAL_TEXT=/data/zetyun/eval/520/val_fw_base_s2025.txt
for config in base t1; do
  CKPT_DIR=/data/zetyun/phase2_520_fineweb_${config}_s2025
  for ckpt in checkpoint-2000 final; do
    [ -d "$CKPT_DIR/$ckpt" ] || continue
    JSON=/data/zetyun/eval/520/520_fw_${config}_${ckpt}_ppl.json
    python3 -c "
import sys; sys.path.insert(0, \"/home/zetyun/nanowhale\")
import torch, json
from transformers import PreTrainedTokenizerFast
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM
tok = PreTrainedTokenizerFast.from_pretrained(\"$CKPT_DIR/$ckpt\")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = DeepseekV4ForCausalLM.from_pretrained(\"$CKPT_DIR/$ckpt\", torch_dtype=torch.float32)
model = model.to(\"cuda\"); model.eval()
with open(\"$VAL_TEXT\") as f: texts = [l.strip() for l in f if l.strip()]
total_nll = 0.0; total_toks = 0
with torch.no_grad():
    for text in texts:
        ids = tok.encode(text, return_tensors=\"pt\").to(\"cuda\")
        if ids.shape[1] < 2: continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n
        total_toks += n
mean_nll = total_nll / max(total_toks, 1)
ppl = float(torch.exp(torch.tensor(mean_nll)).item())
r = {\"config\":\"$config\",\"ckpt\":\"$ckpt\",\"val_nll\":round(mean_nll,4),\"val_ppl\":round(ppl,1)}
with open(\"$JSON\",\"w\") as f: json.dump(r,f)
print(json.dumps(r))
" | grep "val_ppl"
  done
done

echo ""
echo "============================================"
echo "Step C complete: $(date)"
echo "============================================"
