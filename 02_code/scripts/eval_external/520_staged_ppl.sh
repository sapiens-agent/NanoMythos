#!/bin/bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=0
VAL_TEXT=/data/zetyun/eval/519_pilot_base_val.txt
EVAL_DIR=/data/zetyun/eval/520
mkdir -p $EVAL_DIR

for config in base t1; do
  CKPT_DIR=/data/zetyun/phase2_520_long_${config}_s2025
  for ckpt in checkpoint-4000 checkpoint-4500 checkpoint-5000 final; do
    [ -d "$CKPT_DIR/$ckpt" ] || continue
    JSON=$EVAL_DIR/520_staged_${config}_${ckpt}_ppl.json
    echo "Eval: $config $ckpt"
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
" 2>&1 | grep "val_ppl"
  done
done
echo "STAGED PPL DONE"
