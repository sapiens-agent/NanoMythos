#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=6
cd /home/zetyun/nanowhale

LOG_DIR=/data/zetyun/eval/521/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

echo "521 cross-seed: backfill s2025 eval"
python /data/zetyun/scripts/521_backfill_s2025_eval.py 2>&1 | tee "$LOG_DIR/521_backfill_${TS}.log"

echo "521 cross-seed: eval legacy s2027 base (520 ckpt)"
python -c "
import sys; sys.path.insert(0,'/data/zetyun/scripts')
# eval only 520 fw_cs base s2027 at 2k if 5k train not done yet
import importlib.util
spec = importlib.util.spec_from_file_location('cs', '/data/zetyun/scripts/521_fw_cross_seed_5k.py')
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)
cs.eval_only_legacy('/data/zetyun/phase2_520_fw_cs_base_s2027', 'base', 2027)
" 2>&1 | tee -a "$LOG_DIR/521_eval_cs2027_${TS}.log"

echo "521 cross-seed: train seeds 2027,2048"
python /data/zetyun/scripts/521_fw_cross_seed_5k.py --mode all --seeds 2027,2048 \
  2>&1 | tee "$LOG_DIR/521_train_cs_${TS}.log"

echo "521 cross-seed done"
