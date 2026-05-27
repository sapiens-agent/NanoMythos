# T=1 实验复现说明

## 环境

```bash
conda activate nanowhale_runtime
export NANOWHALE_DEVICE_SELECTOR=0
cd /path/to/nanowhale   # 需完整仓库（tokenizer、modeling 与 train_pretrain 同目录）
```

---

## 1. FineWeb-Edu 10K cross-seed（521，主核心展示实验）

数据：`/data/zetyun/datasets/fineweb_edu_10k.parquet`  
Val：`/data/zetyun/eval/520/val_fw_base_s2025.txt`

```bash
python scripts/521_fw_cross_seed_5k.py --mode all --seeds 2027,2048
python scripts/521_fw_cross_seed_5k.py --mode eval --seeds 2025,2027,2048
bash scripts/run_t1_fineweb_cross_seed.sh
```

配置：`configs/519/519_pilot_t1.yaml` vs `519_pilot_base.yaml`

---

## 2. Wikipedia 文本 PPL（518 → 520 → 521 → 522）

数据（已构建）：`/data/zetyun/datasets/nanowhale_518_wiki_text_short50.parquet`  
Val（521/520 共用）：`/data/zetyun/eval/520_wiki50_val_split.txt`

| 步骤 | 脚本（本包 `scripts/wiki/`） | 目的 |
|------|---------------------------|------|
| 518 初探 | `run_phase2_518_wiki.sh` | full / 50 / 100 三档，3k step |
| 520 加深 | `run_phase2_520_wiki50_deep.sh` | Wiki50 跨种子 + T2/T4 + 10k |
| 521 步数 | `run_phase2_521_earlystop.sh` | 1k/2k/5k，定 2k 最佳 |
| 522 确认 | `run_phase2_522.sh` | 2k step，种子 2027/2048 |

配置：`configs/phase2_rec_t1_on.yaml` vs `phase2_smoke_baseline.yaml`

从 HF 构建 WikiText（可选，需网络）：

```bash
python scripts/build_phase2_lm_parquet.py --source wikitext --max_rows 50000
```

原始 **wiki_demo.txt** 管线见 `docs/reports_wiki/518_nanowhale_wiki_text_data_build_20260515.md`。

---

## 3. 其它 T=1 线（脚本在完整仓库，见 EXPERIMENT_INDEX）

- **JD 中文：** `reports/514report` + `run_phase2_518.sh` / `519_run_pilot.sh`
- **长度机制 515：** `run_phase2_515_truncated.sh` 等
- **20ng 519：** `reports/519reports/519_run_pilot.sh`

---

## 核心代码

| 文件 | 作用 |
|------|------|
| `core/modeling_deepseek_v4.py` | recurrent prelude / core×T / coda |
| `core/configuration_deepseek_v4.py` | `recurrent_enabled`, `recurrent_steps` |
| `scripts/train_pretrain.py` | 所有 `run_phase2_*` 的底层入口 |
