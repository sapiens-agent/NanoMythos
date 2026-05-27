# T=1 Bundle 收录清单（2026-05-26 第三轮核对）

## 训练 / 评估脚本

| 路径 | 实验 |
|------|------|
| `scripts/521_fw_cross_seed_5k.py` | FineWeb 10K 5k ★ |
| `scripts/run_t1_fineweb_cross_seed.sh` | 同上入口 |
| `scripts/fineweb/run_phase2_520_train.sh` | LM real wall-match + T=1 1k |
| `scripts/eval_external/520_*`, `521_*` | 520/521 历史训练与 backfill |
| `scripts/wiki/run_phase2_518|520|521|522_*.sh` | Wiki 全链路 ★ |
| `scripts/pilot519/519_run_pilot*.sh` | FineWeb pilot T-depth |
| `scripts/jd_20ng/run_phase2_518*.sh` | JD 中文 3k |
| `scripts/phase2_misc/run_phase2_515|516_*.sh` | 长度 / T-depth 机制 |
| `scripts/train_pretrain.py` | 统一训练入口 |
| `scripts/compute_nll.py` | 单点 PPL |
| `scripts/521_eval_length_binned.py` | 521 分桶 eval |
| `scripts/plot_521_*.py` | 521 图表 |

## 配置 YAML

- `configs/519/*` — pilot + smoke（base/t1/t2/t4）
- `configs/519_pilot_t1.yaml` 等 — 见 `configs/519/`（勿用根目录重复文件）
- `configs/phase2_rec_t{1,2,4}_on.yaml`, `phase2_smoke_baseline.yaml`, `phase2_baseline_L5.yaml`

## 报告与结果

- FineWeb：`docs/521_*`, `results/521_fineweb10k_cross_seed.csv`, `figures/521/*`
- Wiki：`docs/reports_wiki/*`, `docs/reports_518/*`
- JD/20ng：`docs/reports_514/*`
- 长度机制：`docs/reports_515/*`, `reports_518` 短文本摘要
- 杂项：`docs/reports_misc/*`（516 strict val、518 real LM、512 设计）

## 明确未打入包（仅索引）

- **526 Coconut-light**（与 T=1 正交，见主仓库 `reports/526reports/`）
- **GPT-2 训练脚本** `/data/zetyun/scripts/520_gpt2_*.py`（外部对照，见 `521_gpt2_small_standard_baseline.md`）
- **50K FineWeb 主训**（521 已取消；有 `521_fineweb50k_scaleup.md` 文档无新训）
- **完整 `tokenizer/`** — 复现需在 nanowhale 仓库根目录

## 复现前置

见 `docs/DEPENDENCIES.md`。
