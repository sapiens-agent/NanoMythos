# Nanowhale T=1 实验代码、Wiki 结果与报告材料

本整理包面向 Nanowhale T=1 recurrent-core 实验的技术复核、结果展示与对外沟通，包含核心代码、实验配置、结果表、图表、Wiki 实验材料和历史报告链。整理目标是让读者从根目录即可理解：实验做了什么、证据在哪里、如何复核、哪些报告可用于进一步说明。

## 一句话概括

Nanowhale T=1 在 FineWeb-Edu 10K cross-seed 实验中，相比 baseline 在 3/3 个随机种子上获得更低 held-out PPL；在 Wiki 50 字短上下文实验中，也观察到更明显的 PPL 改善。

## 推荐阅读顺序

1. `00_README.md` / `README.md`：总览。
2. `01_quickstart/EXPERIMENT_INDEX.md`：实验地图。
3. `06_reports/main/external_communication_brief_zh.md`：正式对外沟通摘要。
4. `06_reports/wiki/WIKI_REPORTS_INDEX.md`：Wiki 实验与报告入口。
5. `06_reports/all_reports/`：完整历史报告链。
6. `02_code/scripts/wiki/`：Wiki 相关运行脚本。

## 核心结果表

| 实验 | Baseline PPL | T=1 PPL | 结论 |
|---|---:|---:|---|
| FineWeb-Edu 10K seed 2025 @ 5K | 156.7 | **147.7** | T=1 更低 |
| FineWeb-Edu 10K seed 2027 @ 5K | 187.7 | **151.4** | T=1 更低 |
| FineWeb-Edu 10K seed 2048 @ 5K | 154.2 | **145.7** | T=1 更低 |
| Wiki 50 characters | 2398 | **1985** | 约 17% 改善 |

## 目录结构

```text
00_README.md                         # 首读说明，与 README.md 内容一致
README.md                            # 项目总览
01_quickstart/                       # 依赖、实验索引、复现步骤
02_code/                             # 核心模型、训练入口、实验脚本；包含 scripts/wiki
03_configs/                          # baseline、T=1、T=2、T=4 等配置
04_results/                          # 已整理结果表
05_figures/                          # 汇报图表
06_reports/                          # 对外摘要、Wiki 报告、完整历史报告链
07_archive_index/                    # 原始清单的整理版
MANIFEST.md                          # 本整理包文件清单
```

## Wiki 相关材料位置

- Wiki 报告索引：`06_reports/wiki/WIKI_REPORTS_INDEX.md`
- Wiki 专项报告：`06_reports/wiki/reports_wiki/`
- 518 阶段 Wiki 机制与 PPL 报告：`06_reports/wiki/reports_518_wiki/`
- Wiki 脚本：`02_code/scripts/wiki/` 与 `06_reports/wiki/scripts_wiki/`
- 完整报告备份：`06_reports/all_reports/reports_wiki/`、`06_reports/all_reports/reports_518/`

## 建议对外表述

英文可写：

> We organized the Nanowhale T=1 recurrent-core experiment as a reproducible code, results, and report bundle. Across FineWeb-Edu 10K cross-seed tests, T=1 consistently reduced held-out perplexity against the baseline, with a stronger supporting signal on short-context Wiki text.

中文可写：

> 我们整理了 Nanowhale T=1 recurrent-core 实验代码、结果表、图表和报告材料。在 FineWeb-Edu 10K 的三种子测试中，T=1 相比 baseline 稳定降低 held-out PPL；在 Wiki 短上下文文本上，改善也更加明显。

## 边界说明

- 当前证据主要是 held-out PPL，不应表述为完整通用能力评测。
- GPT-2 对比是同数据、同步数训练参照，不是与公开预训练权重的终局能力比较。
- 本包是沟通与复核用整理版，不包含完整训练数据和 tokenizer；复现时需要按 `01_quickstart/REPRODUCE.md` 配置本地路径。
